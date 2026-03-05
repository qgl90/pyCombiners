"""N-body track combination pipeline."""

from __future__ import annotations

__author__ = "Renato Quagliani <rquaglia@cern.ch>"


# ---------------------------------------------------------------------------
# SoA vectorized combination pipeline
# ---------------------------------------------------------------------------


def combine(
    track_pools,
    pvs,
    track_cuts=None,
    combination_cuts=None,
    vertex_cuts=None,
    use_timing=True,
    max_composite_pv_time_residual=0.05,
    max_composite_pv_time_chi2=None,
):
    """Run the n-body combination pipeline.

    track_pools determines the combinatorics: same object twice gives
    C(n,2) pairs, distinct objects give the cartesian product with
    shared-track removal. Each pool must have a "mass" field from
    set_tracks_pid(). Returns a jagged Container (events, var_candidates).
    """
    import awkward as ak
    import numpy as np

    from .models import apply_cuts
    from .physics import (
        composite_pv_association,
        doca_2body,
        doca_nbody,
        invariant_mass,
        lorentz_sum,
        pairwise_time_chi2 as _pair_tchi2,
        pt_eta,
        vertex_fit_xyz,
        vertex_time_fit,
        flatten_daughters,
        make_combinations,
        unflatten_array,
    )

    n_body = len(track_pools)
    n_events = len(pvs["x"])

    # 0. Validate: every pool must carry a "mass" field
    for i, pool in enumerate(track_pools):
        if "mass" not in pool:
            raise ValueError(
                f"track_pools[{i}] is missing a 'mass' field.  "
                "Use set_tracks_pid(tracks, particle_id) to assign mass hypotheses."
            )

    # 1. Track preselection — apply to each unique pool once
    if track_cuts:
        seen: dict[int, dict] = {}
        new_pools = []
        for pool in track_pools:
            pid = id(pool)
            if pid not in seen:
                seen[pid] = apply_cuts(pool, track_cuts)
            new_pools.append(seen[pid])
        track_pools = new_pools

    # 2. Inject _pool_index into each unique pool, then generate combinations
    _seen_pools: set[int] = set()
    for pool in track_pools:
        pid = id(pool)
        if pid not in _seen_pools:
            pool["_pool_index"] = ak.local_index(pool["x"], axis=1)
            _seen_pools.add(pid)

    daughters = make_combinations(track_pools)

    # 3. Flatten to numpy (core physics fields, common to all daughters)
    core_fields = [
        "x",
        "y",
        "z",
        "tx",
        "ty",
        "p",
        "mass",
        "pid",
        "charge",
        "time",
        "sigma_time",
        "track_id",
        "_pool_index",
        "cov_0_0",
        "cov_1_0",
        "cov_1_1",
        "cov_2_0",
        "cov_2_1",
        "cov_2_2",
        "cov_3_0",
        "cov_3_1",
        "cov_3_2",
        "cov_3_3",
    ]
    for extra in ("min_ip", "min_ip_chi2", "best_pv_index"):
        if all(extra in daughter for daughter in daughters):
            core_fields.append(extra)
    avail = [f for f in core_fields if all(f in daughter for daughter in daughters)]
    flat_daughters, counts = flatten_daughters(daughters, fields=avail)
    counts_np = np.asarray(counts)
    N_total = int(counts_np.sum())

    if N_total == 0:
        return _empty_candidates(n_body, n_events, track_pools)

    # 3b. Collect per-daughter source indices for overlap removal
    all_same_pool = all(track_pools[i] is track_pools[0] for i in range(1, n_body))
    source_cols = None
    if not all_same_pool:
        source_cols = []
        for k, daughter in enumerate(daughters):
            daughter_src = []
            daughter_idx_keys = sorted(
                key
                for key in daughter
                if key.startswith("daughter") and key.endswith("_track_id")
            )
            if daughter_idx_keys:
                for key in daughter_idx_keys:
                    daughter_src.append(ak.to_numpy(ak.flatten(daughter[key], axis=1)))
            elif "track_id" in daughter:
                daughter_src.append(
                    ak.to_numpy(ak.flatten(daughter["track_id"], axis=1))
                )
            source_cols.append(daughter_src)

    # 4. Build (N, n_body) arrays
    def _stack(field):
        return np.column_stack([fl[field] for fl in flat_daughters])

    x = _stack("x")
    y = _stack("y")
    z = _stack("z")
    tx = _stack("tx")
    ty = _stack("ty")
    p_arr = _stack("p")
    mass_arr = _stack("mass")
    charge = _stack("charge")
    time_arr = _stack("time")
    sigma_t = _stack("sigma_time")
    tidx = _stack("track_id") if "track_id" in flat_daughters[0] else None
    pidx = _stack("_pool_index") if "_pool_index" in flat_daughters[0] else None
    pid_arr = _stack("pid") if "pid" in flat_daughters[0] else None

    cov_fields = {}
    for ck in (
        "cov_0_0",
        "cov_1_0",
        "cov_1_1",
        "cov_2_0",
        "cov_2_1",
        "cov_2_2",
        "cov_3_0",
        "cov_3_1",
        "cov_3_2",
        "cov_3_3",
    ):
        if ck in flat_daughters[0]:
            cov_fields[ck] = _stack(ck)

    event_idx = np.repeat(np.arange(n_events), counts_np)

    has_min_ip = all("min_ip" in fl for fl in flat_daughters)
    has_best_pv = all("best_pv_index" in fl for fl in flat_daughters)
    if has_min_ip:
        daughter_min_ip = _stack("min_ip")
        daughter_min_ip_chi2 = _stack("min_ip_chi2")
    if has_best_pv:
        daughter_best_pv_index = _stack("best_pv_index")

    # 5. Vertex fit
    vertex_xyz, spatial_chi2, vertex_cov = vertex_fit_xyz(
        x,
        y,
        z,
        tx,
        ty,
        cov_fields,
    )

    # 6. DOCA
    if n_body == 2:
        doca_vals = {"doca12": doca_2body(x, y, z, tx, ty)}
    else:
        doca_vals = doca_nbody(x, y, z, tx, ty, n_body)
    max_doca_arr = np.max(
        np.column_stack(list(doca_vals.values())),
        axis=1,
    )

    # 7. Overlap removal (structural, not a cut)
    geo_mask = np.ones(N_total, dtype=bool)
    if source_cols is not None:
        overlap = np.zeros(N_total, dtype=bool)
        for i in range(n_body):
            for j in range(i + 1, n_body):
                if track_pools[i] is track_pools[j]:
                    continue  # same pool: ak.combinations already de-duped
                for src_i in source_cols[i]:
                    for src_j in source_cols[j]:
                        overlap |= src_i == src_j
        geo_mask &= ~overlap

    total_charge = np.sum(charge.astype(int), axis=1)

    # 8. Build combination dict and apply combination_cuts
    comb = {}
    comb["vertex_x"] = vertex_xyz[:, 0]
    comb["vertex_y"] = vertex_xyz[:, 1]
    comb["vertex_z"] = vertex_xyz[:, 2]
    comb["spatial_chi2"] = spatial_chi2
    comb["max_doca"] = max_doca_arr
    comb["total_charge"] = total_charge.astype(np.float64)
    for dk, dv in doca_vals.items():
        comb[dk] = dv
    for k in range(n_body):
        comb[f"daughter{k}_charge"] = charge[:, k]
        comb[f"daughter{k}_mass"] = mass_arr[:, k]
        comb[f"daughter{k}_p"] = p_arr[:, k]
        norm_k = np.sqrt(1 + tx[:, k] ** 2 + ty[:, k] ** 2)
        comb[f"daughter{k}_pt"] = (
            p_arr[:, k] * np.sqrt(tx[:, k] ** 2 + ty[:, k] ** 2) / norm_k
        )
        if tidx is not None:
            comb[f"daughter{k}_track_id"] = tidx[:, k]
        if pidx is not None:
            comb[f"daughter{k}_pool_index"] = pidx[:, k]
        if pid_arr is not None:
            comb[f"daughter{k}_pid"] = pid_arr[:, k]
        if has_min_ip:
            comb[f"daughter{k}_min_ip"] = daughter_min_ip[:, k]
            comb[f"daughter{k}_min_ip_chi2"] = daughter_min_ip_chi2[:, k]
        if has_best_pv:
            comb[f"daughter{k}_best_pv_index"] = daughter_best_pv_index[:, k]

    if combination_cuts:
        for cfn in combination_cuts:
            geo_mask &= cfn(comb)

    idx_geo = np.where(geo_mask)[0]
    if len(idx_geo) == 0:
        return _empty_candidates(n_body, n_events, track_pools)

    # Apply combination filter to raw arrays
    _g = idx_geo
    x_g = x[_g]
    y_g = y[_g]
    z_g = z[_g]
    tx_g = tx[_g]
    ty_g = ty[_g]
    p_g = p_arr[_g]
    t_g = time_arr[_g]
    st_g = sigma_t[_g]
    mass_g = mass_arr[_g]
    eidx_g = event_idx[_g]
    vxyz_g = vertex_xyz[_g]
    vcov_g = vertex_cov[_g]

    # Filter the combination dict
    out = {k: v[_g] for k, v in comb.items()}
    out["vertex_chi2"] = out.pop("spatial_chi2")  # rename for output

    vcov_sel = vcov_g
    for i in range(3):
        for j in range(i + 1):
            out[f"vertex_cov_{i}_{j}"] = vcov_sel[:, i, j]

    N_geo = len(idx_geo)

    # 9. Timing + kinematics
    if use_timing:
        vt, stt, tchi2 = vertex_time_fit(
            t_g,
            st_g,
            z_g,
            tx_g,
            ty_g,
            p_g,
            mass_g,
            vxyz_g[:, 2],
        )
        ptchi2 = _pair_tchi2(
            t_g,
            st_g,
            z_g,
            tx_g,
            ty_g,
            p_g,
            mass_g,
            vxyz_g[:, 2],
        )
    else:
        vt = np.zeros(N_geo)
        stt = np.zeros(N_geo)
        tchi2 = np.zeros(N_geo)
        ptchi2 = np.zeros(N_geo)

    spx, spy, spz, se = lorentz_sum(p_g, tx_g, ty_g, mass_g)
    mass_v = invariant_mass(spx, spy, spz, se)
    pt_v, eta_v = pt_eta(spx, spy, spz)

    out["vertex_time"] = vt
    out["sigma_time"] = stt
    out["vertex_time_chi2"] = tchi2
    out["pair_time_chi2"] = ptchi2
    out["px"] = spx
    out["py"] = spy
    out["pz"] = spz
    out["energy"] = se
    out["mass"] = mass_v
    out["pt"] = pt_v
    out["eta"] = eta_v

    # 10. Apply vertex_cuts
    if vertex_cuts:
        vmask = np.ones(N_geo, dtype=bool)
        for cfn in vertex_cuts:
            vmask &= cfn(out)
        if not np.all(vmask):
            keep = np.where(vmask)[0]
            out = {k: v[keep] for k, v in out.items()}
            eidx_g = eidx_g[keep]
            vcov_sel = vcov_sel[keep]

    # 11. PV association
    pv_out_counts = np.bincount(eidx_g, minlength=n_events)
    pv_assoc = composite_pv_association(
        vertex_xyz=np.column_stack([out["vertex_x"], out["vertex_y"], out["vertex_z"]]),
        vertex_cov=vcov_sel,
        vertex_time=out["vertex_time"],
        sigma_time=out["sigma_time"],
        px=out["px"],
        py=out["py"],
        pz=out["pz"],
        energy=out["energy"],
        pvs=pvs,
        counts=pv_out_counts,
        max_time_residual=max_composite_pv_time_residual if use_timing else None,
        max_time_chi2=max_composite_pv_time_chi2 if use_timing else None,
    )
    out.update(pv_assoc)

    # 12. Unflatten to jagged
    final_counts = np.bincount(eidx_g, minlength=n_events)
    result = {k: unflatten_array(v, final_counts) for k, v in out.items()}

    # 13. Attach pool references (non-array metadata)
    result["_daughter_pools"] = track_pools

    # 14. Track-compatible fields for staged decays
    _add_track_fields(result)

    # 15. MC truth propagation for hierarchical decays
    _propagate_mc_truth(result)

    return result


def _propagate_mc_truth(result):
    """Propagate MC truth from daughter pools to combined candidates.

    Finds the common MC ancestor across all daughters; sets mc_truth=1
    for signal, 0 otherwise. The remaining ancestor chain (above the
    common mother) is stored in mc_ancestor_pids/keys.
    """
    import awkward as ak
    import numpy as np

    from .models import infer_n_body

    pools = result["_daughter_pools"]
    n_body = infer_n_body(result)
    n_events = len(result["vertex_x"])

    # Guard: skip if any pool lacks MC ancestry fields
    mc_required = {
        "mc_truth",
        "mc_pid",
        "mc_key",
        "mc_pv_key",
        "mc_fromsignal",
        "mc_ancestor_pids",
        "mc_ancestor_keys",
    }
    for pool in pools:
        if not mc_required.issubset(pool.keys()):
            return

    cand_counts = ak.to_numpy(ak.num(result["vertex_x"]))
    N_total = int(cand_counts.sum())
    evt_per_cand = np.repeat(np.arange(n_events), cand_counts)

    # --- Step 1: Gather per-daughter MC fields using offset indexing ---
    d_mc_truth = []
    d_mc_pid = []
    d_mc_key = []
    d_mc_pv_key = []
    d_mc_fromsignal = []
    d_anc_pids = []  # singly-jagged (N_total, var_ancestors)
    d_anc_keys = []

    for k in range(n_body):
        pool = pools[k]
        pool_idx = result[f"daughter{k}_pool_index"]
        idx_flat = ak.to_numpy(ak.flatten(pool_idx))

        pool_counts = ak.to_numpy(ak.num(pool["x"]))
        pool_offsets = np.zeros(len(pool_counts) + 1, dtype=np.int64)
        np.cumsum(pool_counts, out=pool_offsets[1:])

        global_idx = pool_offsets[evt_per_cand] + idx_flat

        # Scalar MC fields
        d_mc_truth.append(np.asarray(ak.flatten(pool["mc_truth"]))[global_idx])
        d_mc_pid.append(np.asarray(ak.flatten(pool["mc_pid"]))[global_idx])
        d_mc_key.append(np.asarray(ak.flatten(pool["mc_key"]))[global_idx])
        d_mc_pv_key.append(np.asarray(ak.flatten(pool["mc_pv_key"]))[global_idx])
        d_mc_fromsignal.append(
            np.asarray(ak.flatten(pool["mc_fromsignal"]))[global_idx]
        )

        # Jagged ancestor fields: (events, tracks, ancestors) → (total_tracks, ancestors)
        flat_anc_pids = ak.flatten(pool["mc_ancestor_pids"], axis=1)
        flat_anc_keys = ak.flatten(pool["mc_ancestor_keys"], axis=1)
        d_anc_pids.append(flat_anc_pids[global_idx])
        d_anc_keys.append(flat_anc_keys[global_idx])

    # --- Step 2: Find common ancestors via awkward broadcasting ---
    # Use daughter0's ancestor chain as reference; ancestors are ordered
    # most-immediate-first, so argmax finds the direct common mother.
    d0_keys = d_anc_keys[0]  # (N_total, var_ancestors)
    d0_pids = d_anc_pids[0]

    # Mask: which of daughter0's ancestors are shared by ALL other daughters
    in_common = ak.ones_like(d0_keys, dtype=bool)
    for k in range(1, n_body):
        # (N, var_d0, 1) == (N, 1, var_dk) → (N, var_d0, var_dk)
        match_k = d0_keys[:, :, np.newaxis] == d_anc_keys[k][:, np.newaxis, :]
        in_common = in_common & ak.any(match_k, axis=-1)

    has_common = ak.to_numpy(ak.any(in_common, axis=-1))  # (N_total,)
    first_idx = ak.to_numpy(ak.fill_none(ak.argmax(in_common, axis=-1), 0))

    # --- Step 3: Extract common ancestor PID/key via flat offset indexing ---
    d0_counts = ak.to_numpy(ak.num(d0_keys))
    d0_offsets = np.zeros(N_total + 1, dtype=np.int64)
    np.cumsum(d0_counts, out=d0_offsets[1:])

    flat_d0_pids = np.asarray(ak.flatten(d0_pids))
    flat_d0_keys = np.asarray(ak.flatten(d0_keys))

    if len(flat_d0_pids) > 0:
        safe_idx = np.where(has_common, d0_offsets[:-1] + first_idx, 0)
        common_pid = np.where(has_common, flat_d0_pids[safe_idx], 0)
        common_key = np.where(has_common, flat_d0_keys[safe_idx], -1)
    else:
        common_pid = np.zeros(N_total, dtype=np.int64)
        common_key = np.full(N_total, -1, dtype=np.int64)

    # --- Step 4: Build remaining ancestor chain (beyond common mother) ---
    remaining_count = np.where(has_common, d0_counts - first_idx - 1, 0).astype(
        np.int64
    )
    remaining_count = np.maximum(remaining_count, 0)
    total_remaining = int(remaining_count.sum())

    if total_remaining > 0 and len(flat_d0_pids) > 0:
        remaining_start = d0_offsets[:-1] + first_idx + 1
        rem_offsets = np.zeros(N_total + 1, dtype=np.int64)
        np.cumsum(remaining_count, out=rem_offsets[1:])

        cand_per_rem = np.repeat(np.arange(N_total), remaining_count)
        local_within = (
            np.arange(total_remaining, dtype=np.int64) - rem_offsets[cand_per_rem]
        )
        flat_rem_idx = remaining_start[cand_per_rem] + local_within

        new_anc_pids = ak.unflatten(flat_d0_pids[flat_rem_idx], remaining_count)
        new_anc_keys = ak.unflatten(flat_d0_keys[flat_rem_idx], remaining_count)
    else:
        dtype = flat_d0_pids.dtype if len(flat_d0_pids) > 0 else np.int64
        new_anc_pids = ak.unflatten(np.array([], dtype=dtype), remaining_count)
        new_anc_keys = ak.unflatten(np.array([], dtype=dtype), remaining_count)

    # --- Step 5: Assemble scalar MC fields ---
    mc_truth = np.where(has_common, 1, 0)
    mc_pid = common_pid
    mc_key = np.where(has_common, common_key, -1)
    mc_pv_key = np.where(has_common, d_mc_pv_key[0], -1)

    all_fromsignal = np.ones(N_total, dtype=bool)
    for k in range(n_body):
        all_fromsignal &= d_mc_fromsignal[k] == 1
    mc_fromsignal = np.where(has_common & all_fromsignal, 1, 0)

    # --- Step 6: Unflatten to (events, candidates) and assign ---
    result["mc_truth"] = ak.unflatten(mc_truth, cand_counts)
    result["mc_pid"] = ak.unflatten(mc_pid, cand_counts)
    result["mc_key"] = ak.unflatten(mc_key, cand_counts)
    result["mc_pv_key"] = ak.unflatten(mc_pv_key, cand_counts)
    result["mc_fromsignal"] = ak.unflatten(mc_fromsignal, cand_counts)
    # Jagged: (N_total, var_ancestors) → (events, candidates, var_ancestors)
    result["mc_ancestor_pids"] = ak.unflatten(new_anc_pids, cand_counts)
    result["mc_ancestor_keys"] = ak.unflatten(new_anc_keys, cand_counts)


def _add_track_fields(result):
    """Add track-compatible fields for staged (hierarchical) decays."""
    import awkward as ak
    import numpy as np

    # Position: vertex → track reference point
    result["x"] = result["vertex_x"]
    result["y"] = result["vertex_y"]
    result["z"] = result["vertex_z"]

    # Slopes and momentum magnitude from 4-momentum
    safe_pz = ak.where(np.abs(result["pz"]) > 1e-12, result["pz"], 1e-12)
    result["tx"] = result["px"] / safe_pz
    result["ty"] = result["py"] / safe_pz
    result["p"] = (result["px"] ** 2 + result["py"] ** 2 + result["pz"] ** 2) ** 0.5

    # Charge and timing
    result["charge"] = result["total_charge"]
    result["time"] = result["vertex_time"]
    # sigma_time already exists from vertex time fit

    # Track identity for overlap removal in next-level combine
    result["track_id"] = ak.local_index(result["vertex_x"], axis=1)

    # Covariance: spatial block from vertex fit, slope block ≈ 0
    result["cov_0_0"] = result["vertex_cov_0_0"]
    result["cov_1_0"] = result["vertex_cov_1_0"]
    result["cov_1_1"] = result["vertex_cov_1_1"]
    zero = result["vertex_x"] * 0
    eps = zero + 1e-6
    result["cov_2_0"] = zero
    result["cov_2_1"] = zero
    result["cov_2_2"] = eps
    result["cov_3_0"] = zero
    result["cov_3_1"] = zero
    result["cov_3_2"] = zero
    result["cov_3_3"] = eps


def _empty_candidates(n_body, n_events, track_pools=None):
    """Return an empty candidate container."""
    import awkward as ak
    import numpy as np

    empty = ak.Array([[] for _ in range(n_events)])
    fields = [
        "vertex_x",
        "vertex_y",
        "vertex_z",
        "vertex_chi2",
        "max_doca",
        "total_charge",
        "vertex_time",
        "sigma_time",
        "vertex_time_chi2",
        "pair_time_chi2",
        "px",
        "py",
        "pz",
        "energy",
        "mass",
        "pt",
        "eta",
        "composite_ip",
        "composite_ip_chi2",
        "time_residual",
        "time_chi2",
        "flight_time",
        "dira",
        "best_pv_x",
        "best_pv_y",
        "best_pv_z",
        # Track-compatible fields
        "x",
        "y",
        "z",
        "tx",
        "ty",
        "p",
        "charge",
        "time",
        "track_id",
        "cov_0_0",
        "cov_1_0",
        "cov_1_1",
        "cov_2_0",
        "cov_2_1",
        "cov_2_2",
        "cov_3_0",
        "cov_3_1",
        "cov_3_2",
        "cov_3_3",
    ]
    for i in range(3):
        for j in range(i + 1):
            fields.append(f"vertex_cov_{i}_{j}")
    for k in range(n_body):
        fields.extend(
            [
                f"daughter{k}_track_id",
                f"daughter{k}_pool_index",
                f"daughter{k}_pid",
                f"daughter{k}_charge",
                f"daughter{k}_mass",
            ]
        )
    if n_body == 2:
        fields.append("doca12")

    # MC truth fields (if pools have them)
    has_mc = track_pools is not None and all(
        "mc_ancestor_keys" in p for p in track_pools
    )
    if has_mc:
        fields.extend(
            [
                "mc_truth",
                "mc_pid",
                "mc_key",
                "mc_pv_key",
                "mc_fromsignal",
            ]
        )

    out = {f: empty for f in fields}

    # Doubly-jagged MC ancestor fields: (events, 0_candidates, var_ancestors)
    if has_mc:
        inner = ak.unflatten(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        empty_nested = ak.unflatten(inner, np.zeros(n_events, dtype=np.int64))
        out["mc_ancestor_pids"] = empty_nested
        out["mc_ancestor_keys"] = empty_nested

    if track_pools is not None:
        out["_daughter_pools"] = track_pools
    return out
