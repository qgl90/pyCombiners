"""N-body track combination pipeline."""

from __future__ import annotations

__author__ = [
    "Renato Quagliani <rquaglia@cern.ch>",
    "Jiahui Zhuo <jiahui.zhuo@cern.ch>",
]

import awkward as ak
import numpy as np

from .models import (
    apply_cuts,
    apply_mask,
    get_daughter,
    n_daughters,
    pick_inner,
    unflatten_container,
)
from .physics import (
    compute_composite_covariance,
    composite_pv_association,
    compute_doca,
    compute_prefit_kinematics,
    consolidate_composite,
    vertex_fit_3d_plus_time,
)


def make_combinations(track_pools):
    """Build flat n-body combination indices from track pools with overlap removal."""

    n_body = len(track_pools)
    n_events = len(track_pools[0]["x"])

    # Group pool positions by identity
    groups: dict[int, list[int]] = {}
    pool_map: dict[int, dict] = {}
    for i, pool in enumerate(track_pools):
        pid = id(pool)
        groups.setdefault(pid, []).append(i)
        pool_map[pid] = pool

    ordered_group_ids = list(groups.keys())

    # Build jagged local indices per daughter
    if len(ordered_group_ids) == 1:
        pool = pool_map[ordered_group_ids[0]]
        idx = ak.local_index(pool["x"], axis=1)
        combo_idx = ak.combinations(idx, n_body, axis=1)
        daughter_local_jag = list(ak.unzip(combo_idx))
    else:
        group_arrays = []
        for gid in ordered_group_ids:
            pool = pool_map[gid]
            positions = groups[gid]
            k = len(positions)
            idx = ak.local_index(pool["x"], axis=1)
            if k == 1:
                group_arrays.append(idx)
            else:
                group_arrays.append(ak.combinations(idx, k, axis=1))

        cart = ak.cartesian(group_arrays, axis=1)
        n_groups = len(ordered_group_ids)
        daughter_local_jag = [None] * n_body

        if n_groups == 1:
            items = ak.unzip(cart)
            for pos, item in zip(groups[ordered_group_ids[0]], items):
                daughter_local_jag[pos] = item
        else:
            for gi, gid in enumerate(ordered_group_ids):
                positions = groups[gid]
                group_part = cart[str(gi)]
                if len(positions) == 1:
                    daughter_local_jag[positions[0]] = group_part
                else:
                    sub_items = ak.unzip(group_part)
                    for pos, sub in zip(positions, sub_items):
                        daughter_local_jag[pos] = sub

    # Flatten and compute global indices
    counts = ak.to_numpy(ak.num(daughter_local_jag[0]))
    event_idx = np.repeat(np.arange(n_events), counts)

    out: dict = {}
    for k in range(n_body):
        pool = track_pools[k]
        pc = ak.to_numpy(ak.num(pool["x"]))
        offsets = np.zeros(len(pc) + 1, dtype=np.int64)
        np.cumsum(pc, out=offsets[1:])

        local_flat = ak.to_numpy(ak.flatten(daughter_local_jag[k]))
        gi = offsets[event_idx] + local_flat.astype(np.int64)
        out[f"daughter{k}_global_index"] = gi

        if "track_id" in pool:
            flat_tid = np.asarray(ak.flatten(pool["track_id"]))
            out[f"daughter{k}_track_id"] = flat_tid[gi]

    out["_daughter_pools"] = track_pools
    out["event_idx"] = event_idx

    # Overlap removal: discard combinations where different pools share a track
    all_same_pool = all(
        track_pools[i] is track_pools[0] for i in range(1, n_body)
    )
    if not all_same_pool:
        source_cols = []
        for k in range(n_body):
            pool = track_pools[k]
            gi = out[f"daughter{k}_global_index"]
            tid_keys = sorted(
                key
                for key in pool
                if key.startswith("daughter") and key.endswith("_track_id")
            )
            if tid_keys:
                source_cols.append(
                    [np.asarray(ak.flatten(pool[key]))[gi] for key in tid_keys]
                )
            elif "track_id" in pool:
                source_cols.append(
                    [np.asarray(ak.flatten(pool["track_id"]))[gi]]
                )
            else:
                source_cols.append([])

        overlap = np.zeros(len(event_idx), dtype=bool)
        for i in range(n_body):
            for j in range(i + 1, n_body):
                if track_pools[i] is track_pools[j]:
                    continue
                for src_i in source_cols[i]:
                    for src_j in source_cols[j]:
                        overlap |= src_i == src_j

        if np.any(overlap):
            keep = ~overlap
            out = apply_mask(out, keep)

    return out


def add_daughter_pv_compatibility(comb):
    """Add daughter best-PV agreement and common eligible-PV information."""
    n_body = n_daughters(comb)
    common = get_daughter(comb, 0, "pv_on_time")
    same_best = get_daughter(comb, 0, "best_pv_index") >= 0

    for k in range(1, n_body):
        other = get_daughter(comb, k, "pv_on_time")
        common = common[
            ak.any(
                common[:, :, np.newaxis] == other[:, np.newaxis, :], axis=-1
            )
        ]
        same_best &= get_daughter(comb, k, "best_pv_index") == get_daughter(
            comb, 0, "best_pv_index"
        )

    comb["daughter_common_pv_on_time"] = common
    comb["daughters_have_common_pv_on_time"] = ak.num(common, axis=-1) > 0
    comb["daughters_have_same_best_pv"] = same_best
    return comb


def combine(
    track_pools,
    pvs,
    track_cuts=None,
    combination_cuts=None,
    composite_cuts=None,
    final_cuts=None,
    vertex_fit_function=vertex_fit_3d_plus_time,
    doca_function=compute_doca,
    consolidate_function=consolidate_composite,
    pv_function=composite_pv_association,
    compute_pv_compatibility=False,
    require_common_pv_on_time=False,
    require_same_best_pv=False,
):
    """Run the n-body combination pipeline and return a jagged composite.

    Daughter PV compatibility is optional metadata and/or a pre-fit candidate
    requirement. Set ``compute_pv_compatibility`` to retain the metadata without
    rejecting candidates.
    After fitting, ``pv_function`` independently associates the new composite
    using its improved time and time uncertainty.
    """

    n_events = len(pvs["x"])

    for i, pool in enumerate(track_pools):
        if "mass" not in pool:
            raise ValueError(
                f"track_pools[{i}] is missing a 'mass' field.  "
                "Use set_tracks_pid(tracks, particle_id) to assign mass hypotheses."
            )

    if track_cuts:
        seen: dict[int, dict] = {}
        new_pools = []
        for pool in track_pools:
            pid = id(pool)
            if pid not in seen:
                seen[pid] = apply_cuts(pool, track_cuts)
            new_pools.append(seen[pid])
        track_pools = new_pools

    comb = make_combinations(track_pools)
    if len(comb["event_idx"]) == 0:
        return None

    if (
        compute_pv_compatibility
        or require_common_pv_on_time
        or require_same_best_pv
    ):
        missing = [
            i
            for i, pool in enumerate(track_pools)
            if "pv_on_time" not in pool or "best_pv_index" not in pool
        ]
        if missing:
            raise ValueError(
                "PV-compatible combinations require tracks/composites to be "
                f"associated to PVs first (missing pools {missing})"
            )
        add_daughter_pv_compatibility(comb)
        compatibility = ak.ones_like(comb["event_idx"], dtype=bool)
        if require_common_pv_on_time:
            compatibility = (
                compatibility & comb["daughters_have_common_pv_on_time"]
            )
        if require_same_best_pv:
            compatibility = compatibility & comb["daughters_have_same_best_pv"]
        comb = apply_mask(comb, compatibility)
        if len(comb["event_idx"]) == 0:
            return None

    doca_function(comb)
    compute_prefit_kinematics(comb)

    if combination_cuts:
        comb = apply_cuts(comb, combination_cuts)
        if len(comb["event_idx"]) == 0:
            return None

    vertex_fit_function(comb)
    consolidate_function(comb)

    if composite_cuts:
        comb = apply_cuts(comb, composite_cuts)
    out = comb

    pv_function(out, pvs)

    if final_cuts:
        out = apply_cuts(out, final_cuts)

    compute_composite_covariance(out)
    _propagate_mc_truth(out)

    for key in [k for k in out if k.startswith("_cached_")]:
        del out[key]

    final_counts = np.bincount(out["event_idx"], minlength=n_events)
    result = unflatten_container(out, final_counts)
    result["_type"] = "composite"

    return result


def _propagate_mc_truth(out):
    """Propagate MC truth from daughter pools to combined candidates."""
    pools = out["_daughter_pools"]
    n_body = n_daughters(out)

    # Guard: skip if any pool lacks MC ancestry fields
    mc_required = {
        "mc_pv_key",
        "mc_fromsignal",
        "mc_ancestor_pids",
        "mc_ancestor_keys",
    }
    for pool in pools:
        if not mc_required.issubset(pool.keys()):
            return

    # Gather per-daughter fields
    d_pv_key = []
    d_fromsignal = []
    d_mc_pid = []
    d_anc_pids = []
    d_anc_keys = []

    for k in range(n_body):
        d_pv_key.append(get_daughter(out, k, "mc_pv_key"))
        d_fromsignal.append(get_daughter(out, k, "mc_fromsignal"))
        d_mc_pid.append(get_daughter(out, k, "mc_pid"))
        d_anc_pids.append(get_daughter(out, k, "mc_ancestor_pids"))
        d_anc_keys.append(get_daughter(out, k, "mc_ancestor_keys"))

    # Find common ancestors
    d0_keys = d_anc_keys[0]
    in_common = ak.ones_like(d0_keys, dtype=bool)
    for k in range(1, n_body):
        match_k = d0_keys[:, :, np.newaxis] == d_anc_keys[k][:, np.newaxis, :]
        in_common = in_common & ak.any(match_k, axis=-1)

    has_common = ak.to_numpy(ak.any(in_common, axis=-1))

    # Among common ancestors, prefer one whose PID differs from all daughters'
    # PIDs — this skips intermediate states (e.g. a pion ancestor when both
    # daughters are pions) and selects the actual mother particle.
    abs_anc = np.abs(d_anc_pids[0])
    not_daughter = ak.ones_like(d0_keys, dtype=bool)
    for k in range(n_body):
        not_daughter = not_daughter & (
            abs_anc != np.abs(d_mc_pid[k])[:, np.newaxis]
        )
    preferred = in_common & not_daughter
    has_preferred = ak.to_numpy(ak.any(preferred, axis=-1))
    pref_idx = ak.to_numpy(ak.fill_none(ak.argmax(preferred, axis=-1), 0))
    fall_idx = ak.to_numpy(ak.fill_none(ak.argmax(in_common, axis=-1), 0))
    best_idx = np.where(has_preferred, pref_idx, fall_idx)

    # Extract common ancestor PID/key
    common_pid = np.where(has_common, pick_inner(d_anc_pids[0], best_idx), 0)
    common_key = np.where(has_common, pick_inner(d_anc_keys[0], best_idx), -1)

    # Remaining ancestor chain (above common mother)
    tail_start = np.where(has_common, best_idx + 1, 0).astype(np.int64)
    local = ak.local_index(d_anc_pids[0], axis=-1)
    tail_mask = local >= tail_start[:, np.newaxis]
    out["mc_ancestor_pids"] = d_anc_pids[0][tail_mask]
    out["mc_ancestor_keys"] = d_anc_keys[0][tail_mask]

    # --- Assemble flat scalar fields ---
    out["mc_truth"] = np.where(has_common, 1, 0)
    out["mc_pid"] = common_pid
    out["mc_key"] = common_key
    out["mc_pv_key"] = np.where(has_common, d_pv_key[0], -1)
    out["mc_fromsignal"] = np.where(has_common & (d_fromsignal[0] == 1), 1, 0)
