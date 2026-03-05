"""Batch physics: IP, vertex fit, DOCA, kinematics, PV association."""

from __future__ import annotations

__author__ = "Renato Quagliani <rquaglia@cern.ch>"

from hepunits import c_light as C_LIGHT_MM_PER_NS  # mm/ns

import numpy as np


# ---------------------------------------------------------------------------
# Shared linear-algebra helpers
# ---------------------------------------------------------------------------


def _mahalanobis_2x2(dx, dy, cxx, cxy, cyy):
    """Batch 2x2 Mahalanobis chi2; falls back to dx^2+dy^2 if singular."""
    cov = np.empty(dx.shape + (2, 2))
    cov[..., 0, 0] = cxx
    cov[..., 0, 1] = cxy
    cov[..., 1, 0] = cxy
    cov[..., 1, 1] = cyy

    det = np.linalg.det(cov)
    singular = np.abs(det) < 1e-18

    r = np.stack([dx, dy], axis=-1)
    cov_inv = np.zeros_like(cov)
    good = ~singular
    if np.any(good):
        cov_inv[good] = np.linalg.inv(cov[good])
    chi2 = np.einsum("...i,...ij,...j->...", r, cov_inv, r)
    return np.where(singular, dx**2 + dy**2, chi2)


# ---------------------------------------------------------------------------
# IP computation
# ---------------------------------------------------------------------------


def ip_to_pvs(tracks, pvs):
    """Compute IP and IP chi2 for all (track, PV) pairs.

    Returns (ip, ip_chi2), both shape (events, tracks, pvs).
    """
    import awkward as ak

    def _bc(t_arr, pv_arr):
        return ak.unzip(ak.cartesian([t_arr, pv_arr], axis=1, nested=True))

    # Broadcast positions and slopes
    t_x, pv_x = _bc(tracks["x"], pvs["x"])
    t_y, pv_y = _bc(tracks["y"], pvs["y"])
    t_z, pv_z = _bc(tracks["z"], pvs["z"])
    t_tx, _ = _bc(tracks["tx"], pvs["z"])
    t_ty, _ = _bc(tracks["ty"], pvs["z"])

    # Track covariance (4x4 lower-tri fields)
    t_c00, _ = _bc(tracks["cov_0_0"], pvs["z"])
    t_c10, _ = _bc(tracks["cov_1_0"], pvs["z"])
    t_c11, _ = _bc(tracks["cov_1_1"], pvs["z"])
    t_c20, _ = _bc(tracks["cov_2_0"], pvs["z"])
    t_c21, _ = _bc(tracks["cov_2_1"], pvs["z"])
    t_c22, _ = _bc(tracks["cov_2_2"], pvs["z"])
    t_c30, _ = _bc(tracks["cov_3_0"], pvs["z"])
    t_c31, _ = _bc(tracks["cov_3_1"], pvs["z"])
    t_c32, _ = _bc(tracks["cov_3_2"], pvs["z"])
    t_c33, _ = _bc(tracks["cov_3_3"], pvs["z"])

    # PV covariance (XY 2x2 only)
    _, pv_c00 = _bc(tracks["z"], pvs["cov_0_0"])
    _, pv_c10 = _bc(tracks["z"], pvs["cov_1_0"])
    _, pv_c11 = _bc(tracks["z"], pvs["cov_1_1"])

    # Extrapolate track to PV z
    dz = pv_z - t_z
    x_ext = t_x + t_tx * dz
    y_ext = t_y + t_ty * dz

    # 2D displacement
    dx = x_ext - pv_x
    dy = y_ext - pv_y

    # Impact parameter
    ip = (dx**2 + dy**2) ** 0.5

    # Propagated track XY covariance at PV z
    # Matches legacy extrapolate_xy_cov:
    #   var_x = c[0][0] + 2*dz*c[0][2] + dz²*c[2][2]
    #   var_y = c[1][1] + 2*dz*c[1][3] + dz²*c[3][3]
    #   cov_xy = c[0][1] + dz*c[0][3] + dz*c[1][2] + dz²*c[2][3]
    var_x = t_c00 + 2.0 * dz * t_c20 + dz**2 * t_c22
    var_y = t_c11 + 2.0 * dz * t_c31 + dz**2 * t_c33
    cov_xy = t_c10 + dz * t_c30 + dz * t_c21 + dz**2 * t_c32

    # Total covariance = track + PV
    tot_xx = var_x + pv_c00
    tot_xy = cov_xy + pv_c10
    tot_yy = var_y + pv_c11

    # Weighted chi2 via np.linalg on flattened arrays
    import numpy as np

    flat_chi2 = _mahalanobis_2x2(
        np.asarray(ak.flatten(dx, axis=None)),
        np.asarray(ak.flatten(dy, axis=None)),
        np.asarray(ak.flatten(tot_xx, axis=None)),
        np.asarray(ak.flatten(tot_xy, axis=None)),
        np.asarray(ak.flatten(tot_yy, axis=None)),
    )
    # Reconstruct the 3D jagged structure (events, tracks, pvs)
    counts_inner = ak.flatten(ak.num(dx, axis=2))  # pvs per track
    counts_outer = ak.num(dx, axis=1)  # tracks per event
    ip_chi2 = ak.unflatten(ak.unflatten(flat_chi2, counts_inner), counts_outer)

    return ip, ip_chi2


def flight_corrected_dt(tracks, pvs):
    """Flight-corrected time residual dt = t_track - t_flight - t_pv.

    Returns shape (events, tracks, pvs).
    """
    import awkward as ak

    def _bc(t_arr, pv_arr):
        return ak.unzip(ak.cartesian([t_arr, pv_arr], axis=1, nested=True))

    t_z, pv_z = _bc(tracks["z"], pvs["z"])
    t_tx, _ = _bc(tracks["tx"], pvs["z"])
    t_ty, _ = _bc(tracks["ty"], pvs["z"])
    t_time, pv_time = _bc(tracks["time"], pvs["time"])

    dz = t_z - pv_z
    speed_factor = (1.0 + t_tx**2 + t_ty**2) ** 0.5
    flight_time = (dz * speed_factor) / C_LIGHT_MM_PER_NS

    return t_time - flight_time - pv_time


def tracks_pv_association(tracks, pvs, max_dt_corrected=0.05):
    """Select best PV per track (min IP) and add best_pv_* fields.

    PVs are optionally pre-filtered by flight-corrected dt; falls back
    to all PVs if none pass. Adds min_ip, min_ip_chi2, best_pv_{x,y,z,...}.
    """
    import awkward as ak
    import numpy as np

    # Vectorized IP and IP chi2: shape (events, tracks, pvs)
    ip_all, ip_chi2_all = ip_to_pvs(tracks, pvs)

    if max_dt_corrected is not None:
        dt_all = flight_corrected_dt(tracks, pvs)  # (events, tracks, pvs)
        time_ok = np.abs(dt_all) < max_dt_corrected
        any_pass = ak.any(time_ok, axis=-1)  # (events, tracks)

        # Where time cut passes for at least one PV, mask out failing PVs;
        # otherwise fall back to all PVs (no masking).
        ip_for_min = ak.where(
            any_pass,
            ak.where(time_ok, ip_all, np.inf),
            ip_all,
        )
    else:
        ip_for_min = ip_all

    best_pv = ak.argmin(ip_for_min, axis=-1, keepdims=True)  # (events, tracks, 1)

    # Handle empty tracks or empty PVs — argmin returns None
    has_pvs = ak.num(ip_all, axis=-1) > 0  # (events, tracks)
    zero = has_pvs * 0.0  # (events, tracks) of zeros

    def _pick(pv_field):
        """Select best-PV value for each track, 0.0 where no PVs."""
        return ak.where(has_pvs, ak.flatten(pv_field[best_pv], axis=-1), zero)

    tracks["min_ip"] = _pick(ip_all)
    tracks["min_ip_chi2"] = _pick(ip_chi2_all)

    # Best PV index (events, tracks) — -1 where no PVs
    bp = ak.flatten(best_pv, axis=-1)  # (events, tracks), may contain None
    bp_safe = ak.fill_none(bp, 0)
    tracks["best_pv_index"] = ak.where(has_pvs, bp_safe, -1)

    # Build flat index for picking best-PV fields from (events, pvs) arrays.
    bp_flat = ak.to_numpy(ak.flatten(bp_safe))
    track_counts = ak.to_numpy(ak.num(bp_safe))
    pv_counts = ak.to_numpy(ak.num(pvs["x"]))
    pv_offsets = np.zeros(len(pv_counts) + 1, dtype=np.int64)
    np.cumsum(pv_counts, out=pv_offsets[1:])
    evt_per_track = np.repeat(np.arange(len(track_counts)), track_counts)
    global_idx = pv_offsets[evt_per_track] + bp_flat

    pv_fields = [
        "x",
        "y",
        "z",
        "time",
        "sigma_time",
        "cov_0_0",
        "cov_1_0",
        "cov_1_1",
        "cov_2_0",
        "cov_2_1",
        "cov_2_2",
        "cov_3_3",
    ]
    for field in pv_fields:
        pv_flat = ak.to_numpy(ak.flatten(pvs[field]))
        picked_flat = pv_flat[global_idx]
        picked = ak.unflatten(picked_flat, track_counts)
        tracks[f"best_pv_{field}"] = ak.where(has_pvs, picked, zero)

    # Store PV container reference for offline lookups
    tracks["_pvs"] = pvs

    return tracks


# ---------------------------------------------------------------------------
# Combination utilities
# ---------------------------------------------------------------------------


def make_combinations(track_pools):
    """Build n-body combinations from track pools.

    Same-object pools use ak.combinations (no self-pairing);
    distinct pools use ak.cartesian (full cross-product).
    Returns a tuple of n_body daughter containers.
    """
    import awkward as ak

    n_body = len(track_pools)

    # Group pool positions by identity
    groups: dict[int, list[int]] = {}
    pool_map: dict[int, dict] = {}
    for i, pool in enumerate(track_pools):
        pid = id(pool)
        groups.setdefault(pid, []).append(i)
        pool_map[pid] = pool

    ordered_group_ids = list(groups.keys())

    if len(ordered_group_ids) == 1:
        # Fast path: all daughters from the same pool → ak.combinations
        pool = pool_map[ordered_group_ids[0]]
        idx = ak.local_index(pool["x"], axis=1)
        combo_idx = ak.combinations(idx, n_body, axis=1)
        daughter_indices = ak.unzip(combo_idx)
        result = []
        for daughter_idx in daughter_indices:
            daughter = {}
            for field, arr in pool.items():
                try:
                    daughter[field] = arr[daughter_idx]
                except (TypeError, IndexError):
                    pass  # skip non-array metadata (_daughter_pools, etc.)
            result.append(daughter)
        return tuple(result)

    # General case: combinations within same-pool groups, cartesian across groups
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

    # Cartesian product across groups
    cart = ak.cartesian(group_arrays, axis=1)

    # Unpack into per-position indices
    # cart fields are "0", "1", ... for each group
    n_groups = len(ordered_group_ids)
    daughter_indices_list: list[tuple[int, ...]] = [None] * n_body  # type: ignore[list-item]

    if n_groups == 1:
        # Already handled above, but just in case
        items = ak.unzip(cart)
        for pos, item in zip(groups[ordered_group_ids[0]], items):
            daughter_indices_list[pos] = item
    else:
        for gi, gid in enumerate(ordered_group_ids):
            positions = groups[gid]
            group_part = cart[str(gi)]
            if len(positions) == 1:
                daughter_indices_list[positions[0]] = group_part
            else:
                sub_items = ak.unzip(group_part)
                for pos, sub in zip(positions, sub_items):
                    daughter_indices_list[pos] = sub

    # Build daughter Containers
    result = []
    for i, daughter_idx in enumerate(daughter_indices_list):
        pool = track_pools[i]
        daughter = {}
        for field, arr in pool.items():
            try:
                daughter[field] = arr[daughter_idx]
            except (TypeError, IndexError):
                pass  # skip non-array metadata (_daughter_pools, etc.)
        result.append(daughter)
    return tuple(result)


def flatten_daughters(daughters, fields=None):
    """Flatten jagged daughters to (flat_daughters, counts).

    Doubly-jagged fields (e.g. mc_ancestor_pids) are skipped.
    """
    import awkward as ak
    import numpy as np

    ref = next(iter(daughters[0].values()))
    counts = ak.to_numpy(ak.num(ref, axis=1))

    if fields is None:
        fields = list(daughters[0].keys())

    flat_daughters = []
    for daughter in daughters:
        flat_daughter = {}
        for field in fields:
            arr = daughter[field]
            flat = ak.flatten(arr, axis=1)
            try:
                flat_daughter[field] = ak.to_numpy(flat)
            except ValueError:
                # Skip doubly-jagged fields (e.g. mc_ancestor_pids)
                continue
        flat_daughters.append(flat_daughter)
    return tuple(flat_daughters), counts


def unflatten_array(flat_arr, counts):
    """Unflatten ``(N,)`` array back to ``(events, var_combos)`` jagged."""
    import awkward as ak

    return ak.unflatten(flat_arr, counts, axis=0)


# ---------------------------------------------------------------------------
# Vertex fit
# ---------------------------------------------------------------------------


def vertex_fit_xyz(x, y, z, tx, ty, cov_fields):
    """Batch least-squares vertex fit in (x, y, z).

    Returns (vertex_xyz, spatial_chi2, cov_xyz).
    """
    import numpy as np

    N, n = x.shape

    # --- Normal equations for [x_v, y_v, z_v] ---
    # Design-matrix rows per track: [1, 0, -tx_i] and [0, 1, -ty_i].
    # ATA and ATb are assembled from these, then solved with np.linalg.
    stx = np.sum(tx, axis=1)
    sty = np.sum(ty, axis=1)
    st2 = np.sum(tx**2 + ty**2, axis=1)

    rx = x - tx * z  # (N, n)
    ry = y - ty * z

    b0 = np.sum(rx, axis=1)
    b1 = np.sum(ry, axis=1)
    b2 = np.sum(-tx * rx - ty * ry, axis=1)

    # Build (N, 3, 3) ATA and (N, 3) ATb
    ATA = np.zeros((N, 3, 3))
    ATA[:, 0, 0] = n
    ATA[:, 1, 1] = n
    ATA[:, 0, 2] = -stx
    ATA[:, 2, 0] = -stx
    ATA[:, 1, 2] = -sty
    ATA[:, 2, 1] = -sty
    ATA[:, 2, 2] = st2
    ATb = np.stack([b0, b1, b2], axis=1)  # (N, 3)

    # Solve; singular systems get fallback values.
    # ATb must be (N, 3, 1) so np.linalg.solve treats the leading axis as
    # batch — otherwise (N, 3) is ambiguous when N == 3.
    try:
        vertex_xyz = np.linalg.solve(ATA, ATb[:, :, np.newaxis])[:, :, 0]
    except np.linalg.LinAlgError:
        # Batch fallback: solve one-by-one, skip singular
        vertex_xyz = np.column_stack(
            [
                np.mean(x, axis=1),
                np.mean(y, axis=1),
                np.mean(z, axis=1),
            ]
        )
        for i in range(N):
            try:
                vertex_xyz[i] = np.linalg.solve(ATA[i], ATb[i])
            except np.linalg.LinAlgError:
                pass  # keeps mean fallback

    # --- Spatial chi2 and weighted vertex covariance ---
    z_v = vertex_xyz[:, 2:3]  # (N, 1)
    x_v = vertex_xyz[:, 0:1]
    y_v = vertex_xyz[:, 1:2]

    dz = z_v - z  # (N, n)
    x_ext = x + tx * dz
    y_ext = y + ty * dz
    dx = x_ext - x_v
    dy = y_ext - y_v

    have_cov = all(
        k in cov_fields
        for k in (
            "cov_0_0",
            "cov_1_1",
            "cov_2_0",
            "cov_2_2",
            "cov_3_1",
            "cov_3_3",
            "cov_1_0",
            "cov_3_0",
            "cov_2_1",
            "cov_3_2",
        )
    )
    if have_cov:
        # Propagate track 2x2 (x,y) covariance to vertex z
        var_x = (
            cov_fields["cov_0_0"]
            + 2.0 * dz * cov_fields["cov_2_0"]
            + dz**2 * cov_fields["cov_2_2"]
        )
        var_y = (
            cov_fields["cov_1_1"]
            + 2.0 * dz * cov_fields["cov_3_1"]
            + dz**2 * cov_fields["cov_3_3"]
        )
        cov_xy = (
            cov_fields["cov_1_0"]
            + dz * cov_fields["cov_3_0"]
            + dz * cov_fields["cov_2_1"]
            + dz**2 * cov_fields["cov_3_2"]
        )
        chi2_leg = _mahalanobis_2x2(dx, dy, var_x, cov_xy, var_y)
        spatial_chi2 = np.sum(chi2_leg, axis=1)

        # Weighted vertex covariance: cov = inv(sum_i H_i^T W_i H_i)
        # H_i = [[1,0,-tx_i],[0,1,-ty_i]], W_i = inv(V_i)
        ATA_w = np.zeros((N, 3, 3))
        for k in range(n):
            det = var_x[:, k] * var_y[:, k] - cov_xy[:, k] ** 2
            safe_det = np.where(np.abs(det) > 1e-30, det, 1.0)
            w00 = var_y[:, k] / safe_det
            w11 = var_x[:, k] / safe_det
            w01 = -cov_xy[:, k] / safe_det
            tx_k = tx[:, k]
            ty_k = ty[:, k]
            ATA_w[:, 0, 0] += w00
            ATA_w[:, 0, 1] += w01
            ATA_w[:, 0, 2] += -tx_k * w00 - ty_k * w01
            ATA_w[:, 1, 0] += w01
            ATA_w[:, 1, 1] += w11
            ATA_w[:, 1, 2] += -tx_k * w01 - ty_k * w11
            ATA_w[:, 2, 0] += -tx_k * w00 - ty_k * w01
            ATA_w[:, 2, 1] += -tx_k * w01 - ty_k * w11
            ATA_w[:, 2, 2] += tx_k**2 * w00 + 2 * tx_k * ty_k * w01 + ty_k**2 * w11

        try:
            cov_xyz_out = np.linalg.inv(ATA_w)
        except np.linalg.LinAlgError:
            cov_xyz_out = np.tile(np.eye(3) * 1e6, (N, 1, 1))
            for i in range(N):
                try:
                    cov_xyz_out[i] = np.linalg.inv(ATA_w[i])
                except np.linalg.LinAlgError:
                    pass
    else:
        spatial_chi2 = np.sum(dx**2 + dy**2, axis=1)
        try:
            cov_xyz_out = np.linalg.inv(ATA)
        except np.linalg.LinAlgError:
            cov_xyz_out = np.tile(np.eye(3) * 1e6, (N, 1, 1))
            for i in range(N):
                try:
                    cov_xyz_out[i] = np.linalg.inv(ATA[i])
                except np.linalg.LinAlgError:
                    pass

    return vertex_xyz, spatial_chi2, cov_xyz_out


# ---------------------------------------------------------------------------
# DOCA
# ---------------------------------------------------------------------------


def doca_2body(x, y, z, tx, ty):
    """Batch distance of closest approach between two straight tracks."""
    import numpy as np

    norm0 = (1.0 + tx[:, 0] ** 2 + ty[:, 0] ** 2) ** 0.5
    norm1 = (1.0 + tx[:, 1] ** 2 + ty[:, 1] ** 2) ** 0.5

    ux0, uy0, uz0 = tx[:, 0] / norm0, ty[:, 0] / norm0, 1.0 / norm0
    ux1, uy1, uz1 = tx[:, 1] / norm1, ty[:, 1] / norm1, 1.0 / norm1

    w0x = x[:, 0] - x[:, 1]
    w0y = y[:, 0] - y[:, 1]
    w0z = z[:, 0] - z[:, 1]

    a = ux0 * ux0 + uy0 * uy0 + uz0 * uz0
    b = ux0 * ux1 + uy0 * uy1 + uz0 * uz1
    c = ux1 * ux1 + uy1 * uy1 + uz1 * uz1
    d = ux0 * w0x + uy0 * w0y + uz0 * w0z
    e = ux1 * w0x + uy1 * w0y + uz1 * w0z

    # Solve 2x2 system [[a, -b], [-b, c]] @ [s, t] = [-d, e] via np.linalg
    # (derived from minimising |w + s*u0 - t*u1|^2)
    N = len(a)
    A = np.stack(
        [np.stack([a, -b], axis=-1), np.stack([-b, c], axis=-1)], axis=-2
    )  # (N, 2, 2)
    rhs = np.stack([-d, e], axis=-1)[:, :, np.newaxis]  # (N, 2, 1)
    det = np.linalg.det(A)
    parallel = np.abs(det) < 1e-12
    st = np.zeros((N, 2))
    good = ~parallel
    if np.any(good):
        st[good] = np.linalg.solve(A[good], rhs[good])[:, :, 0]
    s, t = st[:, 0], st[:, 1]

    diff_x = w0x + s * ux0 - t * ux1
    diff_y = w0y + s * uy0 - t * uy1
    diff_z = w0z + s * uz0 - t * uz1
    doca_np = (diff_x**2 + diff_y**2 + diff_z**2) ** 0.5

    par_x = w0x - d * ux0
    par_y = w0y - d * uy0
    par_z = w0z - d * uz0
    doca_par = (par_x**2 + par_y**2 + par_z**2) ** 0.5

    return np.where(parallel, doca_par, doca_np)


def doca_nbody(x, y, z, tx, ty, n_body):
    """Pairwise DOCAs for all pairs in an n-body combination."""
    import numpy as np

    result = {}
    for i in range(n_body):
        for j in range(i + 1, n_body):
            pair = lambda a: np.stack([a[:, i], a[:, j]], axis=1)
            result[f"doca{i + 1}{j + 1}"] = doca_2body(
                pair(x),
                pair(y),
                pair(z),
                pair(tx),
                pair(ty),
            )
    return result


# ---------------------------------------------------------------------------
# Time fit
# ---------------------------------------------------------------------------


def _propagate_time_to_vertex(time, sigma_time, z, tx, ty, p, masses, vertex_z):
    """Propagate track times to vertex z, correcting for mass-dependent speed."""
    import numpy as np

    if masses.ndim == 1:
        masses = np.broadcast_to(masses[np.newaxis, :], time.shape)

    dz = vertex_z[:, np.newaxis] - z
    speed_factor = (1.0 + tx**2 + ty**2) ** 0.5
    path = dz * speed_factor

    m = np.abs(masses)
    p_safe = np.maximum(p, 0.0)
    energy = (p_safe**2 + m**2) ** 0.5
    beta = np.where(energy > 0, p_safe / energy, 0.0)
    beta = np.clip(beta, 0.0, 1.0)

    denom = beta * C_LIGHT_MM_PER_NS
    valid = denom > 0
    t_prop = np.where(valid, time + path / np.where(valid, denom, 1.0), time)

    return t_prop


def vertex_time_fit(time, sigma_time, z, tx, ty, p, masses, vertex_z):
    """Weighted average of propagated track times at the vertex.

    Returns (vertex_time, sigma_t, time_chi2).
    """
    import numpy as np

    t_prop = _propagate_time_to_vertex(
        time,
        sigma_time,
        z,
        tx,
        ty,
        p,
        masses,
        vertex_z,
    )

    valid_sigma = sigma_time > 0
    w = np.where(valid_sigma, 1.0 / (sigma_time**2), 0.0)

    sum_w = np.sum(w, axis=1)
    sum_wt = np.sum(w * t_prop, axis=1)

    has_w = sum_w > 0
    safe_sw = np.where(has_w, sum_w, 1.0)
    t_v = np.where(has_w, sum_wt / safe_sw, np.mean(t_prop, axis=1))
    sigma_t = np.where(has_w, (1.0 / safe_sw) ** 0.5, 1e3)

    dt = t_prop - t_v[:, np.newaxis]
    chi2_per = np.where(
        valid_sigma, dt**2 / np.where(valid_sigma, sigma_time**2, 1.0), 0.0
    )
    time_chi2 = np.sum(chi2_per, axis=1)

    return t_v, sigma_t, time_chi2


def pairwise_time_chi2(time, sigma_time, z, tx, ty, p, masses, vertex_z):
    """Mean pairwise time chi2 over all unique daughter pairs."""
    import numpy as np

    N, n = time.shape
    if n <= 1:
        return np.zeros(N)

    t_prop = _propagate_time_to_vertex(
        time,
        sigma_time,
        z,
        tx,
        ty,
        p,
        masses,
        vertex_z,
    )

    chi2 = np.zeros(N)
    n_pairs = np.zeros(N)
    for i in range(n):
        for j in range(i + 1, n):
            dt = t_prop[:, i] - t_prop[:, j]
            sig2 = sigma_time[:, i] ** 2 + sigma_time[:, j] ** 2
            valid = sig2 > 0
            chi2 += np.where(valid, dt**2 / np.where(valid, sig2, 1.0), 0.0)
            n_pairs += valid.astype(float)

    return np.where(n_pairs > 0, chi2 / n_pairs, 0.0)


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


def lorentz_sum(p, tx, ty, masses):
    """Sum Lorentz 4-vectors of daughters. Returns (px, py, pz, E)."""
    import numpy as np

    if masses.ndim == 1:
        masses = np.broadcast_to(masses[np.newaxis, :], p.shape)

    norm = (1.0 + tx**2 + ty**2) ** 0.5
    dx = tx / norm
    dy = ty / norm
    dz = 1.0 / norm

    px = p * dx
    py = p * dy
    pz = p * dz
    e = (p**2 + masses**2) ** 0.5

    return np.sum(px, axis=1), np.sum(py, axis=1), np.sum(pz, axis=1), np.sum(e, axis=1)


def invariant_mass(sum_px, sum_py, sum_pz, sum_e):
    """Invariant mass from summed 4-momentum."""
    import numpy as np

    m2 = sum_e**2 - sum_px**2 - sum_py**2 - sum_pz**2
    abs_m2 = np.abs(m2)
    mass = abs_m2**0.5
    return np.where(m2 >= 0, mass, -mass)


def pt_eta(sum_px, sum_py, sum_pz):
    """Transverse momentum and pseudorapidity from 3-momentum."""
    import numpy as np

    pt = (sum_px**2 + sum_py**2) ** 0.5
    p_tot = (sum_px**2 + sum_py**2 + sum_pz**2) ** 0.5

    denom = p_tot - sum_pz
    singular = np.abs(denom) < 1e-30
    safe_denom = np.where(singular, 1e-30, denom)
    eta = np.where(
        singular,
        np.sign(sum_pz) * 1e9,
        0.5 * np.log((p_tot + sum_pz) / safe_denom),
    )
    return pt, eta


# ---------------------------------------------------------------------------
# Composite-PV association
# ---------------------------------------------------------------------------


def composite_pv_association(
    vertex_xyz,
    vertex_cov,
    vertex_time,
    sigma_time,
    px,
    py,
    pz,
    energy,
    pvs,
    counts,
    max_time_residual=0.05,
    max_time_chi2=None,
):
    """Best-PV association for composites (min IP, optional time filter).

    Returns dict of IP, DIRA, fdchi2, flight_eta, mcor, time fields.
    """
    import awkward as ak
    import numpy as np

    N = len(px)
    out = {
        k: np.zeros(N)
        for k in (
            "composite_ip",
            "composite_ip_chi2",
            "time_residual",
            "time_chi2",
            "flight_time",
            "dira",
            "best_pv_index",
            "best_pv_x",
            "best_pv_y",
            "best_pv_z",
            "fdchi2",
            "flight_eta",
            "mcor",
        )
    }
    if N == 0:
        return out

    pv_arrays = {
        k: pvs[k]
        for k in ("x", "y", "z", "time", "sigma_time", "cov_0_0", "cov_1_0", "cov_1_1")
    }
    for k in ("cov_2_0", "cov_2_1", "cov_2_2"):
        pv_arrays[k] = pvs.get(k)

    offset = 0
    for evt_i, nc in enumerate(counts):
        nc = int(nc)
        if nc == 0:
            continue
        sl = slice(offset, offset + nc)
        n_pv = int(ak.num(pvs["x"], axis=1)[evt_i])
        if n_pv == 0:
            offset += nc
            continue

        # Candidate arrays for this event
        vxyz = vertex_xyz[sl]  # (nc, 3)
        vcov = vertex_cov[sl]  # (nc, 3, 3)
        vt = vertex_time[sl]
        vst = sigma_time[sl]
        cpx = px[sl]
        cpy = py[sl]
        cpz = pz[sl]
        ce = energy[sl]

        # PV arrays
        _pv = {
            k: np.asarray(v[evt_i]) if v is not None else np.zeros(n_pv)
            for k, v in pv_arrays.items()
        }

        # Unit momentum direction
        p_mag = np.sqrt(cpx**2 + cpy**2 + cpz**2)
        sp = np.where(p_mag > 1e-16, p_mag, 1e-16)
        ux = cpx / sp
        uy = cpy / sp
        uz = cpz / sp
        suz = np.where(np.abs(uz) > 1e-12, uz, 1e-12)
        txc = ux / suz
        tyc = uy / suz

        # Beta
        m2 = ce**2 - cpx**2 - cpy**2 - cpz**2
        mc = np.sqrt(np.abs(m2))
        ec = np.sqrt(p_mag**2 + mc**2)
        beta = np.clip(np.where(ec > 0, p_mag / ec, 0.0), 0.0, 1.0)
        bc = beta * C_LIGHT_MM_PER_NS

        # Broadcast (nc, 1) x (1, np)
        dz = _pv["z"][np.newaxis, :] - vxyz[:, 2:3]
        x_at = vxyz[:, 0:1] + txc[:, np.newaxis] * dz
        y_at = vxyz[:, 1:2] + tyc[:, np.newaxis] * dz
        dx = x_at - _pv["x"][np.newaxis, :]
        dy = y_at - _pv["y"][np.newaxis, :]

        # Propagated vertex cov
        c = vcov
        var_x = (c[:, 0, 0] + txc**2 * c[:, 2, 2] - 2 * txc * c[:, 0, 2])[:, np.newaxis]
        var_y = (c[:, 1, 1] + tyc**2 * c[:, 2, 2] - 2 * tyc * c[:, 1, 2])[:, np.newaxis]
        cxy = (
            c[:, 0, 1] - tyc * c[:, 0, 2] - txc * c[:, 1, 2] + txc * tyc * c[:, 2, 2]
        )[:, np.newaxis]
        var_x = np.maximum(var_x, 0.0)
        var_y = np.maximum(var_y, 0.0)

        tot_xx = var_x + _pv["cov_0_0"][np.newaxis, :]
        tot_xy = cxy + _pv["cov_1_0"][np.newaxis, :]
        tot_yy = var_y + _pv["cov_1_1"][np.newaxis, :]

        ip = np.sqrt(dx**2 + dy**2)
        ip_chi2 = _mahalanobis_2x2(dx, dy, tot_xx, tot_xy, tot_yy)

        # Flight time & time agreement
        disp_x = vxyz[:, 0:1] - _pv["x"][np.newaxis, :]
        disp_y = vxyz[:, 1:2] - _pv["y"][np.newaxis, :]
        disp_z = vxyz[:, 2:3] - _pv["z"][np.newaxis, :]
        fl = (
            disp_x * ux[:, np.newaxis]
            + disp_y * uy[:, np.newaxis]
            + disp_z * uz[:, np.newaxis]
        )
        vbc = bc[:, np.newaxis] > 0
        sbc = np.where(vbc, bc[:, np.newaxis], 1.0)
        ft = np.where(vbc, fl / sbc, 0.0)

        t_at_pv = vt[:, np.newaxis] - ft
        t_res = t_at_pv - _pv["time"][np.newaxis, :]

        # Sigma flight²
        dvv = (
            ux**2 * c[:, 0, 0]
            + uy**2 * c[:, 1, 1]
            + uz**2 * c[:, 2, 2]
            + 2 * ux * uy * c[:, 0, 1]
            + 2 * ux * uz * c[:, 0, 2]
            + 2 * uy * uz * c[:, 1, 2]
        )
        dvp = (
            ux[:, np.newaxis] ** 2 * _pv["cov_0_0"][np.newaxis, :]
            + uy[:, np.newaxis] ** 2 * _pv["cov_1_1"][np.newaxis, :]
            + uz[:, np.newaxis] ** 2 * _pv["cov_2_2"][np.newaxis, :]
            + 2 * ux[:, np.newaxis] * uy[:, np.newaxis] * _pv["cov_1_0"][np.newaxis, :]
            + 2 * ux[:, np.newaxis] * uz[:, np.newaxis] * _pv["cov_2_0"][np.newaxis, :]
            + 2 * uy[:, np.newaxis] * uz[:, np.newaxis] * _pv["cov_2_1"][np.newaxis, :]
        )
        sf2 = np.where(vbc, np.maximum(dvv[:, np.newaxis] + dvp, 0.0) / sbc**2, 0.0)
        sig2 = (
            np.maximum(vst, 0.0)[:, np.newaxis] ** 2
            + np.maximum(_pv["sigma_time"], 0.0)[np.newaxis, :] ** 2
            + sf2
        )
        ssig2 = np.where(sig2 > 0, sig2, 1.0)
        t_chi2 = np.where(sig2 > 0, t_res**2 / ssig2, t_res**2)

        # Best PV selection
        ip_sel = ip.copy()
        if max_time_residual is not None:
            tok = np.abs(t_res) <= max_time_residual
            ap = np.any(tok, axis=1)
            ip_sel = np.where(tok, ip_sel, np.inf)
            ip_sel = np.where(ap[:, np.newaxis], ip_sel, ip)
        if max_time_chi2 is not None:
            tcok = t_chi2 <= max_time_chi2
            ap2 = np.any(tcok, axis=1)
            ip_sel = np.where(tcok, ip_sel, np.inf)
            ip_sel = np.where(ap2[:, np.newaxis], ip_sel, ip)

        bp = np.argmin(ip_sel, axis=1)
        ar = np.arange(nc)

        out["composite_ip"][sl] = ip[ar, bp]
        out["composite_ip_chi2"][sl] = ip_chi2[ar, bp]
        out["time_residual"][sl] = t_res[ar, bp]
        out["time_chi2"][sl] = t_chi2[ar, bp]
        out["flight_time"][sl] = ft[ar, bp]
        out["best_pv_index"][sl] = bp
        out["best_pv_x"][sl] = _pv["x"][bp]
        out["best_pv_y"][sl] = _pv["y"][bp]
        out["best_pv_z"][sl] = _pv["z"][bp]

        # DIRA
        fx = vxyz[:, 0] - _pv["x"][bp]
        fy = vxyz[:, 1] - _pv["y"][bp]
        fz = vxyz[:, 2] - _pv["z"][bp]
        fm = np.sqrt(fx**2 + fy**2 + fz**2)
        dd = fm * sp
        sdd = np.where(dd > 1e-16, dd, 1.0)
        out["dira"][sl] = np.where(
            dd > 1e-16, (fx * cpx + fy * cpy + fz * cpz) / sdd, 0.0
        )

        # fdchi2 — full 3D Mahalanobis: delta^T (cov_SV + cov_PV)^{-1} delta
        pv_cov = np.zeros((nc, 3, 3))
        pv_cov[:, 0, 0] = _pv["cov_0_0"][bp]
        pv_cov[:, 1, 0] = _pv["cov_1_0"][bp]
        pv_cov[:, 0, 1] = _pv["cov_1_0"][bp]
        pv_cov[:, 1, 1] = _pv["cov_1_1"][bp]
        pv_cov[:, 2, 0] = _pv["cov_2_0"][bp]
        pv_cov[:, 0, 2] = _pv["cov_2_0"][bp]
        pv_cov[:, 2, 1] = _pv["cov_2_1"][bp]
        pv_cov[:, 1, 2] = _pv["cov_2_1"][bp]
        pv_cov[:, 2, 2] = _pv["cov_2_2"][bp]
        total_cov = vcov + pv_cov
        delta = np.column_stack([fx, fy, fz])
        inv_cov = np.linalg.inv(total_cov)
        out["fdchi2"][sl] = np.einsum("ni,nij,nj->n", delta, inv_cov, delta)

        # flight_eta — pseudorapidity of flight direction (PV → SV)
        safe_fm = np.where(fm > 1e-16, fm, 1.0)
        ratio = np.clip(fz / safe_fm, -1 + 1e-7, 1 - 1e-7)
        out["flight_eta"][sl] = np.where(fm > 1e-16, np.arctanh(ratio), 0.0)

        # mcor — corrected mass: sqrt(m_vis^2 + p_perp^2) + sqrt(p_perp^2)
        pperp2 = (
            (cpy * fz - fy * cpz) ** 2
            + (cpz * fx - fz * cpx) ** 2
            + (cpx * fy - fx * cpy) ** 2
        ) / np.maximum(fm**2, 1e-32)
        m_vis2 = np.maximum(ce**2 - cpx**2 - cpy**2 - cpz**2, 0.0)
        out["mcor"][sl] = np.sqrt(m_vis2 + pperp2) + np.sqrt(pperp2)

        offset += nc

    return out
