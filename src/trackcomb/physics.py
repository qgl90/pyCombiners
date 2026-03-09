"""Batch physics: IP, vertex fit, DOCA, kinematics, PV association."""

from __future__ import annotations

__author__ = [
    "Renato Quagliani <rquaglia@cern.ch>",
    "Jiahui Zhuo <jiahui.zhuo@cern.ch>",
]

import awkward as ak
import numpy as np
from hepunits import c_light

from .configurable import configurable
from .models import (
    gather_daughters_stack,
    gather_jagged,
    get_daughter,
    n_daughters,
)
from .linalg import (
    inv_3x3_sym,
    mahalanobis_2x2,
    mahalanobis_3x3,
    solve_2x2,
    solve_3x3_sym,
)


def _flatten_field(container, field):
    """Flatten a jagged field to a 1D numpy array."""
    return np.asarray(ak.flatten(container[field]))


def _pair_indices(counts_a, counts_b):
    """Build flat (a_idx, b_idx, evt_idx, pair_counts) for all cross-event pairs."""
    pair_counts = counts_a * counts_b
    total_pairs = int(pair_counts.sum())

    if total_pairs == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, pair_counts

    # Offsets for each collection
    a_offsets = np.empty(len(counts_a) + 1, dtype=np.int64)
    a_offsets[0] = 0
    np.cumsum(counts_a, out=a_offsets[1:])
    b_offsets = np.empty(len(counts_b) + 1, dtype=np.int64)
    b_offsets[0] = 0
    np.cumsum(counts_b, out=b_offsets[1:])

    # Event index for each pair
    evt_per_pair = np.repeat(np.arange(len(pair_counts)), pair_counts)

    # Within each event's block of pairs, compute local index 0..n_pairs-1
    pair_offsets = np.empty(len(pair_counts) + 1, dtype=np.int64)
    pair_offsets[0] = 0
    np.cumsum(pair_counts, out=pair_offsets[1:])
    local_idx = np.arange(total_pairs) - pair_offsets[evt_per_pair]

    # local_idx = a_local * n_b + b_local
    nb = counts_b[evt_per_pair]
    a_local = local_idx // nb
    b_local = local_idx % nb

    a_idx = a_offsets[evt_per_pair] + a_local
    b_idx = b_offsets[evt_per_pair] + b_local

    return a_idx, b_idx, evt_per_pair, pair_counts


def _unflatten_3d(flat_arr, track_counts, pv_counts):
    """Reshape flat pair array to awkward (events, tracks, pvs)."""
    # Inner: each track has pv_counts[evt] pvs
    inner_counts = np.repeat(pv_counts, track_counts)
    return ak.unflatten(ak.unflatten(flat_arr, inner_counts), track_counts)


def _write_best_pv_fields(container, pvs, pick_fn):
    """Write best_pv_* fields into *container* for every field in *pvs*."""
    for field in pvs:
        if field.startswith("_"):
            continue
        container[f"best_pv_{field}"] = pick_fn(field)


def compute_track_pv_pairs(tracks, pvs):
    """Compute IP, IP chi2, and dt for all (track, PV) pairs."""
    track_counts = ak.to_numpy(ak.num(tracks["x"]))
    pv_counts = ak.to_numpy(ak.num(pvs["x"]))
    t_idx, p_idx, evt_idx, pair_counts = _pair_indices(track_counts, pv_counts)

    # Flatten all needed fields once
    t_x = _flatten_field(tracks, "x")[t_idx]
    t_y = _flatten_field(tracks, "y")[t_idx]
    t_z = _flatten_field(tracks, "z")[t_idx]
    t_tx = _flatten_field(tracks, "tx")[t_idx]
    t_ty = _flatten_field(tracks, "ty")[t_idx]

    p_x = _flatten_field(pvs, "x")[p_idx]
    p_y = _flatten_field(pvs, "y")[p_idx]
    p_z = _flatten_field(pvs, "z")[p_idx]

    # Extrapolate track to PV z
    dz = p_z - t_z
    x_ext = t_x + t_tx * dz
    y_ext = t_y + t_ty * dz
    dx = x_ext - p_x
    dy = y_ext - p_y

    # IP
    ip_flat = np.sqrt(dx**2 + dy**2)

    # Track covariance fields
    t_c00 = _flatten_field(tracks, "cov_0_0")[t_idx]
    t_c10 = _flatten_field(tracks, "cov_1_0")[t_idx]
    t_c11 = _flatten_field(tracks, "cov_1_1")[t_idx]
    t_c20 = _flatten_field(tracks, "cov_2_0")[t_idx]
    t_c21 = _flatten_field(tracks, "cov_2_1")[t_idx]
    t_c22 = _flatten_field(tracks, "cov_2_2")[t_idx]
    t_c30 = _flatten_field(tracks, "cov_3_0")[t_idx]
    t_c31 = _flatten_field(tracks, "cov_3_1")[t_idx]
    t_c32 = _flatten_field(tracks, "cov_3_2")[t_idx]
    t_c33 = _flatten_field(tracks, "cov_3_3")[t_idx]

    # PV covariance
    p_c00 = _flatten_field(pvs, "cov_0_0")[p_idx]
    p_c10 = _flatten_field(pvs, "cov_1_0")[p_idx]
    p_c11 = _flatten_field(pvs, "cov_1_1")[p_idx]

    # Propagated track XY covariance at PV z
    dz2 = dz**2
    var_x = t_c00 + 2.0 * dz * t_c20 + dz2 * t_c22
    var_y = t_c11 + 2.0 * dz * t_c31 + dz2 * t_c33
    cov_xy = t_c10 + dz * t_c30 + dz * t_c21 + dz2 * t_c32

    # Total covariance = track + PV
    chi2_flat = mahalanobis_2x2(
        dx, dy, var_x + p_c00, cov_xy + p_c10, var_y + p_c11
    )

    result = {
        "ip": _unflatten_3d(ip_flat, track_counts, pv_counts),
        "ip_chi2": _unflatten_3d(chi2_flat, track_counts, pv_counts),
        "track_counts": track_counts,
        "pv_counts": pv_counts,
    }

    # Flight-corrected dt (only when both tracks and PVs have time)
    if "time" in tracks and "time" in pvs:
        t_time = _flatten_field(tracks, "time")[t_idx]
        p_time = _flatten_field(pvs, "time")[p_idx]
        dz_flight = t_z - p_z
        speed_factor = np.sqrt(1.0 + t_tx**2 + t_ty**2)
        flight_time = (dz_flight * speed_factor) / c_light
        dt_flat = t_time - flight_time - p_time
        result["dt"] = _unflatten_3d(dt_flat, track_counts, pv_counts)

    return result


def tracks_pv_association(tracks, pvs, max_dt_corrected=0.05):
    """Select best PV per track (min IP) and add best_pv_* fields."""
    pairs = compute_track_pv_pairs(tracks, pvs)
    ip_all = pairs["ip"]
    ip_chi2_all = pairs["ip_chi2"]

    if max_dt_corrected is not None and "dt" in pairs:
        dt_all = pairs["dt"]
        time_ok = np.abs(dt_all) < max_dt_corrected
        any_pass = ak.any(time_ok, axis=-1)
        ip_for_min = ak.where(
            any_pass,
            ak.where(time_ok, ip_all, np.inf),  # Set ip = inf for bad pvs
            ip_all,
        )
    else:
        ip_for_min = ip_all

    best_pv = ak.argmin(ip_for_min, axis=-1, keepdims=True)

    # Handle empty tracks or empty PVs — argmin returns None
    has_pvs = ak.num(ip_all, axis=-1) > 0
    zero = has_pvs * 0.0

    def _pick(pv_field):
        return ak.where(has_pvs, ak.flatten(pv_field[best_pv], axis=-1), zero)

    tracks["min_ip"] = _pick(ip_all)
    tracks["min_ip_chi2"] = _pick(ip_chi2_all)

    # Best PV index (events, tracks) — -1 where no PVs
    bp = ak.flatten(best_pv, axis=-1)
    bp_safe = ak.fill_none(bp, 0)
    tracks["best_pv_index"] = ak.where(has_pvs, bp_safe, -1)

    _write_best_pv_fields(
        tracks,
        pvs,
        lambda f: ak.where(has_pvs, gather_jagged(pvs[f], bp_safe), zero),
    )

    # Store PV container reference for offline lookups
    tracks["_pvs"] = pvs

    return tracks


def vertex_fit_3d(comb):
    """Spatial vertex fit from daughter tracks, writes vertex fields into *comb*."""
    _COV_KEYS = (
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
    )
    pools = comb["_daughter_pools"]
    missing = [ck for ck in _COV_KEYS if ck not in pools[0]]
    if missing:
        raise ValueError(
            f"vertex_fit_3d requires track covariance fields; missing: {missing}"
        )

    x = gather_daughters_stack(comb, "x")
    y = gather_daughters_stack(comb, "y")
    z = gather_daughters_stack(comb, "z")
    tx = gather_daughters_stack(comb, "tx")
    ty = gather_daughters_stack(comb, "ty")
    c00 = gather_daughters_stack(comb, "cov_0_0")
    c10 = gather_daughters_stack(comb, "cov_1_0")
    c11 = gather_daughters_stack(comb, "cov_1_1")
    c20 = gather_daughters_stack(comb, "cov_2_0")
    c21 = gather_daughters_stack(comb, "cov_2_1")
    c22 = gather_daughters_stack(comb, "cov_2_2")
    c30 = gather_daughters_stack(comb, "cov_3_0")
    c31 = gather_daughters_stack(comb, "cov_3_1")
    c32 = gather_daughters_stack(comb, "cov_3_2")
    c33 = gather_daughters_stack(comb, "cov_3_3")

    N, n = x.shape

    # --- Normal equations for [x_v, y_v, z_v] ---
    # Design-matrix rows per track: [1, 0, -tx_i] and [0, 1, -ty_i].
    # ATA and ATb are assembled from these, then solved analytically.
    stx = np.sum(tx, axis=1)
    sty = np.sum(ty, axis=1)
    st2 = np.sum(tx**2 + ty**2, axis=1)

    rx = x - tx * z  # (N, n)
    ry = y - ty * z

    b0 = np.sum(rx, axis=1)
    b1 = np.sum(ry, axis=1)
    b2 = np.sum(-tx * rx - ty * ry, axis=1)

    # ATA is symmetric: [[n, 0, -stx], [0, n, -sty], [-stx, -sty, st2]]
    a00 = np.full(N, float(n))
    a01 = np.zeros(N)
    a02 = -stx
    a11 = np.full(N, float(n))
    a12 = -sty
    a22 = st2

    # Solve ATA @ v = ATb
    vx, vy, vz = solve_3x3_sym(a00, a01, a02, a11, a12, a22, b0, b1, b2)

    # --- Spatial chi2 and weighted vertex covariance ---
    z_v = vz[:, np.newaxis]
    x_v = vx[:, np.newaxis]
    y_v = vy[:, np.newaxis]

    dz = z_v - z  # (N, n)
    x_ext = x + tx * dz
    y_ext = y + ty * dz
    dx = x_ext - x_v
    dy = y_ext - y_v

    # Propagate track 2x2 (x,y) covariance to vertex z
    var_x = c00 + 2.0 * dz * c20 + dz**2 * c22
    var_y = c11 + 2.0 * dz * c31 + dz**2 * c33
    cov_xy = c10 + dz * c30 + dz * c21 + dz**2 * c32
    chi2_leg = mahalanobis_2x2(dx, dy, var_x, cov_xy, var_y)
    spatial_chi2 = np.sum(chi2_leg, axis=1)

    # Weighted vertex covariance: cov = inv(sum_i H_i^T W_i H_i)
    # H_i = [[1,0,-tx_i],[0,1,-ty_i]], W_i = inv(V_i)
    det = var_x * var_y - cov_xy**2
    safe_det = np.where(np.abs(det) > 1e-30, det, 1.0)
    w00 = var_y / safe_det  # (N, n)
    w11 = var_x / safe_det
    w01 = -cov_xy / safe_det

    s00 = np.sum(w00, axis=1)
    s01 = np.sum(w01, axis=1)
    s11 = np.sum(w11, axis=1)
    s02 = np.sum(-tx * w00 - ty * w01, axis=1)
    s12 = np.sum(-tx * w01 - ty * w11, axis=1)
    s22 = np.sum(tx**2 * w00 + 2 * tx * ty * w01 + ty**2 * w11, axis=1)

    vc00, vc01, vc02, vc11, vc12, vc22 = inv_3x3_sym(
        s00, s01, s02, s11, s12, s22
    )

    comb["vertex_x"] = vx
    comb["vertex_y"] = vy
    comb["vertex_z"] = vz
    comb["vertex_chi2"] = spatial_chi2
    comb["vertex_cov_0_0"] = vc00
    comb["vertex_cov_1_0"] = vc01
    comb["vertex_cov_1_1"] = vc11
    comb["vertex_cov_2_0"] = vc02
    comb["vertex_cov_2_1"] = vc12
    comb["vertex_cov_2_2"] = vc22


def doca_2body(x, y, z, tx, ty):
    """Batch distance of closest approach between two straight tracks."""

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

    # Solve 2x2 system [[a, -b], [-b, c]] @ [s, t] = [-d, e]
    # (derived from minimising |w + s*u0 - t*u1|^2)
    s, t, parallel = solve_2x2(a, -b, -b, c, -d, e)

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


def compute_doca(comb):
    """Compute pairwise DOCAs and write doca{i}{j}, max_doca, min_doca into comb."""
    n_body = n_daughters(comb)

    x = gather_daughters_stack(comb, "x")
    y = gather_daughters_stack(comb, "y")
    z = gather_daughters_stack(comb, "z")
    tx = gather_daughters_stack(comb, "tx")
    ty = gather_daughters_stack(comb, "ty")

    if n_body == 2:
        doca_vals = {"doca12": doca_2body(x, y, z, tx, ty)}
    else:
        doca_vals = doca_nbody(x, y, z, tx, ty, n_body)

    for dk, dv in doca_vals.items():
        comb[dk] = dv

    all_doca = np.column_stack(list(doca_vals.values()))
    comb["max_doca"] = np.max(all_doca, axis=1)
    comb["min_doca"] = np.min(all_doca, axis=1)


def compute_composite_covariance(comb):
    """Propagate daughter covariances to composite 5x5 track-state covariance."""
    n_body = n_daughters(comb)
    spx, spy, spz = comb["px"], comb["py"], comb["pz"]
    charge = comb["charge"]
    N = len(spx)

    cov_sub_keys = [
        ("cov_2_2", "cov_3_2", "cov_4_2"),
        ("cov_3_2", "cov_3_3", "cov_4_3"),
        ("cov_4_2", "cov_4_3", "cov_4_4"),
    ]

    # --- Step 1: Sum daughter momentum covariances ---
    # For each daughter, transform cov(tx, ty, qop) → cov(px, py, pz)
    # using Jacobian J, then sum across daughters.
    mom_cov = np.zeros((N, 3, 3))

    for k in range(n_body):
        tx_k = get_daughter(comb, k, "tx")
        ty_k = get_daughter(comb, k, "ty")
        p_k = get_daughter(comb, k, "p")
        qop_k = get_daughter(comb, k, "qop")

        s = np.sqrt(1.0 + tx_k**2 + ty_k**2)
        s3 = s**3
        ps3 = p_k / s3
        px_k = p_k * tx_k / s
        py_k = p_k * ty_k / s
        pz_k = p_k / s
        safe_qop = np.where(np.abs(qop_k) > 1e-30, qop_k, 1e-30)

        # Build 3x3 sub-covariance (tx, ty, qop)
        C = np.zeros((N, 3, 3))
        for i in range(3):
            for j in range(i + 1):
                val = get_daughter(comb, k, cov_sub_keys[i][j])
                C[:, i, j] = val
                C[:, j, i] = val

        # Jacobian d(px,py,pz)/d(tx,ty,qop)
        J = np.zeros((N, 3, 3))
        J[:, 0, 0] = ps3 * (1.0 + ty_k**2)  # dpx/dtx
        J[:, 0, 1] = -ps3 * tx_k * ty_k  # dpx/dty
        J[:, 0, 2] = -px_k / safe_qop  # dpx/dqop
        J[:, 1, 0] = -ps3 * tx_k * ty_k  # dpy/dtx
        J[:, 1, 1] = ps3 * (1.0 + tx_k**2)  # dpy/dty
        J[:, 1, 2] = -py_k / safe_qop  # dpy/dqop
        J[:, 2, 0] = -ps3 * tx_k  # dpz/dtx
        J[:, 2, 1] = -ps3 * ty_k  # dpz/dty
        J[:, 2, 2] = -pz_k / safe_qop  # dpz/dqop

        # mom_cov += J @ C @ J.T
        JC = np.einsum("nij,njk->nik", J, C)
        mom_cov += np.einsum("nij,nkj->nik", JC, J)

    # --- Step 2: Transform sum cov(px,py,pz) → cov(tx,ty,qop) for composite ---
    safe_pz = np.where(np.abs(spz) > 1e-12, spz, 1e-12)
    p_total = np.sqrt(spx**2 + spy**2 + spz**2)
    safe_p3 = np.where(p_total > 1e-12, p_total**3, 1e-36)

    K = np.zeros((N, 3, 3))
    K[:, 0, 0] = 1.0 / safe_pz  # dtx/dpx
    K[:, 0, 2] = -spx / safe_pz**2  # dtx/dpz
    K[:, 1, 1] = 1.0 / safe_pz  # dty/dpy
    K[:, 1, 2] = -spy / safe_pz**2  # dty/dpz
    K[:, 2, 0] = -charge * spx / safe_p3  # dqop/dpx
    K[:, 2, 1] = -charge * spy / safe_p3  # dqop/dpy
    K[:, 2, 2] = -charge * spz / safe_p3  # dqop/dpz

    KC = np.einsum("nij,njk->nik", K, mom_cov)
    slope_cov = np.einsum("nij,nkj->nik", KC, K)  # K @ mom_cov @ K.T

    # --- Step 3: Assemble 5x5 covariance ---
    # Position block from vertex_cov
    comb["cov_0_0"] = comb["vertex_cov_0_0"]
    comb["cov_1_0"] = comb["vertex_cov_1_0"]
    comb["cov_1_1"] = comb["vertex_cov_1_1"]
    # Position-slope cross terms (simplified to zero)
    zero = np.zeros(N)
    comb["cov_2_0"] = zero
    comb["cov_2_1"] = zero
    comb["cov_3_0"] = zero
    comb["cov_3_1"] = zero
    comb["cov_4_0"] = zero
    comb["cov_4_1"] = zero
    # Slope/qop block from propagation
    comb["cov_2_2"] = slope_cov[:, 0, 0]  # var(tx)
    comb["cov_3_2"] = slope_cov[:, 1, 0]  # cov(ty, tx)
    comb["cov_3_3"] = slope_cov[:, 1, 1]  # var(ty)
    comb["cov_4_2"] = slope_cov[:, 2, 0]  # cov(qop, tx)
    comb["cov_4_3"] = slope_cov[:, 2, 1]  # cov(qop, ty)
    comb["cov_4_4"] = slope_cov[:, 2, 2]  # var(qop)


def compute_prefit_kinematics(comb):
    """Compute mass, pt, charge from daughter 4-vectors (no vertex fit needed)."""
    p_arr = gather_daughters_stack(comb, "p")
    tx_arr = gather_daughters_stack(comb, "tx")
    ty_arr = gather_daughters_stack(comb, "ty")
    mass_arr = gather_daughters_stack(comb, "mass")

    spx, spy, spz, se = lorentz_sum(p_arr, tx_arr, ty_arr, mass_arr)
    comb["px"] = spx
    comb["py"] = spy
    comb["pz"] = spz
    comb["energy"] = se
    comb["mass"] = invariant_mass(spx, spy, spz, se)
    pt_v, eta_v = pt_eta(spx, spy, spz)
    comb["pt"] = pt_v
    comb["eta"] = eta_v

    charge_arr = gather_daughters_stack(comb, "charge")
    comb["charge"] = np.sum(charge_arr, axis=1)


def consolidate_composite(comb):
    """Set track-compatible fields from vertex position and pre-computed momenta."""
    if "px" not in comb:
        compute_prefit_kinematics(comb)

    spx, spy, spz = comb["px"], comb["py"], comb["pz"]

    comb["x"] = comb["vertex_x"]
    comb["y"] = comb["vertex_y"]
    comb["z"] = comb["vertex_z"]
    safe_pz = np.where(np.abs(spz) > 1e-12, spz, 1e-12)
    comb["tx"] = spx / safe_pz
    comb["ty"] = spy / safe_pz
    comb["p"] = np.sqrt(spx**2 + spy**2 + spz**2)
    comb["qop"] = comb["charge"] / comb["p"]
    if "vertex_time" in comb:
        comb["time"] = comb["vertex_time"]
        comb["sigma_time"] = comb["vertex_sigma_time"]


def _propagate_time_to_vertex(time, z, tx, ty, p, masses, vertex_z):
    """Propagate track times to vertex z, correcting for mass-dependent speed."""

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

    denom = beta * c_light
    valid = denom > 0
    t_prop = np.where(valid, time + path / np.where(valid, denom, 1.0), time)

    return t_prop


def _vertex_time_fit(t_prop, sigma_time):
    """Weighted average of propagated track times; returns (vertex_time, sigma_t, time_chi2)."""
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


def _pairwise_time_chi2(t_prop, sigma_time):
    """Mean pairwise time chi2 over all unique daughter pairs."""
    N, n = t_prop.shape
    if n <= 1:
        return np.zeros(N)

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


def vertex_fit_3d_plus_time(comb):
    """Spatial vertex fit + time fit if daughters have time/sigma_time."""
    vertex_fit_3d(comb)

    pools = comb["_daughter_pools"]
    has_time = all("time" in p and "sigma_time" in p for p in pools)

    if has_time:
        time_arr = gather_daughters_stack(comb, "time")
        sigma_t = gather_daughters_stack(comb, "sigma_time")

        t_prop = _propagate_time_to_vertex(
            time_arr,
            gather_daughters_stack(comb, "z"),
            gather_daughters_stack(comb, "tx"),
            gather_daughters_stack(comb, "ty"),
            gather_daughters_stack(comb, "p"),
            gather_daughters_stack(comb, "mass"),
            comb["vertex_z"],
        )
        vt, stt, tchi2 = _vertex_time_fit(t_prop, sigma_t)
        comb["vertex_time"] = vt
        comb["vertex_sigma_time"] = stt
        comb["vertex_time_chi2"] = tchi2
        comb["pair_time_chi2"] = _pairwise_time_chi2(t_prop, sigma_t)


def lorentz_sum(p, tx, ty, masses):
    """Sum Lorentz 4-vectors of daughters. Returns (px, py, pz, E)."""

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

    return (
        np.sum(px, axis=1),
        np.sum(py, axis=1),
        np.sum(pz, axis=1),
        np.sum(e, axis=1),
    )


def invariant_mass(sum_px, sum_py, sum_pz, sum_e):
    """Invariant mass from summed 4-momentum."""

    m2 = sum_e**2 - sum_px**2 - sum_py**2 - sum_pz**2
    abs_m2 = np.abs(m2)
    mass = abs_m2**0.5
    return np.where(m2 >= 0, mass, -mass)


def pt_eta(sum_px, sum_py, sum_pz):
    """Transverse momentum and pseudorapidity from 3-momentum."""

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


def compute_composite_pv_pairs(comb, pvs):
    """Compute IP, IP chi2, and time quantities for all (composite, PV) pairs."""
    n_events = len(pvs["x"])
    cand_counts = np.bincount(comb["event_idx"], minlength=n_events)
    pv_counts = ak.to_numpy(ak.num(pvs["x"]))
    has_time = "vertex_time" in comb

    # PV offsets for global indexing
    pv_offsets = np.zeros(n_events + 1, dtype=np.int64)
    np.cumsum(pv_counts, out=pv_offsets[1:])

    evt_per_cand = np.repeat(np.arange(n_events), cand_counts)

    # Flatten PV fields once
    pv_field_names = ["x", "y", "z", "cov_0_0", "cov_1_0", "cov_1_1"]
    if has_time:
        pv_field_names.extend(["time", "sigma_time"])
    pv_cov3_names = ["cov_2_0", "cov_2_1", "cov_2_2", "cov_3_3"]
    pv_mc_names = [k for k in ("mc_x", "mc_y", "mc_z", "mc_key") if k in pvs]

    pv_flat = {}
    for k in pv_field_names + pv_cov3_names + pv_mc_names:
        v = pvs.get(k)
        if v is not None:
            pv_flat[k] = _flatten_field(pvs, k)
        else:
            pv_flat[k] = np.zeros(int(pv_counts.sum()))

    result = {
        "cand_counts": cand_counts,
        "pv_counts": pv_counts,
        "evt_per_cand": evt_per_cand,
        "pv_offsets": pv_offsets,
        "pv_flat": pv_flat,
        "pv_mc_names": pv_mc_names,
    }

    c_idx, p_idx, _, _ = _pair_indices(cand_counts, pv_counts)
    total_pairs = len(c_idx)
    inner_counts = np.repeat(pv_counts, cand_counts)
    result["inner_counts"] = inner_counts

    if total_pairs == 0:
        empty = ak.unflatten(np.zeros(0), inner_counts)
        result["ip"] = empty
        result["ip_chi2"] = empty
        if has_time:
            result["time_residual"] = empty
            result["time_chi2"] = empty
            result["flight_time"] = empty
        return result

    # Vertex position and covariance elements
    vx = comb["vertex_x"]
    vy = comb["vertex_y"]
    vz = comb["vertex_z"]
    c00 = comb["vertex_cov_0_0"]
    c01 = comb["vertex_cov_1_0"]
    c02 = comb["vertex_cov_2_0"]
    c11 = comb["vertex_cov_1_1"]
    c12 = comb["vertex_cov_2_1"]
    c22 = comb["vertex_cov_2_2"]

    px, py, pz = comb["px"], comb["py"], comb["pz"]
    energy = comb["energy"]

    # Per-candidate direction and speed
    p_mag = np.sqrt(px**2 + py**2 + pz**2)
    sp = np.where(p_mag > 1e-16, p_mag, 1e-16)
    ux, uy, uz = px / sp, py / sp, pz / sp
    suz = np.where(np.abs(uz) > 1e-12, uz, 1e-12)
    txc, tyc = ux / suz, uy / suz

    # Propagated vertex XY covariance (per candidate)
    var_x_c = np.maximum(c00 + txc**2 * c22 - 2 * txc * c02, 0.0)
    var_y_c = np.maximum(c11 + tyc**2 * c22 - 2 * tyc * c12, 0.0)
    cxy_c = c01 - tyc * c02 - txc * c12 + txc * tyc * c22

    # Expand to pair level
    vx_p, vy_p, vz_p = vx[c_idx], vy[c_idx], vz[c_idx]
    txc_p, tyc_p = txc[c_idx], tyc[c_idx]
    ux_p, uy_p, uz_p = ux[c_idx], uy[c_idx], uz[c_idx]

    pvx = pv_flat["x"][p_idx]
    pvy = pv_flat["y"][p_idx]
    pvz = pv_flat["z"][p_idx]

    # IP
    dz = pvz - vz_p
    dx = vx_p + txc_p * dz - pvx
    dy = vy_p + tyc_p * dz - pvy
    ip_flat = np.sqrt(dx**2 + dy**2)

    tot_xx = var_x_c[c_idx] + pv_flat["cov_0_0"][p_idx]
    tot_xy = cxy_c[c_idx] + pv_flat["cov_1_0"][p_idx]
    tot_yy = var_y_c[c_idx] + pv_flat["cov_1_1"][p_idx]
    ip_chi2_flat = mahalanobis_2x2(dx, dy, tot_xx, tot_xy, tot_yy)

    result["ip"] = ak.unflatten(ip_flat, inner_counts)
    result["ip_chi2"] = ak.unflatten(ip_chi2_flat, inner_counts)

    # Time residual and chi2 (only when time info exists)
    if has_time:
        beta = np.clip(np.where(energy > 0, p_mag / energy, 0.0), 0.0, 1.0)
        bc_p = (beta * c_light)[c_idx]
        vbc = bc_p > 0
        sbc = np.where(vbc, bc_p, 1.0)

        disp_x, disp_y, disp_z = vx_p - pvx, vy_p - pvy, vz_p - pvz
        fl = disp_x * ux_p + disp_y * uy_p + disp_z * uz_p
        ft_flat = np.where(vbc, fl / sbc, 0.0)

        vt_p = comb["vertex_time"][c_idx]
        t_res_flat = vt_p - ft_flat - pv_flat["time"][p_idx]

        # Flight sigma²
        dvv = (
            ux**2 * c00
            + uy**2 * c11
            + uz**2 * c22
            + 2 * ux * uy * c01
            + 2 * ux * uz * c02
            + 2 * uy * uz * c12
        )
        dvp = (
            ux_p**2 * pv_flat["cov_0_0"][p_idx]
            + uy_p**2 * pv_flat["cov_1_1"][p_idx]
            + uz_p**2 * pv_flat["cov_2_2"][p_idx]
            + 2 * ux_p * uy_p * pv_flat["cov_1_0"][p_idx]
            + 2 * ux_p * uz_p * pv_flat["cov_2_0"][p_idx]
            + 2 * uy_p * uz_p * pv_flat["cov_2_1"][p_idx]
        )
        sf2 = np.where(vbc, np.maximum(dvv[c_idx] + dvp, 0.0) / sbc**2, 0.0)
        vst_p = comb["sigma_time"][c_idx]
        sig2 = (
            np.maximum(vst_p, 0.0) ** 2
            + np.maximum(pv_flat["sigma_time"][p_idx], 0.0) ** 2
            + sf2
        )
        ssig2 = np.where(sig2 > 0, sig2, 1.0)
        t_chi2_flat = np.where(sig2 > 0, t_res_flat**2 / ssig2, t_res_flat**2)

        result["time_residual"] = ak.unflatten(t_res_flat, inner_counts)
        result["time_chi2"] = ak.unflatten(t_chi2_flat, inner_counts)
        result["flight_time"] = ak.unflatten(ft_flat, inner_counts)

    return result


def _composite_flight_vector(comb):
    """Flight vector from best PV to composite vertex."""
    fx = comb["vertex_x"] - comb["best_pv_x"]
    fy = comb["vertex_y"] - comb["best_pv_y"]
    fz = comb["vertex_z"] - comb["best_pv_z"]
    return fx, fy, fz


def compute_composite_dira(comb):
    """Compute DIRA (cosine of angle between flight and momentum)."""
    fx, fy, fz = _composite_flight_vector(comb)
    px, py, pz = comb["px"], comb["py"], comb["pz"]
    fm = np.sqrt(fx**2 + fy**2 + fz**2)
    p_mag = np.sqrt(px**2 + py**2 + pz**2)
    dd = fm * p_mag
    sdd = np.where(dd > 1e-16, dd, 1.0)
    comb["dira"] = np.where(
        dd > 1e-16, (fx * px + fy * py + fz * pz) / sdd, 0.0
    )


def compute_composite_fdchi2(comb):
    """Compute flight distance chi2 (3D Mahalanobis)."""
    fx, fy, fz = _composite_flight_vector(comb)
    z = 0.0
    tc00 = comb["vertex_cov_0_0"] + comb.get("best_pv_cov_0_0", z)
    tc01 = comb["vertex_cov_1_0"] + comb.get("best_pv_cov_1_0", z)
    tc02 = comb["vertex_cov_2_0"] + comb.get("best_pv_cov_2_0", z)
    tc11 = comb["vertex_cov_1_1"] + comb.get("best_pv_cov_1_1", z)
    tc12 = comb["vertex_cov_2_1"] + comb.get("best_pv_cov_2_1", z)
    tc22 = comb["vertex_cov_2_2"] + comb.get("best_pv_cov_2_2", z)
    comb["fdchi2"] = mahalanobis_3x3(
        fx, fy, fz, tc00, tc01, tc02, tc11, tc12, tc22
    )


def compute_composite_flight_eta(comb):
    """Compute pseudorapidity of flight direction."""
    fx, fy, fz = _composite_flight_vector(comb)
    fm = np.sqrt(fx**2 + fy**2 + fz**2)
    safe_fm = np.where(fm > 1e-16, fm, 1.0)
    ratio = np.clip(fz / safe_fm, -1 + 1e-7, 1 - 1e-7)
    comb["flight_eta"] = np.where(fm > 1e-16, np.arctanh(ratio), 0.0)


def compute_composite_mcor(comb):
    """Compute corrected mass."""
    fx, fy, fz = _composite_flight_vector(comb)
    px, py, pz = comb["px"], comb["py"], comb["pz"]
    energy = comb["energy"]
    fm2 = fx**2 + fy**2 + fz**2
    pperp2 = (
        (py * fz - fy * pz) ** 2
        + (pz * fx - fz * px) ** 2
        + (px * fy - fx * py) ** 2
    ) / np.maximum(fm2, 1e-32)
    m_vis2 = np.maximum(energy**2 - px**2 - py**2 - pz**2, 0.0)
    comb["mcor"] = np.sqrt(m_vis2 + pperp2) + np.sqrt(pperp2)


@configurable
def composite_pv_association(
    comb,
    pvs,
    max_time_residual=0.05,
    max_time_chi2=None,
):
    """Best-PV association for composites (min IP, optional time filter)."""
    if len(comb["vertex_x"]) == 0:
        return
    pairs = compute_composite_pv_pairs(comb, pvs)
    if ak.sum(ak.num(pairs["ip"], axis=-1)) == 0:
        return

    # --- Select best PV per composite ---
    has_time = "vertex_time" in comb
    ip_jag = pairs["ip"]
    evt_per_cand = pairs["evt_per_cand"]
    pv_offsets = pairs["pv_offsets"]

    has_pvs = ak.num(ip_jag, axis=-1) > 0
    zero = has_pvs * 0.0

    # Time filter (only when time info exists)
    ip_sel = ak.copy(ip_jag)
    if has_time:
        t_res_jag = pairs["time_residual"]
        t_chi2_jag = pairs["time_chi2"]
        if max_time_residual is not None:
            tok = np.abs(t_res_jag) <= max_time_residual
            any_pass = ak.any(tok, axis=-1)
            ip_sel = ak.where(tok, ip_sel, np.inf)
            ip_sel = ak.where(any_pass, ip_sel, ip_jag)
        if max_time_chi2 is not None:
            tcok = t_chi2_jag <= max_time_chi2
            any_pass2 = ak.any(tcok, axis=-1)
            ip_sel = ak.where(tcok, ip_sel, np.inf)
            ip_sel = ak.where(any_pass2, ip_sel, ip_jag)

    best_pv = ak.argmin(ip_sel, axis=-1, keepdims=True)

    def _pick(jag):
        return ak.to_numpy(
            ak.where(has_pvs, ak.flatten(jag[best_pv], axis=-1), zero)
        )

    comb["composite_ip"] = _pick(ip_jag)
    comb["composite_ip_chi2"] = _pick(pairs["ip_chi2"])

    bp = ak.flatten(best_pv, axis=-1)
    bp_safe = ak.fill_none(bp, 0)
    bp_np = ak.to_numpy(bp_safe)
    comb["best_pv_index"] = ak.to_numpy(ak.where(has_pvs, bp_safe, 0)).astype(
        float
    )

    if has_time:
        comb["time_residual"] = _pick(pairs["time_residual"])
        comb["time_chi2"] = _pick(pairs["time_chi2"])
        comb["flight_time"] = _pick(pairs["flight_time"])

    # Best PV fields — use global PV index
    has_pvs_np = ak.to_numpy(has_pvs)
    best_pv_global = pv_offsets[evt_per_cand] + bp_np

    _write_best_pv_fields(
        comb,
        pvs,
        lambda f: np.where(
            has_pvs_np, _flatten_field(pvs, f)[best_pv_global], 0.0
        ),
    )

    # --- Derived physics quantities ---
    compute_composite_dira(comb)
    compute_composite_fdchi2(comb)
    compute_composite_flight_eta(comb)
    compute_composite_mcor(comb)


def fit_track_t0(tracks):
    """Fit track t0 from TV hits. Adds time and sigma_time to tracks."""
    if "mass" not in tracks:
        raise ValueError(
            "fit_track_t0 requires a 'mass' field — call set_tracks_pid first"
        )

    z = tracks["z"]
    tx, ty = tracks["tx"], tracks["ty"]
    qop = tracks["qop"]
    p = 1.0 / abs(qop)
    mass = tracks["mass"]
    hit_z, hit_t = tracks["tvhits_z"], tracks["tvhits_t"]

    slope_factor = (1.0 + tx**2 + ty**2) ** 0.5
    energy = (p**2 + mass**2) ** 0.5
    beta_c = (p / energy) * c_light

    t0_per_hit = hit_t - (hit_z - z) * slope_factor / beta_c
    n_hits = ak.count(t0_per_hit, axis=-1)

    time = ak.where(
        n_hits > 0,
        ak.sum(t0_per_hit, axis=-1) / ak.where(n_hits > 0, n_hits, 1),
        0.0,
    )

    residuals_sq = (t0_per_hit - time) ** 2
    variance = ak.where(
        n_hits > 1,
        ak.sum(residuals_sq, axis=-1) / ak.where(n_hits > 1, n_hits, 1),
        0.0,
    )
    sigma_time = ak.where(
        n_hits > 1,
        (variance**0.5) / (n_hits**0.5),
        ak.where(n_hits == 1, 1e9, 1e9),
    )
    sigma_time = ak.where(sigma_time < 1e-12, 1e-12, sigma_time)

    tracks["time"] = time
    tracks["sigma_time"] = sigma_time
    return tracks


def compute_default_track_quantities(tracks):
    """Derive p, charge, pt, eta, and t0 from raw track state."""

    qop = tracks["qop"]
    p = 1.0 / abs(qop)
    tracks["p"] = p
    tracks["charge"] = ak.where(qop > 0, 1, -1)

    tx, ty = tracks["tx"], tracks["ty"]
    norm = (1.0 + tx**2 + ty**2) ** 0.5
    dx, dy, dz = tx / norm, ty / norm, 1.0 / norm
    tracks["pt"] = p * (dx**2 + dy**2) ** 0.5

    p_dir = (dx**2 + dy**2 + dz**2) ** 0.5
    denom = ak.where(abs(p_dir - dz) < 1e-30, 1e-30, p_dir - dz)
    tracks["eta"] = 0.5 * np.log((p_dir + dz) / denom)

    return tracks
