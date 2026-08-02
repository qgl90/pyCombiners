#!/usr/bin/env python3
"""Bs -> phi(-> K+K-) gamma cheated reconstruction with PicoCal clusters.

Photon energy is cheated to the matched photon's true energy (the
clusterizer splits ~57% of the showers); direction from the phi vertex
to the cluster, time from the aligned weighted mean (picocal_time.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from trackcomb import (
    apply_mask,
    combine,
    counters,
    load_calo_clusters,
    load_event_info,
    load_pvs,
    load_tracks,
    pdg_id,
    rate_counters,
    run_reconstruction,
    set_composite_pid,
    set_tracks_pid,
    tracks_pv_association,
)
from trackcomb.physics import c_light

import picocal_time

BS_PDG = abs(pdg_id("B(s)0"))
GAMMA_PDG = 22


def _load(chunk):
    tracks = load_tracks(chunk)
    pvs = load_pvs(chunk)
    clusters = load_calo_clusters(chunk, entries=True)
    picocal_time.add_seed_area(clusters)
    info = load_event_info(chunk)
    return tracks, pvs, clusters, info


def _cluster_truth(clusters):
    """Per-cluster truth from the best (highest-weight) MC match.

    Adds: mc_best_pid, mc_best_pe, mc_gamma_from_bs (bool),
    mc_bs_key (ancestor Bs key or -1).
    """
    w = clusters["mc_match_weight"]
    ibest = ak.argmax(w, axis=2, keepdims=True)

    def best(field):
        return ak.firsts(clusters[field][ibest], axis=2)

    clusters["mc_best_pid"] = ak.fill_none(best("mc_match_pid"), 0)
    clusters["mc_best_pe"] = ak.fill_none(best("mc_match_pe"), np.nan)
    anc_pids = best("mc_match_ancestor_pids")
    anc_keys = best("mc_match_ancestor_keys")
    # the phi-gamma photon is a DIRECT Bs daughter: first ancestor is the Bs
    # (fromSignal photons are dominated by pi0 decays and bremsstrahlung)
    first_anc = ak.fill_none(ak.firsts(anc_pids, axis=-1), 0)
    is_direct = abs(first_anc) == BS_PDG
    clusters["mc_gamma_from_bs"] = (
        clusters["mc_best_pid"] == GAMMA_PDG
    ) & is_direct
    clusters["mc_bs_key"] = ak.where(
        is_direct, ak.fill_none(ak.firsts(anc_keys, axis=-1), -1), -1
    )
    return clusters


def _pair_phi_gamma(phi, gam):
    """Build Bs candidates from all (phi, photon cluster) pairs per event.

    Returns a plain dict of jagged per-pair arrays (not a combine container).
    """
    iphi, icl = ak.unzip(
        ak.cartesian(
            [
                ak.local_index(phi["vertex_x"], axis=1),
                ak.local_index(gam["x"], axis=1),
            ],
            axis=1,
        )
    )

    def P(field):
        return phi[field][iphi]

    def G(field):
        return gam[field][icl]

    bs = {}
    vx, vy, vz = P("vertex_x"), P("vertex_y"), P("vertex_z")
    dx, dy, dz = G("x") - vx, G("y") - vy, G("z") - vz
    dist = (dx**2 + dy**2 + dz**2) ** 0.5

    e_gamma = G("e_gamma")
    bs["e_gamma"] = e_gamma
    bs["px"] = P("px") + e_gamma * dx / dist
    bs["py"] = P("py") + e_gamma * dy / dist
    bs["pz"] = P("pz") + e_gamma * dz / dist
    bs["energy"] = P("energy") + e_gamma
    p2 = bs["px"] ** 2 + bs["py"] ** 2 + bs["pz"] ** 2
    m2 = bs["energy"] ** 2 - p2
    bs["mass"] = ak.where(m2 > 0, abs(m2) ** 0.5, -(abs(m2) ** 0.5))
    bs["pt"] = (bs["px"] ** 2 + bs["py"] ** 2) ** 0.5
    bs["p"] = p2**0.5
    bs["eta"] = np.arctanh(bs["pz"] / ak.where(bs["p"] > 0, bs["p"], 1.0))
    bs["vertex_x"], bs["vertex_y"], bs["vertex_z"] = vx, vy, vz
    bs["pt_gamma"] = e_gamma * ((dx**2 + dy**2) ** 0.5) / dist

    # Bs pointing: flight from best PV to phi vertex vs Bs momentum
    pvx, pvy, pvz = P("best_pv_x"), P("best_pv_y"), P("best_pv_z")
    fx, fy, fz = vx - pvx, vy - pvy, vz - pvz
    fmag = (fx**2 + fy**2 + fz**2) ** 0.5
    pmag = p2**0.5
    bs["dira"] = (bs["px"] * fx + bs["py"] * fy + bs["pz"] * fz) / ak.where(
        fmag * pmag > 0, fmag * pmag, 1.0
    )

    # phi-level info
    for field, name in [
        ("mass", "phi_mass"),
        ("vertex_chi2", "phi_vertex_chi2"),
        ("pt", "phi_pt"),
        ("vertex_time", "phi_vertex_time"),
        ("fdchi2", "phi_fdchi2"),
    ]:
        if field in phi:
            bs[name] = P(field)
    bs["cluster_e_raw"] = G("e")
    bs["sum_matched_e_raw"] = G("sum_matched_e_raw")
    bs["cluster_time"] = G("time")

    # ingredients to rebuild the mass with any photon energy offline:
    # phi 4-momentum, photon flight direction, true photon energy
    bs["phi_px"], bs["phi_py"], bs["phi_pz"] = P("px"), P("py"), P("pz")
    bs["phi_energy"] = P("energy")
    bs["gamma_ux"] = dx / dist
    bs["gamma_uy"] = dy / dist
    bs["gamma_uz"] = dz / dist
    bs["gamma_e_true"] = G("mc_best_pe")

    # photon emission time at the phi vertex: per-(area x section)
    # aligned timestamps minus TOF, inverse-variance weighted mean
    def _emission(sec):
        d_sec = (
            (G(f"{sec}_x") - vx) ** 2
            + (G(f"{sec}_y") - vy) ** 2
            + (G(f"{sec}_z") - vz) ** 2
        ) ** 0.5
        return G(f"{sec}_time") - G(f"time_bias_{sec}") - d_sec / c_light

    has_f = G("front_e") > 0
    has_b = G("back_e") > 0
    w_f = ak.where(has_f, 1 / picocal_time.SIGMA_FRONT**2, 0.0)
    w_b = ak.where(has_b, 1 / picocal_time.SIGMA_BACK**2, 0.0)
    t_f = ak.where(has_f, _emission("front"), 0.0)
    t_b = ak.where(has_b, _emission("back"), 0.0)
    w_sum = w_f + w_b
    bs["gamma_time"] = ak.where(
        w_sum > 0,
        (w_f * t_f + w_b * t_b) / ak.where(w_sum > 0, w_sum, 1.0),
        np.nan,
    )
    bs["gamma_time_err"] = ak.where(
        has_f & has_b,
        picocal_time.SIGMA_WMEAN,
        ak.where(
            has_b,
            picocal_time.SIGMA_BACK_ONLY,
            ak.where(has_f, picocal_time.SIGMA_FRONT_ONLY, np.nan),
        ),
    )

    # global-fit ingredients (PV -> phi vertex -> photon)
    for f in ("x", "y", "z", "time", "sigma_time"):
        bs[f"best_pv_{f}"] = P(f"best_pv_{f}")
    for i in range(4):
        for j in range(i + 1):
            bs[f"best_pv_cov_{i}_{j}"] = P(f"best_pv_cov_{i}_{j}")
    for i in range(3):
        for j in range(i + 1):
            bs[f"phi_vertex_cov_{i}_{j}"] = P(f"vertex_cov_{i}_{j}")
    if "vertex_sigma_time" in phi:
        bs["phi_vertex_sigma_time"] = P("vertex_sigma_time")
    # phi as pseudo-track: 5x5 covariance of (x, y, tx, ty, q/p)
    for i in range(5):
        for j in range(i + 1):
            bs[f"phi_cov_{i}_{j}"] = P(f"cov_{i}_{j}")
    # photon cluster geometry: barycentre, both sections, seed identity
    bs["cluster_x"], bs["cluster_y"], bs["cluster_z"] = G("x"), G("y"), G("z")
    for sec in ("front", "back"):
        for f in ("x", "y", "z", "e", "time"):
            bs[f"cluster_{sec}_{f}"] = G(f"{sec}_{f}")
    bs["cluster_area"] = G("area")
    bs["cluster_seed_cellid"] = G("seed_cellid")

    # truth: phi from Bs, photon from Bs, and the SAME Bs (shared key)
    phi_truth = (
        ak.values_astype(P("mc_truth"), bool)
        & (abs(P("mc_pid")) == abs(pdg_id("phi(1020)")))
        & ak.any(abs(P("mc_ancestor_pids")) == BS_PDG, axis=-1)
    )
    gamma_truth = G("mc_gamma_from_bs")
    same_bs = ak.any(
        P("mc_ancestor_keys") == G("mc_bs_key")[..., np.newaxis], axis=-1
    )
    bs["is_signal"] = phi_truth & gamma_truth & same_bs
    return bs


def _to_dataframe(bs, event_info):
    n_per_evt = ak.to_numpy(ak.num(bs["mass"]))
    df = pd.DataFrame({k: ak.to_numpy(ak.flatten(v)) for k, v in bs.items()})
    df["run_number"] = np.repeat(event_info["run_number"], n_per_evt)
    df["event_number"] = np.repeat(event_info["event_number"], n_per_evt)
    return df


def _reconstruct_phi(tracks, pvs, track_cuts=None, phi_cuts=None):
    pos = apply_mask(tracks, tracks["charge"] > 0)
    neg = apply_mask(tracks, tracks["charge"] < 0)
    phi = combine([pos, neg], pvs, track_cuts=track_cuts, **(phi_cuts or {}))
    if phi is not None:
        set_composite_pid(phi, "phi(1020)")
    return phi


def _denominator(tracks, clusters, charge_field):
    """Events with a reconstructed K+K- pair from a true phi<-Bs AND a
    cluster best-matched to the SAME Bs's direct photon.

    charge_field selects reco charge ("charge") or true charge
    ("mc_charge") for the opposite-sign requirement.
    """
    is_kaon = np.abs(tracks["mc_pid"]) == abs(pdg_id("K+"))
    has_phi = ak.any(
        np.abs(tracks["mc_ancestor_pids"]) == abs(pdg_id("phi(1020)")),
        axis=-1,
    )
    has_bs = ak.any(np.abs(tracks["mc_ancestor_pids"]) == BS_PDG, axis=-1)
    sig_kaon = is_kaon & has_phi & has_bs
    bs_keys = tracks["mc_ancestor_keys"][
        np.abs(tracks["mc_ancestor_pids"]) == BS_PDG
    ]
    kaon_bs_key = ak.fill_none(ak.firsts(bs_keys, axis=2), -1)

    pos = kaon_bs_key[sig_kaon & (tracks[charge_field] > 0)]
    neg = kaon_bs_key[sig_kaon & (tracks[charge_field] < 0)]
    a, b = ak.unzip(ak.cartesian([pos, neg], axis=1))
    pair_keys = a[a == b]

    gam_keys = clusters["mc_bs_key"][clusters["mc_gamma_from_bs"]]
    x, y = ak.unzip(ak.cartesian([pair_keys, gam_keys], axis=1))
    return ak.to_numpy(ak.any(x == y, axis=1))


def cheated_reconstruction(chunk):
    tracks, pvs, clusters, event_info = _load(chunk)
    clusters = _cluster_truth(clusters)
    # cheated photon energy; switch back to the cluster energy once the
    # shower splitting is fixed upstream
    clusters["e_gamma"] = clusters["mc_best_pe"]

    # denominator = events the cheated algorithm can reconstruct by
    # construction; kaon charge flips are excluded but counted
    denom = _denominator(tracks, clusters, "charge")
    denom_true_q = _denominator(tracks, clusters, "mc_charge")
    counters("events lost to kaon charge flip").add(
        int(np.sum(denom_true_q & ~denom))
    )
    n_true = int(denom.sum())

    # kaons from Bs (truth) as the combine pools
    is_kaon = np.abs(tracks["mc_pid"]) == abs(pdg_id("K+"))
    has_bs = ak.any(np.abs(tracks["mc_ancestor_pids"]) == BS_PDG, axis=-1)
    tracks = apply_mask(tracks, is_kaon & has_bs)
    set_tracks_pid(tracks, "K+")
    tracks = tracks_pv_association(tracks, pvs)

    phi = _reconstruct_phi(tracks, pvs)
    if phi is None:
        rate_counters("cheated efficiency").add(0, n_true)
        return None

    # pair with all truth photon clusters, keep the leading signal one
    gam = apply_mask(clusters, clusters["mc_gamma_from_bs"])
    # summed raw energy of all clusters split off the same photon
    ki, kj = ak.unzip(
        ak.cartesian([gam["mc_bs_key"], gam["mc_bs_key"]], axis=1, nested=True)
    )
    _, ej = ak.unzip(
        ak.cartesian([gam["mc_bs_key"], gam["e"]], axis=1, nested=True)
    )
    gam["sum_matched_e_raw"] = ak.sum(ak.where(ki == kj, ej, 0.0), axis=2)

    bs = _pair_phi_gamma(phi, gam)
    # cheated energies tie between split clusters -> lead by raw energy
    e_sig = ak.mask(bs["cluster_e_raw"], bs["is_signal"])
    ilead = ak.argmax(e_sig, axis=1, keepdims=True)
    keep = bs["is_signal"] & (
        ak.local_index(bs["e_gamma"], axis=1) == ak.fill_none(ilead, -1)
    )
    bs_sig = {k: v[keep] for k, v in bs.items()}

    n_sig_events = int(ak.sum(ak.any(keep, axis=1)))
    rate_counters("cheated efficiency").add(n_sig_events, n_true)
    if int(ak.sum(ak.num(bs_sig["mass"]))) == 0:
        return None

    df = _to_dataframe(bs_sig, event_info)
    counters("candidates").add(len(df))
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Bs -> phi gamma cheated reconstruction"
    )
    parser.add_argument(
        "--input", required=True, help="ROOT file path (wildcards allowed)"
    )
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out-dir", default="public/bs_to_phigamma/reconstruction"
    )
    parser.add_argument("--out-file", default=None)
    args = parser.parse_args()
    args.max_events = args.max_events or None

    out_path = (
        Path(args.out_file)
        if args.out_file
        else Path(args.out_dir) / "cheated.parquet"
    )

    run_reconstruction(
        cheated_reconstruction,
        input_data=args.input,
        out=out_path,
        max_events=args.max_events,
        chunk_size=args.chunk_size,
        workers=args.workers,
        print_throughput=True,
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
