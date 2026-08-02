#!/usr/bin/env python3
"""Bs -> J/psi(-> mu+mu-) phi(-> K+K-) reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np

from trackcomb import (
    apply_cuts,
    apply_mask,
    candidates_to_dataframe,
    combine,
    compute_bkgcat,
    count_reco_signal,
    count_true_decays,
    counters,
    rate_counters,
    cut_max,
    cut_min,
    cut_range,
    load_event_info,
    load_pvs,
    load_tracks,
    pdg_id,
    run_reconstruction,
    set_composite_pid,
    set_tracks_pid,
    tracks_pv_association,
    all_in_tree,
)


def _load(chunk):
    return load_tracks(chunk), load_pvs(chunk), load_event_info(chunk)


def make_dataframe(candidates, event_info):
    """Convert candidates to DataFrame with event info and truth."""
    df = candidates_to_dataframe(candidates)
    cand_per_evt = ak.to_numpy(ak.num(candidates["vertex_x"]))
    df["run_number"] = np.repeat(event_info["run_number"], cand_per_evt)
    df["event_number"] = np.repeat(event_info["event_number"], cand_per_evt)
    if "bkgcat" not in candidates:
        compute_bkgcat(candidates)
    df["bkgcat"] = ak.to_numpy(ak.flatten(candidates["bkgcat"]))
    df["is_signal"] = df["bkgcat"] <= 10
    return df


def cheated_reconstruction(chunk):
    tracks, pvs, event_info = _load(chunk)

    n_true = int(
        np.sum(count_true_decays(tracks, "B(s)0", ["mu+", "mu-", "K+", "K-"]))
    )

    anc_pids = tracks["mc_ancestor_pids"]
    has_bs = ak.any(np.abs(anc_pids) == abs(pdg_id("B(s)0")), axis=-1)
    tracks = apply_mask(tracks, has_bs)
    tracks = tracks_pv_association(tracks, pvs)

    is_muon = np.abs(tracks["mc_pid"]) == abs(pdg_id("mu+"))
    is_kaon = np.abs(tracks["mc_pid"]) == abs(pdg_id("K+"))

    mu_tracks = apply_mask(tracks, is_muon)
    k_tracks = apply_mask(tracks, is_kaon)

    set_tracks_pid(mu_tracks, "mu+")
    set_tracks_pid(k_tracks, "K+")

    mu_pos = apply_mask(mu_tracks, mu_tracks["charge"] > 0)
    mu_neg = apply_mask(mu_tracks, mu_tracks["charge"] < 0)
    k_pos = apply_mask(k_tracks, k_tracks["charge"] > 0)
    k_neg = apply_mask(k_tracks, k_tracks["charge"] < 0)

    jpsi = combine([mu_pos, mu_neg], pvs)
    if jpsi is None:
        rate_counters("cheated efficiency").add(0, n_true)
        return None
    set_composite_pid(jpsi, "J/psi(1S)")

    phi = combine([k_pos, k_neg], pvs)
    if phi is None:
        rate_counters("cheated efficiency").add(0, n_true)
        return None
    set_composite_pid(phi, "phi(1020)")

    bs = combine([jpsi, phi], pvs)
    if bs is None:
        rate_counters("cheated efficiency").add(0, n_true)
        return None
    set_composite_pid(bs, "B(s)0")

    compute_bkgcat(bs)
    bs = apply_mask(bs, bs["bkgcat"] <= 10)

    n_reco = count_reco_signal(bs, "B(s)0")
    rate_counters("cheated efficiency").add(n_reco, n_true)

    if n_reco == 0:
        return None

    df = make_dataframe(bs, event_info)
    counters("candidates").add(len(df))
    return df


def full_reconstruction(chunk):
    """Full Bs -> J/psi(mu+mu-) phi(K+K-) reconstruction with cuts."""
    tracks, pvs, event_info = _load(chunk)

    tracks = tracks_pv_association(tracks, pvs)
    tracks = apply_cuts(
        tracks,
        [
            cut_min("pt", 500),
            cut_min("min_ip", 0.01),
            cut_min("min_ip_chi2", 4),
        ],
    )

    pos_tracks = apply_mask(tracks, tracks["charge"] > 0)
    neg_tracks = apply_mask(tracks, tracks["charge"] < 0)

    mu_pos, mu_neg = {**pos_tracks}, {**neg_tracks}
    set_tracks_pid(mu_pos, "mu+")
    set_tracks_pid(mu_neg, "mu+")
    k_pos, k_neg = {**pos_tracks}, {**neg_tracks}
    set_tracks_pid(k_pos, "K+")
    set_tracks_pid(k_neg, "K+")

    jpsi = combine(
        [mu_pos, mu_neg],
        pvs,
        combination_cuts=[
            cut_max("max_doca", 0.1),
            cut_range("mass", 2800, 3400),
        ],
        composite_cuts=[
            cut_max("vertex_chi2", 16),
            cut_min("pt", 500),
        ],
    )
    if jpsi is None:
        return None
    set_composite_pid(jpsi, "J/psi(1S)")

    phi = combine(
        [k_pos, k_neg],
        pvs,
        combination_cuts=[
            cut_max("max_doca", 0.1),
            cut_range("mass", 990, 1060),
        ],
        composite_cuts=[
            cut_max("vertex_chi2", 16),
        ],
    )
    if phi is None:
        return None
    set_composite_pid(phi, "phi(1020)")

    bs = combine(
        [jpsi, phi],
        pvs,
        combination_cuts=[
            cut_max("max_doca", 0.1),
            cut_range("mass", 5100, 5600),
        ],
        composite_cuts=[
            cut_max("vertex_chi2", 15),
            cut_max("pair_time_chi2", 15),
            cut_min("pt", 2000),
        ],
        final_cuts=[
            cut_min("dira", 0.9995),
            cut_max("composite_ip", 0.1),
            cut_max("composite_ip_chi2", 16),
            cut_min("fdchi2", 2),
        ],
    )
    if bs is None:
        return None
    set_composite_pid(bs, "B(s)0")

    if int(ak.sum(ak.num(bs["vertex_x"]))) == 0:
        return None

    df = make_dataframe(bs, event_info)
    counters("candidates").add(len(df))
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Bs -> J/psi phi reconstruction"
    )
    parser.add_argument("--mode", required=True, choices=["cheated", "full"])
    parser.add_argument(
        "--input", required=True, help="ROOT file path (wildcards allowed)"
    )
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out-dir", default="public/bs_to_jpsiphi/reconstruction"
    )
    parser.add_argument("--out-file", default=None)
    args = parser.parse_args()
    args.max_events = args.max_events or None

    if args.mode == "cheated":
        reco_fn = cheated_reconstruction
    else:
        reco_fn = full_reconstruction

    out_path = (
        Path(args.out_file)
        if args.out_file
        else Path(args.out_dir) / f"{args.mode}.parquet"
    )

    run_reconstruction(
        reco_fn,
        input_data=args.input,
        out=out_path,
        max_events=args.max_events,
        chunk_size=args.chunk_size,
        workers=args.workers,
        print_throughput=True,
    )

    print(f"\nMode: {args.mode}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
