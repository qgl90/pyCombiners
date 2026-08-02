#!/usr/bin/env python3
"""Ks -> pi+pi- reconstruction: run a single combiner mode, save to Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np

from trackcomb import (
    configurable,
    apply_cuts,
    apply_mask,
    candidates_to_dataframe,
    combine,
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
    run_reconstruction,
    set_composite_pid,
    set_tracks_pid,
    tracks_pv_association,
    compute_bkgcat,
    pdg_id,
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


def rich_electron_veto(tracks):
    """Reject electron-like tracks; tracks without RICH info are kept.

    RICH DLLs are relative to the pion hypothesis (DLL_Pion == 0 by
    construction), so pion ID for Ks daughters is expressed as vetoes.
    Kaon/proton vetoes were scanned and rejected: Ks background is
    dominated by genuine pion combinatorics, PID cannot remove it.
    """
    return ~(tracks["rich_dll_electron"] > 0.0)


@configurable
def cheated_reconstruction(chunk, candidate_cuts=None):
    tracks, pvs, event_info = _load(chunk)

    n_true = int(np.sum(count_true_decays(tracks, "K(S)0", ["pi+", "pi-"])))

    # Pre-filter to tracks with Ks ancestor (for speed)
    anc_pids = tracks["mc_ancestor_pids"]
    has_ks = ak.any(np.abs(anc_pids) == abs(pdg_id("K(S)0")), axis=-1)
    tracks = apply_mask(tracks, has_ks)

    set_tracks_pid(tracks, "pi+")
    tracks = tracks_pv_association(tracks, pvs)
    pos_tracks = apply_mask(tracks, tracks["charge"] > 0)
    neg_tracks = apply_mask(tracks, tracks["charge"] < 0)

    candidates = combine([pos_tracks, neg_tracks], pvs)
    if candidates is None:
        rate_counters("cheated efficiency").add(0, n_true)
        return None
    set_composite_pid(candidates, "K(S)0")

    if candidate_cuts:
        candidates = apply_cuts(candidates, candidate_cuts)
        if int(ak.sum(ak.num(candidates["vertex_x"]))) == 0:
            rate_counters("cheated efficiency").add(0, n_true)
            return None

    compute_bkgcat(candidates)
    candidates = apply_mask(candidates, candidates["bkgcat"] <= 10)

    n_reco = count_reco_signal(candidates, "K(S)0")
    rate_counters("cheated efficiency").add(n_reco, n_true)

    if n_reco == 0:
        return None

    df = make_dataframe(candidates, event_info)
    counters("candidates").add(len(df))
    return df


@configurable
def reconstruction(chunk, mode="full"):
    tracks, pvs, event_info = _load(chunk)

    tracks = tracks_pv_association(tracks, pvs)
    set_tracks_pid(tracks, "pi+")
    pos_tracks = apply_mask(tracks, tracks["charge"] > 0)
    neg_tracks = apply_mask(tracks, tracks["charge"] < 0)

    if mode == "full":
        cuts = {
            "track_cuts": [
                cut_min("pt", 50),
                cut_min("min_ip", 0.1),
                rich_electron_veto,
            ],
            "combination_cuts": [
                cut_max("max_doca", 0.15),
                cut_range("mass", 470, 520),
            ],
            "composite_cuts": [
                cut_max("vertex_chi2", 10.0),
                cut_max("pair_time_chi2", 10.0),
            ],
            "final_cuts": [
                cut_min("dira", 0.999),
                cut_max("composite_ip", 0.5),
                cut_min("fdchi2", 100),
            ],
        }
    elif mode == "dist":
        cuts = {
            "track_cuts": [cut_min("pt", 60), cut_min("min_ip", 0.08)],
            "combination_cuts": [
                cut_max("max_doca", 0.3),
                cut_range("mass", 470, 520),
            ],
            "composite_cuts": [
                cut_max("vertex_chi2", 10.0),
                cut_max("pair_time_chi2", 10.0),
            ],
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    candidates = combine([pos_tracks, neg_tracks], pvs, **cuts)
    if candidates is None:
        return None
    set_composite_pid(candidates, "K(S)0")

    if int(ak.sum(ak.num(candidates["vertex_x"]))) == 0:
        return None

    df = make_dataframe(candidates, event_info)
    counters("candidates").add(len(df))
    return df


def main():
    parser = argparse.ArgumentParser(description="Ks -> pi+pi- reconstruction")
    parser.add_argument(
        "--mode", required=True, choices=["cheated", "full", "dist"]
    )
    parser.add_argument(
        "--input", required=True, help="ROOT file path (wildcards allowed)"
    )
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out-dir", default="public/ks_to_pipi/reconstruction"
    )
    parser.add_argument(
        "--out-file", default=None, help="Override output file path"
    )
    args = parser.parse_args()
    args.max_events = args.max_events or None

    if args.mode == "cheated":
        reco_fn = cheated_reconstruction
    else:
        reconstruction.global_bind(mode=args.mode)
        reco_fn = reconstruction

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
