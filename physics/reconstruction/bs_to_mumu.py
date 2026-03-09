#!/usr/bin/env python3
"""Bs -> mu+mu- reconstruction: run a single combiner mode, save to Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

import pandas as pd

from trackcomb import (
    configurable,
    apply_mask,
    apply_cuts,
    all_in_tree,
    any_in_tree,
    candidates_to_dataframe,
    combine,
    compute_bkgcat,
    count_reco_signal,
    count_true_decays,
    counters,
    rate_counters,
    onnx_models,
    cut_max,
    cut_min,
    cut_range,
    get_daughter,
    pdg_id,
    run_reconstruction,
    set_composite_pid,
    set_tracks_pid,
    sum_in_tree,
    tracks_pv_association,
)


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


def cheated_reconstruction(events):
    tracks, pvs = events["tracks"], events["pvs"]
    event_info = {k: events[k] for k in ("run_number", "event_number")}

    n_true = int(np.sum(count_true_decays(tracks, "B(s)0", ["mu+", "mu-"])))

    is_muon = np.abs(tracks["mc_pid"]) == abs(pdg_id("mu+"))
    anc_pids = tracks["mc_ancestor_pids"]
    has_bs = ak.any(np.abs(anc_pids) == abs(pdg_id("B(s)0")), axis=-1)
    tracks = apply_mask(tracks, is_muon & has_bs)

    set_tracks_pid(tracks, "mu+")
    tracks = tracks_pv_association(tracks, pvs)
    pos_tracks = apply_mask(tracks, tracks["charge"] > 0)
    neg_tracks = apply_mask(tracks, tracks["charge"] < 0)

    candidates = combine([pos_tracks, neg_tracks], pvs)
    if candidates is None:
        rate_counters("cheated efficiency").add(0, n_true)
        return None
    set_composite_pid(candidates, "B(s)0")

    compute_bkgcat(candidates)
    candidates = apply_mask(candidates, candidates["bkgcat"] <= 10)

    n_reco = count_reco_signal(candidates, "B(s)0")
    rate_counters("cheated efficiency").add(n_reco, n_true)

    if n_reco == 0:
        return None

    df = make_dataframe(candidates, event_info)
    counters("candidates").add(len(df))
    return df


@configurable
def reconstruction(events, mode="full"):
    tracks, pvs = events["tracks"], events["pvs"]
    event_info = {k: events[k] for k in ("run_number", "event_number")}

    set_tracks_pid(tracks, "mu+")
    if mode == "dist":
        tracks = tracks_pv_association(tracks, pvs)
    pos_tracks = apply_mask(tracks, tracks["charge"] > 0)
    neg_tracks = apply_mask(tracks, tracks["charge"] < 0)

    if mode == "full":
        cuts: dict[str, Any] = {
            "track_cuts": [cut_min("pt", 1000)],
            "combination_cuts": [
                cut_max("max_doca", 0.05),
                cut_range("mass", 4700, 6000),
            ],
            "composite_cuts": [
                cut_max("vertex_chi2", 4.0),
                cut_max("pair_time_chi2", 4.0),
                cut_min("pt", 1000),
                any_in_tree(cut_min("pt", 2000)),
            ],
            "final_cuts": [
                cut_min("dira", 0.9995),
                cut_max("composite_ip", 0.1),
                cut_max("composite_ip_chi2", 16),
            ],
        }
    elif mode == "full_notime":
        cuts = {
            "track_cuts": [cut_min("pt", 1000)],
            "combination_cuts": [
                cut_max("max_doca", 0.05),
                cut_range("mass", 4700, 6000),
            ],
            "composite_cuts": [
                cut_max("vertex_chi2", 4.0),
                cut_min("pt", 1000),
                any_in_tree(cut_min("pt", 2000)),
            ],
            "final_cuts": [
                cut_min("dira", 0.9995),
                cut_max("composite_ip", 0.1),
                cut_max("composite_ip_chi2", 16),
            ],
        }
    elif mode == "dist":
        cuts = {
            "track_cuts": [cut_min("pt", 800)],
            "combination_cuts": [
                cut_max("max_doca", 0.2),
                cut_range("mass", 4500, 6000),
            ],
            "composite_cuts": [
                cut_max("vertex_chi2", 10.0),
                cut_max("pair_time_chi2", 10.0),
                cut_min("pt", 1000),
                any_in_tree(cut_min("pt", 2000)),
            ],
            "final_cuts": [
                cut_min("dira", 0.9995),
                cut_max("composite_ip", 0.1),
                cut_max("composite_ip_chi2", 16),
            ],
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    candidates = combine([pos_tracks, neg_tracks], pvs, **cuts)
    set_composite_pid(candidates, "B(s)0")

    if int(ak.sum(ak.num(candidates["vertex_x"]))) == 0:
        return None

    df = make_dataframe(candidates, event_info)
    counters("candidates").add(len(df))
    return df


def with_pvtag(fn, model_path, mva_cut):
    """Wrap a reconstruction function with PV tagging prefilter."""

    def wrapped(events):
        events = pvtag_events(events, model_path, mva_cut)
        return fn(events) if events is not None else None

    return wrapped


def pvtag_events(events, model_path, mva_cut):
    """Filter events by TwoTrackMVA PV tagging. Returns new events or None."""
    tracks, pvs = events["tracks"], events["pvs"]
    n_events = len(events["run_number"])

    # PV association for all tracks
    tracks = tracks_pv_association(tracks, pvs)

    # MVA track preselection
    mva_tracks = apply_cuts(
        tracks,
        [
            cut_min("pt", 200),
            cut_min("min_ip", 0.06),
            cut_max("chi2ndof", 10),
        ],
    )
    set_tracks_pid(mva_tracks, "pi+")

    mva_cands = combine(
        [mva_tracks, mva_tracks],
        pvs,
        combination_cuts=[
            cut_max("max_doca", 1.0),
            cut_min(sum_in_tree("pt"), 400),
            cut_min("pt", 1000),
        ],
        composite_cuts=[
            cut_max("vertex_chi2", 20.0),
            cut_max("max_doca", 0.2),
            cut_min("vertex_z", -330),
        ],
        final_cuts=[
            cut_range("flight_eta", 2, 5),
            cut_min("mcor", 1000),
            all_in_tree(cut_min("min_ip_chi2", 4)),
            all_in_tree(cut_min("pt", 200)),
            cut_max("composite_ip_chi2", 16),
        ],
    )

    if int(ak.sum(ak.num(mva_cands["vertex_x"]))) == 0:
        return None

    # MVA feature extraction — directly from candidates, no DataFrame
    cand_per_evt = ak.to_numpy(ak.num(mva_cands["vertex_x"]))
    fdchi2 = ak.to_numpy(ak.flatten(mva_cands["fdchi2"]))
    vertex_chi2 = ak.to_numpy(ak.flatten(mva_cands["vertex_chi2"]))
    d0_pt = ak.to_numpy(ak.flatten(get_daughter(mva_cands, 0, "pt")))
    d1_pt = ak.to_numpy(ak.flatten(get_daughter(mva_cands, 1, "pt")))
    min_ipchi2 = np.minimum(
        ak.to_numpy(ak.flatten(get_daughter(mva_cands, 0, "min_ip_chi2"))),
        ak.to_numpy(ak.flatten(get_daughter(mva_cands, 1, "min_ip_chi2"))),
    )

    features = np.column_stack(
        [
            np.log(np.maximum(fdchi2, 1e-10)),
            (d0_pt + d1_pt) / 1000,  # MeV -> GeV (model trained in GeV)
            np.maximum(vertex_chi2, 1e-10),
            np.log(np.maximum(min_ipchi2, 1e-10)),
        ]
    ).astype(np.float32)

    mva_response = onnx_models(model_path).run(features)
    mva_pass = mva_response > mva_cut
    if not mva_pass.any():
        return None

    # Tagged (event_idx, pv_index) pairs — vectorized with key encoding
    event_idx = np.repeat(np.arange(n_events, dtype=np.int64), cand_per_evt)
    best_pv = ak.to_numpy(ak.flatten(mva_cands["best_pv_index"])).astype(
        np.int64
    )

    flat_pv_idx = ak.to_numpy(ak.flatten(pvs["pv_index"])).astype(np.int64)
    stride = int(flat_pv_idx.max()) + 1 if len(flat_pv_idx) > 0 else 1
    tagged_keys = np.unique(event_idx[mva_pass] * stride + best_pv[mva_pass])

    # Filter tracks and PVs by tagged pairs
    flat_bpi = ak.to_numpy(ak.flatten(tracks["best_pv_index"])).astype(
        np.int64
    )
    tracks_per_event = ak.to_numpy(ak.num(tracks["best_pv_index"]))
    track_ei = np.repeat(np.arange(n_events, dtype=np.int64), tracks_per_event)
    track_mask = ak.unflatten(
        np.isin(track_ei * stride + flat_bpi, tagged_keys), tracks_per_event
    )

    pvs_per_event = ak.to_numpy(ak.num(pvs["pv_index"]))
    pv_ei = np.repeat(np.arange(n_events, dtype=np.int64), pvs_per_event)
    pv_mask = ak.unflatten(
        np.isin(pv_ei * stride + flat_pv_idx, tagged_keys), pvs_per_event
    )

    return {
        **events,
        "tracks": apply_mask(tracks, track_mask),
        "pvs": apply_mask(pvs, pv_mask),
    }


def main():
    parser = argparse.ArgumentParser(description="Bs -> mu+mu- reconstruction")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["cheated", "full", "full_notime", "dist"],
    )
    parser.add_argument("--input", required=True, help="ROOT file path")
    parser.add_argument("--tree", default="BestLongTracks/TrackTuple")
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--slice-size", type=int, default=1000)
    parser.add_argument(
        "--out-dir", default="public/1p5e34/reconstruction/bs_to_mumu"
    )
    parser.add_argument(
        "--out-file", default=None, help="Override output file path"
    )
    parser.add_argument(
        "--pvtag", action="store_true", help="Enable PV tagging prefilter"
    )
    parser.add_argument(
        "--model", default=None, help="ONNX model path (--pvtag)"
    )
    parser.add_argument(
        "--mva-cut", type=float, default=0.9569, help="MVA cut (--pvtag)"
    )
    args = parser.parse_args()
    args.max_events = args.max_events or None

    if args.pvtag and not args.model:
        parser.error("--model is required when using --pvtag")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "cheated":
        reco_fn = cheated_reconstruction
    else:
        reconstruction.global_bind(mode=args.mode)
        if args.mode == "full_notime":
            set_tracks_pid.global_bind(fit_track_time=False)
        reco_fn = reconstruction

    if args.pvtag:
        reco_fn = with_pvtag(reco_fn, args.model, args.mva_cut)

    results = run_reconstruction(
        reco_fn,
        input_data=args.input,
        tree_name=args.tree,
        max_events=args.max_events,
        slice_size=args.slice_size,
        print_throughput=True,
    )

    dfs = [r for r in (results or []) if r is not None]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    label = f"{args.mode}_pvtag" if args.pvtag else args.mode
    out_path = (
        Path(args.out_file) if args.out_file else out_dir / f"{label}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"\nMode: {label}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
