#!/usr/bin/env python3
"""TwoTrackMVA reconstruction: inclusive 2-body SV selection with MVA."""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np

import pandas as pd

from trackcomb import (
    all_in_tree,
    apply_cuts,
    candidates_to_dataframe,
    combine,
    configurable,
    counters,
    cut_max,
    cut_min,
    cut_range,
    get_daughter,
    onnx_models,
    run_reconstruction,
    set_tracks_pid,
    sum_in_tree,
    tracks_pv_association,
)


@configurable
def reconstruction(
    events, model_path="models/two_track_mva.onnx", mva_cut=0.0
):
    """Reconstruct 2-body SVs with MVA selection."""
    tracks, pvs = events["tracks"], events["pvs"]
    event_info = {k: events[k] for k in ("run_number", "event_number")}

    set_tracks_pid(tracks, "pi+")

    tracks = tracks_pv_association(tracks, pvs)

    tracks = apply_cuts(
        tracks,
        [
            cut_min("pt", 200),
            cut_min("min_ip", 0.06),
            cut_max("chi2ndof", 10),
        ],
    )

    candidates = combine(
        [tracks, tracks],
        pvs,
        combination_cuts=[
            cut_max("max_doca", 1.0),
            cut_min(sum_in_tree("pt"), 400),
            cut_min("pt", 1000),
            lambda c: (
                get_daughter(c, 0, "best_pv_index")
                == get_daughter(c, 1, "best_pv_index")
            ),
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

    n_cands = int(ak.sum(ak.num(candidates["vertex_x"])))
    counters("after selection").add(n_cands)
    if n_cands == 0:
        return None

    # MVA features — directly from candidates, no DataFrame
    fdchi2 = ak.to_numpy(ak.flatten(candidates["fdchi2"]))
    vertex_chi2 = ak.to_numpy(ak.flatten(candidates["vertex_chi2"]))
    d0_pt = ak.to_numpy(ak.flatten(get_daughter(candidates, 0, "pt")))
    d1_pt = ak.to_numpy(ak.flatten(get_daughter(candidates, 1, "pt")))
    min_ipchi2 = np.minimum(
        ak.to_numpy(ak.flatten(get_daughter(candidates, 0, "min_ip_chi2"))),
        ak.to_numpy(ak.flatten(get_daughter(candidates, 1, "min_ip_chi2"))),
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
    counters("after MVA").add(int(mva_pass.sum()))

    if not mva_pass.any():
        return None

    # Build DataFrame only for passing candidates
    df = candidates_to_dataframe(candidates)
    cand_per_evt = ak.to_numpy(ak.num(candidates["vertex_x"]))
    df["run_number"] = np.repeat(event_info["run_number"], cand_per_evt)
    df["event_number"] = np.repeat(event_info["event_number"], cand_per_evt)
    df["mva_response"] = mva_response
    return df[mva_pass].copy()


def main():
    parser = argparse.ArgumentParser(
        description="TwoTrackMVA inclusive 2-body SV reconstruction"
    )
    parser.add_argument("--input", required=True, help="ROOT file path")
    parser.add_argument("--tree", default="BestLongTracks/TrackTuple")
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument(
        "--model", default="models/two_track_mva.onnx", help="ONNX model path"
    )
    parser.add_argument("--mva-cut", type=float, default=0.0)
    # parser.add_argument("--mva-cut", type=float, default=0.9569)
    parser.add_argument("--slice-size", type=int, default=1000)
    parser.add_argument(
        "--out-dir", default="public/1p5e34/reconstruction/two_track_mva"
    )
    args = parser.parse_args()
    args.max_events = args.max_events or None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reconstruction.global_bind(model_path=args.model, mva_cut=args.mva_cut)

    results = run_reconstruction(
        reconstruction,
        input_data=args.input,
        tree_name=args.tree,
        max_events=args.max_events,
        slice_size=args.slice_size,
        print_throughput=True,
    )

    dfs = [r for r in (results or []) if r is not None]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df.to_parquet(out_dir / "mva.parquet", index=False)

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
