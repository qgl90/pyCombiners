#!/usr/bin/env python3
"""TwoTrackMVA reconstruction: inclusive 2-body SV selection with MVA."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import awkward as ak
import numpy as np

from trackcomb import (
    DEFAULT_MAX_DT_CHI2,
    all_in_tree,
    apply_cuts,
    candidates_to_dataframe,
    combine,
    composite_pv_association,
    configurable,
    counters,
    cut_max,
    cut_max_ip_chi2,
    cut_min,
    cut_min_ip,
    cut_min_ip_chi2,
    cut_range,
    get_daughter,
    load_event_info,
    load_pvs,
    load_tracks,
    onnx_models,
    run_reconstruction,
    set_tracks_pid,
    sum_in_tree,
    tracks_pv_association,
)

TIMED_COMPOSITE_PV = partial(
    composite_pv_association, max_dt_chi2=DEFAULT_MAX_DT_CHI2
)


@configurable
def reconstruction(chunk, model_path="models/two_track_mva.onnx", mva_cut=0.0):
    """Reconstruct 2-body SVs with MVA selection."""
    tracks = load_tracks(chunk)
    pvs = load_pvs(chunk)
    event_info = load_event_info(chunk)

    set_tracks_pid(tracks, "pi+")

    tracks = tracks_pv_association(
        tracks, pvs, max_dt_chi2=DEFAULT_MAX_DT_CHI2
    )

    tracks = apply_cuts(
        tracks,
        [
            cut_min("pt", 200),
            cut_min_ip(0.06, dt_chi2=DEFAULT_MAX_DT_CHI2),
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
        ],
        # combine() may return None when a cut stage empties the chunk
        composite_cuts=[
            cut_max("vertex_chi2", 20.0),
            cut_max("max_doca", 0.2),
            cut_min("vertex_z", -330),
        ],
        final_cuts=[
            cut_range("flight_eta", 2, 5),
            cut_min("mcor", 1000),
            all_in_tree(cut_min_ip_chi2(4, dt_chi2=DEFAULT_MAX_DT_CHI2)),
            all_in_tree(cut_min("pt", 200)),
            cut_max_ip_chi2(16, dt_chi2=DEFAULT_MAX_DT_CHI2),
        ],
        pv_function=TIMED_COMPOSITE_PV,
        require_same_best_pv=True,
    )

    if candidates is None:
        return None
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
    parser.add_argument(
        "--input", required=True, help="ROOT file path (wildcards allowed)"
    )
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument(
        "--model", default="models/two_track_mva.onnx", help="ONNX model path"
    )
    parser.add_argument("--mva-cut", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out-dir", default="public/bs_to_mumu/reconstruction"
    )
    args = parser.parse_args()
    args.max_events = args.max_events or None

    out_dir = Path(args.out_dir)

    reconstruction.global_bind(model_path=args.model, mva_cut=args.mva_cut)

    run_reconstruction(
        reconstruction,
        input_data=args.input,
        out=out_dir / "mva.parquet",
        max_events=args.max_events,
        chunk_size=args.chunk_size,
        workers=args.workers,
        print_throughput=True,
    )

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
