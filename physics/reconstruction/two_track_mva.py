#!/usr/bin/env python3
"""TwoTrackMVA reconstruction: inclusive 2-body SV selection with MVA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import awkward as ak
import numpy as np

from trackcomb import (
    apply_mask,
    candidates_to_dataframe,
    combine,
    cut_max,
    extract_daughter_fields,
    load_events_root,
    set_tracks_pid,
    tracks_pv_association,
)


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
    parser.add_argument(
        "--out-dir", default="public/1p5e34/reconstruction/two_track_mva"
    )
    args = parser.parse_args()
    args.max_events = args.max_events or None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # 1. Load events
    tracks, pvs, event_info = load_events_root(args.input, args.tree, args.max_events)

    # 2. Track-PV association
    tracks = tracks_pv_association(tracks, pvs)

    # 3. Track preselection
    mask = (tracks["pt"] > 200) & (tracks["min_ip"] > 0.06)
    if "chi2ndof" in tracks:
        mask = mask & (tracks["chi2ndof"] < 10)
    tracks = apply_mask(tracks, mask)

    # 4. Pion mass hypothesis (inclusive, no charge split)
    tracks = set_tracks_pid(tracks, "pi+")

    # 5. Combine all unique pairs
    def cut_same_pv(c):
        return c["daughter0_best_pv_index"] == c["daughter1_best_pv_index"]

    def cut_sum_pt(c):
        return c["daughter0_pt"] + c["daughter1_pt"] > 400

    candidates = combine(
        [tracks, tracks],
        pvs,
        combination_cuts=[
            cut_max("max_doca", 1.0),
            # cut_same_pv, # Remove the same pv cut
            cut_sum_pt,
        ],
        vertex_cuts=[
            cut_max("vertex_chi2", 20.0),
            lambda c: c["pt"] > 1000,
        ],
    )

    n_events = len(event_info["run_number"])
    n_cands = int(ak.sum(ak.num(candidates["vertex_x"])))

    if n_cands == 0:
        import pandas as pd

        pd.DataFrame().to_parquet(out_dir / "mva.parquet", index=False)
        (out_dir / "mva_summary.json").write_text(
            json.dumps(
                {
                    "n_events": n_events,
                    "n_candidates_presel": 0,
                    "n_candidates_mva": 0,
                },
                indent=2,
            )
        )
        elapsed = time.perf_counter() - t0
        print(f"No candidates found in {n_events} events ({elapsed:.2f}s)")
        return

    # 6. Flatten for line selection
    df = candidates_to_dataframe(candidates)
    cand_per_evt = ak.to_numpy(ak.num(candidates["vertex_x"]))
    df["run_number"] = np.repeat(event_info["run_number"], cand_per_evt)
    df["event_number"] = np.repeat(event_info["event_number"], cand_per_evt)
    for k, v in extract_daughter_fields(candidates).items():
        df[k] = v

    # 7. Line selection cuts (post-combine, only cuts needing PV association)
    sel = np.ones(len(df), dtype=bool)
    sel &= (df["flight_eta"].values > 2) & (df["flight_eta"].values < 5)
    sel &= df["mcor"].values > 1000  # mcor > 1000 MeV
    sel &= df["max_doca"].values < 0.2
    sel &= df["vertex_z"].values >= -330.0

    min_daughter_ipchi2 = np.minimum(
        df["daughter0_min_ip_chi2"].values, df["daughter1_min_ip_chi2"].values
    )
    sel &= min_daughter_ipchi2 > 4

    min_daughter_pt = np.minimum(df["daughter0_pt"].values, df["daughter1_pt"].values)
    sel &= min_daughter_pt > 200

    sel &= df["composite_ip_chi2"].values < 16

    n_presel = int(sel.sum())
    print(f"Line selection: {n_presel} / {n_cands} candidates pass")

    if n_presel == 0:
        import pandas as pd

        pd.DataFrame().to_parquet(out_dir / "mva.parquet", index=False)
        (out_dir / "mva_summary.json").write_text(
            json.dumps(
                {
                    "n_events": n_events,
                    "n_candidates_presel": 0,
                    "n_candidates_mva": 0,
                },
                indent=2,
            )
        )
        elapsed = time.perf_counter() - t0
        print(f"Time: {elapsed:.2f}s ({n_events / elapsed:.1f} evt/s)")
        return

    # 8. MVA features (on pre-selected candidates)
    fdchi2 = df["fdchi2"].values[sel]
    vertex_chi2 = df["vertex_chi2"].values[sel]
    sumpt = (df["daughter0_pt"].values + df["daughter1_pt"].values)[sel]
    features = np.column_stack(
        [
            np.log(np.maximum(fdchi2, 1e-10)),
            sumpt,
            np.maximum(vertex_chi2, 1e-10),
            np.log(np.maximum(min_daughter_ipchi2[sel], 1e-10)),
        ]
    ).astype(np.float32)

    # 9. ONNX inference
    import onnxruntime as ort

    session = ort.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name
    mva_response = session.run(None, {input_name: features})[0].flatten()

    # 10. MVA cut
    mva_pass = mva_response > args.mva_cut
    n_mva = int(mva_pass.sum())
    print(f"MVA cut (> {args.mva_cut}): {n_mva} / {n_presel} candidates pass")

    # 11. Build output DataFrame
    df_sel = df.iloc[np.where(sel)[0]].copy()
    df_sel["mva_response"] = mva_response
    df_final = df_sel[mva_pass].copy()

    df_final.to_parquet(out_dir / "mva.parquet", index=False)

    (out_dir / "mva_summary.json").write_text(
        json.dumps(
            {
                "n_events": n_events,
                "n_candidates_presel": n_presel,
                "n_candidates_mva": n_mva,
            },
            indent=2,
        )
    )

    elapsed = time.perf_counter() - t0
    print(f"\nEvents: {n_events}")
    print(f"Candidates after combine: {n_cands}")
    print(f"Candidates after line selection: {n_presel}")
    print(f"Candidates after MVA: {n_mva}")
    print(f"Time: {elapsed:.2f}s ({n_events / elapsed:.1f} evt/s)")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
