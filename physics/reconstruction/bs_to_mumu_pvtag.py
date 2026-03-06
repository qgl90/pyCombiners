#!/usr/bin/env python3
"""Bs -> mu+mu- reconstruction with TwoTrackMVA PV-tagging pre-filter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

from trackcomb import (
    apply_mask,
    candidates_to_dataframe,
    combine,
    cut_max,
    cut_min,
    cut_range,
    extract_daughter_fields,
    load_events_root,
    pdg_id,
    set_tracks_pid,
    tracks_pv_association,
)
from trackcomb.truth import bkgcat, count_true_decays


def main():
    parser = argparse.ArgumentParser(
        description="Bs -> mu+mu- reconstruction with PV-tagging"
    )
    parser.add_argument("--input", required=True, help="ROOT file path")
    parser.add_argument("--tree", default="BestLongTracks/TrackTuple")
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument(
        "--model", default="models/two_track_mva.onnx", help="ONNX model path"
    )
    parser.add_argument(
        "--mva-cut", type=float, default=0.9569, help="MVA cut threshold"
    )
    parser.add_argument("--out-dir", default="public/1p5e34/reconstruction/bs_to_mumu")
    args = parser.parse_args()
    args.max_events = args.max_events or None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ================================================================
    # Load events
    # ================================================================
    tracks, pvs, event_info = load_events_root(args.input, args.tree, args.max_events)
    all_tracks = tracks
    n_events = len(event_info["run_number"])

    # Track-PV association (timing-based)
    tracks = tracks_pv_association(tracks, pvs)

    # ================================================================
    # Stage 1: TwoTrackMVA PV Tagging
    # ================================================================

    # MVA track preselection
    mva_mask = (tracks["pt"] > 200) & (tracks["min_ip"] > 0.06)
    if "chi2ndof" in tracks:
        mva_mask = mva_mask & (tracks["chi2ndof"] < 10)
    mva_tracks = apply_mask(tracks, mva_mask)
    mva_tracks = set_tracks_pid(mva_tracks, "pi+")

    # Combine all unique pairs (inclusive)
    def cut_same_pv(c):
        return c["daughter0_best_pv_index"] == c["daughter1_best_pv_index"]

    def cut_sum_pt(c):
        return c["daughter0_pt"] + c["daughter1_pt"] > 400

    mva_cands = combine(
        [mva_tracks, mva_tracks],
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

    n_mva_raw = int(ak.sum(ak.num(mva_cands["vertex_x"])))

    # Flatten and apply line selection + MVA
    tagged_pairs = set()  # (event_idx, pv_index)

    if n_mva_raw > 0:
        df_mva = candidates_to_dataframe(mva_cands)
        cand_per_evt = ak.to_numpy(ak.num(mva_cands["vertex_x"]))
        df_mva["event_idx"] = np.repeat(np.arange(n_events), cand_per_evt)
        for k, v in extract_daughter_fields(mva_cands).items():
            df_mva[k] = v

        # Line selection (same as two_track_mva.py)
        sel = np.ones(len(df_mva), dtype=bool)
        sel &= (df_mva["flight_eta"].values > 2) & (df_mva["flight_eta"].values < 5)
        sel &= df_mva["mcor"].values > 1000
        sel &= df_mva["max_doca"].values < 0.2
        sel &= df_mva["vertex_z"].values >= -330.0

        min_daughter_ipchi2 = np.minimum(
            df_mva["daughter0_min_ip_chi2"].values,
            df_mva["daughter1_min_ip_chi2"].values,
        )
        sel &= min_daughter_ipchi2 > 4

        min_daughter_pt = np.minimum(
            df_mva["daughter0_pt"].values, df_mva["daughter1_pt"].values
        )
        sel &= min_daughter_pt > 200

        sel &= df_mva["composite_ip_chi2"].values < 16

        n_presel = int(sel.sum())

        if n_presel > 0:
            # MVA inference
            import onnxruntime as ort

            fdchi2 = df_mva["fdchi2"].values[sel]
            vertex_chi2 = df_mva["vertex_chi2"].values[sel]
            sumpt = (df_mva["daughter0_pt"].values + df_mva["daughter1_pt"].values)[sel]
            features = np.column_stack(
                [
                    np.log(np.maximum(fdchi2, 1e-10)),
                    sumpt,
                    np.maximum(vertex_chi2, 1e-10),
                    np.log(np.maximum(min_daughter_ipchi2[sel], 1e-10)),
                ]
            ).astype(np.float32)

            session = ort.InferenceSession(args.model)
            input_name = session.get_inputs()[0].name
            mva_response = session.run(None, {input_name: features})[0].flatten()

            mva_pass = mva_response > args.mva_cut
            n_mva_pass = int(mva_pass.sum())

            # Extract tagged PV indices
            sel_indices = np.where(sel)[0]
            pass_indices = sel_indices[mva_pass]
            for idx in pass_indices:
                ei = int(df_mva["event_idx"].iloc[idx])
                pi = int(df_mva["best_pv_index"].iloc[idx])
                tagged_pairs.add((ei, pi))
        else:
            n_mva_pass = 0
    else:
        n_presel = 0
        n_mva_pass = 0

    n_tagged_pvs = len(tagged_pairs)
    n_events_with_tag = len({ei for ei, _ in tagged_pairs})
    print(f"\nMVA: {n_mva_raw} raw → {n_presel} presel → {n_mva_pass} pass")
    print(f"Tagged PVs: {n_tagged_pvs} across {n_events_with_tag} events")

    # ================================================================
    # Stage 2: Track + PV Filtering
    # ================================================================
    n_tracks_before = int(ak.sum(ak.num(tracks["pt"])))

    if n_tagged_pvs == 0:
        # No tagged PVs → no candidates possible
        n_tracks_after = 0
        filtered_tracks = None
        filtered_pvs = None
    else:
        # Filter tracks: keep only those with best_pv_index in tagged set
        flat_bpi = ak.to_numpy(ak.flatten(tracks["best_pv_index"]))
        tracks_per_event = ak.to_numpy(ak.num(tracks["best_pv_index"]))
        event_indices = np.repeat(np.arange(n_events), tracks_per_event)
        flat_track_mask = np.array(
            [
                (int(ei), int(pi)) in tagged_pairs
                for ei, pi in zip(event_indices, flat_bpi)
            ]
        )
        track_mask = ak.unflatten(flat_track_mask, tracks_per_event)
        filtered_tracks = apply_mask(tracks, track_mask)

        # Filter PVs: keep only tagged PVs
        flat_pv_idx = ak.to_numpy(ak.flatten(pvs["pv_index"]))
        pvs_per_event = ak.to_numpy(ak.num(pvs["pv_index"]))
        pv_event_indices = np.repeat(np.arange(n_events), pvs_per_event)
        flat_pv_mask = np.array(
            [
                (int(ei), int(pi)) in tagged_pairs
                for ei, pi in zip(pv_event_indices, flat_pv_idx)
            ]
        )
        pv_mask = ak.unflatten(flat_pv_mask, pvs_per_event)
        filtered_pvs = apply_mask(pvs, pv_mask)

        n_tracks_after = int(ak.sum(ak.num(filtered_tracks["pt"])))

    print(f"Tracks: {n_tracks_before} → {n_tracks_after}")

    # ================================================================
    # Stage 3: Bs→μμ Reconstruction
    # ================================================================
    n_cands_raw = 0
    df = None

    if filtered_tracks is not None:
        bs_tracks = set_tracks_pid(filtered_tracks, "mu+")
        pos = apply_mask(bs_tracks, bs_tracks["charge"] > 0)
        neg = apply_mask(bs_tracks, bs_tracks["charge"] < 0)

        candidates = combine(
            [pos, neg],
            filtered_pvs,
            track_cuts=[cut_min("pt", 1000)],
            combination_cuts=[
                cut_max("max_doca", 0.05),
                cut_max("spatial_chi2", 4.0),
            ],
            vertex_cuts=[
                cut_range("mass", 4700, 6000),
                cut_max("pair_time_chi2", 4.0),
            ],
        )
        candidates["pid"] = pdg_id("B(s)0")

        n_cands_raw = int(ak.sum(ak.num(candidates["vertex_x"])))

        if n_cands_raw > 0:
            df_all = candidates_to_dataframe(candidates)
            cand_per_evt = ak.to_numpy(ak.num(candidates["vertex_x"]))
            df_all["run_number"] = np.repeat(event_info["run_number"], cand_per_evt)
            df_all["event_number"] = np.repeat(event_info["event_number"], cand_per_evt)
            for k, v in extract_daughter_fields(candidates).items():
                df_all[k] = v
            df_all["bkgcat"] = ak.to_numpy(ak.flatten(bkgcat(candidates)))
            df_all["is_signal"] = df_all["bkgcat"] <= 10

            # Post-combine cuts (same as bs_to_mumu "full")
            psel = np.ones(len(df_all), dtype=bool)
            psel &= df_all["dira"].isna() | (df_all["dira"] >= 0.9995)
            psel &= df_all["composite_ip"].isna() | (df_all["composite_ip"] < 0.1)
            psel &= df_all["composite_ip_chi2"].isna() | (
                df_all["composite_ip_chi2"] < 16
            )
            psel &= df_all["pt"] >= 1000
            daughter_pt_cols = sorted(
                c
                for c in df_all.columns
                if c.endswith("_pt") and c.startswith("daughter")
            )
            if daughter_pt_cols:
                psel &= np.nanmax(df_all[daughter_pt_cols].values, axis=1) >= 2000

            df = df_all[psel].copy()

    # ================================================================
    # Save results
    # ================================================================
    n_true = int(count_true_decays(all_tracks, "B(s)0", ["mu+", "mu-"]).sum())

    out_path = out_dir / "pvtag.parquet"
    if df is None or len(df) == 0:
        import pandas as pd

        pd.DataFrame().to_parquet(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    # ================================================================
    # Cheated reference (same events, unfiltered)
    # ================================================================
    cheated_path = out_dir / "pvtag_cheated.parquet"
    ch_tracks = all_tracks
    is_muon = np.abs(ch_tracks["mc_pid"]) == abs(pdg_id("mu+"))
    anc_pids = ch_tracks["mc_ancestor_pids"]
    has_bs = ak.any(np.abs(anc_pids) == abs(pdg_id("B(s)0")), axis=-1)
    ch_tracks = apply_mask(ch_tracks, is_muon & has_bs)
    ch_tracks = set_tracks_pid(ch_tracks, "mu+")
    ch_pos = apply_mask(ch_tracks, ch_tracks["charge"] > 0)
    ch_neg = apply_mask(ch_tracks, ch_tracks["charge"] < 0)
    ch_cands = combine([ch_pos, ch_neg], pvs)
    ch_cands["pid"] = pdg_id("B(s)0")
    n_ch = int(ak.sum(ak.num(ch_cands["vertex_x"])))
    if n_ch > 0:
        df_ch = candidates_to_dataframe(ch_cands)
        cpe = ak.to_numpy(ak.num(ch_cands["vertex_x"]))
        df_ch["run_number"] = np.repeat(event_info["run_number"], cpe)
        df_ch["event_number"] = np.repeat(event_info["event_number"], cpe)
        for k, v in extract_daughter_fields(ch_cands).items():
            df_ch[k] = v
        df_ch["bkgcat"] = ak.to_numpy(ak.flatten(bkgcat(ch_cands)))
        df_ch["is_signal"] = df_ch["bkgcat"] <= 10
        df_ch.to_parquet(cheated_path, index=False)
    else:
        import pandas as pd

        pd.DataFrame().to_parquet(cheated_path, index=False)

    # ================================================================
    # Summary
    # ================================================================
    n_cands = len(df) if df is not None else 0

    (out_dir / "pvtag_summary.json").write_text(
        json.dumps(
            {
                "n_true_decays": n_true,
                "n_events": n_events,
                "n_mva_candidates": n_mva_raw,
                "n_mva_pass": n_mva_pass,
                "n_tagged_pvs": n_tagged_pvs,
                "n_tracks_before": n_tracks_before,
                "n_tracks_after": n_tracks_after,
                "n_candidates_raw": n_cands_raw,
                "n_candidates": n_cands,
            },
            indent=2,
        )
    )

    elapsed = time.perf_counter() - t0
    print(f"\nTrue Bs->mumu: {n_true}")
    print(f"Candidates after combine: {n_cands_raw}")
    print(f"Candidates after selection: {n_cands}")
    print(f"Events: {n_events}")
    print(f"Time: {elapsed:.2f}s ({n_events / elapsed:.1f} evt/s)")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
