#!/usr/bin/env python3
"""Bs -> mu+mu- reconstruction: run a single combiner mode, save to Parquet."""

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
)
from trackcomb.truth import bkgcat, count_true_decays


def _build_mode(mode: str) -> dict[str, Any]:
    """Return combine kwargs for the given mode."""
    if mode == "cheated":
        return {}

    if mode == "full":
        return {
            "track_cuts": [cut_min("pt", 1.0)],
            "combination_cuts": [
                cut_max("max_doca", 0.05),
                cut_max("spatial_chi2", 4.0),
            ],
            "vertex_cuts": [
                cut_range("mass", 4.7, 6.0),
                cut_max("pair_time_chi2", 4.0),
            ],
        }

    if mode == "full_notime":
        return {
            "track_cuts": [cut_min("pt", 1.0)],
            "combination_cuts": [
                cut_max("max_doca", 0.05),
                cut_max("spatial_chi2", 4.0),
            ],
            "vertex_cuts": [
                cut_range("mass", 4.7, 6.0),
            ],
            "use_timing": False,
        }

    if mode == "dist":
        return {
            "track_cuts": [cut_min("pt", 0.8)],
            "combination_cuts": [
                cut_max("max_doca", 0.2),
                cut_max("spatial_chi2", 10.0),
            ],
            "vertex_cuts": [
                cut_range("mass", 4.5, 6.0),
                cut_max("pair_time_chi2", 10.0),
            ],
        }

    raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Bs -> mu+mu- reconstruction")
    parser.add_argument(
        "--mode", required=True, choices=["cheated", "full", "full_notime", "dist"]
    )
    parser.add_argument("--input", required=True, help="ROOT file path")
    parser.add_argument("--tree", default="BestLongTracks/TrackTuple")
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--out-dir", default="public/1p5e34/reconstruction/bs_to_mumu")
    args = parser.parse_args()
    args.max_events = args.max_events or None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    tracks, pvs, event_info = load_events_root(args.input, args.tree, args.max_events)
    all_tracks = tracks

    mode_cfg = _build_mode(args.mode)

    # Cheated mode: filter tracks by MC PID (muon) AND Bs ancestry
    if args.mode == "cheated":
        is_muon = np.abs(tracks["mc_pid"]) == abs(pdg_id("mu+"))
        anc_pids = tracks["mc_ancestor_pids"]
        has_bs = ak.any(np.abs(anc_pids) == abs(pdg_id("B(s)0")), axis=-1)
        tracks = apply_mask(tracks, is_muon & has_bs)

    # Split into positive and negative charge pools
    tracks = set_tracks_pid(tracks, "mu+")
    pos_tracks = apply_mask(tracks, tracks["charge"] > 0)
    neg_tracks = apply_mask(tracks, tracks["charge"] < 0)

    candidates = combine([pos_tracks, neg_tracks], pvs, **mode_cfg)
    candidates["pid"] = pdg_id("B(s)0")

    # ---- Post-combine cuts (need PV association fields) ----
    n_cands_raw = int(ak.sum(ak.num(candidates["vertex_x"])))

    if args.mode != "cheated" and n_cands_raw > 0:
        df_all = candidates_to_dataframe(candidates)
        cand_per_evt = ak.to_numpy(ak.num(candidates["vertex_x"]))
        df_all["run_number"] = np.repeat(event_info["run_number"], cand_per_evt)
        df_all["event_number"] = np.repeat(event_info["event_number"], cand_per_evt)
        for k, v in extract_daughter_fields(candidates).items():
            df_all[k] = v
        df_all["bkgcat"] = ak.to_numpy(ak.flatten(bkgcat(candidates)))
        df_all["is_signal"] = df_all["bkgcat"] <= 10

        sel = np.ones(len(df_all), dtype=bool)
        sel &= df_all["dira"].isna() | (df_all["dira"] >= 0.9995)
        sel &= df_all["composite_ip"].isna() | (df_all["composite_ip"] < 0.1)
        sel &= df_all["composite_ip_chi2"].isna() | (df_all["composite_ip_chi2"] < 16)
        sel &= df_all["pt"] >= 1.0
        daughter_pt_cols = sorted(
            c for c in df_all.columns if c.endswith("_pt") and c.startswith("daughter")
        )
        if daughter_pt_cols:
            sel &= np.nanmax(df_all[daughter_pt_cols].values, axis=1) >= 2.0

        df = df_all[sel].copy()
    else:
        df = None

    # ---- Save results ----
    n_true = int(count_true_decays(all_tracks, "B(s)0", ["mu+", "mu-"]).sum())
    n_events = len(event_info["run_number"])

    out_path = out_dir / f"{args.mode}.parquet"
    if df is None or len(df) == 0:
        if args.mode == "cheated" and n_cands_raw > 0:
            # Cheated: no post-cuts, save all
            df = candidates_to_dataframe(candidates)
            cand_per_evt = ak.to_numpy(ak.num(candidates["vertex_x"]))
            df["run_number"] = np.repeat(event_info["run_number"], cand_per_evt)
            df["event_number"] = np.repeat(event_info["event_number"], cand_per_evt)
            for k, v in extract_daughter_fields(candidates).items():
                df[k] = v
            df["bkgcat"] = ak.to_numpy(ak.flatten(bkgcat(candidates)))
            df["is_signal"] = df["bkgcat"] <= 10
            df.to_parquet(out_path, index=False)
        else:
            import pandas as pd

            pd.DataFrame().to_parquet(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    # ---- Cheated reference (same events, for efficiency denominator) ----
    cheated_path = out_dir / f"{args.mode}_cheated.parquet"
    if args.mode == "cheated":
        # Main output is already the cheated reference
        if df is not None:
            df.to_parquet(cheated_path, index=False)
        else:
            import pandas as pd

            pd.DataFrame().to_parquet(cheated_path, index=False)
    else:
        # Run cheated combine on same events
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

    n_cands = len(df) if df is not None else 0

    (out_dir / f"{args.mode}_summary.json").write_text(
        json.dumps(
            {
                "n_true_decays": n_true,
                "n_events": n_events,
                "n_candidates_raw": n_cands_raw,
                "n_candidates": n_cands,
            },
            indent=2,
        )
    )

    elapsed = time.perf_counter() - t0
    print(f"\nMode: {args.mode}")
    print(f"True Bs->mumu: {n_true}")
    print(f"Candidates after combine: {n_cands_raw}")
    print(f"Candidates after selection: {n_cands}")
    print(f"Events: {n_events}")
    print(f"Time: {elapsed:.2f}s ({n_events / elapsed:.1f} evt/s)")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
