#!/usr/bin/env python3
"""Bs -> mu+mu- reconstruction: run a single combiner mode, save to Parquet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from trackcomb import (
    CombinationCuts,
    TrackPreselection,
    candidates_to_dataframe,
    combine,
    count_true_decays,
    iter_events_root,
    make_decay,
    truth_match,
)

BS_PDG = 531
MUON_PDG = 13


def _bs_muon_filter(t):
    """Keep only muons (|mc_pid|=13) with Bs ancestry."""
    if abs(t.metadata.get("mc_pid", 0)) != MUON_PDG:
        return False
    pids = t.metadata.get("mc_ancestor_pids", [])
    return any(abs(p) == BS_PDG for p in pids)


def _build_decay(mode: str):
    """Return (decay, track_filter) for the given mode."""
    if mode == "cheated":
        decay = make_decay(
            ["mu", "mu"],
            cuts=CombinationCuts(allowed_charge_patterns=("+-", "-+")),
        )
        return decay, _bs_muon_filter

    if mode == "full":
        decay = make_decay(
            ["mu", "mu"],
            preselection=TrackPreselection(min_pt=1.0, min_ip_to_any_pv=0.0),
            cuts=CombinationCuts(
                min_mass=4.7, max_mass=6.0,
                max_doca=0.05, max_vertex_chi2=4.0,
                max_pair_time_chi2=4.0,
                allowed_charge_patterns=("+-", "-+"),
            ),
        )
        return decay, None

    if mode == "full_notime":
        decay = make_decay(
            ["mu", "mu"],
            preselection=TrackPreselection(min_pt=1.0, min_ip_to_any_pv=0.0),
            cuts=CombinationCuts(
                min_mass=4.7, max_mass=6.0,
                max_doca=0.05, max_vertex_chi2=4.0,
                allowed_charge_patterns=("+-", "-+"),
                use_timing=False,
            ),
        )
        return decay, None

    if mode == "dist":
        decay = make_decay(
            ["mu", "mu"],
            preselection=TrackPreselection(min_pt=0.8, min_ip_to_any_pv=0.0),
            cuts=CombinationCuts(
                min_mass=4.5, max_mass=6.0,
                max_doca=0.2, max_vertex_chi2=10.0,
                max_pair_time_chi2=10.0,
                allowed_charge_patterns=("+-", "-+"),
            ),
        )
        return decay, None

    raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Bs -> mu+mu- reconstruction (single mode)",
    )
    parser.add_argument("--mode", required=True,
                        choices=["cheated", "full", "full_notime", "dist"])
    parser.add_argument("--input", required=True, help="ROOT file path")
    parser.add_argument("--tree", default="BestLongTracks/TrackTuple")
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--out-dir", default="public/1p5e32/reconstruct/bs_to_mumu")
    args = parser.parse_args()
    args.max_events = args.max_events or None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decay, track_filter = _build_decay(args.mode)
    events = iter_events_root(args.input, args.tree, max_events=args.max_events)

    candidates = []
    tracks_list = []
    maps_list = []
    n_true_total = 0
    n_events = 0

    for event in tqdm(events, desc=f"Bs {args.mode}", total=args.max_events):
        tracks = list(event.tracks)
        pvs = list(event.primary_vertices)
        track_map = {t.track_id: t for t in tracks}
        n_events += 1

        n_true_total += count_true_decays(
            tracks, mother_pdg=BS_PDG, daughter_pdgs=[MUON_PDG, MUON_PDG],
        )

        for c in combine(decay, tracks, pvs,
                         event_id=event.event_id, track_filter=track_filter):
            candidates.append(c)
            tracks_list.append(tracks)
            maps_list.append(track_map)

    # ---- Save ----
    _save_candidates(candidates, tracks_list, maps_list,
                     BS_PDG, [MUON_PDG, MUON_PDG],
                     out_dir / f"{args.mode}.parquet")

    summary = {
        "n_true_decays": n_true_total,
        "n_events": n_events,
        "n_candidates": len(candidates),
    }
    (out_dir / f"{args.mode}_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nMode: {args.mode}")
    print(f"True Bs->mumu: {n_true_total}")
    print(f"Candidates: {len(candidates)}")
    print(f"Events: {n_events}")
    print(f"Saved to {out_dir}")


def _save_candidates(candidates, tracks_list, map_list, mother_pdg, daughter_pdgs, path):
    """Truth-match and save candidates to Parquet."""
    if not candidates:
        import pandas as pd
        pd.DataFrame().to_parquet(path, index=False)
        print(f"  {path.name}: 0 candidates")
        return
    df = candidates_to_dataframe(candidates)
    df["is_signal"] = [
        truth_match(c, trks, mother_pdg, daughter_pdgs)
        for c, trks in zip(candidates, tracks_list)
    ]
    df["max_track_pt"] = [
        max(m[tid].pt for tid in c.source_track_ids)
        for c, m in zip(candidates, map_list)
    ]
    df["min_track_pt"] = [
        min(m[tid].pt for tid in c.source_track_ids)
        for c, m in zip(candidates, map_list)
    ]
    df.to_parquet(path, index=False)
    print(f"  {path.name}: {len(df)} candidates")


if __name__ == "__main__":
    main()
