#!/usr/bin/env python3
"""Track-to-PV association reconstruction: save per-track stats to Parquet."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from trackcomb import iter_events_root
from trackcomb.physics import impact_parameter_to_pv

C_MM_PER_NS = 299.792458


def main():
    parser = argparse.ArgumentParser(
        description="Track-to-PV association reconstruction (save stats to Parquet)",
    )
    parser.add_argument("--input", required=True, help="ROOT file path")
    parser.add_argument("--tree", default="BestLongTracks/TrackTuple")
    parser.add_argument("--max-events", type=int, default=200)
    parser.add_argument("--out-dir", default="public/1p5e32/reconstruct/track_pv_association")
    args = parser.parse_args()
    args.max_events = args.max_events or None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load PV_mc_key separately (not in the standard framework loader)
    import uproot
    tree = uproot.open(f"{args.input}:{args.tree}")
    pv_mc_keys = tree["PV_mc_key"].array(
        entry_stop=args.max_events, library="ak",
    ).tolist()

    events = iter_events_root(args.input, args.tree, max_events=args.max_events)

    # Per-track observables
    rows = []
    # Per-track association data (for efficiency study)
    assoc_rows = []

    n_tracks_total = 0
    n_truth = 0
    n_matched = 0
    n_no_true_pv = 0
    n_events = 0

    for evt_idx, event in enumerate(tqdm(events, desc="Processing",
                                          total=args.max_events)):
        tracks = list(event.tracks)
        pvs = list(event.primary_vertices)
        n_events += 1

        # Build PV lookup: mc_key -> PV object
        evt_pv_mc_keys = pv_mc_keys[evt_idx]
        pv_by_mc_key: dict[int, object] = {}
        for pi, pv in enumerate(pvs):
            mc_key = int(evt_pv_mc_keys[pi])
            pv_by_mc_key[mc_key] = pv

        for t in tracks:
            n_tracks_total += 1
            mc_truth = t.metadata.get("mc_truth", 0)
            if not mc_truth:
                continue
            n_truth += 1

            mc_pv_key = t.metadata.get("mc_pv_key", -1)
            if mc_pv_key == -1:
                n_no_true_pv += 1
                continue

            true_pv = pv_by_mc_key.get(mc_pv_key)
            if true_pv is None:
                n_no_true_pv += 1
                continue
            n_matched += 1

            # IP and IP chi2 w.r.t. true PV
            ip, ip_chi2 = impact_parameter_to_pv(t, true_pv)

            # Time difference
            dt = t.time - true_pv.time

            # Flight-corrected time
            dz = t.z - true_pv.z
            speed_factor = math.sqrt(1.0 + t.tx * t.tx + t.ty * t.ty)
            flight_time = (dz * speed_factor) / C_MM_PER_NS
            dt_corrected = t.time - flight_time - true_pv.time

            rows.append({
                "event_id": event.event_id,
                "track_id": t.track_id,
                "true_pv_id": true_pv.pv_id,
                "ip": ip,
                "ip_chi2": ip_chi2,
                "dt_raw": dt,
                "dt_corrected": dt_corrected,
                "track_pt": t.pt,
                "track_eta": t.eta,
            })

            # Per-PV association data
            for pv in pvs:
                pv_ip, pv_ip_chi2 = impact_parameter_to_pv(t, pv)
                pv_dz = t.z - pv.z
                pv_flight = (pv_dz * speed_factor) / C_MM_PER_NS
                pv_dt_corr = t.time - pv_flight - pv.time
                assoc_rows.append({
                    "event_id": event.event_id,
                    "track_id": t.track_id,
                    "true_pv_id": true_pv.pv_id,
                    "pv_id": pv.pv_id,
                    "is_true_pv": pv.pv_id == true_pv.pv_id,
                    "ip": pv_ip,
                    "ip_chi2": pv_ip_chi2,
                    "dt_corrected": pv_dt_corr,
                })

    # ---- Save Parquet ----
    import pandas as pd

    df_tracks = pd.DataFrame(rows) if rows else pd.DataFrame()
    df_tracks.to_parquet(out_dir / "pv_stats.parquet", index=False)
    print(f"Saved {len(df_tracks)} track-PV entries to pv_stats.parquet")

    df_assoc = pd.DataFrame(assoc_rows) if assoc_rows else pd.DataFrame()
    df_assoc.to_parquet(out_dir / "pv_assoc.parquet", index=False)
    print(f"Saved {len(df_assoc)} track-PV association entries to pv_assoc.parquet")

    # Summary
    summary = {
        "n_tracks_total": n_tracks_total,
        "n_truth": n_truth,
        "n_matched": n_matched,
        "n_no_true_pv": n_no_true_pv,
        "n_events": n_events,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"Total tracks:              {n_tracks_total}")
    print(f"Truth tracks:              {n_truth}")
    print(f"Matched to true PV:        {n_matched}")
    print(f"No true PV found:          {n_no_true_pv}")
    print(f"{'='*60}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
