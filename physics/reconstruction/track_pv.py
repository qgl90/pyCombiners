#!/usr/bin/env python3
"""Track-to-PV association reconstruction: save per-track stats to Parquet."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from trackcomb import load_events_root, pick_along_inner
from trackcomb.physics import ip_to_pvs, flight_corrected_dt


def main():
    parser = argparse.ArgumentParser(
        description="Track-to-PV association reconstruction (save stats to Parquet)",
    )
    parser.add_argument("--input", required=True, help="ROOT file path")
    parser.add_argument("--tree", default="BestLongTracks/TrackTuple")
    parser.add_argument("--max-events", type=int, default=200)
    parser.add_argument(
        "--out-dir", default="public/1p5e34/reconstruction/track_pv_association"
    )
    args = parser.parse_args()
    args.max_events = args.max_events or None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    tracks, pvs, event_info = load_events_root(args.input, args.tree, args.max_events)
    n_events = len(event_info["run_number"])

    # Load PV_mc_key separately (not in the standard framework loader)
    tree = uproot.open(f"{args.input}:{args.tree}")
    pv_mc_keys = tree["PV_mc_key"].array(entry_stop=args.max_events, library="ak")

    # Batch-compute IP and flight-corrected dt for all (track, PV) pairs
    ip_all, ip_chi2_all = ip_to_pvs(tracks, pvs)  # (events, tracks, pvs)
    dt_all = flight_corrected_dt(tracks, pvs)  # (events, tracks, pvs)

    # ---- PV matching (fully vectorized via awkward broadcasting) ----
    # (events, tracks, 1) == (events, 1, pvs) → (events, tracks, pvs)
    match = tracks["mc_pv_key"][:, :, np.newaxis] == pv_mc_keys[:, np.newaxis, :]
    has_match = ak.any(match, axis=2)
    true_pv_idx = ak.fill_none(ak.argmax(match, axis=2), 0)

    sel_mask = ak.values_astype(
        tracks["mc_truth"] & (tracks["mc_pv_key"] != -1) & has_match, bool
    )

    n_tracks_total = int(ak.sum(ak.num(tracks["mc_truth"], axis=1)))
    n_truth = int(ak.sum(tracks["mc_truth"]))
    n_matched = int(ak.sum(sel_mask))
    n_no_true_pv = n_truth - n_matched

    # ---- Extract at true PV using pick_along_inner ----
    ip_at_true = pick_along_inner(ip_all, true_pv_idx)
    chi2_at_true = pick_along_inner(ip_chi2_all, true_pv_idx)
    dt_at_true = pick_along_inner(dt_all, true_pv_idx)

    # pv_time at true PV — 2D gather (pvs["time"] is (events, pvs), not 3D)
    flat_pv_time = np.append(np.asarray(ak.flatten(pvs["time"])), 0.0)  # sentinel
    n_pvs_per_evt = np.asarray(ak.num(pvs["time"], axis=1))
    pv_offsets = np.empty(len(n_pvs_per_evt), dtype=np.int64)
    pv_offsets[0] = 0
    np.cumsum(n_pvs_per_evt[:-1], out=pv_offsets[1:])
    n_trk_per_evt = np.asarray(ak.num(true_pv_idx, axis=1))
    flat_pv_offsets = np.repeat(pv_offsets, n_trk_per_evt)
    flat_true_pv = np.asarray(ak.flatten(true_pv_idx, axis=None))
    pv_time_at_true = ak.unflatten(
        flat_pv_time[flat_pv_offsets + flat_true_pv], n_trk_per_evt
    )
    dt_raw_all = tracks["time"] - pv_time_at_true

    # Mask to selected tracks and flatten
    ip_sel = np.asarray(ak.flatten(ip_at_true[sel_mask]))
    chi2_sel = np.asarray(ak.flatten(chi2_at_true[sel_mask]))
    dt_sel = np.asarray(ak.flatten(dt_at_true[sel_mask]))
    dt_raw_sel = np.asarray(ak.flatten(dt_raw_all[sel_mask]))
    pt_sel = np.asarray(ak.flatten(tracks["pt"][sel_mask]))
    eta_sel = np.asarray(ak.flatten(tracks["eta"][sel_mask]))

    # Build string IDs for track rows
    trk_local_idx = np.asarray(ak.flatten(ak.local_index(sel_mask, axis=1)[sel_mask]))
    pv_idx_sel = np.asarray(ak.flatten(true_pv_idx[sel_mask]))
    n_sel_per_evt = np.asarray(ak.sum(sel_mask, axis=1))
    evt_idx_flat = np.repeat(np.arange(n_events), n_sel_per_evt)

    event_ids = [
        f"run{event_info['run_number'][e]}_evt{event_info['event_number'][e]}"
        for e in evt_idx_flat
    ]
    track_ids = [f"trk_{i}" for i in trk_local_idx]
    true_pv_ids = [f"pv_{i}" for i in pv_idx_sel]

    df_tracks = pd.DataFrame(
        {
            "event_id": event_ids,
            "track_id": track_ids,
            "true_pv_id": true_pv_ids,
            "ip": ip_sel,
            "ip_chi2": chi2_sel,
            "dt_raw": dt_raw_sel,
            "dt_corrected": dt_sel,
            "track_pt": pt_sel,
            "track_eta": eta_sel,
        }
    )
    df_tracks.to_parquet(out_dir / "pv_stats.parquet", index=False)
    print(f"Saved {len(df_tracks)} track-PV entries to pv_stats.parquet")

    # ---- Phase 3: Assoc rows (selected tracks × all PVs, vectorized) ----
    ip_sel_all = ip_all[sel_mask]  # (events, sel_tracks, pvs)
    chi2_sel_all = ip_chi2_all[sel_mask]
    dt_sel_all = dt_all[sel_mask]

    # is_true_pv: local PV index == true PV index for that track
    local_pv = ak.local_index(ip_sel_all, axis=2)
    true_pv_sel = true_pv_idx[sel_mask]
    is_true = local_pv == true_pv_sel[:, :, np.newaxis]

    # Flatten everything to 1D
    assoc_ip = np.asarray(ak.flatten(ip_sel_all, axis=None))
    assoc_chi2 = np.asarray(ak.flatten(chi2_sel_all, axis=None))
    assoc_dt = np.asarray(ak.flatten(dt_sel_all, axis=None))
    assoc_is_true = np.asarray(ak.flatten(is_true, axis=None))

    # Repeat track-level info by n_pvs per track
    n_pvs_per_track = np.asarray(ak.flatten(ak.num(ip_sel_all, axis=2)))
    assoc_trk_idx = np.repeat(trk_local_idx, n_pvs_per_track)
    assoc_true_pv = np.repeat(pv_idx_sel, n_pvs_per_track)
    assoc_evt = np.repeat(evt_idx_flat, n_pvs_per_track)
    assoc_pv_idx = np.asarray(ak.flatten(local_pv, axis=None))

    df_assoc = pd.DataFrame(
        {
            "event_id": [
                f"run{event_info['run_number'][e]}_evt{event_info['event_number'][e]}"
                for e in assoc_evt
            ],
            "track_id": [f"trk_{i}" for i in assoc_trk_idx],
            "true_pv_id": [f"pv_{i}" for i in assoc_true_pv],
            "pv_id": [f"pv_{i}" for i in assoc_pv_idx],
            "is_true_pv": assoc_is_true,
            "ip": assoc_ip,
            "ip_chi2": assoc_chi2,
            "dt_corrected": assoc_dt,
        }
    )
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

    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 60}")
    print(f"Total tracks:              {n_tracks_total}")
    print(f"Truth tracks:              {n_truth}")
    print(f"Matched to true PV:        {n_matched}")
    print(f"No true PV found:          {n_no_true_pv}")
    print(f"Events: {n_events}")
    print(f"Time: {elapsed:.2f}s ({n_events / elapsed:.1f} evt/s)")
    print(f"{'=' * 60}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
