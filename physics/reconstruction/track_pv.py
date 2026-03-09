#!/usr/bin/env python3
"""Track-to-PV association reconstruction: save per-track stats to Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from trackcomb import counters, run_reconstruction, set_tracks_pid
from trackcomb.physics import compute_track_pv_pairs


def reconstruction(events):
    """Compute track-to-PV association stats for one slice."""
    tracks, pvs = events["tracks"], events["pvs"]
    event_info = {k: events[k] for k in ("run_number", "event_number")}

    pv_mc_keys = pvs["mc_key"]

    # Assign pion PID so time fields are populated
    set_tracks_pid(tracks, "pi+")

    # Batch-compute IP, IP chi2, and flight-corrected dt in one pass
    pairs = compute_track_pv_pairs(tracks, pvs)

    # PV matching: find true PV index for each track
    match = (
        tracks["mc_pv_key"][:, :, np.newaxis] == pv_mc_keys[:, np.newaxis, :]
    )
    has_match = ak.any(match, axis=2)
    true_pv_idx = ak.fill_none(ak.argmax(match, axis=2), 0)

    mc_truth = ak.values_astype(tracks["mc_truth"], bool)
    sel_mask = mc_truth & (tracks["mc_pv_key"] != -1) & has_match

    counters("total tracks").add(
        int(ak.sum(ak.num(tracks["mc_truth"], axis=1)))
    )
    counters("truth tracks").add(int(ak.sum(mc_truth)))
    counters("matched to true PV").add(int(ak.sum(sel_mask)))

    # Selected tracks x all PVs
    ip_sel = pairs["ip"][sel_mask]
    chi2_sel = pairs["ip_chi2"][sel_mask]
    dt_sel = pairs["dt"][sel_mask]

    local_pv = ak.local_index(ip_sel, axis=2)
    true_pv_sel = true_pv_idx[sel_mask]
    is_true = local_pv == true_pv_sel[:, :, np.newaxis]

    # Flatten to (track, pv) rows
    assoc_ip = np.asarray(ak.flatten(ip_sel, axis=None))
    assoc_chi2 = np.asarray(ak.flatten(chi2_sel, axis=None))
    assoc_dt = np.asarray(ak.flatten(dt_sel, axis=None))
    assoc_is_true = np.asarray(ak.flatten(is_true, axis=None))

    if len(assoc_ip) == 0:
        return None

    # Build event/track/pv indices
    n_events = len(event_info["run_number"])
    trk_local_idx = np.asarray(
        ak.flatten(ak.local_index(sel_mask, axis=1)[sel_mask])
    )
    pv_idx_sel = np.asarray(ak.flatten(true_pv_idx[sel_mask]))
    n_sel_per_evt = np.asarray(ak.sum(sel_mask, axis=1))
    evt_idx_flat = np.repeat(np.arange(n_events), n_sel_per_evt)

    n_pvs_per_track = np.asarray(ak.flatten(ak.num(ip_sel, axis=2)))
    assoc_trk_idx = np.repeat(trk_local_idx, n_pvs_per_track)
    assoc_true_pv = np.repeat(pv_idx_sel, n_pvs_per_track)
    assoc_evt = np.repeat(evt_idx_flat, n_pvs_per_track)
    assoc_pv_idx = np.asarray(ak.flatten(local_pv, axis=None))

    return pd.DataFrame(
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


def main():
    parser = argparse.ArgumentParser(
        description="Track-to-PV association reconstruction (save stats to Parquet)",
    )
    parser.add_argument("--input", required=True, help="ROOT file path")
    parser.add_argument("--tree", default="BestLongTracks/TrackTuple")
    parser.add_argument("--max-events", type=int, default=200)
    parser.add_argument("--slice-size", type=int, default=1000)
    parser.add_argument(
        "--out-dir",
        default="public/1p5e34/reconstruction/track_pv_association",
    )
    args = parser.parse_args()
    args.max_events = args.max_events or None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    df.to_parquet(out_dir / "pv_assoc.parquet", index=False)

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
