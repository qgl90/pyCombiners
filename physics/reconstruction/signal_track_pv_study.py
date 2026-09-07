#!/usr/bin/env python3
"""Single signal-track to PV timing-association scan."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from trackcomb import (
    compute_track_pv_pairs,
    counters,
    load_event_info,
    load_pvs,
    load_tracks,
    pick_inner,
    reduce_track_pv_pairs,
    run_reconstruction,
    set_track_pv_ip_statistics,
    set_tracks_pid,
    track_pv_time_mask,
    tracks_on_time_for_pvs,
)


DEFAULT_DT_THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5)
DEFAULT_DT_CHI2_THRESHOLDS = (1.0, 4.0, 9.0, 16.0, 25.0, 50.0, 100.0)
NO_TIMING_METRIC = "none"
NO_TIMING_THRESHOLD = np.inf


def _build_scans(dt_thresholds, dt_chi2_thresholds):
    """Build requested timing scans plus one explicit all-PV reference."""
    scans = [("dt", value) for value in dt_thresholds]
    scans += [("dt_chi2", value) for value in dt_chi2_thresholds]
    if not scans:
        raise ValueError("at least one dt or dt-chi2 threshold is required")
    return [*scans, (NO_TIMING_METRIC, NO_TIMING_THRESHOLD)]


def _pv_values_for_tracks(values_by_pv, pv_indices, missing=-1):
    """Gather event-local PV values for every track with empty-event safety."""
    out = []
    for evt_values, evt_indices in zip(
        ak.to_list(values_by_pv), ak.to_list(pv_indices)
    ):
        out.append(
            [
                evt_values[index] if 0 <= index < len(evt_values) else missing
                for index in evt_indices
            ]
        )
    return ak.Array(out)


def _flat_selected(values, signal_mask):
    return np.asarray(ak.flatten(values[signal_mask]))


def _list_selected(values, signal_mask):
    """Flatten events/tracks but retain the considered-PV list per track."""
    return ak.to_list(ak.flatten(values[signal_mask], axis=1))


def _scan_dataframe(
    tracks,
    pvs,
    event_info,
    pairs,
    signal_mask,
    true_pv_idx,
    has_true_pv,
    metric,
    threshold,
):
    if metric == "dt":
        kwargs = {"max_dt": threshold}
    elif metric == "dt_chi2":
        kwargs = {"max_dt_chi2": threshold}
    elif metric == NO_TIMING_METRIC:
        kwargs = {}
    else:
        raise ValueError(f"unknown timing metric: {metric}")
    # This mask defines the PV subset considered by this scan point. All IP
    # rankings below are computed only inside this subset.
    considered_mask = track_pv_time_mask(pairs, **kwargs)
    considered_pvs = reduce_track_pv_pairs(pairs, considered_mask)
    set_track_pv_ip_statistics(
        tracks, pairs, prefix="scan", pv_mask=considered_mask
    )

    # Exercise both directions of the association and retain their
    # multiplicities in the signal-track rows.
    on_time_tracks = tracks_on_time_for_pvs(tracks, pvs, pairs=pairs, **kwargs)
    tracks_per_pv = ak.num(on_time_tracks, axis=2)
    tracks_at_true_pv = _pv_values_for_tracks(tracks_per_pv, true_pv_idx)
    tracks_at_best_pv = _pv_values_for_tracks(
        tracks_per_pv, tracks["scan_best_pv_index"]
    )

    mean_tracks_per_pv = np.array(
        [
            np.mean(values) if values else 0.0
            for values in ak.to_list(tracks_per_pv)
        ]
    )
    max_tracks_per_pv = np.array(
        [max(values) if values else 0 for values in ak.to_list(tracks_per_pv)]
    )

    n_events = len(event_info["run_number"])
    signal_counts = ak.to_numpy(ak.sum(signal_mask, axis=1))
    event_index = np.repeat(np.arange(n_events), signal_counts)
    track_index = _flat_selected(
        ak.local_index(tracks["x"], axis=1), signal_mask
    )

    best_ip = _flat_selected(tracks["scan_min_ip"], signal_mask)
    second_ip = _flat_selected(tracks["scan_second_min_ip"], signal_mask)
    best_idx = _flat_selected(tracks["scan_best_pv_index"], signal_mask)
    true_idx = _flat_selected(true_pv_idx, signal_mask)
    has_true = _flat_selected(has_true_pv, signal_mask).astype(bool)
    n_tracks_at_true = _flat_selected(tracks_at_true_pv, signal_mask)

    true_on_time = pick_inner(considered_mask, true_pv_idx)

    return pd.DataFrame(
        {
            "run_number": event_info["run_number"][event_index],
            "event_number": event_info["event_number"][event_index],
            "track_index": track_index,
            "mc_pid": _flat_selected(tracks["mc_pid"], signal_mask),
            "mc_key": _flat_selected(tracks["mc_key"], signal_mask),
            "mc_pv_key": _flat_selected(tracks["mc_pv_key"], signal_mask),
            "p": _flat_selected(tracks["p"], signal_mask),
            "pt": _flat_selected(tracks["pt"], signal_mask),
            "eta": _flat_selected(tracks["eta"], signal_mask),
            "time": _flat_selected(tracks["time"], signal_mask),
            "sigma_time": _flat_selected(tracks["sigma_time"], signal_mask),
            "n_tracks_event": pairs["track_counts"][event_index],
            "n_signal_tracks_event": signal_counts[event_index],
            "n_pvs_event": pairs["pv_counts"][event_index],
            "timing_metric": metric,
            "timing_threshold": threshold,
            "n_pvs_considered": _flat_selected(
                tracks["scan_n_pvs"], signal_mask
            ),
            "considered_pv_indices": _list_selected(
                considered_pvs["pv_index"], signal_mask
            ),
            "considered_pv_ip": _list_selected(
                considered_pvs["ip"], signal_mask
            ),
            "considered_pv_ip_chi2": _list_selected(
                considered_pvs["ip_chi2"], signal_mask
            ),
            "considered_pv_dt": _list_selected(
                considered_pvs["dt"], signal_mask
            ),
            "considered_pv_dt_chi2": _list_selected(
                considered_pvs["dt_chi2"], signal_mask
            ),
            "min_ip": best_ip,
            "second_min_ip": second_ip,
            "second_minus_min_ip": second_ip - best_ip,
            "min_ip_pv_index": best_idx,
            "second_min_ip_pv_index": _flat_selected(
                tracks["scan_second_pv_index"], signal_mask
            ),
            "min_ip_all_pvs": _flat_selected(
                tracks["all_pvs_min_ip"], signal_mask
            ),
            "second_min_ip_all_pvs": _flat_selected(
                tracks["all_pvs_second_min_ip"], signal_mask
            ),
            "min_ip_all_pvs_index": _flat_selected(
                tracks["all_pvs_best_pv_index"], signal_mask
            ),
            "has_true_pv": has_true,
            "true_pv_index": true_idx,
            "true_pv_on_time": has_true
            & _flat_selected(true_on_time, signal_mask).astype(bool),
            "best_is_true_pv": has_true & (best_idx == true_idx),
            "n_tracks_on_time_true_pv": np.where(
                has_true, n_tracks_at_true, -1
            ),
            "n_tracks_on_time_best_pv": _flat_selected(
                tracks_at_best_pv, signal_mask
            ),
            "mean_tracks_on_time_per_pv_event": mean_tracks_per_pv[
                event_index
            ],
            "max_tracks_on_time_per_pv_event": max_tracks_per_pv[event_index],
        }
    )


def reconstruction(chunk, scans):
    """Build signal-track rows for all requested timing scans in one chunk."""
    tracks = load_tracks(chunk)
    pvs = load_pvs(chunk)
    event_info = load_event_info(chunk)
    set_tracks_pid(tracks, "pi+")

    signal_mask = ak.values_astype(tracks["mc_fromsignal"], bool)
    n_signal = int(ak.sum(signal_mask))
    counters("signal tracks").add(n_signal)
    if n_signal == 0:
        return None

    pairs = compute_track_pv_pairs(tracks, pvs)
    if "dt_chi2" not in pairs:
        raise ValueError(
            "track and PV time uncertainties are required for this study"
        )

    match = (
        tracks["mc_pv_key"][:, :, np.newaxis]
        == pvs["mc_key"][:, np.newaxis, :]
    )
    has_true_pv = ak.any(match, axis=2)
    true_pv_idx = ak.fill_none(ak.argmax(match, axis=2), 0)

    set_track_pv_ip_statistics(tracks, pairs, prefix="all_pvs")
    frames = [
        _scan_dataframe(
            tracks,
            pvs,
            event_info,
            pairs,
            signal_mask,
            true_pv_idx,
            has_true_pv,
            metric,
            threshold,
        )
        for metric, threshold in scans
    ]
    result = pd.concat(frames, ignore_index=True)
    counters("scan rows").add(len(result))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Signal-track to PV timing and IP scan",
    )
    parser.add_argument(
        "--input", required=True, help="ROOT path; wildcards allowed"
    )
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--dt-thresholds",
        type=float,
        nargs="*",
        default=DEFAULT_DT_THRESHOLDS,
        help="absolute flight-corrected dt cuts in ns",
    )
    parser.add_argument(
        "--dt-chi2-thresholds",
        type=float,
        nargs="*",
        default=DEFAULT_DT_CHI2_THRESHOLDS,
    )
    parser.add_argument(
        "--out",
        default="public/pv_association/signal_track_pv_scan.parquet",
    )
    args = parser.parse_args()
    args.max_events = args.max_events or None

    try:
        scans = _build_scans(args.dt_thresholds, args.dt_chi2_thresholds)
    except ValueError as error:
        parser.error(str(error))

    out = Path(args.out)
    run_reconstruction(
        partial(reconstruction, scans=scans),
        input_data=args.input,
        out=out,
        max_events=args.max_events,
        chunk_size=args.chunk_size,
        workers=args.workers,
        print_throughput=True,
    )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
