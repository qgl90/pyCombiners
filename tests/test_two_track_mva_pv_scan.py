"""Tests for event-denominator-safe TwoTrackMVA scans."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).parents[1]
    / "physics/analysis/two_track_mva_pv_study/analyze.py"
)
SPEC = importlib.util.spec_from_file_location(
    "two_track_mva_pv_analyze", MODULE_PATH
)
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


def test_scan_keeps_empty_events_and_counts_distinct_pvs():
    frame = pd.DataFrame(
        [
            {
                "run_number": 1,
                "event_number": 10,
                "track_sample": "all_tracks",
                "sample_label": "test",
                "is_candidate": False,
                "signal_pv_indices_json": "[2]",
                "n_pvs_event": 5,
            },
            {
                "run_number": 1,
                "event_number": 11,
                "track_sample": "all_tracks",
                "sample_label": "test",
                "is_candidate": False,
                "signal_pv_indices_json": "[1]",
                "n_pvs_event": 4,
            },
            {
                "run_number": 1,
                "event_number": 12,
                "track_sample": "all_tracks",
                "sample_label": "test",
                "is_candidate": False,
                "signal_pv_indices_json": "[]",
                "n_pvs_event": 3,
            },
            {
                "run_number": 1,
                "event_number": 10,
                "track_sample": "all_tracks",
                "sample_label": "test",
                "is_candidate": True,
                "mva_response": 0.9,
                "best_pv_index": 2,
                "daughters_share_pv_on_time": True,
                "daughters_same_best_pv": False,
                "same_sign": False,
            },
            {
                "run_number": 1,
                "event_number": 10,
                "track_sample": "all_tracks",
                "sample_label": "test",
                "is_candidate": True,
                "mva_response": 0.95,
                "best_pv_index": 2,
                "daughters_share_pv_on_time": True,
                "daughters_same_best_pv": True,
                "same_sign": True,
            },
            {
                "run_number": 1,
                "event_number": 12,
                "track_sample": "all_tracks",
                "sample_label": "test",
                "is_candidate": True,
                "mva_response": 0.92,
                "best_pv_index": 0,
                "daughters_share_pv_on_time": True,
                "daughters_same_best_pv": True,
                "same_sign": False,
            },
        ]
    )

    scan = ANALYZE.build_scan(frame, thresholds=[0.8])
    row = scan.loc[
        np.isclose(scan["mva_cut"], 0.8)
        & (scan["pv_policy"] == "common_on_time")
        & (scan["charge_selection"] == "all")
    ].iloc[0]

    assert row["n_events"] == 3
    assert row["n_candidates"] == 3
    assert row["candidates_per_event"] == 1.0
    assert row["mean_selected_pvs_per_event"] == 2 / 3
    assert row["pv_selected_event_fraction"] == 2 / 3
    assert row["mean_unique_pvs_per_selected_event"] == 1.0
    assert row["n_signal_evaluable_selected_events"] == 1
    assert row["n_signal_evaluable_events"] == 2
    assert row["n_selected_events_without_signal_pv_truth"] == 1
    assert row["n_selected_events_signal_pv_found"] == 1
    assert row["signal_pv_efficiency_given_selected_event"] == 1.0
    assert row["signal_pv_correctness_given_selected_event"] == 1.0
    assert row["signal_pv_retention_efficiency"] == 0.5
    assert row["signal_pv_efficiency"] == 0.5

    curve = scan.loc[
        (scan["pv_policy"] == "common_on_time")
        & (scan["charge_selection"] == "all")
    ].sort_values("mva_cut")
    assert (
        curve["n_selected_events_signal_pv_found"].diff().dropna() <= 0
    ).all()
    assert (curve["signal_pv_retention_efficiency"].diff().dropna() <= 0).all()

    multiplicity = ANALYZE.build_pv_multiplicity_rows(
        frame, thresholds=[0.8], policy="common_on_time", charge="all"
    )
    at_threshold = multiplicity.loc[np.isclose(multiplicity["mva_cut"], 0.8)]
    assert sorted(at_threshold["n_selected_pvs"]) == [1, 1]
    assert sorted(at_threshold["n_pvs_event"]) == [3, 5]

    processed = ANALYZE.build_processed_event_pv_multiplicity(
        frame, mva_cut=0.8, policy="common_on_time", charge="all"
    )
    assert sorted(processed["n_selected_pvs"]) == [0, 1, 1]
