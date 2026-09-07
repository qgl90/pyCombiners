"""Checks for the HLT2-like Bs -> J/psi phi selection definition."""

from unittest.mock import patch

import awkward as ak
import numpy as np
import pandas as pd

from physics.analysis.decay_performance_study import decay_performance
from physics.reconstruction import bs_to_jpsiphi as selection


def _combine_arguments(builder, config=selection.DEFAULT_SELECTION):
    with patch.object(selection, "combine", return_value={}) as mocked:
        builder({}, {}, {}, apply_hlt2_cuts=True, config=config)
    return mocked.call_args.kwargs


def _passes(cuts, candidate):
    return all(bool(np.asarray(cut(candidate))[0]) for cut in cuts)


def test_hlt2_intermediate_cut_definitions():
    jpsi = _combine_arguments(selection._combine_jpsi)
    assert jpsi["require_common_pv_on_time"]
    assert _passes(
        jpsi["combination_cuts"],
        {
            "mass": np.array([3097.0]),
            "pt": np.array([1.0]),
            "max_doca_chi2": np.array([29.0]),
        },
    )
    assert _passes(
        jpsi["composite_cuts"],
        {
            "vertex_chi2": np.array([29.0]),
            "vertex_time_chi2": np.array([15.0]),
        },
    )
    assert not _passes(
        jpsi["composite_cuts"],
        {
            "vertex_chi2": np.array([29.0]),
            "vertex_time_chi2": np.array([17.0]),
        },
    )
    assert _passes(jpsi["final_cuts"], {"fdchi2": np.array([31.0])})
    assert not _passes(jpsi["final_cuts"], {"fdchi2": np.array([29.0])})

    phi = _combine_arguments(selection._combine_phi)
    assert phi["require_common_pv_on_time"]
    assert _passes(
        phi["combination_cuts"],
        {
            "mass": np.array([1020.0]),
            "pt": np.array([401.0]),
            "max_doca_chi2": np.array([29.0]),
        },
    )
    assert _passes(
        phi["composite_cuts"],
        {
            "vertex_chi2": np.array([24.0]),
            "vertex_time_chi2": np.array([15.0]),
        },
    )
    assert not _passes(
        phi["composite_cuts"],
        {
            "vertex_chi2": np.array([24.0]),
            "vertex_time_chi2": np.array([17.0]),
        },
    )


def test_bs_ip_chi2_uses_timing_qualified_pvs():
    bs = _combine_arguments(selection._combine_bs)
    assert _passes(
        bs["composite_cuts"],
        {
            "vertex_chi2": np.array([8.0]),
            "vertex_time_chi2": np.array([15.0]),
        },
    )
    assert not _passes(
        bs["composite_cuts"],
        {
            "vertex_chi2": np.array([8.0]),
            "vertex_time_chi2": np.array([17.0]),
        },
    )
    candidate = {
        "dira": np.array([0.9999]),
        "pv_ip": ak.Array([[0.01, 0.02]]),
        "pv_ip_chi2": ak.Array([[100.0, 20.0]]),
        "pv_dt_chi2": ak.Array([[1.0, 15.0]]),
    }
    assert _passes(bs["final_cuts"], candidate)

    candidate["pv_dt_chi2"] = ak.Array([[1.0, 17.0]])
    assert not _passes(bs["final_cuts"], candidate)
    assert bs["pv_function"].keywords == {"max_dt_chi2": 16.0}
    assert bs["require_common_pv_on_time"]


def test_vertex_time_chi2_and_no_timing_are_configurable():
    config = selection.SelectionConfig(
        max_dt_chi2=None,
        require_common_pv_on_time=False,
        max_jpsi_vertex_time_chi2=7.0,
        max_phi_vertex_time_chi2=8.0,
        max_bs_vertex_time_chi2=9.0,
    )

    for builder, threshold in (
        (selection._combine_jpsi, 7.0),
        (selection._combine_phi, 8.0),
        (selection._combine_bs, 9.0),
    ):
        arguments = _combine_arguments(builder, config)
        assert _passes(
            arguments["composite_cuts"],
            {
                "vertex_chi2": np.array([1.0]),
                "vertex_time_chi2": np.array([threshold - 0.1]),
            },
        )
        assert arguments["pv_function"].keywords == {"max_dt_chi2": None}
        assert not arguments["require_common_pv_on_time"]


def test_signal_cutflow_counts_events_once_and_is_cumulative():
    frame = pd.DataFrame(
        {
            "has_four_signal_long_tracks": [True, True, True, False],
            "has_cheated_signal_candidate": [True, True, True, False],
            "passes_signal_muon_selection": [True, True, False, False],
            "passes_four_signal_track_selection": [True, False, False, False],
            "passes_signal_jpsi_selection": [True, False, False, False],
            "passes_signal_phi_selection": [True, False, False, False],
            "passes_signal_bs_selection": [True, False, False, False],
        }
    )
    cutflow = decay_performance.build_signal_cutflow(frame)

    assert cutflow["selected_signal_events"].tolist() == [3, 3, 2, 1, 1, 1, 1]
    assert cutflow["cumulative_efficiency"].tolist() == [
        1.0,
        1.0,
        2 / 3,
        1 / 3,
        1 / 3,
        1 / 3,
        1 / 3,
    ]
    assert (cutflow["selected_signal_events"].diff().dropna() <= 0).all()


def test_official_bkgcat_presentation_groups():
    assert decay_performance.bkgcat_group(0) == "signal-like (0,10,50)"
    assert decay_performance.bkgcat_group(50) == "signal-like (0,10,50)"
    assert (
        decay_performance.bkgcat_group(40) == "physics background (20,30,40)"
    )
    assert decay_performance.bkgcat_group(100) == "combinatorial / fake (>=60)"
    assert decay_performance.bkgcat_group(-1) == "undefined / other"
    assert (
        decay_performance.bkgcat_group(1000) == "combinatorial / fake (>=60)"
    )
