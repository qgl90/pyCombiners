"""Unit tests for the PID and tracking performance dataframe analyses."""

import numpy as np
import pandas as pd

from pid.performance import PIDPerformance
from tracking.compare_tracking_efficiencies import _integrated_summary
from tracking.tracking_efficiencies import _set_track_type_tags
from tracking.tracking_efficiencies import (
    _fit_gaussian_core,
    _momentum_resolution_table,
    _performance_tables,
)


def test_pid_roc_uses_truth_and_rich_selected_species(tmp_path):
    frame = pd.DataFrame(
        {
            "mc_truth": [True] * 8,
            "rich_has_info": [True] * 8,
            "mc_pid": [321] * 4 + [211] * 4,
            "rich_dll_kaon": [5.0, 4.0, 1.0, -1.0, 3.0, 0.0, -2.0, -4.0],
            "p": np.full(8, 10_000.0),
            "pt": np.full(8, 1_000.0),
            "eta": np.full(8, 3.0),
        }
    )

    performance = PIDPerformance(frame, out_dir=tmp_path)
    roc = performance.global_roc(cuts=np.array([2.0]))

    np.testing.assert_allclose(roc["efficiency"], [0.5])
    np.testing.assert_allclose(roc["misid"], [0.25])


def test_tracking_summary_counts_unique_matches_and_ghosts():
    frame = pd.DataFrame(
        {
            "row_type": [
                "reconstructible",
                "reconstructible",
                "long",
                "long",
                "long",
            ],
            "truth_matched": [False, False, True, True, False],
            "is_unique_truth_match": [False, False, True, False, False],
            "from_signal": [True, True, True, True, False],
            "has_velo": [True] * 5,
            "has_ut": [False] * 5,
            "has_mp": [False] * 5,
            "has_ft": [False] * 5,
            "has_t": [True] * 5,
        }
    )
    _set_track_type_tags(frame)

    summary = _integrated_summary(frame, "sample", "long", ["from_signal"])

    assert summary["efficiency_numerator"] == 1
    assert summary["efficiency_denominator"] == 2
    assert summary["efficiency"] == 0.5
    assert summary["ghost_numerator"] == 1
    assert summary["ghost_denominator"] == 3


def test_ghost_rate_uses_reconstructed_phi_for_both_counts():
    frame = pd.DataFrame(
        {
            "row_type": ["reconstructible", "long", "long", "long"],
            "truth_matched": [False, True, True, False],
            "is_unique_truth_match": [False, True, True, False],
            "from_signal": [True, True, True, False],
            "has_velo": [True] * 4,
            "has_ut": [False] * 4,
            "has_mp": [False] * 4,
            "has_ft": [False] * 4,
            "has_t": [True] * 4,
            "truth_pt": [1_000.0, 1_000.0, 2_000.0, np.nan],
            "truth_eta": [3.0, 3.0, 3.1, np.nan],
            "truth_p": [10_000.0, 10_000.0, 20_000.0, np.nan],
            "truth_phi": [0.1, 0.1, 0.2, np.nan],
            "reco_pt": [np.nan, 1_000.0, 2_000.0, 3_000.0],
            "reco_eta": [np.nan, 3.0, 3.1, 3.2],
            "reco_p": [np.nan, 10_000.0, 20_000.0, 30_000.0],
            "reco_phi": [np.nan, 0.1, 0.2, 0.3],
        }
    )

    table = _performance_tables(frame, "long", ["from_signal"])
    phi = table[table["variable"] == "phi"]

    assert int(phi["fake_numerator"].sum()) == 1
    assert int(phi["fake_denominator"].sum()) == 3


def test_momentum_resolution_uses_gaussian_mean_and_width():
    residual = np.tile(np.array([-0.02, -0.01, 0.0, 0.01, 0.02]), 10)
    n_fit, mean, sigma, mean_error, sigma_error = _fit_gaussian_core(
        residual, min_entries=20
    )

    assert n_fit == 50
    np.testing.assert_allclose(mean, 0.0, atol=1e-15)
    np.testing.assert_allclose(sigma, np.sqrt(2e-4))
    assert mean_error > 0.0
    assert sigma_error > 0.0

    frame = pd.DataFrame(
        {
            "row_type": ["long"] * len(residual),
            "truth_matched": [True] * len(residual),
            "truth_p": np.full(len(residual), 10_000.0),
            "reco_p": 10_000.0 * (1.0 + residual),
            "truth_eta": np.full(len(residual), 3.0),
            "truth_phi": np.full(len(residual), 0.2),
        }
    )
    table = _momentum_resolution_table(frame, min_entries=20)
    fitted_p = table[(table["variable"] == "p") & (table["n_fit"] > 0)].iloc[0]

    np.testing.assert_allclose(fitted_p["bias_percent"], 0.0, atol=1e-13)
    np.testing.assert_allclose(
        fitted_p["resolution_percent"], 100.0 * np.sqrt(2e-4)
    )


def test_momentum_resolution_does_not_fit_an_underpopulated_robust_core():
    residual = np.concatenate([np.linspace(-0.01, 0.01, 19), [1_000.0]])
    n_fit, mean, sigma, mean_error, sigma_error = _fit_gaussian_core(
        residual, min_entries=20
    )

    assert n_fit == 19
    assert all(
        np.isnan(value) for value in (mean, sigma, mean_error, sigma_error)
    )
