"""Unit tests for the standalone PV-resolution study."""

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from pv.pv_resolution import _add_pull_columns, _fit_table, _pv_frame


def _aligned_pvs():
    pvs = {
        "pv_index": ak.Array([[0, 1], [0]]),
        "mc_key": ak.Array([[10, -1], [20]]),
        "x": ak.Array([[0.11, 9.0], [0.19]]),
        "y": ak.Array([[0.20, 9.0], [0.31]]),
        "z": ak.Array([[1.03, 9.0], [1.96]]),
        "time": ak.Array([[0.005, 9.0], [0.012]]),
        "mc_x": ak.Array([[0.10, np.nan], [0.20]]),
        "mc_y": ak.Array([[0.20, np.nan], [0.30]]),
        "mc_z": ak.Array([[1.00, np.nan], [2.00]]),
        "mc_time": ak.Array([[0.004, np.nan], [0.010]]),
        "ndof": ak.Array([[20.0, 4.0], [30.0]]),
        "chi2ndof": ak.Array([[1.0, 2.0], [1.2]]),
    }
    for i in range(4):
        for j in range(i + 1):
            pvs[f"cov_{i}_{j}"] = ak.Array([[0.01, 0.02], [0.03]])
    pvs["cov_0_0"] = ak.Array([[0.01, 0.02], [0.04]])
    pvs["cov_1_1"] = ak.Array([[0.09, 0.02], [0.09]])
    pvs["cov_2_2"] = ak.Array([[0.16, 0.02], [0.16]])
    pvs["cov_3_3"] = ak.Array([[1e-6, 0.02], [1e-6]])
    return pvs


def test_pv_frame_preserves_alignment_and_drops_unmatched_entries():
    frame = _pv_frame(
        _aligned_pvs(),
        {"run_number": [1, 1], "event_number": [100, 101]},
    )

    assert frame["mc_key"].tolist() == [10, 20]
    assert frame["pv_index"].tolist() == [0, 0]
    assert frame["event_number"].tolist() == [100, 101]
    np.testing.assert_allclose(frame["delta_x"], [0.01, -0.01])
    np.testing.assert_allclose(frame["delta_time"], [0.001, 0.002])
    np.testing.assert_allclose(frame["sigma_x"], [0.1, 0.2])
    np.testing.assert_allclose(frame["sigma_y"], [0.3, 0.3])
    np.testing.assert_allclose(frame["sigma_z"], [0.4, 0.4])
    np.testing.assert_allclose(frame["sigma_time"], [0.001, 0.001])
    np.testing.assert_allclose(frame["pull_x"], [0.1, -0.05])
    np.testing.assert_allclose(frame["pull_time"], [1.0, 2.0])


def test_pv_frame_rejects_misaligned_state_and_truth_collections():
    pvs = _aligned_pvs()
    pvs["mc_key"] = ak.Array([[10], [20]])
    with pytest.raises(ValueError, match="multiplicities differ"):
        _pv_frame(
            pvs,
            {"run_number": [1, 1], "event_number": [100, 101]},
        )


def test_pv_fit_table_reports_gaussian_bias_and_width():
    residual = np.tile(np.array([-0.02, -0.01, 0.0, 0.01, 0.02]), 10)
    frame = pd.DataFrame(
        {
            "ndof": np.full(len(residual), 25.0),
            "delta_x": residual,
            "delta_y": residual,
            "delta_z": residual,
            "delta_time": residual,
        }
    )
    table = _fit_table(frame, np.array([0.0, 50.0]), min_entries=20)
    fitted = table[table["variable"] == "x"].iloc[0]

    assert fitted["fit_status"] == "fitted"
    assert fitted["n_fit"] == 50
    np.testing.assert_allclose(fitted["bias"], 0.0, atol=1e-15)
    np.testing.assert_allclose(fitted["resolution"], np.sqrt(2e-4))


def test_nonpositive_pv_variance_produces_no_pull():
    frame = pd.DataFrame(
        {
            "delta_x": [1.0],
            "delta_y": [1.0],
            "delta_z": [1.0],
            "delta_time": [1.0],
            "cov_0_0": [0.0],
            "cov_1_1": [-1.0],
            "cov_2_2": [np.nan],
            "cov_3_3": [4.0],
        }
    )
    _add_pull_columns(frame)

    assert np.isnan(frame.loc[0, "pull_x"])
    assert np.isnan(frame.loc[0, "pull_y"])
    assert np.isnan(frame.loc[0, "pull_z"])
    assert frame.loc[0, "pull_time"] == 0.5
