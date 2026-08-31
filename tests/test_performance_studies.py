"""Unit tests for the PID and tracking performance dataframe analyses."""

import numpy as np
import pandas as pd

from pid.performance import PIDPerformance
from tracking.compare_tracking_efficiencies import _integrated_summary
from tracking.tracking_efficiencies import _set_track_type_tags


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
