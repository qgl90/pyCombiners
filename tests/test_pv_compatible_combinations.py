"""Daughter PV compatibility tests."""

import awkward as ak
import numpy as np

import trackcomb.combiner as combiner
from trackcomb import add_daughter_pv_compatibility


def test_daughter_pv_compatibility_keeps_both_selection_modes_separate():
    pool0 = {
        "pv_on_time": ak.Array([[[0, 1], [2]]]),
        "best_pv_index": ak.Array([[0, 2]]),
    }
    pool1 = {
        "pv_on_time": ak.Array([[[1, 3], [4]]]),
        "best_pv_index": ak.Array([[0, 2]]),
    }
    combinations = {
        "_daughter_pools": [pool0, pool1],
        "daughter0_global_index": np.array([0, 1]),
        "daughter1_global_index": np.array([0, 1]),
    }

    add_daughter_pv_compatibility(combinations)

    assert ak.to_list(combinations["daughter_common_pv_on_time"]) == [[1], []]
    assert ak.to_list(combinations["daughters_have_common_pv_on_time"]) == [
        True,
        False,
    ]
    assert ak.to_list(combinations["daughters_have_same_best_pv"]) == [
        True,
        True,
    ]


def test_same_best_pv_rejects_unassociated_daughters():
    pool = {
        "pv_on_time": ak.Array([[[], []]]),
        "best_pv_index": ak.Array([[-1, -1]]),
    }
    combinations = {
        "_daughter_pools": [pool, pool],
        "daughter0_global_index": np.array([0]),
        "daughter1_global_index": np.array([1]),
    }

    add_daughter_pv_compatibility(combinations)

    assert ak.to_list(combinations["daughters_have_same_best_pv"]) == [False]
    assert ak.to_list(combinations["daughters_have_common_pv_on_time"]) == [
        False
    ]


def test_combine_can_compute_pv_compatibility_without_filtering(monkeypatch):
    pool = {
        "mass": ak.Array([[139.6, 139.6]]),
        "pv_on_time": ak.Array([[[0], [1]]]),
        "best_pv_index": ak.Array([[0, 1]]),
    }
    combinations = {
        "event_idx": np.array([0]),
        "_daughter_pools": [pool, pool],
        "daughter0_global_index": np.array([0]),
        "daughter1_global_index": np.array([1]),
    }
    monkeypatch.setattr(combiner, "make_combinations", lambda _: combinations)
    monkeypatch.setattr(combiner, "compute_prefit_kinematics", lambda _: None)
    monkeypatch.setattr(
        combiner, "compute_composite_covariance", lambda _: None
    )
    monkeypatch.setattr(combiner, "_propagate_mc_truth", lambda _: None)
    monkeypatch.setattr(combiner, "unflatten_container", lambda out, _: out)

    result = combiner.combine(
        [pool, pool],
        {"x": ak.Array([[0.0]])},
        compute_pv_compatibility=True,
        doca_function=lambda _: None,
        vertex_fit_function=lambda _: None,
        consolidate_function=lambda _: None,
        pv_function=lambda *_: None,
    )

    assert ak.to_list(result["daughters_have_common_pv_on_time"]) == [False]
    assert len(result["event_idx"]) == 1
