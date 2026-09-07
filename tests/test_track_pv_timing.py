"""Track-PV timing-qualified IP tests."""

import awkward as ak
import numpy as np
import pytest

from physics.analysis.pv_association_study.track_pv_timing_scan import (
    _label_suffix,
    _no_cut_position,
)
from physics.reconstruction.signal_track_pv_study import (
    NO_TIMING_METRIC,
    _build_scans,
)
from trackcomb import (
    min_ip,
    min_ip_chi2,
    pdg_mass,
    pvs_on_time_for_tracks,
    reduce_track_pv_pairs,
    set_track_pv_ip_statistics,
    track_pv_time_mask,
    tracks_on_time_for_pvs,
    tracks_pv_association,
)
from trackcomb.physics import compute_track_pv_pairs


def test_signal_track_scan_has_bounded_no_timing_endpoint():
    scans = _build_scans([0.05], [4.0])

    assert scans[:-1] == [("dt", 0.05), ("dt_chi2", 4.0)]
    assert scans[-1][0] == NO_TIMING_METRIC
    assert np.isinf(scans[-1][1])
    assert np.isclose(_no_cut_position([1.0, 4.0, 9.0, 100.0]), 300.0)


def test_timing_scan_label_suffix_avoids_trailing_underscore():
    assert _label_suffix("") == ""
    assert _label_suffix("1p0_timed") == "_1p0_timed"


def _inputs():
    tracks = {
        "x": ak.Array([[0.0, 0.0]]),
        "y": ak.Array([[0.0, 0.0]]),
        "z": ak.Array([[0.0, 0.0]]),
        "tx": ak.Array([[0.0, 0.0]]),
        "ty": ak.Array([[0.0, 0.0]]),
        "time": ak.Array([[0.02, 0.19]]),
        "sigma_time": ak.Array([[0.01, 0.01]]),
    }
    for i in range(4):
        for j in range(i + 1):
            value = 1e-4 if i == j else 0.0
            tracks[f"cov_{i}_{j}"] = ak.Array([[value, value]])

    pvs = {
        "x": ak.Array([[0.2, 0.01]]),
        "y": ak.Array([[0.0, 0.0]]),
        "z": ak.Array([[0.0, 0.0]]),
        "time": ak.Array([[0.0, 0.2]]),
        "sigma_time": ak.Array([[0.01, 0.01]]),
        "cov_0_0": ak.Array([[1e-4, 1e-4]]),
        "cov_1_0": ak.Array([[0.0, 0.0]]),
        "cov_1_1": ak.Array([[1e-4, 1e-4]]),
    }
    return tracks, pvs


def test_loaded_tracks_have_default_pion_mass_and_time(_root_data):
    tracks, _, _ = _root_data
    mass = np.asarray(ak.flatten(tracks["mass"]))

    np.testing.assert_allclose(mass, pdg_mass("pi+"))
    assert "time" in tracks
    assert "sigma_time" in tracks


def test_pair_table_contains_ip_and_both_timing_metrics():
    tracks, pvs = _inputs()
    pairs = compute_track_pv_pairs(tracks, pvs)

    assert {"ip", "ip_chi2", "dt", "dt_chi2"} <= pairs.keys()
    expected = np.array([[2.0, 162.0], [180.5, 0.5]])
    np.testing.assert_allclose(np.asarray(pairs["dt_chi2"][0]), expected)


def test_bidirectional_timing_lists_and_track_subset():
    tracks, pvs = _inputs()
    pairs = compute_track_pv_pairs(tracks, pvs)

    assert ak.to_list(
        pvs_on_time_for_tracks(tracks, pvs, max_dt=0.05, pairs=pairs)
    ) == [[[0], [1]]]
    assert ak.to_list(
        tracks_on_time_for_pvs(tracks, pvs, max_dt=0.05, pairs=pairs)
    ) == [[[0], [1]]]
    assert ak.to_list(
        tracks_on_time_for_pvs(
            tracks,
            pvs,
            max_dt=0.05,
            pairs=pairs,
            track_mask=ak.Array([[True, False]]),
        )
    ) == [[[0], []]]


def test_masked_ip_statistics_have_no_empty_selection_fallback():
    tracks, pvs = _inputs()
    pairs = compute_track_pv_pairs(tracks, pvs)
    mask = track_pv_time_mask(pairs, max_dt=0.001)
    set_track_pv_ip_statistics(tracks, pairs, prefix="selected", pv_mask=mask)

    assert ak.to_list(tracks["selected_n_pvs"]) == [[0, 0]]
    assert np.all(np.isnan(np.asarray(tracks["selected_min_ip"][0])))
    assert np.all(np.isnan(np.asarray(tracks["selected_second_min_ip"][0])))
    assert ak.to_list(tracks["selected_best_pv_index"]) == [[-1, -1]]
    assert ak.to_list(tracks["selected_second_pv_index"]) == [[-1, -1]]


def test_ip_ranking_is_only_within_timing_subset():
    tracks = {"x": ak.Array([[0.0]])}
    pairs = {
        "ip": ak.Array([[[0.3, 0.1, 0.2]]]),
        "dt": ak.Array([[[0.01, 0.20, 0.02]]]),
    }
    allowed = track_pv_time_mask(pairs, max_dt=0.05)
    reduced = reduce_track_pv_pairs(pairs, allowed)
    set_track_pv_ip_statistics(
        tracks, pairs, pv_mask=allowed, prefix="selected"
    )

    assert ak.to_list(reduced["pv_index"]) == [[[0, 2]]]
    assert ak.to_list(reduced["ip"]) == [[[0.3, 0.2]]]
    assert ak.to_list(tracks["selected_n_pvs"]) == [[2]]
    assert ak.to_list(tracks["selected_min_ip"]) == [[0.2]]
    assert ak.to_list(tracks["selected_second_min_ip"]) == [[0.3]]
    assert ak.to_list(tracks["selected_best_pv_index"]) == [[2]]
    assert ak.to_list(tracks["selected_second_pv_index"]) == [[0]]


def test_best_pv_is_chosen_only_inside_timing_window():
    tracks, pvs = _inputs()
    tracks_pv_association(tracks, pvs, max_dt=0.05)

    assert ak.to_list(tracks["best_pv_index"]) == [[0, 1]]
    assert ak.to_list(tracks["pv_on_time"]) == [[[0], [1]]]
    assert ak.to_list(tracks["n_pvs_considered"]) == [[1, 1]]
    assert {"pv_ip", "pv_ip_chi2", "pv_dt", "pv_dt_chi2"} <= tracks.keys()

    tracks, pvs = _inputs()
    tracks_pv_association(tracks, pvs, max_dt_chi2=3.5)
    assert ak.to_list(tracks["best_pv_index"]) == [[0, 1]]

    with pytest.raises(ValueError, match="choose at most one"):
        tracks_pv_association(tracks, pvs, max_dt=0.05, max_dt_chi2=4.0)


def test_no_timing_cut_associates_using_all_pvs_and_spatial_ip_only():
    tracks, pvs = _inputs()
    del tracks["time"]
    del tracks["sigma_time"]
    del pvs["time"]
    del pvs["sigma_time"]

    tracks_pv_association(tracks, pvs)

    assert ak.to_list(tracks["best_pv_index"]) == [[1, 1]]
    assert ak.to_list(tracks["pv_on_time"]) == [[[0, 1], [0, 1]]]
    assert ak.to_list(tracks["n_pvs_considered"]) == [[2, 2]]
    assert {"pv_ip", "pv_ip_chi2"} <= tracks.keys()
    assert "pv_dt" not in tracks


def test_empty_timing_window_has_no_all_pv_fallback():
    tracks, pvs = _inputs()
    tracks_pv_association(tracks, pvs, max_dt=0.001)

    assert ak.to_list(tracks["best_pv_index"]) == [[-1, -1]]
    assert ak.to_list(tracks["n_pvs_considered"]) == [[0, 0]]
    assert np.all(np.isnan(np.asarray(tracks["min_ip"][0])))
    assert np.all(np.isnan(np.asarray(tracks["min_ip_chi2"][0])))


def test_dynamic_minimum_uses_all_pvs_when_no_timing_metric_is_given():
    container = {
        "pv_ip": ak.Array([[[0.3, 0.1, 0.2]]]),
        "pv_ip_chi2": ak.Array([[[9.0, 1.0, 4.0]]]),
        "pv_dt": ak.Array([[[0.01, 0.20, 0.02]]]),
        "pv_dt_chi2": ak.Array([[[1.0, 100.0, 4.0]]]),
    }

    assert ak.to_list(min_ip(container)) == [[0.1]]
    assert ak.to_list(min_ip_chi2(container)) == [[1.0]]
    assert ak.to_list(min_ip(container, dt=0.05)) == [[0.2]]
    assert ak.to_list(min_ip_chi2(container, dt_chi2=4.0)) == [[4.0]]

    with pytest.raises(ValueError, match="choose at most one"):
        min_ip(container, dt=0.05, dt_chi2=4.0)
