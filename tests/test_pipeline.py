"""SoA pipeline tests: combine correctness, edge cases, multi-pool, staged decay."""

from __future__ import annotations

import unittest

import pytest

import awkward as ak
import numpy as np

from particle import literals as lp

from trackcomb.combiner import combine
from trackcomb.models import cut_max, cut_range
from trackcomb.pid import set_tracks_pid

_PION_MASS = lp.pi_plus.mass  # MeV


# ---------------------------------------------------------------------------
# Slow tests (use session-scoped fixtures from conftest.py)
# ---------------------------------------------------------------------------


class TestCombineSoASanity:
    """Sanity checks on combine output: mass window, charge, PV fields, DIRA."""

    def test_has_candidates(self, ks_candidates):
        total = sum(
            len(ks_candidates["mass"][i]) for i in range(len(ks_candidates["mass"]))
        )
        assert total > 0

    def test_mass_in_window(self, ks_candidates):
        for i in range(len(ks_candidates["mass"])):
            masses = np.asarray(ks_candidates["mass"][i])
            if len(masses) > 0:
                assert np.all(masses >= 400), f"Event {i}: mass below 400"
                assert np.all(masses <= 600), f"Event {i}: mass above 600"

    def test_charge_pattern(self, ks_candidates):
        for i in range(len(ks_candidates["mass"])):
            n = len(ks_candidates["mass"][i])
            for j in range(n):
                q0 = float(ks_candidates["daughter0_charge"][i][j])
                q1 = float(ks_candidates["daughter1_charge"][i][j])
                assert np.sign(q0) != np.sign(q1), (
                    f"Event {i}, cand {j}: same-sign charges {q0}, {q1}"
                )

    def test_pv_fields_present(self, ks_candidates):
        for field in (
            "composite_ip",
            "composite_ip_chi2",
            "dira",
            "best_pv_x",
            "best_pv_y",
            "best_pv_z",
        ):
            assert field in ks_candidates, f"Missing field: {field}"

    def test_composite_ip_nonneg(self, ks_candidates):
        for i in range(len(ks_candidates["composite_ip"])):
            vals = np.asarray(ks_candidates["composite_ip"][i])
            if len(vals) > 0:
                assert np.all(vals >= 0), f"Event {i}: negative composite_ip"

    def test_dira_reasonable(self, ks_candidates):
        for i in range(len(ks_candidates["dira"])):
            dira = np.asarray(ks_candidates["dira"][i])
            if len(dira) > 0:
                assert np.all(dira >= -1.0 - 1e-10), f"Event {i}: dira < -1"
                assert np.all(dira <= 1.0 + 1e-10), f"Event {i}: dira > 1"

    def test_timing_reduces_candidates(self, ks_candidates, ks_candidates_timing):
        n_events = len(ks_candidates["mass"])
        total_no_time = sum(len(ks_candidates["mass"][i]) for i in range(n_events))
        total_time = sum(len(ks_candidates_timing["mass"][i]) for i in range(n_events))
        assert total_time <= total_no_time

    def test_vertex_time_present_with_timing(self, ks_candidates_timing):
        assert "vertex_time" in ks_candidates_timing


class TestCandidateTrackFields:
    """Verify combine output includes track-compatible fields for staged decays."""

    def test_track_fields_present(self, ks_candidates):
        for field in (
            "x",
            "y",
            "z",
            "tx",
            "ty",
            "p",
            "charge",
            "time",
            "sigma_time",
            "track_id",
            "cov_0_0",
            "cov_1_1",
            "cov_2_2",
            "cov_3_3",
        ):
            assert field in ks_candidates, f"Missing: {field}"

    def test_position_from_vertex(self, ks_candidates):
        for i in range(len(ks_candidates["mass"])):
            n = len(ks_candidates["mass"][i])
            for j in range(min(n, 5)):
                np.testing.assert_allclose(
                    float(ks_candidates["x"][i][j]),
                    float(ks_candidates["vertex_x"][i][j]),
                )
                np.testing.assert_allclose(
                    float(ks_candidates["z"][i][j]),
                    float(ks_candidates["vertex_z"][i][j]),
                )

    def test_momentum_direction(self, ks_candidates):
        for i in range(len(ks_candidates["mass"])):
            n = len(ks_candidates["mass"][i])
            for j in range(min(n, 5)):
                px = float(ks_candidates["px"][i][j])
                py = float(ks_candidates["py"][i][j])
                pz = float(ks_candidates["pz"][i][j])
                if abs(pz) > 1e-12:
                    np.testing.assert_allclose(
                        float(ks_candidates["tx"][i][j]),
                        px / pz,
                        rtol=1e-10,
                    )
                    np.testing.assert_allclose(
                        float(ks_candidates["ty"][i][j]),
                        py / pz,
                        rtol=1e-10,
                    )

    def test_can_recombine(self, ks_candidates, root_pvs):
        if all(len(ks_candidates["x"][i]) < 2 for i in range(len(ks_candidates["x"]))):
            pytest.skip("No events with >= 2 candidates")
        result = combine(
            [ks_candidates, ks_candidates],
            root_pvs,
            combination_cuts=[
                cut_max("max_doca", 0.05),
                cut_max("spatial_chi2", 5.0),
            ],
            vertex_cuts=[cut_range("mass", 1000, 2000)],
            use_timing=False,
        )
        assert "mass" in result


class TestStagedDecay:
    """Test staged decay: candidates used directly as composite track pool."""

    def test_cross_pool_combine_runs(self, staged_candidates):
        assert "mass" in staged_candidates

    def test_cross_pool_has_fewer_than_cartesian(
        self, staged_candidates, ks_candidates, root_tracks
    ):
        for i in range(len(staged_candidates["mass"])):
            n_ks = len(ks_candidates["x"][i])
            n_trk = len(root_tracks["x"][i])
            n_cand = len(staged_candidates["mass"][i])
            full_cartesian = n_ks * n_trk
            if n_ks > 0 and n_trk > 0:
                assert n_cand <= full_cartesian, (
                    f"Event {i}: {n_cand} > {full_cartesian} (overlap not removed?)"
                )

    def test_composite_daughter_indices_preserved(self, staged_candidates):
        assert "daughter0_track_id" in staged_candidates
        assert "daughter1_track_id" in staged_candidates


# ---------------------------------------------------------------------------
# Fast tests (synthetic data, no ROOT file needed)
# ---------------------------------------------------------------------------


class TestCombineSoAEdgeCases(unittest.TestCase):
    """Edge cases for combine."""

    def test_empty_tracks(self):
        """Empty track container should produce empty output."""
        tracks = {
            "x": ak.Array([[]]),
            "y": ak.Array([[]]),
            "z": ak.Array([[]]),
            "tx": ak.Array([[]]),
            "ty": ak.Array([[]]),
            "p": ak.Array([[]]),
            "mass": ak.Array([[]]),
            "charge": ak.Array([[]]),
            "time": ak.Array([[]]),
            "sigma_time": ak.Array([[]]),
            "track_id": ak.Array([[]]),
        }
        for ck in (
            "cov_0_0",
            "cov_1_0",
            "cov_1_1",
            "cov_2_0",
            "cov_2_1",
            "cov_2_2",
            "cov_3_0",
            "cov_3_1",
            "cov_3_2",
            "cov_3_3",
        ):
            tracks[ck] = ak.Array([[]])

        pvs = {
            "x": ak.Array([[0.0]]),
            "y": ak.Array([[0.0]]),
            "z": ak.Array([[0.0]]),
            "time": ak.Array([[0.0]]),
            "sigma_time": ak.Array([[0.01]]),
            "cov_0_0": ak.Array([[0.01]]),
            "cov_1_0": ak.Array([[0.0]]),
            "cov_1_1": ak.Array([[0.01]]),
        }

        result = combine([tracks, tracks], pvs, use_timing=False)
        self.assertEqual(len(result["mass"][0]), 0)

    def test_single_track_no_combination(self):
        """One track with n_body=2 should yield zero candidates."""
        tracks = {
            "x": ak.Array([[1.0]]),
            "y": ak.Array([[0.0]]),
            "z": ak.Array([[100.0]]),
            "tx": ak.Array([[0.01]]),
            "ty": ak.Array([[0.0]]),
            "p": ak.Array([[5.0]]),
            "mass": ak.Array([[_PION_MASS]]),
            "charge": ak.Array([[1]]),
            "time": ak.Array([[1.0]]),
            "sigma_time": ak.Array([[0.07]]),
            "track_id": ak.Array([[0]]),
        }
        for ck in (
            "cov_0_0",
            "cov_1_0",
            "cov_1_1",
            "cov_2_0",
            "cov_2_1",
            "cov_2_2",
            "cov_3_0",
            "cov_3_1",
            "cov_3_2",
            "cov_3_3",
        ):
            tracks[ck] = ak.Array(
                [
                    [
                        0.01
                        if "0_0" in ck or "1_1" in ck or "2_2" in ck or "3_3" in ck
                        else 0.0
                    ]
                ]
            )

        pvs = {
            "x": ak.Array([[0.0]]),
            "y": ak.Array([[0.0]]),
            "z": ak.Array([[0.0]]),
            "time": ak.Array([[0.0]]),
            "sigma_time": ak.Array([[0.01]]),
            "cov_0_0": ak.Array([[0.01]]),
            "cov_1_0": ak.Array([[0.0]]),
            "cov_1_1": ak.Array([[0.01]]),
        }

        result = combine([tracks, tracks], pvs, use_timing=False)
        self.assertEqual(len(result["mass"][0]), 0)

    def test_lambda_candidate_cut(self):
        """Custom lambda candidate cut must filter results."""
        tracks = {
            "x": ak.Array([[1.0, -1.0, 0.5]]),
            "y": ak.Array([[0.0, 0.0, 0.1]]),
            "z": ak.Array([[100.0, 100.0, 200.0]]),
            "tx": ak.Array([[0.01, -0.01, 0.005]]),
            "ty": ak.Array([[0.0, 0.0, 0.001]]),
            "p": ak.Array([[5.0, 5.0, 10.0]]),
            "mass": ak.Array([[_PION_MASS, _PION_MASS, _PION_MASS]]),
            "charge": ak.Array([[1, -1, 1]]),
            "time": ak.Array([[1.0, 1.0, 1.0]]),
            "sigma_time": ak.Array([[0.07, 0.07, 0.07]]),
            "track_id": ak.Array([[0, 1, 2]]),
        }
        for ck in ("cov_0_0", "cov_1_1", "cov_2_2", "cov_3_3"):
            tracks[ck] = ak.Array([[0.01, 0.01, 0.01]])
        for ck in ("cov_1_0", "cov_2_0", "cov_2_1", "cov_3_0", "cov_3_1", "cov_3_2"):
            tracks[ck] = ak.Array([[0.0, 0.0, 0.0]])

        pvs = {
            "x": ak.Array([[0.0]]),
            "y": ak.Array([[0.0]]),
            "z": ak.Array([[0.0]]),
            "time": ak.Array([[0.0]]),
            "sigma_time": ak.Array([[0.01]]),
            "cov_0_0": ak.Array([[0.01]]),
            "cov_1_0": ak.Array([[0.0]]),
            "cov_1_1": ak.Array([[0.01]]),
        }

        no_cut = combine([tracks, tracks], pvs, use_timing=False)
        n_no_cut = len(no_cut["mass"][0])

        with_cut = combine(
            [tracks, tracks],
            pvs,
            use_timing=False,
            vertex_cuts=[lambda c: c["mass"] > 999000],
        )
        n_with_cut = len(with_cut["mass"][0])

        self.assertGreater(n_no_cut, 0, "Should have candidates without cut")
        self.assertEqual(n_with_cut, 0, "All candidates should be removed by cut")


class TestMultiPoolCombine(unittest.TestCase):
    """Test pool-based combine with distinct track pools."""

    def _make_tracks(self, n_tracks_per_event, seed=0):
        rng = np.random.RandomState(seed)
        n_events = len(n_tracks_per_event)
        tracks = {}
        for field in ("x", "y", "z", "tx", "ty"):
            tracks[field] = ak.Array([rng.randn(n) * 0.1 for n in n_tracks_per_event])
        tracks["p"] = ak.Array(
            [np.abs(rng.randn(n)) * 10 + 5 for n in n_tracks_per_event]
        )
        tracks["charge"] = ak.Array(
            [rng.choice([1.0, -1.0], size=n) for n in n_tracks_per_event]
        )
        tracks["time"] = ak.Array([rng.randn(n) * 0.01 for n in n_tracks_per_event])
        tracks["sigma_time"] = ak.Array([[0.07] * n for n in n_tracks_per_event])
        tracks["mass"] = ak.Array([[_PION_MASS] * n for n in n_tracks_per_event])
        tracks["track_id"] = ak.Array([list(range(n)) for n in n_tracks_per_event])
        for i in range(4):
            for j in range(i + 1):
                if i == j:
                    tracks[f"cov_{i}_{j}"] = ak.Array(
                        [[0.001] * n for n in n_tracks_per_event]
                    )
                else:
                    tracks[f"cov_{i}_{j}"] = ak.Array(
                        [[0.0] * n for n in n_tracks_per_event]
                    )
        return tracks

    def _make_pvs(self, n_events):
        return {
            "x": ak.Array([[0.0]] * n_events),
            "y": ak.Array([[0.0]] * n_events),
            "z": ak.Array([[0.0]] * n_events),
            "time": ak.Array([[0.0]] * n_events),
            "sigma_time": ak.Array([[0.01]] * n_events),
            "cov_0_0": ak.Array([[0.01]] * n_events),
            "cov_1_0": ak.Array([[0.0]] * n_events),
            "cov_1_1": ak.Array([[0.01]] * n_events),
        }

    def test_same_pool_gives_combinations(self):
        tracks = self._make_tracks([4, 3])
        pvs = self._make_pvs(2)
        result = combine([tracks, tracks], pvs, use_timing=False)
        self.assertEqual(len(result["mass"][0]), 6)
        self.assertEqual(len(result["mass"][1]), 3)

    def test_diff_pools_gives_cartesian(self):
        pool_a = self._make_tracks([3, 2], seed=0)
        pool_b = self._make_tracks([2, 3], seed=42)
        pool_a["track_id"] = ak.Array([[0, 1, 2], [0, 1]])
        pool_b["track_id"] = ak.Array([[10, 11], [10, 11, 12]])
        pvs = self._make_pvs(2)
        result = combine([pool_a, pool_b], pvs, use_timing=False)
        self.assertEqual(len(result["mass"][0]), 6)
        self.assertEqual(len(result["mass"][1]), 6)

    def test_cross_pool_overlap_removed(self):
        pool_a = self._make_tracks([3], seed=0)
        pool_b = self._make_tracks([3], seed=42)
        pvs = self._make_pvs(1)
        result = combine([pool_a, pool_b], pvs, use_timing=False)
        self.assertEqual(len(result["mass"][0]), 6)


class TestSetTracksPid(unittest.TestCase):
    """Test set_tracks_pid function."""

    def _make_tracks(self):
        return {
            "x": ak.Array([[1.0, 2.0], [3.0]]),
            "y": ak.Array([[0.1, 0.2], [0.3]]),
        }

    def test_string_lookup(self):
        tracks = set_tracks_pid(self._make_tracks(), "pi+")
        assert "mass" in tracks
        assert "pid" in tracks
        np.testing.assert_allclose(
            ak.to_numpy(ak.flatten(tracks["mass"])),
            [_PION_MASS, _PION_MASS, _PION_MASS],
            rtol=1e-6,
        )

    def test_int_lookup(self):
        tracks = set_tracks_pid(self._make_tracks(), 211)
        np.testing.assert_allclose(
            ak.to_numpy(ak.flatten(tracks["mass"])),
            [_PION_MASS, _PION_MASS, _PION_MASS],
            rtol=1e-6,
        )
        pid_flat = ak.to_numpy(ak.flatten(tracks["pid"]))
        np.testing.assert_array_equal(pid_flat, [211, 211, 211])

    def test_preserves_existing_fields(self):
        orig = self._make_tracks()
        out = set_tracks_pid(orig, "K+")
        assert ak.to_list(out["x"]) == ak.to_list(orig["x"])

    def test_bad_name_raises(self):
        with self.assertRaises(ValueError):
            set_tracks_pid(self._make_tracks(), "quark_soup")


if __name__ == "__main__":
    unittest.main()
