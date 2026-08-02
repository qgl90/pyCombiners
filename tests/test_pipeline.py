"""SoA pipeline tests: combine correctness, edge cases, multi-pool, staged decay."""

from __future__ import annotations

import unittest

import pytest

import awkward as ak
import numpy as np

from trackcomb.combiner import combine
from trackcomb.models import cut_max, cut_range, get_daughter
from trackcomb.pid import pdg_mass

_PION_MASS = pdg_mass("pi+")


# ---------------------------------------------------------------------------
# Slow tests (use session-scoped fixtures from conftest.py)
# ---------------------------------------------------------------------------


class TestCombineSoASanity:
    """Sanity checks on combine output: mass window, charge, PV fields, DIRA."""

    def test_has_candidates(self, ks_candidates):
        total = sum(
            len(ks_candidates["mass"][i])
            for i in range(len(ks_candidates["mass"]))
        )
        assert total > 0

    def test_mass_in_window(self, ks_candidates):
        for i in range(len(ks_candidates["mass"])):
            masses = np.asarray(ks_candidates["mass"][i])
            if len(masses) > 0:
                assert np.all(masses >= 400), f"Event {i}: mass below 400"
                assert np.all(masses <= 600), f"Event {i}: mass above 600"

    def test_charge_pattern(self, ks_candidates):
        d0_charge = get_daughter(ks_candidates, 0, "charge")
        d1_charge = get_daughter(ks_candidates, 1, "charge")
        for i in range(len(ks_candidates["mass"])):
            n = len(ks_candidates["mass"][i])
            for j in range(n):
                q0 = float(d0_charge[i][j])
                q1 = float(d1_charge[i][j])
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

    def test_vertex_time_present(self, ks_candidates):
        assert "vertex_time" in ks_candidates


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
            "qop",
            "time",
            "sigma_time",
            "cov_0_0",
            "cov_1_1",
            "cov_2_2",
            "cov_3_3",
            "cov_4_4",
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
        if all(
            len(ks_candidates["x"][i]) < 2
            for i in range(len(ks_candidates["x"]))
        ):
            pytest.skip("No events with >= 2 candidates")
        result = combine(
            [ks_candidates, ks_candidates],
            root_pvs,
            combination_cuts=[
                cut_max("max_doca", 0.05),
                cut_range("mass", 1000, 2000),
            ],
            composite_cuts=[
                cut_max("vertex_chi2", 5.0),
            ],
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
        # daughter0 is a composite (no track_id), daughter1 is a track pool
        assert "daughter0_track_id" not in staged_candidates
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
            "qop": ak.Array([[]]),
        }
        for i in range(5):
            for j in range(i + 1):
                tracks[f"cov_{i}_{j}"] = ak.Array([[]])

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

        result = combine([tracks, tracks], pvs)
        self.assertIsNone(result)

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
            "qop": ak.Array([[1.0 / 5.0]]),
        }
        for i in range(5):
            for j in range(i + 1):
                tracks[f"cov_{i}_{j}"] = ak.Array([[0.01 if i == j else 0.0]])

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

        result = combine([tracks, tracks], pvs)
        self.assertIsNone(result)

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
            "qop": ak.Array([[1.0 / 5.0, -1.0 / 5.0, 1.0 / 10.0]]),
        }
        for i in range(5):
            for j in range(i + 1):
                tracks[f"cov_{i}_{j}"] = ak.Array(
                    [[0.01 if i == j else 0.0] * 3]
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

        no_cut = combine([tracks, tracks], pvs)
        n_no_cut = len(no_cut["mass"][0])

        with_cut = combine(
            [tracks, tracks],
            pvs,
            combination_cuts=[lambda c: c["mass"] > 999000],
        )

        self.assertGreater(n_no_cut, 0, "Should have candidates without cut")
        self.assertIsNone(with_cut, "All candidates should be removed by cut")


class TestMultiPoolCombine(unittest.TestCase):
    """Test pool-based combine with distinct track pools."""

    def _make_tracks(self, n_tracks_per_event, seed=0):
        rng = np.random.RandomState(seed)
        tracks = {}
        for field in ("x", "y", "z", "tx", "ty"):
            tracks[field] = ak.Array(
                [rng.randn(n) * 0.1 for n in n_tracks_per_event]
            )
        tracks["p"] = ak.Array(
            [np.abs(rng.randn(n)) * 10 + 5 for n in n_tracks_per_event]
        )
        tracks["charge"] = ak.Array(
            [rng.choice([1.0, -1.0], size=n) for n in n_tracks_per_event]
        )
        tracks["time"] = ak.Array(
            [rng.randn(n) * 0.01 for n in n_tracks_per_event]
        )
        tracks["sigma_time"] = ak.Array(
            [[0.07] * n for n in n_tracks_per_event]
        )
        tracks["mass"] = ak.Array(
            [[_PION_MASS] * n for n in n_tracks_per_event]
        )
        tracks["track_id"] = ak.Array(
            [list(range(n)) for n in n_tracks_per_event]
        )
        tracks["qop"] = ak.Array(
            [
                1.0
                / np.abs(rng.randn(n) * 10 + 5)
                * rng.choice([1, -1], size=n)
                for n in n_tracks_per_event
            ]
        )
        for i in range(5):
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
        result = combine([tracks, tracks], pvs)
        self.assertEqual(len(result["mass"][0]), 6)
        self.assertEqual(len(result["mass"][1]), 3)

    def test_diff_pools_gives_cartesian(self):
        pool_a = self._make_tracks([3, 2], seed=0)
        pool_b = self._make_tracks([2, 3], seed=42)
        pool_a["track_id"] = ak.Array([[0, 1, 2], [0, 1]])
        pool_b["track_id"] = ak.Array([[10, 11], [10, 11, 12]])
        pvs = self._make_pvs(2)
        result = combine([pool_a, pool_b], pvs)
        self.assertEqual(len(result["mass"][0]), 6)
        self.assertEqual(len(result["mass"][1]), 6)

    def test_cross_pool_overlap_removed(self):
        pool_a = self._make_tracks([3], seed=0)
        pool_b = self._make_tracks([3], seed=42)
        pvs = self._make_pvs(1)
        result = combine([pool_a, pool_b], pvs)
        self.assertEqual(len(result["mass"][0]), 6)


if __name__ == "__main__":
    unittest.main()
