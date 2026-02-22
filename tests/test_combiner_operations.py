"""Unit tests for core combiner operations and helper behavior."""

from __future__ import annotations

import math
import unittest

from trackcomb import CombinationCuts, ParticleCombiner, PrimaryVertex, TrackState, TrackPreselection
from trackcomb.physics import fit_vertex_xyz_t


class TestCombinerOperations(unittest.TestCase):
    """Validate combination math, outputs, preselection, and charge filters."""

    @staticmethod
    def _cov4(scale: float = 1.0):
        """Build a simple diagonal 4x4 covariance matrix."""
        return (
            (scale, 0.0, 0.0, 0.0),
            (0.0, scale, 0.0, 0.0),
            (0.0, 0.0, scale, 0.0),
            (0.0, 0.0, 0.0, scale),
        )

    @staticmethod
    def _pvs() -> list[PrimaryVertex]:
        """Provide a default one-PV container for tests."""
        return [
            PrimaryVertex(
                pv_id="pv0",
                x=0.0,
                y=0.0,
                z=0.0,
                cov3=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
                time=1.0,
                sigma_time=0.1,
            )
        ]

    def test_combiner_counts_all_2body_and_hypotheses(self) -> None:
        """Ensure combination count matches combinatorics times mass hypotheses."""
        tracks = [
            TrackState("t1", z=0.0, x=0.0, y=0.0, tx=0.01, ty=0.01, time=1.0, cov4=self._cov4(), sigma_time=0.1, p=10.0),
            TrackState("t2", z=0.0, x=0.1, y=0.0, tx=-0.01, ty=0.02, time=1.1, cov4=self._cov4(), sigma_time=0.1, p=8.0),
            TrackState("t3", z=0.0, x=0.0, y=0.2, tx=0.03, ty=-0.01, time=1.2, cov4=self._cov4(), sigma_time=0.1, p=7.0),
            TrackState("t4", z=0.0, x=-0.2, y=0.1, tx=0.00, ty=0.00, time=1.3, cov4=self._cov4(), sigma_time=0.1, p=9.0),
        ]
        mass_hypotheses = [[0.13957, 0.13957], [0.49367, 0.13957]]

        results = ParticleCombiner().combine(
            tracks=tracks,
            primary_vertices=self._pvs(),
            n_body=2,
            mass_hypotheses=mass_hypotheses,
        )

        self.assertEqual(len(results), 12)  # C(4,2)=6, times 2 mass hypotheses

    def test_two_body_lorentz_sum_and_vertex_extrapolation(self) -> None:
        """Check basic 2-body kinematics and fitted vertex z."""
        tracks = [
            TrackState("a", z=0.0, x=0.0, y=0.0, tx=0.0, ty=0.0, time=1.0, cov4=self._cov4(), sigma_time=0.1, p=3.0),
            TrackState("b", z=0.0, x=1.0, y=-1.0, tx=0.0, ty=0.0, time=1.0, cov4=self._cov4(), sigma_time=0.1, p=4.0),
        ]
        masses = [[1.0, 2.0]]

        [result] = ParticleCombiner().combine(
            tracks=tracks,
            primary_vertices=self._pvs(),
            n_body=2,
            mass_hypotheses=masses,
        )

        self.assertAlmostEqual(result.vertex_xyz[2], 0.0, places=12)
        expected_e = math.sqrt(3.0 * 3.0 + 1.0 * 1.0) + math.sqrt(4.0 * 4.0 + 2.0 * 2.0)
        self.assertAlmostEqual(result.candidate_p4.px, 0.0, places=12)
        self.assertAlmostEqual(result.candidate_p4.py, 0.0, places=12)
        self.assertAlmostEqual(result.candidate_p4.pz, 7.0, places=12)
        self.assertAlmostEqual(result.candidate_p4.e, expected_e, places=12)
        self.assertAlmostEqual(
            result.candidate_p4.mass,
            math.sqrt(expected_e * expected_e - 49.0),
            places=12,
        )

    def test_vertex_fit_xyz_time(self) -> None:
        """Check that simple symmetric inputs produce expected 4D vertex fit."""
        tracks = [
            TrackState("a", z=0.0, x=0.0, y=0.0, tx=0.0, ty=0.0, time=1.0, cov4=self._cov4(), sigma_time=0.1, p=1.0),
            TrackState("b", z=0.0, x=1.0, y=0.0, tx=0.0, ty=0.0, time=1.2, cov4=self._cov4(), sigma_time=0.1, p=1.0),
        ]
        xyz, t, _, _ = fit_vertex_xyz_t(tracks)
        self.assertAlmostEqual(xyz[0], 0.5, places=12)
        self.assertAlmostEqual(xyz[1], 0.0, places=12)
        self.assertAlmostEqual(xyz[2], 0.0, places=12)
        self.assertAlmostEqual(t, 1.1, places=12)

    def test_invalid_mass_hypothesis_length_raises(self) -> None:
        """Invalid mass hypothesis size must raise a ValueError."""
        tracks = [
            TrackState("a", z=0.0, x=0.0, y=0.0, tx=0.0, ty=0.0, time=0.0, cov4=self._cov4(), sigma_time=0.1, p=1.0),
            TrackState("b", z=0.0, x=0.0, y=0.0, tx=0.0, ty=0.0, time=0.0, cov4=self._cov4(), sigma_time=0.1, p=1.0),
        ]
        with self.assertRaises(ValueError):
            ParticleCombiner().combine(
                tracks=tracks,
                primary_vertices=self._pvs(),
                n_body=2,
                mass_hypotheses=[[0.13957, 0.13957, 0.13957]],
            )

    def test_combiner_reports_doca_and_ip_metrics(self) -> None:
        """Combination outputs must include DOCA and IP/IPchi2 metrics."""
        tracks = [
            TrackState("a", z=0.0, x=0.0, y=0.0, tx=0.0, ty=0.0, time=0.0, cov4=self._cov4(0.01), sigma_time=0.1, p=1.0),
            TrackState("b", z=0.0, x=0.2, y=0.0, tx=0.0, ty=0.0, time=0.0, cov4=self._cov4(0.01), sigma_time=0.1, p=1.0),
            TrackState("c", z=0.0, x=0.1, y=0.1, tx=0.0, ty=0.0, time=0.0, cov4=self._cov4(0.01), sigma_time=0.1, p=1.0),
        ]
        pv = PrimaryVertex(
            pv_id="pvX",
            x=0.0,
            y=0.0,
            z=0.0,
            cov3=((0.001, 0.0, 0.0), (0.0, 0.001, 0.0), (0.0, 0.0, 0.001)),
            time=0.0,
            sigma_time=0.1,
        )
        [res] = ParticleCombiner().combine(
            tracks=tracks,
            primary_vertices=[pv],
            n_body=3,
            mass_hypotheses=[[0.1, 0.1, 0.1]],
        )
        self.assertIn("doca12", res.doca_pairs)
        self.assertIn("doca13", res.doca_pairs)
        self.assertIn("doca23", res.doca_pairs)
        self.assertIn("a", res.track_min_ip)
        self.assertIn("a", res.track_min_ip_chi2)

    def test_preselection_min_track_pt(self) -> None:
        """Track preselection by minimum pT must keep only qualifying tracks."""
        tracks = [
            TrackState("a", z=0.0, x=0.0, y=0.0, tx=0.0, ty=0.0, time=0.0, cov4=self._cov4(), sigma_time=0.1, p=1.0),
            TrackState("b", z=0.0, x=0.0, y=0.0, tx=1.0, ty=0.0, time=0.0, cov4=self._cov4(), sigma_time=0.1, p=3.0),
        ]
        out = ParticleCombiner().preselect_tracks(
            tracks=tracks,
            primary_vertices=list(self._pvs()),
            preselection=TrackPreselection(min_pt=2.0),
        )
        self.assertEqual([t.track_id for t in out], ["b"])

    def test_charge_pattern_filter_for_2body(self) -> None:
        """2-body charge-pattern filtering should retain only allowed pairs."""
        tracks = [
            TrackState("p1", z=0.0, x=0.0, y=0.0, tx=0.0, ty=0.0, time=0.0, cov4=self._cov4(), sigma_time=0.1, p=1.0, charge=1),
            TrackState("p2", z=0.0, x=0.1, y=0.0, tx=0.0, ty=0.0, time=0.0, cov4=self._cov4(), sigma_time=0.1, p=1.0, charge=1),
            TrackState("m1", z=0.0, x=0.2, y=0.0, tx=0.0, ty=0.0, time=0.0, cov4=self._cov4(), sigma_time=0.1, p=1.0, charge=-1),
        ]
        results = ParticleCombiner().combine(
            tracks=tracks,
            primary_vertices=self._pvs(),
            n_body=2,
            mass_hypotheses=[[0.1, 0.1]],
            cuts=CombinationCuts(allowed_charge_patterns=("+-", "-+",)),
        )
        self.assertTrue(all(r.charge_pattern in {"+-", "-+"} for r in results))

    def test_charge_pattern_filter_for_3body(self) -> None:
        """3-body charge-pattern filtering should retain only allowed triplets."""
        tracks = [
            TrackState("p1", z=0.0, x=0.0, y=0.0, tx=0.0, ty=0.0, time=0.0, cov4=self._cov4(), sigma_time=0.1, p=1.0, charge=1),
            TrackState("p2", z=0.0, x=0.1, y=0.0, tx=0.0, ty=0.0, time=0.0, cov4=self._cov4(), sigma_time=0.1, p=1.0, charge=1),
            TrackState("m1", z=0.0, x=0.2, y=0.0, tx=0.0, ty=0.0, time=0.0, cov4=self._cov4(), sigma_time=0.1, p=1.0, charge=-1),
            TrackState("m2", z=0.0, x=0.3, y=0.0, tx=0.0, ty=0.0, time=0.0, cov4=self._cov4(), sigma_time=0.1, p=1.0, charge=-1),
        ]
        results = ParticleCombiner().combine(
            tracks=tracks,
            primary_vertices=self._pvs(),
            n_body=3,
            mass_hypotheses=[[0.1, 0.1, 0.1]],
            cuts=CombinationCuts(allowed_charge_patterns=("++-", "+--")),
        )
        self.assertTrue(all(r.charge_pattern in {"++-", "+--"} for r in results))


if __name__ == "__main__":
    unittest.main()
