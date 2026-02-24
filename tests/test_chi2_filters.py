"""Unit tests for candidate-level chi2-based filtering behavior."""

from __future__ import annotations

import unittest

from trackcomb import (
    CombinationCuts,
    ParticleCombiner,
    PrimaryVertex,
    TrackState,
    make_pion,
    make_proton,
)


def _cov4(scale: float = 1.0):
    """Build a diagonal covariance matrix used in simple test tracks."""
    return (
        (scale, 0.0, 0.0, 0.0),
        (0.0, scale, 0.0, 0.0),
        (0.0, 0.0, scale, 0.0),
        (0.0, 0.0, 0.0, scale),
    )


def _pv() -> PrimaryVertex:
    """Create a baseline PV hypothesis for filter tests."""
    return PrimaryVertex(
        pv_id="pv0",
        x=0.0,
        y=0.0,
        z=0.0,
        cov3=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
        time=1.0,
        sigma_time=0.1,
    )


def _tracks_for_filter_tests() -> list[TrackState]:
    """Create a small track set where only one pair is strongly compatible."""
    return [
        TrackState("t1", z=0.0, x=0.0, y=0.0, tx=0.0, ty=0.0, time=1.00, cov4=_cov4(0.01), sigma_time=0.1, p=5.0),
        TrackState("t2", z=0.0, x=0.1, y=0.0, tx=0.0, ty=0.0, time=1.02, cov4=_cov4(0.01), sigma_time=0.1, p=5.0),
        TrackState("t3", z=0.0, x=10.0, y=0.0, tx=0.0, ty=0.0, time=3.00, cov4=_cov4(0.01), sigma_time=0.1, p=5.0),
    ]


class TestChi2Filters(unittest.TestCase):
    """Validate vertex/time chi2 cuts individually and in combination."""

    def test_vertex_chi2_filter_keeps_only_spatially_compatible_pair(self) -> None:
        """Spatial chi2 cut should reject geometrically incompatible pairs."""
        results = ParticleCombiner().combine(
            tracks=_tracks_for_filter_tests(),
            primary_vertices=[_pv()],
            n_body=2,
            mass_hypotheses=[[0.13957, 0.13957]],
            cuts=CombinationCuts(max_vertex_chi2=100.0),
        )

        self.assertEqual([r.track_ids for r in results], [("t1", "t2")])
        self.assertLess(results[0].vertex_chi2, 100.0)

    def test_time_chi2_filter_keeps_only_time_compatible_pair(self) -> None:
        """Timing chi2 cut should reject temporally incompatible pairs."""
        results = ParticleCombiner().combine(
            tracks=_tracks_for_filter_tests(),
            primary_vertices=[_pv()],
            n_body=2,
            mass_hypotheses=[[0.13957, 0.13957]],
            cuts=CombinationCuts(max_pair_time_chi2=1.0),
        )

        self.assertEqual([r.track_ids for r in results], [("t1", "t2")])
        self.assertLess(results[0].pair_time_chi2, 1.0)

    def test_vertex_and_time_filters_can_be_applied_together(self) -> None:
        """Joint chi2 cuts should keep only pairs satisfying both constraints."""
        results = ParticleCombiner().combine(
            tracks=_tracks_for_filter_tests(),
            primary_vertices=[_pv()],
            n_body=2,
            mass_hypotheses=[[0.13957, 0.13957]],
            cuts=CombinationCuts(max_vertex_chi2=100.0, max_pair_time_chi2=1.0),
        )

        self.assertEqual([r.track_ids for r in results], [("t1", "t2")])

    def test_time_chi2_uses_beta_corrected_vertex_propagation(self) -> None:
        """Pair/vertex time chi2 should include mass-dependent beta correction."""
        tracks = [
            TrackState(
                "slow",
                z=10.0,
                x=1.0,
                y=0.0,
                tx=0.1,
                ty=0.0,
                time=0.0,
                cov4=_cov4(0.01),
                sigma_time=0.1,
                p=0.25,
            ),
            TrackState(
                "fast",
                z=10.0,
                x=-1.0,
                y=0.0,
                tx=-0.1,
                ty=0.0,
                time=0.0,
                cov4=_cov4(0.01),
                sigma_time=0.1,
                p=5.0,
            ),
        ]
        [res] = ParticleCombiner().combine(
            tracks=tracks,
            primary_vertices=[_pv()],
            n_body=2,
            mass_hypotheses=[[make_proton(), make_pion()]],
        )
        self.assertGreater(res.pair_time_chi2, 0.1)
        self.assertGreater(res.vertex_time_chi2, 0.1)


if __name__ == "__main__":
    unittest.main()
