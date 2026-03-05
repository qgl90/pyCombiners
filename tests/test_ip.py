"""Tests for IP computation."""

from __future__ import annotations

import unittest

import awkward as ak
import numpy as np

from trackcomb.physics import ip_to_pvs


class TestBatchIPEdgeCases(unittest.TestCase):
    """Edge cases for batch IP functions."""

    def test_empty_events(self):
        """Batch IP should handle events with no tracks or no PVs."""
        tracks = {
            "x": ak.Array([[], [1.0, 2.0]]),
            "y": ak.Array([[], [0.0, 0.0]]),
            "z": ak.Array([[], [100.0, 200.0]]),
            "tx": ak.Array([[], [0.01, -0.01]]),
            "ty": ak.Array([[], [0.0, 0.0]]),
            "cov_0_0": ak.Array([[], [0.01, 0.01]]),
            "cov_1_0": ak.Array([[], [0.0, 0.0]]),
            "cov_1_1": ak.Array([[], [0.01, 0.01]]),
            "cov_2_0": ak.Array([[], [0.0, 0.0]]),
            "cov_2_1": ak.Array([[], [0.0, 0.0]]),
            "cov_2_2": ak.Array([[], [0.001, 0.001]]),
            "cov_3_0": ak.Array([[], [0.0, 0.0]]),
            "cov_3_1": ak.Array([[], [0.0, 0.0]]),
            "cov_3_2": ak.Array([[], [0.0, 0.0]]),
            "cov_3_3": ak.Array([[], [0.001, 0.001]]),
            "time": ak.Array([[], [1.0, 2.0]]),
        }
        pvs = {
            "x": ak.Array([[], [0.0]]),
            "y": ak.Array([[], [0.0]]),
            "z": ak.Array([[], [0.0]]),
            "cov_0_0": ak.Array([[], [0.01]]),
            "cov_1_0": ak.Array([[], [0.0]]),
            "cov_1_1": ak.Array([[], [0.01]]),
            "time": ak.Array([[], [0.0]]),
        }
        ip, chi2 = ip_to_pvs(tracks, pvs)
        self.assertEqual(len(ip[0]), 0)  # event 0: no tracks
        self.assertEqual(len(ip[1]), 2)  # event 1: 2 tracks

    def test_single_track_single_pv(self):
        """Verify a simple analytic case: track at (1,0,0) with tx=ty=0, PV at origin."""
        tracks = {
            "x": ak.Array([[1.0]]),
            "y": ak.Array([[0.0]]),
            "z": ak.Array([[0.0]]),
            "tx": ak.Array([[0.0]]),
            "ty": ak.Array([[0.0]]),
            "cov_0_0": ak.Array([[0.0]]),
            "cov_1_0": ak.Array([[0.0]]),
            "cov_1_1": ak.Array([[0.0]]),
            "cov_2_0": ak.Array([[0.0]]),
            "cov_2_1": ak.Array([[0.0]]),
            "cov_2_2": ak.Array([[0.0]]),
            "cov_3_0": ak.Array([[0.0]]),
            "cov_3_1": ak.Array([[0.0]]),
            "cov_3_2": ak.Array([[0.0]]),
            "cov_3_3": ak.Array([[0.0]]),
        }
        pvs = {
            "x": ak.Array([[0.0]]),
            "y": ak.Array([[0.0]]),
            "z": ak.Array([[0.0]]),
            "cov_0_0": ak.Array([[0.0]]),
            "cov_1_0": ak.Array([[0.0]]),
            "cov_1_1": ak.Array([[0.0]]),
        }
        ip, chi2 = ip_to_pvs(tracks, pvs)
        # Track at (1,0) relative to PV at (0,0) — same z, no extrapolation
        np.testing.assert_allclose(float(ip[0][0][0]), 1.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
