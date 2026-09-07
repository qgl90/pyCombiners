"""Focused tests for straight-line covariance propagation in DOCA chi2."""

from __future__ import annotations

import unittest

import awkward as ak
import numpy as np

from trackcomb.physics import (
    _propagate_track_xy_covariance,
    compute_doca,
)


class TestDocaChi2(unittest.TestCase):
    def test_full_state_covariance_indexing(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(5, 5))
        full_cov = a @ a.T
        dz = np.array([7.5])
        cov = {
            f"cov_{i}_{j}": np.array([full_cov[i, j]])
            for i in range(5)
            for j in range(i + 1)
        }

        cxx, cxy, cyy = _propagate_track_xy_covariance(cov, dz)

        transport = np.eye(5)
        transport[0, 2] = dz[0]
        transport[1, 3] = dz[0]
        expected = transport @ full_cov @ transport.T
        np.testing.assert_allclose(
            [cxx[0], cxy[0], cyy[0]],
            [expected[0, 0], expected[0, 1], expected[1, 1]],
            rtol=1e-14,
        )

    def test_doca_chi2_uses_each_track_reference_z(self):
        # Parallel tracks: the chosen POCA is at track 1's reference z=30.
        # Track 0 therefore propagates by dz=20, while track 1 has dz=0.
        pools = []
        for x, z in ((1.0, 10.0), (0.0, 30.0)):
            pool = {
                "x": ak.Array([[x]]),
                "y": ak.Array([[0.0]]),
                "z": ak.Array([[z]]),
                "tx": ak.Array([[0.0]]),
                "ty": ak.Array([[0.0]]),
            }
            for i in range(5):
                for j in range(i + 1):
                    pool[f"cov_{i}_{j}"] = ak.Array([[1.0 if i == j else 0.0]])
            pools.append(pool)

        comb = {
            "_daughter_pools": pools,
            "daughter0_global_index": np.array([0]),
            "daughter1_global_index": np.array([0]),
        }
        compute_doca(comb)

        # var_x(track0 at z=30) = 1 + 20^2; track1 contributes 1.
        expected_chi2 = 1.0 / 402.0
        np.testing.assert_allclose(comb["doca12"], [1.0])
        np.testing.assert_allclose(comb["doca12_chi2"], [expected_chi2])
        np.testing.assert_allclose(comb["min_doca_chi2"], [expected_chi2])
        np.testing.assert_allclose(comb["max_doca_chi2"], [expected_chi2])


if __name__ == "__main__":
    unittest.main()
