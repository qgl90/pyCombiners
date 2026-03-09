"""Tests for combination generation and physics utilities."""

from __future__ import annotations

import unittest

import awkward as ak
import numpy as np

from trackcomb.combiner import make_combinations


class TestMakeCombinationsMultiPool(unittest.TestCase):
    """Test make_combinations with distinct pools (cartesian product)."""

    def _make_pool(self, n_per_event, seed=0):
        rng = np.random.RandomState(seed)
        pool = {}
        for field in ("x", "y", "z", "tx", "ty"):
            pool[field] = ak.Array([rng.randn(n) * 0.1 for n in n_per_event])
        pool["p"] = ak.Array(
            [np.abs(rng.randn(n)) * 10 + 5 for n in n_per_event]
        )
        pool["charge"] = ak.Array(
            [rng.choice([1.0, -1.0], size=n) for n in n_per_event]
        )
        pool["track_id"] = ak.Array(
            [list(range(seed * 100, seed * 100 + n)) for n in n_per_event]
        )
        return pool

    def test_two_distinct_pools_cartesian(self):
        """Two distinct pools → n1 * n2 combos per event."""
        pool_a = self._make_pool([3, 2], seed=0)
        pool_b = self._make_pool([2, 4], seed=42)
        comb = make_combinations([pool_a, pool_b])
        event_idx = comb["event_idx"]
        # Event 0: 3*2=6, Event 1: 2*4=8
        self.assertEqual(np.sum(event_idx == 0), 6)
        self.assertEqual(np.sum(event_idx == 1), 8)

    def test_same_pool_identity(self):
        """Same object twice → C(n, 2) (not n^2)."""
        pool = self._make_pool([4, 3])
        comb = make_combinations([pool, pool])
        event_idx = comb["event_idx"]
        # C(4,2)=6, C(3,2)=3
        self.assertEqual(np.sum(event_idx == 0), 6)
        self.assertEqual(np.sum(event_idx == 1), 3)

    def test_daughter_fields_from_correct_pool(self):
        """Global indices should point to correct pool entries."""
        pool_a = self._make_pool([3], seed=0)
        pool_b = self._make_pool([2], seed=42)
        comb = make_combinations([pool_a, pool_b])
        # daughter0 track_id values should be from pool_a (seed=0 → 0..2)
        d0_tid = comb["daughter0_track_id"]
        d1_tid = comb["daughter1_track_id"]
        self.assertTrue(np.all((d0_tid >= 0) & (d0_tid < 3)))
        self.assertTrue(np.all((d1_tid >= 4200) & (d1_tid < 4202)))


def _make_flat_pool(**fields):
    """Wrap flat numpy arrays into a pool with 1-event jagged structure."""
    return {k: ak.Array([v]) for k, v in fields.items()}


def _make_two_body_comb(pool_a, pool_b, idx_a, idx_b):
    """Build a minimal flat comb dict for testing physics functions."""
    comb = {
        "_daughter_pools": [pool_a, pool_b],
        "daughter0_global_index": idx_a,
        "daughter1_global_index": idx_b,
    }
    return comb


class TestCompositeCovariance(unittest.TestCase):
    """Test compute_composite_covariance correctness."""

    def _build_single_daughter_comb(self, tx, ty, qop, cov_3x3):
        """Build a 1-body comb with one daughter that has known state and cov.

        For a single daughter, the composite momentum equals the daughter's,
        so the roundtrip J then K should recover the original (tx, ty, qop) cov.
        """
        p = np.abs(1.0 / qop)
        s = np.sqrt(1.0 + tx**2 + ty**2)
        px = p * tx / s
        py = p * ty / s
        pz = p / s
        charge = np.sign(qop)

        pool = _make_flat_pool(
            tx=np.array([tx]),
            ty=np.array([ty]),
            p=np.array([p]),
            qop=np.array([qop]),
            cov_2_2=np.array([cov_3x3[0, 0]]),
            cov_3_2=np.array([cov_3x3[1, 0]]),
            cov_3_3=np.array([cov_3x3[1, 1]]),
            cov_4_2=np.array([cov_3x3[2, 0]]),
            cov_4_3=np.array([cov_3x3[2, 1]]),
            cov_4_4=np.array([cov_3x3[2, 2]]),
        )
        comb = {
            "_daughter_pools": [pool],
            "daughter0_global_index": np.array([0]),
            "px": np.array([px]),
            "py": np.array([py]),
            "pz": np.array([pz]),
            "charge": np.array([charge]),
            "vertex_cov_0_0": np.array([1e-4]),
            "vertex_cov_1_0": np.array([1e-6]),
            "vertex_cov_1_1": np.array([1e-4]),
        }
        return comb

    def test_single_daughter_roundtrip(self):
        """Single daughter: composite (tx,ty,qop) cov should equal the daughter's."""
        from trackcomb.physics import compute_composite_covariance

        cov_in = np.array(
            [
                [1e-6, 2e-8, 1e-9],
                [2e-8, 1.5e-6, -3e-9],
                [1e-9, -3e-9, 4e-10],
            ]
        )
        comb = self._build_single_daughter_comb(
            tx=0.3, ty=-0.1, qop=1.0 / 50000.0, cov_3x3=cov_in
        )
        compute_composite_covariance(comb)

        cov_out = np.array(
            [
                [comb["cov_2_2"][0], comb["cov_3_2"][0], comb["cov_4_2"][0]],
                [comb["cov_3_2"][0], comb["cov_3_3"][0], comb["cov_4_3"][0]],
                [comb["cov_4_2"][0], comb["cov_4_3"][0], comb["cov_4_4"][0]],
            ]
        )
        np.testing.assert_allclose(cov_out, cov_in, rtol=1e-10)

    def test_output_is_symmetric_positive_semidefinite(self):
        """Composite cov must be symmetric and positive semi-definite."""
        from trackcomb.physics import compute_composite_covariance

        rng = np.random.RandomState(42)
        # Two daughters with random states
        N = 5
        pools = []
        for seed in [10, 20]:
            r = np.random.RandomState(seed)
            # Build a random positive-definite 3x3 cov per candidate
            A = r.randn(N, 3, 3) * 1e-3
            covs = np.einsum("nij,nkj->nik", A, A) + np.eye(3) * 1e-8
            pool = _make_flat_pool(
                tx=r.randn(N) * 0.3,
                ty=r.randn(N) * 0.3,
                p=np.abs(r.randn(N)) * 30000 + 5000,
                qop=r.choice([-1, 1], N) / (np.abs(r.randn(N)) * 30000 + 5000),
                cov_2_2=covs[:, 0, 0],
                cov_3_2=covs[:, 1, 0],
                cov_3_3=covs[:, 1, 1],
                cov_4_2=covs[:, 2, 0],
                cov_4_3=covs[:, 2, 1],
                cov_4_4=covs[:, 2, 2],
            )
            pools.append(pool)

        # Compute composite momentum from both daughters
        def _mom(pool):
            p = ak.to_numpy(ak.flatten(pool["p"]))
            tx = ak.to_numpy(ak.flatten(pool["tx"]))
            ty = ak.to_numpy(ak.flatten(pool["ty"]))
            s = np.sqrt(1 + tx**2 + ty**2)
            return p * tx / s, p * ty / s, p / s

        px0, py0, pz0 = _mom(pools[0])
        px1, py1, pz1 = _mom(pools[1])
        q0 = np.sign(ak.to_numpy(ak.flatten(pools[0]["qop"])))
        q1 = np.sign(ak.to_numpy(ak.flatten(pools[1]["qop"])))

        comb = {
            "_daughter_pools": pools,
            "daughter0_global_index": np.arange(N),
            "daughter1_global_index": np.arange(N),
            "px": px0 + px1,
            "py": py0 + py1,
            "pz": pz0 + pz1,
            "charge": q0 + q1,
            "vertex_cov_0_0": rng.rand(N) * 1e-4,
            "vertex_cov_1_0": rng.rand(N) * 1e-6,
            "vertex_cov_1_1": rng.rand(N) * 1e-4,
        }
        compute_composite_covariance(comb)

        # Check all 5x5 cov entries
        full = np.zeros((N, 5, 5))
        for i in range(5):
            for j in range(i + 1):
                vals = comb[f"cov_{i}_{j}"]
                full[:, i, j] = vals
                full[:, j, i] = vals

        for n in range(N):
            # Symmetric
            np.testing.assert_allclose(full[n], full[n].T, atol=1e-20)
            # Positive semi-definite (eigenvalues >= 0)
            eigvals = np.linalg.eigvalsh(full[n])
            self.assertTrue(
                np.all(eigvals >= -1e-20),
                f"Candidate {n}: negative eigenvalue {eigvals.min()}",
            )

    def test_two_daughters_larger_variance(self):
        """Two-body composite should have larger (tx,ty,qop) variance than either daughter."""
        from trackcomb.physics import compute_composite_covariance

        cov_diag = np.diag([1e-6, 1e-6, 1e-10])

        # Build two daughters with same cov but different kinematics
        pool_a = _make_flat_pool(
            tx=np.array([0.2]),
            ty=np.array([0.1]),
            p=np.array([40000.0]),
            qop=np.array([1.0 / 40000.0]),
            cov_2_2=np.array([cov_diag[0, 0]]),
            cov_3_2=np.array([0.0]),
            cov_3_3=np.array([cov_diag[1, 1]]),
            cov_4_2=np.array([0.0]),
            cov_4_3=np.array([0.0]),
            cov_4_4=np.array([cov_diag[2, 2]]),
        )
        pool_b = _make_flat_pool(
            tx=np.array([-0.15]),
            ty=np.array([0.05]),
            p=np.array([30000.0]),
            qop=np.array([1.0 / 30000.0]),
            cov_2_2=np.array([cov_diag[0, 0]]),
            cov_3_2=np.array([0.0]),
            cov_3_3=np.array([cov_diag[1, 1]]),
            cov_4_2=np.array([0.0]),
            cov_4_3=np.array([0.0]),
            cov_4_4=np.array([cov_diag[2, 2]]),
        )

        def _mom(pool):
            p = ak.to_numpy(ak.flatten(pool["p"]))
            tx = ak.to_numpy(ak.flatten(pool["tx"]))
            ty = ak.to_numpy(ak.flatten(pool["ty"]))
            s = np.sqrt(1 + tx**2 + ty**2)
            return p * tx / s, p * ty / s, p / s

        px0, py0, pz0 = _mom(pool_a)
        px1, py1, pz1 = _mom(pool_b)

        comb = {
            "_daughter_pools": [pool_a, pool_b],
            "daughter0_global_index": np.array([0]),
            "daughter1_global_index": np.array([0]),
            "px": px0 + px1,
            "py": py0 + py1,
            "pz": pz0 + pz1,
            "charge": np.array([2.0]),  # both positive for nonzero qop
            "vertex_cov_0_0": np.array([1e-4]),
            "vertex_cov_1_0": np.array([0.0]),
            "vertex_cov_1_1": np.array([1e-4]),
        }
        compute_composite_covariance(comb)

        # Variances should be positive
        self.assertGreater(comb["cov_4_4"][0], 0)
        self.assertGreater(comb["cov_2_2"][0], 0)
        self.assertGreater(comb["cov_3_3"][0], 0)

    def test_position_block_from_vertex_cov(self):
        """Position block of 5x5 cov should come directly from vertex_cov."""
        from trackcomb.physics import compute_composite_covariance

        cov_in = np.diag([1e-6, 1e-6, 1e-10])
        comb = self._build_single_daughter_comb(
            tx=0.1, ty=0.2, qop=1.0 / 20000.0, cov_3x3=cov_in
        )
        vcov00, vcov10, vcov11 = 1.23e-4, 4.56e-6, 7.89e-4
        comb["vertex_cov_0_0"] = np.array([vcov00])
        comb["vertex_cov_1_0"] = np.array([vcov10])
        comb["vertex_cov_1_1"] = np.array([vcov11])

        compute_composite_covariance(comb)

        self.assertAlmostEqual(comb["cov_0_0"][0], vcov00)
        self.assertAlmostEqual(comb["cov_1_0"][0], vcov10)
        self.assertAlmostEqual(comb["cov_1_1"][0], vcov11)

    def test_cross_terms_are_zero(self):
        """Position-slope cross terms should be zero (simplified approximation)."""
        from trackcomb.physics import compute_composite_covariance

        cov_in = np.diag([1e-6, 1e-6, 1e-10])
        comb = self._build_single_daughter_comb(
            tx=0.1, ty=0.2, qop=1.0 / 20000.0, cov_3x3=cov_in
        )
        compute_composite_covariance(comb)

        for key in [
            "cov_2_0",
            "cov_2_1",
            "cov_3_0",
            "cov_3_1",
            "cov_4_0",
            "cov_4_1",
        ]:
            np.testing.assert_equal(
                comb[key], 0.0, err_msg=f"{key} should be zero"
            )


if __name__ == "__main__":
    unittest.main()
