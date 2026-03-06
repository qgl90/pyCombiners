"""Tests for combination generation and physics utilities."""

from __future__ import annotations

import unittest

import awkward as ak
import numpy as np

from trackcomb.physics import (
    make_combinations,
    flatten_daughters,
    unflatten_array,
)


class TestFlattenUnflatten(unittest.TestCase):
    """Test flatten/unflatten roundtrip."""

    def test_roundtrip(self):
        """Unflatten(flatten(x)) must reproduce the original jagged structure."""
        original = ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0]])
        counts = ak.to_numpy(ak.num(original, axis=1))
        flat = ak.to_numpy(ak.flatten(original, axis=1))
        recovered = unflatten_array(flat, counts)
        self.assertEqual(ak.to_list(recovered), ak.to_list(original))


class TestMakeCombinationsMultiPool(unittest.TestCase):
    """Test make_combinations with distinct pools (cartesian product)."""

    def _make_pool(self, n_per_event, seed=0):
        rng = np.random.RandomState(seed)
        pool = {}
        for field in ("x", "y", "z", "tx", "ty"):
            pool[field] = ak.Array([rng.randn(n) * 0.1 for n in n_per_event])
        pool["p"] = ak.Array([np.abs(rng.randn(n)) * 10 + 5 for n in n_per_event])
        pool["charge"] = ak.Array(
            [rng.choice([1.0, -1.0], size=n) for n in n_per_event]
        )
        pool["track_id"] = ak.Array([list(range(n)) for n in n_per_event])
        return pool

    def test_two_distinct_pools_cartesian(self):
        """Two distinct pools → n1 * n2 combos per event."""
        pool_a = self._make_pool([3, 2], seed=0)
        pool_b = self._make_pool([2, 4], seed=42)
        daughters = make_combinations([pool_a, pool_b])
        # Event 0: 3*2=6, Event 1: 2*4=8
        self.assertEqual(len(daughters[0]["x"][0]), 6)
        self.assertEqual(len(daughters[0]["x"][1]), 8)

    def test_same_pool_identity(self):
        """Same object twice → C(n, 2) (not n^2)."""
        pool = self._make_pool([4, 3])
        daughters = make_combinations([pool, pool])
        # C(4,2)=6, C(3,2)=3
        self.assertEqual(len(daughters[0]["x"][0]), 6)
        self.assertEqual(len(daughters[0]["x"][1]), 3)

    def test_daughter_fields_from_correct_pool(self):
        """Each daughter should contain fields from its source pool."""
        pool_a = self._make_pool([3], seed=0)
        pool_b = self._make_pool([2], seed=42)
        daughters = make_combinations([pool_a, pool_b])
        # Check that daughter track_id values stay in expected range
        daughter0_idx = ak.to_numpy(ak.flatten(daughters[0]["track_id"]))
        daughter1_idx = ak.to_numpy(ak.flatten(daughters[1]["track_id"]))
        self.assertTrue(np.all(daughter0_idx < 3))  # pool_a has 3 tracks
        self.assertTrue(np.all(daughter1_idx < 2))  # pool_b has 2 tracks


if __name__ == "__main__":
    unittest.main()
