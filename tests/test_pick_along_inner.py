"""Tests for pick_inner — jagged inner-axis selection."""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from trackcomb import pick_inner


class TestPickAlongInner:
    """Verify O(N) flat-offset indexing against naive element-by-element."""

    def test_basic(self):
        # 2 events: event0 has 3 tracks × 4 pvs, event1 has 2 tracks × 3 pvs
        arr = ak.Array(
            [
                [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
                [[40, 41, 42], [50, 51, 52]],
            ]
        )
        idx = ak.Array([[2, 0, 3], [1, 2]])
        result = pick_inner(arr, idx)
        expected = ak.Array([[12, 20, 33], [41, 52]])
        assert result.tolist() == expected.tolist()

    def test_single_event(self):
        arr = ak.Array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        idx = ak.Array([[0, 2]])
        result = pick_inner(arr, idx)
        assert result.tolist() == [[1.0, 6.0]]

    def test_single_pv(self):
        # Each track sees exactly 1 PV — idx must be 0 everywhere
        arr = ak.Array([[[100], [200], [300]]])
        idx = ak.Array([[0, 0, 0]])
        result = pick_inner(arr, idx)
        assert result.tolist() == [[100, 200, 300]]

    def test_empty_event(self):
        # Event 0 has tracks, event 1 is empty
        arr = ak.Array([[[1, 2], [3, 4]], []])
        idx = ak.Array([[1, 0], []])
        result = pick_inner(arr, idx)
        assert result.tolist() == [[2, 3], []]

    def test_many_events(self):
        """Randomized stress test: compare against naive per-element access."""
        rng = np.random.default_rng(42)
        n_events = 50
        lists = []
        idx_lists = []
        expected = []
        for _ in range(n_events):
            n_trk = rng.integers(0, 20)
            n_pv = rng.integers(1, 40)
            evt = rng.standard_normal((n_trk, n_pv)).tolist()
            evt_idx = rng.integers(0, n_pv, size=n_trk).tolist()
            lists.append(evt)
            idx_lists.append(evt_idx)
            expected.append([evt[t][evt_idx[t]] for t in range(n_trk)])

        arr = ak.Array(lists)
        idx = ak.Array(idx_lists)
        result = pick_inner(arr, idx)

        for e in range(n_events):
            np.testing.assert_allclose(result[e].tolist(), expected[e])
