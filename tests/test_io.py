"""Tests for data model and IO."""

from __future__ import annotations

import functools
import unittest

from pathlib import Path

import awkward as ak
import numpy as np

from trackcomb.io import (
    load_events_root,
    iter_events_root,
)
from trackcomb.models import (
    Container,
    apply_cuts,
    apply_mask,
    cut_min,
    cut_max,
    cut_range,
)

# Path to test ROOT file
_ROOT_FILE = Path(__file__).resolve().parent / "input" / "minbias_2evts.root"
_TREE = "BestLongTracks/TrackTuple"
_MAX_EVENTS = 2


def _root_available() -> bool:
    return _ROOT_FILE.exists()


@functools.lru_cache(maxsize=None)
def _cached_events():
    return load_events_root(_ROOT_FILE, _TREE, max_events=_MAX_EVENTS)


@unittest.skipUnless(_root_available(), "ROOT test file not available")
class TestIterEventsRoot(unittest.TestCase):
    """Validate chunked reading matches single-pass loading."""

    def test_chunked_equals_full(self):
        full_tracks, full_pvs, full_info = _cached_events()

        # Collect chunks
        chunk_tracks_list = []
        chunk_pvs_list = []
        chunk_run_nums = []
        chunk_evt_nums = []
        for t, p, info in iter_events_root(
            _ROOT_FILE,
            _TREE,
            max_events=_MAX_EVENTS,
            chunk_size=3,
        ):
            chunk_tracks_list.append(t)
            chunk_pvs_list.append(p)
            chunk_run_nums.append(info["run_number"])
            chunk_evt_nums.append(info["event_number"])

        # Concatenate chunks
        all_run = np.concatenate(chunk_run_nums)
        all_evt = np.concatenate(chunk_evt_nums)
        self.assertEqual(len(all_run), len(full_info["run_number"]))

        # Check event info matches
        np.testing.assert_array_equal(all_run, full_info["run_number"])
        np.testing.assert_array_equal(all_evt, full_info["event_number"])

        # Check track counts match
        all_x = ak.concatenate([c["x"] for c in chunk_tracks_list])
        for i in range(_MAX_EVENTS):
            self.assertEqual(
                int(ak.count(all_x[i])),
                int(ak.count(full_tracks["x"][i])),
            )


class TestModelUtilities(unittest.TestCase):
    """Test Container utilities (apply_mask, apply_cuts, etc.)."""

    def _make_test_tracks(self):
        """Create a simple test container: 2 events, variable tracks."""
        return {
            "pt": ak.Array([[0.1, 0.5, 1.0], [0.05, 2.0]]),
            "eta": ak.Array([[2.5, 3.0, 4.5], [1.5, 3.5]]),
            "x": ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0]]),
        }

    def test_apply_mask(self):
        tracks = self._make_test_tracks()
        mask = tracks["pt"] > 0.3
        filtered = apply_mask(tracks, mask)
        # Event 0: 0.5, 1.0 survive; Event 1: 2.0 survives
        self.assertEqual(ak.to_list(filtered["pt"]), [[0.5, 1.0], [2.0]])

    def test_apply_cuts_single(self):
        tracks = self._make_test_tracks()
        filtered = apply_cuts(tracks, [lambda t: t["pt"] > 0.3])
        self.assertEqual(ak.to_list(filtered["pt"]), [[0.5, 1.0], [2.0]])

    def test_apply_cuts_composable(self):
        tracks = self._make_test_tracks()
        filtered = apply_cuts(
            tracks,
            [
                cut_min("pt", 0.1),
                cut_max("eta", 4.0),
            ],
        )
        # Event 0: pt>=0.1 AND eta<=4.0 → 0.1(eta=2.5✓), 0.5(eta=3.0✓), 1.0(eta=4.5✗)
        # Event 1: pt>=0.1 AND eta<=4.0 → 0.05(pt<0.1✗), 2.0(eta=3.5✓)
        self.assertEqual(ak.to_list(filtered["pt"]), [[0.1, 0.5], [2.0]])

    def test_cut_range(self):
        tracks = self._make_test_tracks()
        filtered = apply_cuts(tracks, [cut_range("eta", 2.0, 4.0)])
        self.assertEqual(ak.to_list(filtered["eta"]), [[2.5, 3.0], [3.5]])

    def test_apply_cuts_empty(self):
        """No cuts should return unchanged container."""
        tracks = self._make_test_tracks()
        result = apply_cuts(tracks, [])
        self.assertEqual(ak.to_list(result["pt"]), ak.to_list(tracks["pt"]))


if __name__ == "__main__":
    unittest.main()
