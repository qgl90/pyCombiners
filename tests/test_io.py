"""Tests for data model and IO."""

from __future__ import annotations

import functools
import tempfile
import unittest

from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from trackcomb.io import event_stream, load_events, read
from trackcomb.components import (
    load_tracks,
    load_pvs,
    load_event_info,
    load_reconstructible_tracks,
)
from trackcomb.runner import run_reconstruction, ParquetSink

# Path to test ROOT file (new EventTuple format)
_ROOT_FILE = (
    Path(__file__).resolve().parent / "input" / "eventtuple_minbias_2evts.root"
)
_MAX_EVENTS = 2


def _root_available() -> bool:
    return _ROOT_FILE.exists()


@functools.lru_cache(maxsize=None)
def _cached_events():
    return load_events(_ROOT_FILE, max_events=_MAX_EVENTS)


@unittest.skipUnless(_root_available(), "ROOT test file not available")
class TestEventStream(unittest.TestCase):
    """Validate chunk cursors and chunked reading vs one-shot loading."""

    def test_chunk_cursors_are_plain_data(self):
        chunks = list(event_stream(_ROOT_FILE, chunk_size=1))
        self.assertEqual(len(chunks), 2)
        for i, c in enumerate(chunks):
            self.assertEqual(c["entry_start"], i)
            self.assertEqual(c["entry_stop"], i + 1)
            self.assertEqual(c["path"], str(_ROOT_FILE))

    def test_max_events_truncates(self):
        chunks = list(event_stream(_ROOT_FILE, chunk_size=100, max_events=1))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["entry_stop"], 1)

    def test_wildcard_expansion(self):
        pattern = str(_ROOT_FILE.parent / "eventtuple_*_2evts.root")
        chunks = list(event_stream(pattern, chunk_size=100))
        self.assertEqual(len(chunks), 1)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            list(event_stream("/nonexistent/nope_*.root"))

    def test_chunked_equals_full(self):
        full_tracks, full_pvs, full_info = _cached_events()

        run_nums, evt_nums, xs = [], [], []
        for chunk in event_stream(_ROOT_FILE, chunk_size=1):
            tracks = load_tracks(chunk)
            info = load_event_info(chunk)
            run_nums.append(info["run_number"])
            evt_nums.append(info["event_number"])
            xs.append(tracks["x"])

        np.testing.assert_array_equal(
            np.concatenate(run_nums), full_info["run_number"]
        )
        np.testing.assert_array_equal(
            np.concatenate(evt_nums), full_info["event_number"]
        )
        all_x = ak.concatenate(xs)
        for i in range(_MAX_EVENTS):
            self.assertEqual(
                int(ak.count(all_x[i])),
                int(ak.count(full_tracks["x"][i])),
            )

    def test_read_by_full_path(self):
        chunk = next(iter(event_stream(_ROOT_FILE, chunk_size=100)))
        x = read(chunk, "BestLongState_FirstMeasurement/x")
        self.assertEqual(len(x), _MAX_EVENTS)


@unittest.skipUnless(_root_available(), "ROOT test file not available")
class TestComponentLoaders(unittest.TestCase):
    def _chunk(self):
        return next(iter(event_stream(_ROOT_FILE, chunk_size=100)))

    def test_tracks_fields(self):
        tracks = load_tracks(self._chunk())
        for field in (
            "x",
            "y",
            "z",
            "tx",
            "ty",
            "qop",
            "p",
            "pt",
            "eta",
            "charge",
            "chi2ndof",
            "track_id",
            "mc_pid",
            "mc_ancestor_pids",
            "rich_dll_muon",
            "tvhits_n",
            "tvhits_t",
            "tvhits_z",
        ):
            self.assertIn(field, tracks)
        self.assertEqual(tracks["_type"], "tracks")
        # hits are doubly jagged, sized by tvhits_n
        self.assertTrue(
            ak.all(ak.num(tracks["tvhits_t"], axis=2) == tracks["tvhits_n"])
        )
        # state upcast to float64 for fit numerics
        self.assertIn("float64", str(ak.type(tracks["x"])))

    def test_tracks_granularity_switches(self):
        tracks = load_tracks(self._chunk(), hits=(), rich=False, mc=False)
        self.assertNotIn("tvhits_t", tracks)
        self.assertNotIn("rich_dll_muon", tracks)
        self.assertNotIn("mc_pid", tracks)
        self.assertIn("x", tracks)

    def test_true_hits(self):
        tracks = load_tracks(self._chunk(), true_hits=True)
        self.assertIn("true_tvhits_t", tracks)
        self.assertTrue(
            ak.all(
                ak.num(tracks["true_tvhits_t"], axis=2) == tracks["tvhits_n"]
            )
        )

    def test_pvs_fields(self):
        pvs = load_pvs(self._chunk())
        for field in (
            "x",
            "y",
            "z",
            "time",
            "sigma_time",
            "cov_3_3",
            "mc_key",
            "pv_index",
        ):
            self.assertIn(field, pvs)
        self.assertEqual(pvs["_type"], "pvs")

    def test_reconstructible_tracks(self):
        recon = load_reconstructible_tracks(self._chunk())
        for field in (
            "pid",
            "pt",
            "eta",
            "from_signal",
            "charge",
            "has_velo",
            "reconstructible_id",
        ):
            self.assertIn(field, recon)
        # charge unpacked from flags is +-1
        flat = ak.flatten(recon["charge"])
        self.assertTrue(ak.all((flat == 1) | (flat == -1)))


def _count_reco(chunk):
    """Module-level (picklable) reco for the workers test."""
    info = load_event_info(chunk)
    tracks = load_tracks(chunk, hits=(), rich=False, mc=False)
    return pd.DataFrame(
        {
            "event_number": info["event_number"],
            "n_tracks": ak.num(tracks["x"]).to_numpy(),
        }
    )


@unittest.skipUnless(_root_available(), "ROOT test file not available")
class TestRunReconstruction(unittest.TestCase):
    """Validate run_reconstruction interface."""

    def test_collects_return_values(self):
        def my_reco(chunk):
            tracks = load_tracks(chunk, hits=(), rich=False, mc=False)
            return len(tracks["x"])

        results = run_reconstruction(
            my_reco,
            input_data=_ROOT_FILE,
            max_events=_MAX_EVENTS,
            chunk_size=1,
        )
        self.assertIsInstance(results, list)
        self.assertEqual(sum(results), _MAX_EVENTS)

    def test_returns_none_when_user_returns_nothing(self):
        called = []

        def my_reco(chunk):
            called.append(1)

        result = run_reconstruction(
            my_reco, input_data=_ROOT_FILE, max_events=_MAX_EVENTS
        )
        self.assertIsNone(result)
        self.assertGreater(len(called), 0)

    def test_streaming_sink_writes_single_parquet(self):
        def my_reco(chunk):
            info = load_event_info(chunk)
            return pd.DataFrame({"event_number": info["event_number"]})

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.parquet"
            ret = run_reconstruction(
                my_reco,
                input_data=_ROOT_FILE,
                out=out,
                max_events=_MAX_EVENTS,
                chunk_size=1,
            )
            self.assertEqual(ret, out)
            df = pd.read_parquet(out)
            self.assertEqual(len(df), _MAX_EVENTS)

    def test_workers_match_serial(self):
        """Process-pool results match serial (up to row order)."""
        with tempfile.TemporaryDirectory() as tmp:
            serial, parallel = Path(tmp) / "s.parquet", Path(tmp) / "p.parquet"
            run_reconstruction(
                _count_reco,
                input_data=_ROOT_FILE,
                out=serial,
                chunk_size=1,
            )
            run_reconstruction(
                _count_reco,
                input_data=_ROOT_FILE,
                out=parallel,
                chunk_size=1,
                workers=2,
            )
            df_s = pd.read_parquet(serial).sort_values("event_number")
            df_p = pd.read_parquet(parallel).sort_values("event_number")
            np.testing.assert_array_equal(
                df_s["event_number"].values, df_p["event_number"].values
            )
            np.testing.assert_array_equal(
                df_s["n_tracks"].values, df_p["n_tracks"].values
            )

    def test_sink_writes_empty_parquet_when_no_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "empty.parquet"
            with ParquetSink(out):
                pass
            self.assertTrue(out.exists())
            self.assertEqual(len(pd.read_parquet(out)), 0)


if __name__ == "__main__":
    unittest.main()
