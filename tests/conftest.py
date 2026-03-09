"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from trackcomb.io import load_events
from trackcomb.combiner import combine
from trackcomb.models import cut_max, cut_min, cut_range, get_daughter
from trackcomb.pid import set_tracks_pid

ROOT_FILE = Path(__file__).resolve().parent / "input" / "minbias_2evts.root"
TREE = "BestLongTracks/TrackTuple"
MAX_EVENTS = 2


@pytest.fixture(scope="session")
def _root_data():
    if not ROOT_FILE.exists():
        pytest.skip("ROOT test file not available")
    return load_events(str(ROOT_FILE), TREE, max_events=MAX_EVENTS)


@pytest.fixture(scope="session")
def root_tracks(_root_data):
    tracks, _, _ = _root_data
    set_tracks_pid(tracks, "pi+")
    return tracks


@pytest.fixture(scope="session")
def root_pvs(_root_data):
    _, pvs, _ = _root_data
    return pvs


@pytest.fixture(scope="session")
def ks_candidates(root_tracks, root_pvs):
    """Ks-like candidates with standard cuts."""
    return combine(
        [root_tracks, root_tracks],
        root_pvs,
        track_cuts=[cut_min("pt", 500)],
        combination_cuts=[
            cut_max("max_doca", 0.5),
            lambda c: (
                get_daughter(c, 0, "charge") * get_daughter(c, 1, "charge") < 0
            ),
            cut_range("mass", 400, 600),
        ],
        composite_cuts=[
            cut_max("vertex_chi2", 25.0),
        ],
    )


@pytest.fixture(scope="session")
def staged_candidates(ks_candidates, root_tracks, root_pvs):
    """Staged decay: candidates used directly as composite tracks."""
    return combine(
        [ks_candidates, root_tracks],
        root_pvs,
        combination_cuts=[
            cut_max("max_doca", 0.1),
        ],
        composite_cuts=[
            cut_max("vertex_chi2", 10.0),
        ],
    )
