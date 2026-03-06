"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from trackcomb.io import load_tracks_root, load_pvs_root
from trackcomb.combiner import combine
from trackcomb.models import cut_max, cut_range
from trackcomb.pid import set_tracks_pid

ROOT_FILE = Path(__file__).resolve().parent / "input" / "minbias_2evts.root"
TREE = "BestLongTracks/TrackTuple"
MAX_EVENTS = 2


@pytest.fixture(scope="session")
def root_tracks():
    if not ROOT_FILE.exists():
        pytest.skip("ROOT test file not available")
    tracks = load_tracks_root(str(ROOT_FILE), TREE, max_events=MAX_EVENTS)
    return set_tracks_pid(tracks, "pi+")


@pytest.fixture(scope="session")
def root_pvs():
    if not ROOT_FILE.exists():
        pytest.skip("ROOT test file not available")
    return load_pvs_root(str(ROOT_FILE), TREE, max_events=MAX_EVENTS)


@pytest.fixture(scope="session")
def ks_candidates(root_tracks, root_pvs):
    """Ks-like candidates with standard cuts (no timing)."""
    return combine(
        [root_tracks, root_tracks],
        root_pvs,
        combination_cuts=[
            cut_max("spatial_chi2", 25.0),
            cut_max("max_doca", 0.5),
            lambda c: c["daughter0_charge"] * c["daughter1_charge"] < 0,
        ],
        vertex_cuts=[
            cut_range("mass", 400, 600),
        ],
        use_timing=False,
    )


@pytest.fixture(scope="session")
def ks_candidates_timing(root_tracks, root_pvs):
    """Ks-like candidates with timing enabled."""
    return combine(
        [root_tracks, root_tracks],
        root_pvs,
        combination_cuts=[
            cut_max("spatial_chi2", 25.0),
            cut_max("max_doca", 0.5),
            lambda c: c["daughter0_charge"] * c["daughter1_charge"] < 0,
        ],
        vertex_cuts=[
            cut_range("mass", 400, 600),
        ],
        use_timing=True,
    )


@pytest.fixture(scope="session")
def staged_candidates(ks_candidates, root_tracks, root_pvs):
    """Staged decay: candidates used directly as composite tracks."""
    return combine(
        [ks_candidates, root_tracks],
        root_pvs,
        combination_cuts=[
            cut_max("max_doca", 0.1),
            cut_max("spatial_chi2", 10.0),
        ],
        use_timing=False,
    )
