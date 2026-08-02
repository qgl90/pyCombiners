"""reconstructible_tracks container: MC particles reconstructible as tracks."""

from __future__ import annotations

import awkward as ak

from ..io import read
from ..models import Container


def load_reconstructible_tracks(chunk: Container) -> Container:
    r: Container = {}

    r["pid"] = read(chunk, "ReconstructibleTracks/pid")
    r["p"] = read(chunk, "ReconstructibleTracks/p")
    r["pt"] = read(chunk, "ReconstructibleTracks/pt")
    r["eta"] = read(chunk, "ReconstructibleTracks/eta")
    r["phi"] = read(chunk, "ReconstructibleTracks/phi")
    r["n_tv"] = read(chunk, "ReconstructibleTracks/n_tv")
    r["n_up"] = read(chunk, "ReconstructibleTracks/n_up")
    r["n_mp"] = read(chunk, "ReconstructibleTracks/n_mp")
    r["n_ft"] = read(chunk, "ReconstructibleTracks/n_ft")
    r["ancestor_pid"] = read(chunk, "ReconstructibleTracks/ancestor_pid")
    r["ancestor_key"] = read(chunk, "ReconstructibleTracks/ancestor_key")

    flags = read(chunk, "ReconstructibleTracks/flags")
    r["has_velo"] = (flags & (1 << 0)) != 0
    r["has_ut"] = (flags & (1 << 1)) != 0
    r["has_mp"] = (flags & (1 << 2)) != 0
    r["has_ft"] = (flags & (1 << 3)) != 0
    r["has_t"] = (flags & (1 << 4)) != 0
    r["from_signal"] = (flags & (1 << 5)) != 0
    r["charge"] = ak.where((flags & (1 << 6)) != 0, 1, -1)
    r["from_beauty"] = (flags & (1 << 7)) != 0
    r["from_charm"] = (flags & (1 << 8)) != 0
    r["from_strange"] = (flags & (1 << 9)) != 0

    r["_type"] = "reconstructible_tracks"
    r["reconstructible_id"] = ak.local_index(r["pid"], axis=1)
    return r
