"""Ancestry-based truth matching for particle combinations.

Truth matching works by searching for a common (ancestor_key, ancestor_pid)
pair across all final-state tracks of a candidate, where |ancestor_pid|
matches the requested mother PDG ID.

This approach works at any decay depth because each track stores its full
MC ancestry chain (``mc_ancestor_pids``, ``mc_ancestor_keys`` in metadata).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .models import CombinationResult, TrackState


def truth_match(
    candidate: CombinationResult,
    tracks: Sequence[TrackState],
    mother_pdg: int,
    daughter_pdgs: Sequence[int] | None = None,
) -> bool:
    """Check if a candidate is truth-matched to a given decay.

    A candidate is truth-matched when all its source tracks share a common
    MC ancestor whose ``|pid| == mother_pdg``.  If ``daughter_pdgs`` is given,
    the set of MC PIDs of the candidate's source tracks must match exactly.

    Parameters
    ----------
    candidate : CombinationResult
        The reconstructed candidate.
    tracks : sequence of TrackState
        Full track list for the event (used to look up metadata by track_id).
    mother_pdg : int
        Absolute PDG ID of the mother particle (e.g. 310 for Ks, 443 for J/psi).
    daughter_pdgs : sequence of int, optional
        Expected absolute PDG IDs of the daughters, as a multiset.
        E.g. ``[211, 211]`` for Ks -> pi+ pi-.

    Returns
    -------
    bool
    """
    track_map = {t.track_id: t for t in tracks}
    source_tracks = []
    for tid in candidate.source_track_ids:
        if tid not in track_map:
            return False
        source_tracks.append(track_map[tid])

    if not source_tracks:
        return False

    common = _common_ancestor(source_tracks, mother_pdg)
    if common is None:
        return False

    if daughter_pdgs is not None:
        actual = sorted(abs(t.metadata.get("mc_pid", 0)) for t in source_tracks)
        expected = sorted(abs(p) for p in daughter_pdgs)
        if actual != expected:
            return False

    return True


def truth_match_candidates(
    candidates: Sequence[CombinationResult],
    tracks: Sequence[TrackState],
    mother_pdg: int,
    daughter_pdgs: Sequence[int] | None = None,
) -> list[tuple[CombinationResult, bool]]:
    """Tag each candidate with a truth-match boolean.

    Returns a list of ``(candidate, is_matched)`` pairs.
    """
    return [
        (c, truth_match(c, tracks, mother_pdg, daughter_pdgs))
        for c in candidates
    ]


def count_true_decays(
    tracks: Sequence[TrackState],
    mother_pdg: int,
    daughter_pdgs: Sequence[int] | None = None,
) -> int:
    """Count how many true decays of a given type exist in the track list.

    Groups tracks by common ancestor (key, pid) where ``|pid| == mother_pdg``,
    optionally requiring that the daughter PIDs match.
    """
    groups = _group_by_ancestor(tracks, mother_pdg)
    if daughter_pdgs is None:
        return len(groups)
    expected = sorted(abs(p) for p in daughter_pdgs)
    count = 0
    for group_tracks in groups.values():
        actual = sorted(abs(t.metadata.get("mc_pid", 0)) for t in group_tracks)
        if actual == expected:
            count += 1
    return count


def filter_tracks_by_ancestor(
    tracks: Sequence[TrackState],
    ancestor_pdg: int,
) -> list[TrackState]:
    """Return tracks that have ``ancestor_pdg`` in their ancestry chain.

    Useful for building a "cheated" track list (all daughters of a given
    mother type) to compute the efficiency denominator.
    """
    out: list[TrackState] = []
    for t in tracks:
        pids = t.metadata.get("mc_ancestor_pids", [])
        if any(abs(p) == abs(ancestor_pdg) for p in pids):
            out.append(t)
    return out


def get_true_decay_groups(
    tracks: Sequence[TrackState],
    mother_pdg: int,
    daughter_pdgs: Sequence[int] | None = None,
) -> dict[int, list[TrackState]]:
    """Return groups of tracks forming true decays.

    Keys are the MC ancestor keys; values are the track lists.
    Optionally filtered by ``daughter_pdgs``.
    """
    groups = _group_by_ancestor(tracks, mother_pdg)
    if daughter_pdgs is None:
        return groups
    expected = sorted(abs(p) for p in daughter_pdgs)
    return {
        key: trks
        for key, trks in groups.items()
        if sorted(abs(t.metadata.get("mc_pid", 0)) for t in trks) == expected
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _common_ancestor(
    tracks: list[TrackState],
    mother_pdg: int,
) -> tuple[int, int] | None:
    """Find a common (key, pid) ancestor across all tracks.

    Returns the first ``(key, pid)`` pair that appears in every track's
    ancestry with ``|pid| == mother_pdg``, or ``None``.
    """
    if not tracks:
        return None

    # Build candidate set from the first track
    first = tracks[0]
    pids = first.metadata.get("mc_ancestor_pids", [])
    keys = first.metadata.get("mc_ancestor_keys", [])
    candidates: set[tuple[int, int]] = set()
    for k, p in zip(keys, pids):
        if abs(p) == abs(mother_pdg):
            candidates.add((k, p))

    if not candidates:
        return None

    # Intersect with remaining tracks
    for t in tracks[1:]:
        pids = t.metadata.get("mc_ancestor_pids", [])
        keys = t.metadata.get("mc_ancestor_keys", [])
        t_set: set[tuple[int, int]] = set()
        for k, p in zip(keys, pids):
            if abs(p) == abs(mother_pdg):
                t_set.add((k, p))
        candidates &= t_set
        if not candidates:
            return None

    return next(iter(candidates))


def _group_by_ancestor(
    tracks: Sequence[TrackState],
    mother_pdg: int,
) -> dict[int, list[TrackState]]:
    """Group tracks by ancestor key where |ancestor_pid| == mother_pdg."""
    groups: dict[int, list[TrackState]] = defaultdict(list)
    for t in tracks:
        pids = t.metadata.get("mc_ancestor_pids", [])
        keys = t.metadata.get("mc_ancestor_keys", [])
        for k, p in zip(keys, pids):
            if abs(p) == abs(mother_pdg):
                groups[k].append(t)
                break  # one track contributes once per mother type
    return dict(groups)
