"""Overlap checking utilities for particle candidates."""

from __future__ import annotations

from typing import Sequence

from .models import CombinationResult


def has_shared_tracks(a: CombinationResult, b: CombinationResult) -> bool:
    """Check if two candidates share any source tracks."""
    return bool(set(a.source_track_ids) & set(b.source_track_ids))


def remove_overlaps(
    candidates: Sequence[CombinationResult],
    key=None,
) -> list[CombinationResult]:
    """Remove overlapping candidates, keeping the best by ``key``.

    Iterates through candidates sorted by ``key`` (ascending).  For each
    candidate, if it shares source tracks with any already-accepted
    candidate, it is rejected.

    Parameters
    ----------
    candidates : sequence of CombinationResult
    key : callable, optional
        Function mapping ``CombinationResult`` to a sort value.
        Default: ``vertex_chi2`` (lower is better).

    Returns
    -------
    list[CombinationResult]
    """
    if key is None:
        key = lambda c: c.vertex_chi2

    sorted_cands = sorted(candidates, key=key)
    accepted: list[CombinationResult] = []
    used_sources: set[str] = set()

    for c in sorted_cands:
        c_sources = set(c.source_track_ids)
        if c_sources & used_sources:
            continue
        accepted.append(c)
        used_sources |= c_sources

    return accepted
