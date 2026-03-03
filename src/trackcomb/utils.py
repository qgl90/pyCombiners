"""Small general-purpose utilities for the particle-combination framework."""

from __future__ import annotations

from typing import Sequence

from .models import CombinationResult


def filter_candidates(
    candidates: Sequence[CombinationResult],
    **cuts,
) -> list[CombinationResult]:
    """Filter candidates by simple attribute cuts.

    Each keyword maps a ``CombinationResult`` attribute name to a
    ``(min_val, max_val)`` tuple.  Use ``None`` for an open bound.

    Example
    -------
    >>> good = filter_candidates(results, vertex_chi2=(None, 10.0), pair_pt=(0.5, None))
    """
    out: list[CombinationResult] = []
    for c in candidates:
        keep = True
        for attr, bounds in cuts.items():
            val = getattr(c, attr)
            lo, hi = bounds
            if lo is not None and val < lo:
                keep = False
                break
            if hi is not None and val > hi:
                keep = False
                break
        if keep:
            out.append(c)
    return out


def best_candidate(
    candidates: Sequence[CombinationResult],
    key=None,
) -> CombinationResult | None:
    """Return the best candidate by ``key`` (default: lowest vertex_chi2)."""
    if not candidates:
        return None
    if key is None:
        key = lambda c: c.vertex_chi2
    return min(candidates, key=key)
