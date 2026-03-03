"""Decay definition and unified combine() function for particle reconstruction.

Provides:
- ``Decay``: immutable bundle of n_body, mass hypotheses, preselection, cuts.
- ``make_decay()``: factory that builds a ``Decay`` from daughter names.
- ``combine()``: unified combination function for flat and pool-based inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable, Iterator, Sequence

from .combiner import ParticleCombiner
from .models import (
    CombinationCuts,
    CombinationResult,
    ParticleHypothesis,
    PrimaryVertex,
    TrackPreselection,
    TrackState,
)
from .pid import particle_hypothesis_from_name


@dataclass(frozen=True)
class Decay:
    """Decay definition bundling n_body, hypotheses, preselection, and cuts."""

    n_body: int
    mass_hypotheses: tuple[tuple[ParticleHypothesis, ...], ...]
    preselection: TrackPreselection | None = None
    cuts: CombinationCuts | None = None


def make_decay(
    daughters: Sequence[ParticleHypothesis | str],
    preselection: TrackPreselection | None = None,
    cuts: CombinationCuts | None = None,
) -> Decay:
    """Create a Decay from daughter particle names or hypotheses.

    Parameters
    ----------
    daughters : sequence of str or ParticleHypothesis
        One entry per daughter.  Strings are resolved via
        ``particle_hypothesis_from_name`` (e.g. ``"pi"``, ``"K"``, ``"mu"``).
        For composite daughters pass ``ParticleHypothesis`` directly.
    preselection : TrackPreselection, optional
        Track-level preselection applied before combinatorics.
    cuts : CombinationCuts, optional
        Candidate-level cuts applied after vertex fitting.

    Examples
    --------
    >>> ks = make_decay(["pi", "pi"], cuts=CombinationCuts(min_mass=0.4, max_mass=0.6))
    >>> jpsi = make_decay(["mu", "mu"])
    """
    hyps: list[ParticleHypothesis] = []
    for d in daughters:
        if isinstance(d, str):
            hyps.append(particle_hypothesis_from_name(d))
        elif isinstance(d, ParticleHypothesis):
            hyps.append(d)
        else:
            raise TypeError(f"Expected str or ParticleHypothesis, got {type(d)}")

    return Decay(
        n_body=len(hyps),
        mass_hypotheses=(tuple(hyps),),
        preselection=preselection,
        cuts=cuts,
    )


# ---------------------------------------------------------------------------
# Unified combine
# ---------------------------------------------------------------------------

def combine(
    decay: Decay,
    tracks,
    primary_vertices: Sequence[PrimaryVertex],
    event_id: str | None = None,
    track_filter: Callable[[TrackState], bool] | None = None,
    candidate_filter: Callable[[CombinationResult], bool] | None = None,
) -> list[CombinationResult]:
    """Unified combination function.

    Parameters
    ----------
    decay : Decay
        Decay definition (daughters, cuts, preselection).
    tracks : flat or pool input
        *Flat*: ``Sequence[TrackState | CombinationResult]``
            Uses C(n, n_body) combinations from one pool.
        *Pools*: ``Sequence[Sequence[TrackState | CombinationResult]]``
            Product across distinct pools; combinations within same-identity
            pools (detected via ``is``).  Candidates with shared source tracks
            are automatically skipped.
    primary_vertices : sequence of PrimaryVertex
    event_id : str, optional
    track_filter : callable, optional
        Custom track-level filter applied before combinatorics.
        ``lambda t: ...`` returning True to keep the track.
        Useful for truth-based selection (cheated studies).
    candidate_filter : callable, optional
        Custom candidate-level filter applied after building each result.
        ``lambda c: ...`` returning True to keep the candidate.

    Returns
    -------
    list[CombinationResult]
    """
    if not tracks:
        return []

    combiner = ParticleCombiner()

    # Detect flat vs pool input
    first = tracks[0]
    if isinstance(first, (TrackState, CombinationResult)):
        return _combine_flat(combiner, decay, tracks, primary_vertices, event_id,
                             track_filter, candidate_filter)
    return _combine_pools(combiner, decay, tracks, primary_vertices, event_id,
                          track_filter, candidate_filter)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _combine_flat(combiner, decay, tracks, primary_vertices, event_id,
                   track_filter, candidate_filter):
    """Flat mode: C(n, n_body) from one pool."""
    flat_tracks = [_to_track_state(t) for t in tracks]
    if track_filter is not None:
        flat_tracks = [t for t in flat_tracks if track_filter(t)]
    results = combiner.combine(
        tracks=flat_tracks,
        primary_vertices=primary_vertices,
        n_body=decay.n_body,
        mass_hypotheses=list(decay.mass_hypotheses),
        preselection=decay.preselection,
        cuts=decay.cuts,
        event_id=event_id,
    )
    if candidate_filter is not None:
        results = [r for r in results if candidate_filter(r)]
    return results


def _combine_pools(combiner, decay, raw_pools, primary_vertices, event_id,
                    track_filter, candidate_filter):
    """Pool mode: product across pools with same-pool combination optimization."""
    # Convert items to TrackState, preserving pool identity for same-pool detection.
    pool_cache: dict[int, list[TrackState]] = {}
    pools: list[list[TrackState]] = []
    for raw_pool in raw_pools:
        pid = id(raw_pool)
        if pid in pool_cache:
            pools.append(pool_cache[pid])
        else:
            converted = [_to_track_state(t) for t in raw_pool]
            if track_filter is not None:
                converted = [t for t in converted if track_filter(t)]
            pool_cache[pid] = converted
            pools.append(converted)

    if len(pools) != decay.n_body:
        raise ValueError(
            f"Number of pools ({len(pools)}) does not match "
            f"decay n_body ({decay.n_body})."
        )

    pvs = list(primary_vertices)
    if not pvs:
        raise ValueError("At least one primary vertex is required.")

    cuts = decay.cuts or CombinationCuts()

    # Preselect each pool
    if decay.preselection is not None:
        pools = [combiner.preselect_tracks(p, pvs, decay.preselection,
                                           use_timing=cuts.use_timing) for p in pools]

    if any(not p for p in pools):
        return []

    valid_hypotheses = combiner._validate_hypotheses(
        list(decay.mass_hypotheses), decay.n_body,
    )
    combiner._validate_charge_patterns(cuts.allowed_charge_patterns, decay.n_body)

    candidate_iter = _enumerate_pool_candidates(pools)
    results = combiner._combine_tuples(
        candidate_iter, pvs, valid_hypotheses, cuts, event_id,
    )
    if candidate_filter is not None:
        results = [r for r in results if candidate_filter(r)]
    return results


def _to_track_state(item):
    """Convert a TrackState or CombinationResult to TrackState."""
    if isinstance(item, TrackState):
        return item
    if isinstance(item, CombinationResult):
        if item.composite_track is None:
            raise ValueError(
                "CombinationResult has no composite_track. "
                "Ensure results were produced by ParticleCombiner.combine()."
            )
        return item.composite_track
    raise TypeError(f"Expected TrackState or CombinationResult, got {type(item)}")


def _enumerate_pool_candidates(
    pools: list[list[TrackState]],
) -> Iterator[tuple[TrackState, ...]]:
    """Enumerate valid candidate tuples from pools.

    Pools sharing the same identity (``is``) use ``combinations`` to avoid
    self-pairing and duplicate ordering.  Candidates with shared source
    track IDs are skipped.
    """
    # Group pool positions by identity
    groups: dict[int, list[int]] = {}
    for i, pool in enumerate(pools):
        groups.setdefault(id(pool), []).append(i)

    ordered_groups = list(groups.values())
    group_combos: list[list[tuple[TrackState, ...]]] = []
    for positions in ordered_groups:
        pool = pools[positions[0]]
        k = len(positions)
        group_combos.append(list(combinations(pool, k)))

    # Product across groups, reconstruct candidate in original position order
    for parts in product(*group_combos):
        candidate: list[TrackState | None] = [None] * len(pools)
        for positions, tracks_group in zip(ordered_groups, parts):
            for pos, track in zip(positions, tracks_group):
                candidate[pos] = track

        tup = tuple(t for t in candidate if t is not None)
        if len(tup) != len(pools):
            continue

        if _has_shared_sources(tup):
            continue

        yield tup


def _has_shared_sources(tracks: tuple[TrackState, ...]) -> bool:
    """Check if any tracks share source track IDs."""
    seen: set[str] = set()
    for t in tracks:
        ids = set(t.source_track_ids) if t.source_track_ids else {t.track_id}
        if seen & ids:
            return True
        seen |= ids
    return False
