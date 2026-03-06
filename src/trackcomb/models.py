"""SoA data model: Container type alias and shared utilities."""

from __future__ import annotations

from typing import Any, Callable

# ---------------------------------------------------------------------------
# Core container type
# ---------------------------------------------------------------------------

Container = dict[str, Any]
CutFunction = Callable[[Container], Any]


# Lower-triangular covariance index pairs for 4x4 (used by IO)
COV4_LOWER_TRI = [(i, j) for i in range(4) for j in range(i + 1)]

# Lower-triangular for 5x5 (ROOT branch naming uses 5x5)
COV5_LOWER_TRI = [(i, j) for i in range(5) for j in range(i + 1)]


# ---------------------------------------------------------------------------
# Container utilities
# ---------------------------------------------------------------------------


def infer_n_body(candidates: Container) -> int:
    """Count daughter{k}_pool_index fields to get n_body."""
    n = 0
    while f"daughter{n}_pool_index" in candidates:
        n += 1
    return n


# ---------------------------------------------------------------------------


def apply_mask(container: Container, mask) -> Container:
    """Apply a boolean mask to every array in the container.

    Non-array metadata (e.g. _daughter_pools) is carried forward unchanged.
    """
    out = {}
    for key, val in container.items():
        try:
            out[key] = val[mask]
        except (TypeError, IndexError):
            out[key] = val  # non-array metadata: carry forward
    return out


def apply_cuts(container: Container, cuts: list[CutFunction]) -> Container:
    """AND-combine cut functions and return the filtered container."""
    if not cuts:
        return container
    import awkward as ak

    # Start with all-True mask matching the shape of any field
    ref = next(iter(container.values()))
    mask = ak.ones_like(ref, dtype=bool)
    for cut_fn in cuts:
        mask = mask & cut_fn(container)
    return apply_mask(container, mask)


# ---------------------------------------------------------------------------
# Cut convenience factories
# ---------------------------------------------------------------------------


def cut_min(field: str, threshold: float) -> CutFunction:
    """Cut: field >= threshold."""
    return lambda c: c[field] >= threshold


def cut_max(field: str, threshold: float) -> CutFunction:
    """Cut: field <= threshold."""
    return lambda c: c[field] <= threshold


def cut_range(field: str, lo: float, hi: float) -> CutFunction:
    """Cut: lo <= field <= hi."""
    return lambda c: (c[field] >= lo) & (c[field] <= hi)


# ---------------------------------------------------------------------------
# Jagged array utilities
# ---------------------------------------------------------------------------


def pick_along_inner(arr, idx):
    """Pick arr[e][t][idx[e][t]] for a 3-deep jagged arr and 2-deep idx.

    Uses flat offset arithmetic — O(N_tracks), no inner-axis iteration.
    """
    import awkward as ak
    import numpy as np

    # Number of inner elements per (event, track) pair
    counts = ak.flatten(ak.num(arr, axis=2))
    counts_np = np.asarray(counts)

    # Cumulative offsets: where each track's inner block starts in the flat array
    offsets = np.empty(len(counts_np), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts_np[:-1], out=offsets[1:])

    flat_idx = offsets + np.asarray(ak.flatten(idx, axis=None))
    flat_arr = np.asarray(ak.flatten(arr, axis=None))

    result = flat_arr[flat_idx]

    # Restore jagged structure: (total_tracks,) → (events, tracks)
    return ak.unflatten(result, ak.num(idx, axis=1))
