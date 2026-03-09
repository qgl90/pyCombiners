"""SoA data model: Container type alias and shared utilities."""

from __future__ import annotations

from typing import Any, Callable

import awkward as ak
import numpy as np

Container = dict[str, Any]
CutFunction = Callable[[Container], Any]


# Lower-triangular covariance index pairs for 4x4 (used by IO)
COV4_LOWER_TRI = [(i, j) for i in range(4) for j in range(i + 1)]

# Lower-triangular for 5x5 (ROOT branch naming uses 5x5)
COV5_LOWER_TRI = [(i, j) for i in range(5) for j in range(i + 1)]


def n_daughters(candidates: Container) -> int:
    """Count daughter{k}_global_index fields to get n_body."""
    n = 0
    while f"daughter{n}_global_index" in candidates:
        n += 1
    return n


def get_daughter(candidates: Container, k: int, field: str):
    """Look up a daughter field via global_index into the pool."""
    cache_key = f"cached_daughter{k}_{field}"
    if cache_key in candidates:
        return candidates[cache_key]

    pool = candidates["_daughter_pools"][k]
    global_idx = candidates[f"daughter{k}_global_index"]
    # Flatten the event axis of the pool → one entry per track, then index
    flat_pool = ak.flatten(pool[field])
    result = flat_pool[global_idx]
    # Flat container + 1D field → convert to numpy for downstream compatibility
    if isinstance(global_idx, np.ndarray) and flat_pool.ndim == 1:
        result = np.asarray(result)

    candidates[cache_key] = result
    return result


def gather_daughters_stack(candidates: Container, field: str):
    """Gather a field from all daughters and column-stack into (N, n_body)."""
    cache_key = f"_cached_daughters_stack_{field}"
    if cache_key in candidates:
        return candidates[cache_key]

    n_body = n_daughters(candidates)
    result = np.column_stack(
        [get_daughter(candidates, k, field) for k in range(n_body)]
    )
    candidates[cache_key] = result
    return result


def unflatten_container(container: Container, counts) -> Container:
    """Unflatten every public array to jagged structure; '_' keys are carried forward unchanged."""
    out = {}
    for key, val in container.items():
        if key.startswith("_"):
            out[key] = val
        else:
            out[key] = ak.unflatten(val, counts, axis=0)
    return out


def apply_mask(container: Container, mask) -> Container:
    """Apply a boolean mask to every array; non-array metadata is carried forward."""
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

    # Start with all-True mask matching the shape of any field
    ref = next(iter(container.values()))
    mask = ak.ones_like(ref, dtype=bool)
    for cut_fn in cuts:
        mask = mask & cut_fn(container)
    return apply_mask(container, mask)


def _resolve(field, c):
    """Resolve a field: call if callable, otherwise container lookup."""
    return field(c) if callable(field) else c[field]


def cut_min(field, threshold: float) -> CutFunction:
    """Cut: field >= threshold. field can be a string or callable."""
    return lambda c: _resolve(field, c) >= threshold


def cut_max(field, threshold: float) -> CutFunction:
    """Cut: field <= threshold. field can be a string or callable."""
    return lambda c: _resolve(field, c) <= threshold


def cut_range(field, lo: float, hi: float) -> CutFunction:
    """Cut: lo <= field <= hi. field can be a string or callable."""
    return lambda c: (_resolve(field, c) >= lo) & (_resolve(field, c) <= hi)


class _DaughterView:
    """Proxy that redirects c[field] to get_daughter, so generic cuts work."""

    __slots__ = ("_c", "_k")

    def __init__(self, c, k):
        self._c = c
        self._k = k

    def __getitem__(self, key):
        return get_daughter(self._c, self._k, key)

    def __contains__(self, key):
        return key in self._c["_daughter_pools"][self._k]


def any_in_tree(cut: CutFunction) -> CutFunction:
    """Wrap a cut to apply per-daughter: passes if ANY daughter satisfies it."""

    def _cut(c):
        mask = None
        k = 0
        while f"daughter{k}_global_index" in c:
            result = cut(_DaughterView(c, k))
            mask = result if mask is None else (mask | result)
            k += 1
        return mask

    return _cut


def all_in_tree(cut: CutFunction) -> CutFunction:
    """Wrap a cut to apply per-daughter: passes only if ALL daughters satisfy it."""

    def _cut(c):
        mask = None
        k = 0
        while f"daughter{k}_global_index" in c:
            result = cut(_DaughterView(c, k))
            mask = result if mask is None else (mask & result)
            k += 1
        return mask

    return _cut


def sum_in_tree(field: str) -> Callable[[Container], Any]:
    """Sum a field across all daughters."""

    def _fn(c):
        total = None
        k = 0
        while f"daughter{k}_global_index" in c:
            val = get_daughter(c, k, field)
            total = val if total is None else (total + val)
            k += 1
        return total

    return _fn


def pick_inner(arr, idx):
    """Pick one element from the innermost axis of a jagged array per row."""
    # Inner-axis counts and cumulative offsets
    inner_counts = ak.num(arr, axis=-1)
    counts_np = np.asarray(ak.flatten(inner_counts, axis=None))
    offsets = np.zeros(len(counts_np), dtype=np.int64)
    if len(counts_np) > 1:
        np.cumsum(counts_np[:-1], out=offsets[1:])

    flat_arr = np.asarray(ak.flatten(arr, axis=None))
    flat_idx = offsets + np.asarray(ak.flatten(idx, axis=None))
    # Clamp to valid range (empty inner rows produce out-of-bounds idx)
    if len(flat_arr) > 0:
        np.clip(flat_idx, 0, len(flat_arr) - 1, out=flat_idx)
        result = flat_arr[flat_idx]
    else:
        result = np.zeros_like(flat_idx)

    # Restore the outer structure of idx (could be flat, 1-deep jagged, etc.)
    if isinstance(idx, np.ndarray):
        return result
    return ak.unflatten(result, ak.num(idx, axis=-1))


def gather_jagged(source, indices):
    """Pick from source[evt, :] using indices[evt, :], returning result[evt, :]."""
    src_flat = np.asarray(ak.flatten(source))
    idx_flat = np.asarray(ak.flatten(indices))
    src_counts = ak.to_numpy(ak.num(source))
    idx_counts = ak.to_numpy(ak.num(indices))

    offsets = np.zeros(len(src_counts) + 1, dtype=np.int64)
    np.cumsum(src_counts, out=offsets[1:])
    evt_per = np.repeat(np.arange(len(idx_counts)), idx_counts)

    picked = src_flat[offsets[evt_per] + idx_flat]
    return ak.unflatten(picked, idx_counts)
