"""I/O core: stream EventTuple files in chunks, read branches, export DataFrames.

Reading rule (see .claude/IO_DESIGN.md): branches are read ONLY via their full
path ("Group/leaf"). Never use filter_name/wildcard reads — leaf basenames
repeat across groups and uproot silently merges same-named fields.
"""

from __future__ import annotations

import functools
import glob
import warnings
from pathlib import Path
from typing import Any, Iterator

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .models import Container, n_daughters
from .counters import counters

TREE_NAME = "EventTuple"


@functools.lru_cache(maxsize=64)
def _open_tree(path: str):
    return uproot.open(f"{path}:{TREE_NAME}")


def read(chunk: Container, branch: str):
    """Read one branch (full path) for a chunk's entry range."""
    tree = _open_tree(chunk["path"])
    return tree[branch].array(
        entry_start=chunk["entry_start"], entry_stop=chunk["entry_stop"]
    )


def read_float64(chunk: Container, branch: str):
    """Read a branch and upcast to float64 (for fit numerics)."""
    return ak.values_astype(read(chunk, branch), "float64")


def unflatten_2d(flat_data, sizes):
    """Reconstruct doubly-jagged array from per-event flat data and sizes."""
    per_object = ak.unflatten(ak.flatten(flat_data), ak.flatten(sizes))
    return ak.unflatten(per_object, ak.num(sizes))


def expand_files(path: str | Path) -> list[str]:
    """Expand a path (may contain wildcards) to a sorted file list."""
    pattern = str(path)
    if any(c in pattern for c in "*?["):
        files = sorted(glob.glob(pattern))
    else:
        files = [pattern]
    if not files:
        raise FileNotFoundError(f"no files match {path}")
    return files


def event_stream(
    path: str | Path,
    chunk_size: int = 100,
    max_events: int | None = None,
) -> Iterator[Container]:
    """Yield picklable chunk cursors {path, entry_start, entry_stop}.

    Unreadable files are skipped with a warning and counted in
    counters("skipped corrupt files"). Chunks never span files, so a
    trailing chunk may be smaller than chunk_size.
    """
    n_seen = 0
    n_opened = 0
    for file_path in expand_files(path):
        try:
            n_entries = _open_tree(file_path).num_entries
        except Exception as exc:
            warnings.warn(f"skipping unreadable file {file_path}: {exc}")
            counters("skipped corrupt files").add(1)
            continue
        n_opened += 1
        start = 0
        while start < n_entries:
            stop = (
                min(start + chunk_size, n_entries) if chunk_size else n_entries
            )
            if max_events is not None:
                stop = min(stop, start + max_events - n_seen)
            yield {
                "path": file_path,
                "entry_start": start,
                "entry_stop": stop,
            }
            n_seen += stop - start
            if max_events is not None and n_seen >= max_events:
                return
            start = stop
    if n_opened == 0:
        raise FileNotFoundError(f"no readable files match {path}")


def n_chunk_events(chunk: Container) -> int:
    return chunk["entry_stop"] - chunk["entry_start"]


def load_events(
    path: str | Path,
    max_events: int | None = None,
    **loader_kwargs,
) -> tuple[Container, Container, dict]:
    """One-shot convenience loader: (tracks, pvs, event_info).

    For notebooks and tests. Streams chunks internally and concatenates.
    """
    from .components.tracks import load_tracks
    from .components.pvs import load_pvs
    from .components.event_info import load_event_info

    parts = []
    for chunk in event_stream(path, chunk_size=0, max_events=max_events):
        parts.append(
            (
                load_tracks(chunk, **loader_kwargs),
                load_pvs(chunk),
                load_event_info(chunk),
            )
        )
    if len(parts) == 1:
        return parts[0]

    def _concat(containers):
        out = dict(containers[0])
        for key, val in out.items():
            if key.startswith("_"):
                continue
            arrays = [c[key] for c in containers]
            if isinstance(val, np.ndarray):
                out[key] = np.concatenate(arrays)
            else:
                out[key] = ak.concatenate(arrays)
        return out

    tracks = _concat([p[0] for p in parts])
    pvs = _concat([p[1] for p in parts])
    info = {k: np.concatenate([p[2][k] for p in parts]) for k in parts[0][2]}
    return tracks, pvs, info


def candidates_to_dataframe(candidates: Container, _prefix="daughter") -> Any:
    """Convert a candidate container to a pandas DataFrame."""
    flat: dict[str, Any] = {}

    # Candidate-level fields
    for key, arr in candidates.items():
        if key.startswith("_") or key.startswith("cached_"):
            continue
        try:
            val = ak.to_numpy(ak.flatten(arr, axis=1))
            if val.ndim == 1:
                flat[key] = val
        except Exception:
            pass

    # Daughter fields via pool lookup using global_index
    if "_daughter_pools" in candidates:
        n_body = n_daughters(candidates)
        pools = candidates["_daughter_pools"]

        for k in range(n_body):
            pool = pools[k]
            global_idx = ak.to_numpy(
                ak.flatten(candidates[f"daughter{k}_global_index"])
            )
            d_prefix = f"{_prefix}{k}"

            for field in pool:
                if field.startswith("_") or field.startswith("cached_"):
                    continue
                try:
                    val = ak.to_numpy(ak.flatten(pool[field]))[global_idx]
                    if val.ndim == 1:
                        flat[f"{d_prefix}_{field}"] = val
                except Exception:
                    continue

            if "_daughter_pools" in pool:
                sub = candidates_to_dataframe(
                    pool, _prefix=f"_{d_prefix}_sub_"
                )
                for fname, arr in sub.items():
                    flat[f"{d_prefix}_{fname}"] = np.asarray(arr)[global_idx]

    if _prefix == "daughter":
        return pd.DataFrame(flat)
    return flat
