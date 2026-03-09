"""I/O helpers: read ROOT into SoA containers, export to DataFrame."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .models import Container, COV5_LOWER_TRI, n_daughters
from .configurable import configurable
from .physics import compute_default_track_quantities


def _to_float64(arr):
    """Upcast awkward array to float64."""
    return ak.values_astype(arr, "float64")


def _unflatten_2d(flat_data, sizes):
    """Reconstruct doubly-jagged array from flat data and sizes."""
    flat_1d = ak.flatten(flat_data)
    sizes_1d = ak.flatten(sizes)
    per_element = ak.unflatten(flat_1d, sizes_1d)
    return ak.unflatten(per_element, ak.num(sizes))


@configurable
def make_tracks(
    data,
    hit_types=("TVHits",),
    state_name="FirstMeasurement",
    compute_track_quantities=compute_default_track_quantities,
) -> Container:
    """Build track container from uproot awkward arrays."""

    tracks: Container = {}
    _handled = set()

    def _load_1d(branch):
        _handled.add(branch)
        return _to_float64(data[branch])

    def _load_2d(branch, size_branch):
        _handled.add(branch)
        _handled.add(size_branch)
        return _to_float64(_unflatten_2d(data[branch], data[size_branch]))

    # Track state
    for field in ("x", "y", "z", "tx", "ty", "qop"):
        tracks[field] = _load_1d(f"{state_name}_{field}")

    # State cov
    for i, j in COV5_LOWER_TRI:
        tracks[f"cov_{i}_{j}"] = _load_1d(f"{state_name}_cov_{i}_{j}")

    # Hits
    hit_info_map = {
        "TVHits": ["x", "y", "z", "t"],
        "UPHits": ["x", "y", "z"],
        "FTHits": ["x", "z"],
        "MPHits": ["x", "y", "z"],
    }
    for htype in hit_types:
        assert htype in hit_info_map, f"{htype} is an invalid hit type"
        for info in hit_info_map[htype]:
            tracks[f"{htype}_{info}".lower()] = _load_2d(
                f"{htype}_{info}", f"{htype}_n"
            )

    # MC truth
    _mc_branches = [b for b in data.fields if b.startswith("MC_")]
    if _mc_branches:
        if "MC_ancestor_pids" in data.fields:
            tracks["mc_ancestor_pids"] = _load_2d(
                "MC_ancestor_pids", "MC_n_ancestors"
            )

        if "MC_ancestor_keys" in data.fields:
            tracks["mc_ancestor_keys"] = _load_2d(
                "MC_ancestor_keys", "MC_n_ancestors"
            )

        for branch in _mc_branches:
            if branch in _handled:
                continue
            tracks[branch.lower()] = _load_1d(branch)

    # Auto-add remaining Track_* branches
    for branch in data.fields:
        if branch in _handled or not branch.startswith("Track_"):
            continue
        field = branch[6:].lower()
        if field not in tracks:
            tracks[field] = data[branch]

    # Metadata
    tracks["_type"] = "tracks"
    tracks["track_id"] = ak.local_index(tracks["x"], axis=1)

    # Derive physics quantities
    compute_track_quantities(tracks)

    return tracks


def make_pvs(data) -> Container:
    """Build PV container from uproot awkward arrays."""
    pvs: Container = {}

    # Auto-load all PV_* branches, stripping prefix
    for branch in data.fields:
        if not branch.startswith("PV_"):
            continue
        field = branch[3:].lower()
        pvs[field] = _to_float64(data[branch])

    # PV_t → time (rename for consistency with tracks)
    if "t" in pvs:
        pvs["time"] = pvs.pop("t")

    # sigma_time = sqrt(max(cov_3_3, 0))
    if "cov_3_3" in pvs:
        pvs["sigma_time"] = ak.where(
            pvs["cov_3_3"] > 0.0,
            pvs["cov_3_3"] ** 0.5,
            0.0,
        )

    # Metadata
    pvs["_type"] = "pvs"
    pvs["pv_index"] = ak.local_index(pvs["x"], axis=1)

    return pvs


def load_events(
    path: str | Path,
    tree_name: str = "BestLongTracks/TrackTuple",
    max_events: int | None = None,
) -> tuple[Container, Container, dict]:
    """Load tracks and PVs from a ROOT file."""
    tree = uproot.open(f"{path}:{tree_name}")

    entry_stop = max_events if max_events else None
    data = tree.arrays(library="ak", entry_stop=entry_stop)

    tracks = make_tracks(data)
    pvs = make_pvs(data)

    event_info = {
        "run_number": ak.to_numpy(data["RunNumber"]).astype(np.int64),
        "event_number": ak.to_numpy(data["EventNumber"]).astype(np.int64),
    }

    return tracks, pvs, event_info


def load_events_in_slices(
    path: str | Path,
    tree_name: str = "BestLongTracks/TrackTuple",
    max_events: int | None = None,
    slice_size: int = 100,
) -> Iterator[tuple[Container, Container, dict]]:
    """Yield (tracks, pvs, event_info) in slices from a ROOT file."""

    if slice_size == 0 or (
        max_events is not None and max_events <= slice_size
    ):
        yield load_events(path, tree_name, max_events)
        return

    tree = uproot.open(f"{path}:{tree_name}")

    entry_stop = max_events if max_events else None

    for chunk in tree.iterate(
        library="ak",
        step_size=slice_size,
        entry_stop=entry_stop,
    ):
        tracks = make_tracks(chunk)
        pvs = make_pvs(chunk)

        event_info = {
            "run_number": ak.to_numpy(chunk["RunNumber"]).astype(np.int64),
            "event_number": ak.to_numpy(chunk["EventNumber"]).astype(np.int64),
        }

        yield tracks, pvs, event_info


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
