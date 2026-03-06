"""I/O helpers: read ROOT/JSON into SoA containers, export to Parquet."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

from .models import Container, COV4_LOWER_TRI, COV5_LOWER_TRI

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


def _progress(iterable, **kwargs):
    if _tqdm is not None:
        return _tqdm(iterable, **kwargs)
    return iterable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from hepunits import c_light as _C_LIGHT_MM_PER_NS  # mm/ns
from particle import literals as _lp

_PION_MASS_MEV = _lp.pi_plus.mass  # MeV


# ---------------------------------------------------------------------------
# ROOT reading — tracks
# ---------------------------------------------------------------------------


def load_tracks_root(
    path: str | Path,
    tree_name: str = "BestLongTracks/TrackTuple",
    max_events: int | None = None,
) -> Container:
    """Load tracks from a ROOT TTree into a SoA container."""
    uproot, ak = _require_uproot()
    import numpy as np

    tree = uproot.open(f"{path}:{tree_name}")

    # --- branches ---
    track_branches = [
        "FirstMeasurement_x",
        "FirstMeasurement_y",
        "FirstMeasurement_z",
        "FirstMeasurement_tx",
        "FirstMeasurement_ty",
        "FirstMeasurement_qop",
        "Track_chi2ndof",
    ]
    cov_branches = [f"FirstMeasurement_cov_{i}_{j}" for i, j in COV5_LOWER_TRI]
    hit_branches = ["TVHits_z", "TVHits_t"]
    mc_branches = [
        "MC_truth",
        "MC_pid",
        "MC_key",
        "MC_pv_key",
        "MC_fromSignal",
        "MC_px",
        "MC_py",
        "MC_pz",
        "MC_pe",
        "MC_charge",
        "MC_ovtx_x",
        "MC_ovtx_y",
        "MC_ovtx_z",
    ]
    mc_jagged_branches = ["MC_ancestor_pids", "MC_ancestor_keys"]
    event_branches = ["EventNumber", "RunNumber"]

    # Auto-detect flat+sizes format (RNTuple) vs doubly-jagged (TTree)
    available = set(tree.keys())
    flat_format = "MC_n_ancestors" in available
    size_branches = []
    if flat_format:
        size_branches = ["MC_n_ancestors", "TVHits_n"]

    all_branches = (
        track_branches
        + cov_branches
        + hit_branches
        + mc_branches
        + mc_jagged_branches
        + event_branches
        + size_branches
    )
    entry_stop = max_events if max_events else None
    data = tree.arrays(expressions=all_branches, library="ak", entry_stop=entry_stop)

    return _build_track_container(data, ak, np, flat_format=flat_format)


def _to_float64(arr, ak):
    """Upcast awkward array to float64 if needed (ROOT stores float32)."""
    return ak.values_astype(arr, "float64")


def _unflatten_2d(flat_data, sizes, ak):
    """Reconstruct doubly-jagged from flat+sizes (RNTuple format)."""
    flat_1d = ak.flatten(flat_data)
    sizes_1d = ak.flatten(sizes)
    per_element = ak.unflatten(flat_1d, sizes_1d)
    return ak.unflatten(per_element, ak.num(sizes))


def _build_track_container(data, ak, np, flat_format=False) -> Container:
    """Build track SoA container from raw uproot awkward arrays."""
    # Filter invalid tracks: qop == 0
    qop = _to_float64(data["FirstMeasurement_qop"], ak)
    valid = qop != 0.0

    tracks: Container = {}

    # Raw state — upcast to float64 for physics precision (ROOT stores float32)
    tracks["x"] = _to_float64(data["FirstMeasurement_x"][valid], ak)
    tracks["y"] = _to_float64(data["FirstMeasurement_y"][valid], ak)
    tracks["z"] = _to_float64(data["FirstMeasurement_z"][valid], ak)
    tracks["tx"] = _to_float64(data["FirstMeasurement_tx"][valid], ak)
    tracks["ty"] = _to_float64(data["FirstMeasurement_ty"][valid], ak)

    qop_valid = qop[valid]

    # Momentum and charge (MeV, native LHCb unit)
    p_mev = 1.0 / abs(qop_valid)
    tracks["p"] = p_mev
    tracks["charge"] = ak.where(qop_valid > 0, 1, -1)

    # Covariance: map 5x5 ROOT branches to 4x4 fields (indices 0-3 only)
    for i, j in COV4_LOWER_TRI:
        root_name = f"FirstMeasurement_cov_{i}_{j}"
        tracks[f"cov_{i}_{j}"] = _to_float64(data[root_name][valid], ak)

    # Derived kinematics (matches legacy TrackState.direction() + properties)
    tx = tracks["tx"]
    ty = tracks["ty"]
    p = tracks["p"]
    norm = (1.0 + tx**2 + ty**2) ** 0.5
    dx = tx / norm
    dy = ty / norm
    dz = 1.0 / norm
    tracks["pt"] = p * (dx**2 + dy**2) ** 0.5

    # Pseudorapidity — match legacy: p_dir = sqrt(dx²+dy²+dz²), eta = 0.5*log((p+pz)/(p-pz))
    # p_dir is ≈1 but not exactly 1 in floating-point, matching legacy rounding behaviour.
    p_dir = (dx**2 + dy**2 + dz**2) ** 0.5
    # Clamp (p_dir - dz) away from zero for very forward tracks
    denom = ak.where(abs(p_dir - dz) < 1e-30, 1e-30, p_dir - dz)
    tracks["eta"] = 0.5 * np.log((p_dir + dz) / denom)

    # Track t0 from TV hits (vectorized)
    if flat_format:
        # Flat+sizes: two-step unflatten (flat → per-track → per-event),
        # then apply the valid mask on the track axis.
        hit_z = _to_float64(
            _unflatten_2d(data["TVHits_z"], data["TVHits_n"], ak)[valid], ak
        )
        hit_t = _to_float64(
            _unflatten_2d(data["TVHits_t"], data["TVHits_n"], ak)[valid], ak
        )
    else:
        hit_z = _to_float64(data["TVHits_z"][valid], ak)
        hit_t = _to_float64(data["TVHits_t"][valid], ak)
    time, sigma_time = _fit_track_t0(
        tracks["z"],
        tx,
        ty,
        p_mev,
        hit_z,
        hit_t,
        ak,
        np,
    )
    tracks["time"] = time
    tracks["sigma_time"] = sigma_time

    # MC truth (scalar per track)
    _int_mc = {"MC_truth", "MC_pid", "MC_key", "MC_pv_key", "MC_fromSignal"}
    mc_field_map = {
        "MC_truth": "mc_truth",
        "MC_pid": "mc_pid",
        "MC_key": "mc_key",
        "MC_pv_key": "mc_pv_key",
        "MC_fromSignal": "mc_fromsignal",
        "MC_px": "mc_px",
        "MC_py": "mc_py",
        "MC_pz": "mc_pz",
        "MC_pe": "mc_pe",
        "MC_charge": "mc_charge",
        "MC_ovtx_x": "mc_ovtx_x",
        "MC_ovtx_y": "mc_ovtx_y",
        "MC_ovtx_z": "mc_ovtx_z",
    }
    for root_name, field_name in mc_field_map.items():
        tracks[field_name] = data[root_name][valid]

    # MC truth (jagged per track — doubly-jagged overall)
    if flat_format:
        # Flat+sizes: two-step unflatten, then apply valid mask
        tracks["mc_ancestor_pids"] = _unflatten_2d(
            data["MC_ancestor_pids"], data["MC_n_ancestors"], ak
        )[valid]
        tracks["mc_ancestor_keys"] = _unflatten_2d(
            data["MC_ancestor_keys"], data["MC_n_ancestors"], ak
        )[valid]
    else:
        tracks["mc_ancestor_pids"] = data["MC_ancestor_pids"][valid]
        tracks["mc_ancestor_keys"] = data["MC_ancestor_keys"][valid]

    # Track quality
    if "Track_chi2ndof" in data.fields:
        tracks["chi2ndof"] = _to_float64(data["Track_chi2ndof"][valid], ak)

    # Track index within event
    tracks["track_id"] = ak.local_index(tracks["x"], axis=1)

    return tracks


def _fit_track_t0(z, tx, ty, p_mev, hit_z, hit_t, ak, np):
    """Fit track t0 from TV hits. Returns (time, sigma_time)."""
    slope_factor = (1.0 + tx**2 + ty**2) ** 0.5
    energy = (p_mev**2 + _PION_MASS_MEV**2) ** 0.5
    beta_c = (p_mev / energy) * _C_LIGHT_MM_PER_NS

    # t0 per hit: propagate hit time back to reference z
    # hit_z, hit_t are (events, tracks, hits); z, slope_factor, beta_c are (events, tracks)
    # awkward broadcasts automatically along the innermost (hits) axis
    t0_per_hit = hit_t - (hit_z - z) * slope_factor / beta_c

    # Count valid hits per track
    n_hits = ak.count(t0_per_hit, axis=-1)  # (events, tracks)

    # Mean over hits axis
    time = ak.where(
        n_hits > 0,
        ak.sum(t0_per_hit, axis=-1) / ak.where(n_hits > 0, n_hits, 1),
        0.0,
    )

    # Sigma: std / sqrt(n) for n > 1
    # Compute variance manually to handle edge cases
    mean_broadcast = time  # (events, tracks) — broadcasts vs (events, tracks, hits)
    residuals_sq = (t0_per_hit - mean_broadcast) ** 2
    variance = ak.where(
        n_hits > 1,
        ak.sum(residuals_sq, axis=-1) / ak.where(n_hits > 1, n_hits, 1),
        0.0,
    )
    sigma_time = ak.where(
        n_hits > 1,
        (variance**0.5) / (n_hits**0.5),
        ak.where(n_hits == 1, 1e9, 1e9),
    )
    # Clamp to minimum
    sigma_time = ak.where(sigma_time < 1e-12, 1e-12, sigma_time)

    return time, sigma_time


# ---------------------------------------------------------------------------
# ROOT reading — primary vertices
# ---------------------------------------------------------------------------


def load_pvs_root(
    path: str | Path,
    tree_name: str = "BestLongTracks/TrackTuple",
    max_events: int | None = None,
) -> Container:
    """Load primary vertices from a ROOT TTree into a SoA container."""
    uproot, ak = _require_uproot()
    import numpy as np

    tree = uproot.open(f"{path}:{tree_name}")

    pv_branches = [
        "PV_x",
        "PV_y",
        "PV_z",
        "PV_t",
        "PV_cov_0_0",
        "PV_cov_1_0",
        "PV_cov_1_1",
        "PV_cov_2_0",
        "PV_cov_2_1",
        "PV_cov_2_2",
        "PV_cov_3_3",
    ]
    event_branches = ["EventNumber", "RunNumber"]
    entry_stop = max_events if max_events else None
    data = tree.arrays(
        expressions=pv_branches + event_branches,
        library="ak",
        entry_stop=entry_stop,
    )

    return _build_pv_container(data, ak, np)


def _build_pv_container(data, ak, np) -> Container:
    """Build PV SoA container from raw uproot awkward arrays."""
    pvs: Container = {}
    pvs["x"] = _to_float64(data["PV_x"], ak)
    pvs["y"] = _to_float64(data["PV_y"], ak)
    pvs["z"] = _to_float64(data["PV_z"], ak)
    pvs["time"] = _to_float64(data["PV_t"], ak)

    pvs["cov_0_0"] = _to_float64(data["PV_cov_0_0"], ak)
    pvs["cov_1_0"] = _to_float64(data["PV_cov_1_0"], ak)
    pvs["cov_1_1"] = _to_float64(data["PV_cov_1_1"], ak)
    pvs["cov_2_0"] = _to_float64(data["PV_cov_2_0"], ak)
    pvs["cov_2_1"] = _to_float64(data["PV_cov_2_1"], ak)
    pvs["cov_2_2"] = _to_float64(data["PV_cov_2_2"], ak)
    pvs["cov_3_3"] = _to_float64(data["PV_cov_3_3"], ak)

    # sigma_time = sqrt(max(cov_3_3, 0))
    pvs["sigma_time"] = ak.where(
        pvs["cov_3_3"] > 0.0,
        pvs["cov_3_3"] ** 0.5,
        0.0,
    )

    # PV index within event
    pvs["pv_index"] = ak.local_index(pvs["x"], axis=1)

    return pvs


# ---------------------------------------------------------------------------
# ROOT reading — combined (tracks + PVs + event IDs)
# ---------------------------------------------------------------------------


def _get_all_branches(tree, include_pvs=False):
    """Return (branch_list, flat_format) for a ROOT tree/RNTuple."""
    track_branches = [
        "FirstMeasurement_x",
        "FirstMeasurement_y",
        "FirstMeasurement_z",
        "FirstMeasurement_tx",
        "FirstMeasurement_ty",
        "FirstMeasurement_qop",
        "Track_chi2ndof",
    ]
    cov_branches = [f"FirstMeasurement_cov_{i}_{j}" for i, j in COV5_LOWER_TRI]
    hit_branches = ["TVHits_z", "TVHits_t"]
    mc_branches = [
        "MC_truth",
        "MC_pid",
        "MC_key",
        "MC_pv_key",
        "MC_fromSignal",
        "MC_px",
        "MC_py",
        "MC_pz",
        "MC_pe",
        "MC_charge",
        "MC_ovtx_x",
        "MC_ovtx_y",
        "MC_ovtx_z",
    ]
    mc_jagged_branches = ["MC_ancestor_pids", "MC_ancestor_keys"]
    event_branches = ["EventNumber", "RunNumber"]

    available = set(tree.keys())
    flat_format = "MC_n_ancestors" in available
    size_branches = ["MC_n_ancestors", "TVHits_n"] if flat_format else []

    all_branches = (
        track_branches
        + cov_branches
        + hit_branches
        + mc_branches
        + mc_jagged_branches
        + size_branches
        + event_branches
    )
    if include_pvs:
        all_branches += [
            "PV_x",
            "PV_y",
            "PV_z",
            "PV_t",
            "PV_cov_0_0",
            "PV_cov_1_0",
            "PV_cov_1_1",
            "PV_cov_2_0",
            "PV_cov_2_1",
            "PV_cov_2_2",
            "PV_cov_3_3",
        ]
    return all_branches, flat_format


def load_events_root(
    path: str | Path,
    tree_name: str = "BestLongTracks/TrackTuple",
    max_events: int | None = None,
) -> tuple[Container, Container, dict]:
    """Load tracks and PVs from a ROOT file in a single pass.

    Returns (tracks, pvs, event_info).
    """
    uproot, ak = _require_uproot()
    import numpy as np

    tree = uproot.open(f"{path}:{tree_name}")

    all_branches, flat_format = _get_all_branches(tree, include_pvs=True)
    entry_stop = max_events if max_events else None
    data = tree.arrays(expressions=all_branches, library="ak", entry_stop=entry_stop)

    tracks = _build_track_container(data, ak, np, flat_format=flat_format)
    pvs = _build_pv_container(data, ak, np)

    event_info = {
        "run_number": ak.to_numpy(data["RunNumber"]).astype(np.int64),
        "event_number": ak.to_numpy(data["EventNumber"]).astype(np.int64),
    }

    return tracks, pvs, event_info


def iter_events_root(
    path: str | Path,
    tree_name: str = "BestLongTracks/TrackTuple",
    max_events: int | None = None,
    chunk_size: int = 100,
) -> Iterator[tuple[Container, Container, dict]]:
    """Yield (tracks, pvs, event_info) from ROOT in chunks."""
    uproot, ak = _require_uproot()
    import numpy as np

    tree = uproot.open(f"{path}:{tree_name}")

    all_branches, flat_format = _get_all_branches(tree, include_pvs=True)
    entry_stop = max_events if max_events else None

    for chunk in tree.iterate(
        expressions=all_branches,
        library="ak",
        step_size=chunk_size,
        entry_stop=entry_stop,
    ):
        tracks = _build_track_container(chunk, ak, np, flat_format=flat_format)
        pvs = _build_pv_container(chunk, ak, np)

        event_info = {
            "run_number": ak.to_numpy(chunk["RunNumber"]).astype(np.int64),
            "event_number": ak.to_numpy(chunk["EventNumber"]).astype(np.int64),
        }

        yield tracks, pvs, event_info


# ---------------------------------------------------------------------------
# JSON reading
# ---------------------------------------------------------------------------


def load_tracks_json(path: str | Path) -> Container:
    """Load tracks from a single-event JSON file."""
    import awkward as ak
    import numpy as np

    data = _load_json(path)
    tracks_data = data.get("tracks")
    if not isinstance(tracks_data, list):
        raise ValueError("Input JSON must contain a list under key 'tracks'.")

    return _json_tracks_to_container([tracks_data], ak, np)


def load_events_json(path: str | Path) -> tuple[Container, Container, dict]:
    """Load multi-event JSON. Returns (tracks, pvs, event_info)."""
    import awkward as ak
    import numpy as np

    data = _load_json(path)
    events_data = data.get("events")
    if not isinstance(events_data, list):
        raise ValueError("Events JSON must contain a list under key 'events'.")

    all_tracks: list[list[dict]] = []
    all_pvs: list[list[dict]] = []
    run_numbers: list[int] = []
    event_numbers: list[int] = []

    for idx, event in enumerate(events_data):
        if not isinstance(event, dict):
            raise ValueError(f"Event entry at index {idx} must be an object.")
        run_numbers.append(int(event.get("run_number", 0)))
        evt_num = event.get("event_number", idx)
        try:
            event_numbers.append(int(evt_num))
        except (ValueError, TypeError):
            event_numbers.append(idx)
        tracks_data = event.get("tracks")
        if not isinstance(tracks_data, list):
            raise ValueError(
                f"Event at index {idx} must contain a list under key 'tracks'."
            )
        all_tracks.append(tracks_data)

        pvs_data = _extract_primary_vertices_payload(event, allow_object_fallback=False)
        all_pvs.append(pvs_data)

    tracks = _json_tracks_to_container(all_tracks, ak, np)
    pvs = _json_pvs_to_container(all_pvs, ak, np)
    event_info = {
        "run_number": np.array(run_numbers, dtype=np.int64),
        "event_number": np.array(event_numbers, dtype=np.int64),
    }

    return tracks, pvs, event_info


def _json_tracks_to_container(events_tracks: list[list[dict]], ak, np) -> Container:
    """Convert per-event lists of track dicts to SoA container."""
    fields: dict[str, list[list]] = {
        "x": [],
        "y": [],
        "z": [],
        "tx": [],
        "ty": [],
        "p": [],
        "charge": [],
        "time": [],
        "sigma_time": [],
    }
    # Covariance fields
    for i, j in COV4_LOWER_TRI:
        fields[f"cov_{i}_{j}"] = []

    for event_tracks in events_tracks:
        event_vals = {k: [] for k in fields}
        for item in event_tracks:
            state = item.get("state", item)
            event_vals["x"].append(float(state["x"]))
            event_vals["y"].append(float(state["y"]))
            event_vals["z"].append(float(item["z"]))
            event_vals["tx"].append(float(state["tx"]))
            event_vals["ty"].append(float(state["ty"]))
            event_vals["p"].append(float(item["p"]))
            event_vals["charge"].append(int(item.get("charge", 0)))

            # Time
            if "time" in state:
                event_vals["time"].append(float(state["time"]))
            elif "t" in item:
                event_vals["time"].append(float(item["t"]))
            elif "time" in item:
                event_vals["time"].append(float(item["time"]))
            else:
                event_vals["time"].append(0.0)
            event_vals["sigma_time"].append(
                float(item.get("sigma_time", item.get("sigma_t", 1.0)))
            )

            # Covariance
            cov4 = item.get("cov4", [[0] * 4] * 4)
            for i, j in COV4_LOWER_TRI:
                event_vals[f"cov_{i}_{j}"].append(float(cov4[i][j]))

        for k in fields:
            fields[k].append(event_vals[k])

    container: Container = {}
    for k, v in fields.items():
        container[k] = ak.Array(v)

    # Derive pt, eta (matches legacy TrackState formula)
    tx = container["tx"]
    ty = container["ty"]
    p = container["p"]
    norm = (1.0 + tx**2 + ty**2) ** 0.5
    dx = tx / norm
    dy = ty / norm
    dz = 1.0 / norm
    container["pt"] = p * (dx**2 + dy**2) ** 0.5
    p_dir = (dx**2 + dy**2 + dz**2) ** 0.5
    denom = ak.where(abs(p_dir - dz) < 1e-30, 1e-30, p_dir - dz)
    container["eta"] = 0.5 * np.log((p_dir + dz) / denom)

    container["track_id"] = ak.local_index(container["x"], axis=1)

    return container


def _json_pvs_to_container(events_pvs: list[list[dict]], ak, np) -> Container:
    """Convert per-event lists of PV dicts to SoA container."""
    fields: dict[str, list[list]] = {
        "x": [],
        "y": [],
        "z": [],
        "time": [],
        "sigma_time": [],
    }
    cov_keys = [
        "cov_0_0",
        "cov_1_0",
        "cov_1_1",
        "cov_2_0",
        "cov_2_1",
        "cov_2_2",
        "cov_3_3",
    ]
    for k in cov_keys:
        fields[k] = []

    for event_pvs in events_pvs:
        event_vals = {k: [] for k in fields}
        for item in event_pvs:
            event_vals["x"].append(float(item["x"]))
            event_vals["y"].append(float(item["y"]))
            event_vals["z"].append(float(item["z"]))
            event_vals["time"].append(float(item["time"]))
            event_vals["sigma_time"].append(float(item["sigma_time"]))
            cov3 = item.get("cov3", [[0] * 3] * 3)
            event_vals["cov_0_0"].append(float(cov3[0][0]))
            event_vals["cov_1_0"].append(float(cov3[1][0]))
            event_vals["cov_1_1"].append(float(cov3[1][1]))
            event_vals["cov_2_0"].append(float(cov3[2][0]))
            event_vals["cov_2_1"].append(float(cov3[2][1]))
            event_vals["cov_2_2"].append(float(cov3[2][2]))
            event_vals["cov_3_3"].append(float(item.get("sigma_time", 0.0)) ** 2)
        for k in fields:
            fields[k].append(event_vals[k])

    container: Container = {}
    for k, v in fields.items():
        container[k] = ak.Array(v)
    container["pv_index"] = ak.local_index(container["x"], axis=1)
    return container


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def candidates_to_parquet(candidates: Container, path: str | Path) -> None:
    """Flatten and write a candidate container to Parquet."""
    import awkward as ak

    flat: dict[str, Any] = {}
    for key, arr in candidates.items():
        if key.startswith("_"):
            continue
        try:
            flat[key] = ak.flatten(arr, axis=1)
        except Exception:
            flat[key] = arr
    ak.to_parquet(ak.Array(flat), str(path))


def extract_daughter_fields(candidates: Container) -> dict[str, Any]:
    """Extract per-daughter track fields using pool_index lookups.

    Returns a flat dict {field_name: numpy_array} for DataFrame merging.
    """
    import awkward as ak
    import numpy as np
    from .models import infer_n_body

    n_body = infer_n_body(candidates)
    daughter_pools = candidates["_daughter_pools"]

    # Fields to extract per daughter (only if present in pool)
    _FIELDS = [
        # kinematics
        "pt",
        "eta",
        "x",
        "y",
        "z",
        "tx",
        "ty",
        "time",
        "sigma_time",
        # track-PV association
        "min_ip",
        "min_ip_chi2",
        "best_pv_x",
        "best_pv_y",
        "best_pv_z",
        "best_pv_time",
        "best_pv_sigma_time",
        # MC truth (scalars only)
        "mc_pid",
        "mc_truth",
        "mc_key",
        "mc_pv_key",
        "mc_fromsignal",
    ]

    result: dict[str, Any] = {}

    for k in range(n_body):
        pool = daughter_pools[k]
        pool_idx = candidates[f"daughter{k}_pool_index"]  # jagged (events, cands)

        # Flatten pool_index to 1D numpy
        cand_counts = ak.to_numpy(ak.num(pool_idx))
        idx_flat = ak.to_numpy(ak.flatten(pool_idx))

        # Pool offsets: pool arrays are jagged (events, tracks), we need
        # global indices = pool_offset[event] + pool_local_index
        pool_counts = ak.to_numpy(ak.num(pool["x"]))
        pool_offsets = np.zeros(len(pool_counts) + 1, dtype=np.int64)
        np.cumsum(pool_counts, out=pool_offsets[1:])

        # Map each flat candidate to its event index
        evt_per_cand = np.repeat(np.arange(len(cand_counts)), cand_counts)
        global_idx = pool_offsets[evt_per_cand] + idx_flat

        for field in _FIELDS:
            if field not in pool:
                continue
            pool_arr = pool[field]
            try:
                pool_flat = ak.to_numpy(ak.flatten(pool_arr))
            except Exception:
                continue  # skip non-flat fields
            result[f"daughter{k}_{field}"] = pool_flat[global_idx]

    return result


def candidates_to_dataframe(candidates: Container) -> Any:
    """Convert a candidate container to a pandas DataFrame."""
    import awkward as ak
    import pandas as pd

    flat: dict[str, Any] = {}
    for key, arr in candidates.items():
        if key.startswith("_"):
            continue
        try:
            flat[key] = ak.to_numpy(ak.flatten(arr, axis=1))
        except Exception:
            pass  # skip variable-length fields (e.g. mc_ancestor_pids)
    return pd.DataFrame(flat)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_uproot():
    """Import uproot and awkward lazily."""
    try:
        import uproot
        import awkward as ak
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "uproot and awkward are required to read ROOT files. "
            "Install them with: pip install 'track-combination-framework[root]'"
        ) from exc
    return uproot, ak


def _extract_primary_vertices_payload(
    data: dict[str, Any],
    allow_object_fallback: bool = True,
) -> list[Any]:
    pvs_data = data.get("primary_vertices", data.get("pvs"))
    if pvs_data is None:
        if not allow_object_fallback:
            raise ValueError("Event payload must contain 'primary_vertices' list.")
        pv_data = data.get("primary_vertex", data)
        if not isinstance(pv_data, dict):
            raise ValueError(
                "Primary vertex JSON must contain 'primary_vertices' list."
            )
        pvs_data = [pv_data]
    if not isinstance(pvs_data, list):
        raise ValueError("'primary_vertices' must be a list.")
    return pvs_data


def _load_json(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"JSON document at {path} must be an object.")
    return data
