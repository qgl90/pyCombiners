"""Input/output helpers for JSON inputs, ROOT inputs, and tabular result export."""

from __future__ import annotations
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


def _progress(iterable, **kwargs):
    """Wrap an iterable with tqdm if available, otherwise pass through."""
    if _tqdm is not None:
        return _tqdm(iterable, **kwargs)
    return iterable

from .models import CombinationResult, EventInput, Matrix4x4, ParticleHypothesis, PrimaryVertex, TrackState
from .pid import particle_hypothesis_from_name


def load_tracks_json(path: str | Path) -> list[TrackState]:
    """Load track container JSON into `TrackState` objects."""
    data = _load_json(path)
    tracks_data = data.get("tracks")
    if not isinstance(tracks_data, list):
        raise ValueError("Input JSON must contain a list under key 'tracks'.")
    return [
        _parse_track_item(item=item, idx=idx, context=f"{path}")
        for idx, item in enumerate(tracks_data)
    ]


def load_primary_vertices_json(path: str | Path) -> list[PrimaryVertex]:
    """Load PV container JSON into `PrimaryVertex` objects.

    Supports both:
    - modern list format: `primary_vertices: [...]`
    - backward-compatible single PV object.
    """
    data = _load_json(path)
    pvs_data = _extract_primary_vertices_payload(data)
    return [
        _parse_primary_vertex_item(item=pv_data, idx=idx, context=f"{path}")
        for idx, pv_data in enumerate(pvs_data)
    ]


def load_events_json(path: str | Path) -> list[EventInput]:
    """Load multi-event input JSON into `EventInput` objects.

    Expected shape:
    {
      "events": [
        {"event_id": "...", "tracks": [...], "primary_vertices": [...]},
        ...
      ]
    }
    """
    data = _load_json(path)
    events_data = data.get("events")
    if not isinstance(events_data, list):
        raise ValueError("Events JSON must contain a list under key 'events'.")
    out: list[EventInput] = []
    for idx, event in enumerate(events_data):
        if not isinstance(event, dict):
            raise ValueError(f"Event entry at index {idx} must be an object.")
        event_id = str(event.get("event_id", f"evt{idx}"))
        tracks_data = event.get("tracks")
        if not isinstance(tracks_data, list):
            raise ValueError(f"Event '{event_id}' must contain a list under key 'tracks'.")
        pvs_data = _extract_primary_vertices_payload(event, allow_object_fallback=False)
        tracks = tuple(
            _parse_track_item(item=track_item, idx=tidx, context=f"event '{event_id}'")
            for tidx, track_item in enumerate(tracks_data)
        )
        pvs = tuple(
            _parse_primary_vertex_item(item=pv_item, idx=pidx, context=f"event '{event_id}'")
            for pidx, pv_item in enumerate(pvs_data)
        )
        out.append(EventInput(event_id=event_id, tracks=tracks, primary_vertices=pvs))
    return out


def load_mass_hypotheses_json(
    path: str | Path,
) -> list[tuple[float | ParticleHypothesis, ...]]:
    """Load mass-hypothesis sets from JSON.

    Supported per-entry values in each hypothesis list:
    - numeric mass (float/int)
    - string particle name (`"pi"`, `"kaon"`, `"mu"`, ...)
    - object with explicit mass (`{"name": "...", "mass": ...}`)
    - object with particle alias (`{"pid": "pi"}`)
    """
    data = _load_json(path)
    masses_data = data.get("mass_hypotheses")
    if not isinstance(masses_data, list):
        raise ValueError("Mass JSON must contain a list under key 'mass_hypotheses'.")
    parsed: list[tuple[float | ParticleHypothesis, ...]] = []
    for idx, hyp in enumerate(masses_data):
        if not isinstance(hyp, list):
            raise ValueError(f"Mass hypothesis at index {idx} must be a list.")
        # Keep rich entries (named hypotheses) for downstream provenance fields.
        parsed.append(tuple(_parse_mass_entry(entry) for entry in hyp))
    return parsed


def write_results_table(path: str | Path, results: list[CombinationResult]) -> None:
    """Write combination results into Parquet/CSV/Pickle table."""
    pd = _require_pandas()
    df = pd.DataFrame(_result_rows(results))
    out = Path(path)
    suffix = out.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(out, index=False)
    elif suffix in (".pkl", ".pickle"):
        df.to_pickle(out)
    elif suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        raise ValueError(
            f"Unsupported output format '{suffix}'. Use .parquet, .csv, or .pkl"
        )


def _result_rows(results: list[CombinationResult]) -> list[dict[str, Any]]:
    """Flatten rich combination objects into DataFrame-ready row dictionaries."""
    rows: list[dict[str, Any]] = []
    for res in results:
        row: dict[str, Any] = {
            "event_id": res.event_id,
            "track_ids": ",".join(res.track_ids),
            "source_track_ids": ",".join(res.source_track_ids),
            "masses": ",".join(str(x) for x in res.masses),
            "particle_hypotheses": ",".join(res.particle_hypotheses),
            "vertex_x": res.vertex_xyz[0],
            "vertex_y": res.vertex_xyz[1],
            "vertex_z": res.vertex_xyz[2],
            "vertex_cov_xx": res.vertex_cov_xyz[0][0],
            "vertex_cov_xy": res.vertex_cov_xyz[0][1],
            "vertex_cov_xz": res.vertex_cov_xyz[0][2],
            "vertex_cov_yy": res.vertex_cov_xyz[1][1],
            "vertex_cov_yz": res.vertex_cov_xyz[1][2],
            "vertex_cov_zz": res.vertex_cov_xyz[2][2],
            "vertex_time": res.vertex_time,
            "vertex_sigma_time": res.vertex_sigma_time,
            "vertex_chi2": res.vertex_chi2,
            "vertex_time_chi2": res.vertex_time_chi2,
            "pair_time_chi2": res.pair_time_chi2,
            "px": res.candidate_p4.px,
            "py": res.candidate_p4.py,
            "pz": res.candidate_p4.pz,
            "energy": res.candidate_p4.e,
            "candidate_mass": res.candidate_p4.mass,
            "pair_pt": res.pair_pt,
            "pair_eta": res.pair_eta,
            "charge_pattern": res.charge_pattern,
            "total_charge": res.total_charge,
            "best_pv_id": res.best_pv_id,
            "preselected_pv_ids": ",".join(res.preselected_pv_ids),
            "composite_min_ip": res.composite_min_ip,
            "composite_min_ip_chi2": res.composite_min_ip_chi2,
            "composite_pv_time_chi2": res.composite_pv_time_chi2,
            "composite_pv_time_residual": res.composite_pv_time_residual,
            "composite_pv_flight_time": res.composite_pv_flight_time,
        }
        for idx, (x, y) in enumerate(res.vertices_xy):
            row[f"v{idx}_x"] = x
            row[f"v{idx}_y"] = y
        for k, v in res.doca_pairs.items():
            row[k] = v
        for track_id, ip in res.track_min_ip.items():
            row[f"ip_{track_id}"] = ip
        for track_id, ipchi2 in res.track_min_ip_chi2.items():
            row[f"ipchi2_{track_id}"] = ipchi2
        for idx, tid in enumerate(res.track_ids, start=1):
            pid = res.track_pid_info.get(tid, {})
            row[f"trk{idx}_id"] = tid
            row[f"trk{idx}_charge"] = res.track_charges.get(tid)
            row[f"trk{idx}_hasRICH1"] = pid.get("hasRICH1")
            row[f"trk{idx}_hasRICH2"] = pid.get("hasRICH2")
            row[f"trk{idx}_richDLL_pi"] = pid.get("richDLL_pi")
            row[f"trk{idx}_richDLL_k"] = pid.get("richDLL_k")
            row[f"trk{idx}_richDLL_p"] = pid.get("richDLL_p")
            row[f"trk{idx}_richDLL_e"] = pid.get("richDLL_e")
            row[f"trk{idx}_hasCALO"] = pid.get("hasCALO")
            row[f"trk{idx}_caloDLL_e"] = pid.get("caloDLL_e")
        rows.append(row)
    return rows


def load_tracks_root(
    path: str | Path,
    tree_name: str = "BestLongTracks/TrackTuple",
    max_events: int | None = None,
    track_type: str = "long",
) -> list[tuple[str, list[TrackState]]]:
    """Load tracks per event from a ROOT TTree.

    Returns a list of ``(event_id, tracks)`` tuples, one per TTree entry.

    Parameters
    ----------
    path : str or Path
        Path to the ROOT file.
    tree_name : str
        ``"directory/tree"`` path inside the file.
    max_events : int or None
        If set, read at most this many entries.
    track_type : str
        Prefix used in track IDs (e.g. ``"long"`` → ``evt0_long3``).
    """
    uproot, ak = _require_uproot()
    tree = uproot.open(f"{path}:{tree_name}")

    # -- branches to read ------------------------------------------------
    track_branches = [
        "FirstMeasurement_x", "FirstMeasurement_y", "FirstMeasurement_z",
        "FirstMeasurement_tx", "FirstMeasurement_ty", "FirstMeasurement_qop",
    ]
    cov_branches = [
        f"FirstMeasurement_cov_{i}_{j}"
        for i in range(5) for j in range(i + 1)
    ]
    hit_branches = ["TVHits_z", "TVHits_t"]
    mc_branches = [
        "MC_truth", "MC_pid",
        "MC_key",
        "MC_px", "MC_py", "MC_pz", "MC_pe", "MC_charge",
        "MC_ovtx_x", "MC_ovtx_y", "MC_ovtx_z",
    ]
    mc_jagged_branches = ["MC_ancestor_pids", "MC_ancestor_keys"]
    event_branches = ["EventNumber", "RunNumber"]

    all_branches = track_branches + cov_branches + hit_branches + mc_branches + mc_jagged_branches + event_branches
    entry_stop = max_events if max_events is not None else None
    data = tree.arrays(expressions=all_branches, library="ak", entry_stop=entry_stop)

    # Convert all awkward arrays to Python lists in one pass (C-level bulk conversion).
    # This avoids per-element awkward indexing which dominates the runtime.
    py = {br: data[br].tolist() for br in all_branches}

    n_entries = len(py["EventNumber"])
    result: list[tuple[str, list[TrackState]]] = []

    # Pre-build covariance index pairs
    _cov_ij = [(i, j) for i in range(5) for j in range(i + 1)]
    _int_mc = {"MC_truth", "MC_pid", "MC_key"}

    for entry_idx in _progress(range(n_entries), desc="Loading tracks", total=n_entries):
        evt_num = int(py["EventNumber"][entry_idx])
        run_num = int(py["RunNumber"][entry_idx])
        event_id = f"run{run_num}_evt{evt_num}_idx{entry_idx}"

        qop_arr = py["FirstMeasurement_qop"][entry_idx]
        n_tracks = len(qop_arr)
        tracks: list[TrackState] = []

        # Pre-fetch event-level lists (Python list indexing is ~10x faster than awkward)
        ev_x = py["FirstMeasurement_x"][entry_idx]
        ev_y = py["FirstMeasurement_y"][entry_idx]
        ev_z = py["FirstMeasurement_z"][entry_idx]
        ev_tx = py["FirstMeasurement_tx"][entry_idx]
        ev_ty = py["FirstMeasurement_ty"][entry_idx]
        ev_cov = {(i, j): py[f"FirstMeasurement_cov_{i}_{j}"][entry_idx]
                  for i, j in _cov_ij}
        ev_hz = py["TVHits_z"][entry_idx]
        ev_ht = py["TVHits_t"][entry_idx]
        ev_mc = {k: py[k][entry_idx] for k in mc_branches}
        ev_mc_jag = {k: py[k][entry_idx] for k in mc_jagged_branches}

        for ti in range(n_tracks):
            qop = qop_arr[ti]
            if qop == 0.0:
                continue
            p_mev = 1.0 / abs(qop)
            charge = 1 if qop > 0 else -1

            x = ev_x[ti]
            y = ev_y[ti]
            z = ev_z[ti]
            tx = ev_tx[ti]
            ty = ev_ty[ti]

            # Reconstruct symmetric 4x4 covariance from lower-triangular elements
            cov_vals = {ij: ev_cov[ij][ti] for ij in _cov_ij}
            cov4 = _build_sym_cov4(cov_vals)

            # Fit track t0 from TV hits (uses MeV internally)
            time, sigma_time = _fit_track_t0(z, tx, ty, p_mev, ev_hz[ti], ev_ht[ti])

            # MC truth metadata
            metadata: dict[str, Any] = {}
            for mc_key in mc_branches:
                val = ev_mc[mc_key][ti]
                metadata[mc_key.lower()] = int(val) if mc_key in _int_mc else val
            for jb in mc_jagged_branches:
                metadata[jb.lower()] = [int(v) for v in ev_mc_jag[jb][ti]]

            # Convert momentum to GeV to match framework convention
            p_gev = p_mev / 1000.0

            track_id = f"evt{entry_idx}_{track_type}{ti}"
            tracks.append(TrackState(
                track_id=track_id,
                z=z, x=x, y=y, tx=tx, ty=ty,
                time=time,
                cov4=cov4,
                sigma_time=sigma_time,
                p=p_gev,
                charge=charge,
                source_track_ids=(track_id,),
                metadata=metadata,
            ))

        result.append((event_id, tracks))

    return result


def load_pvs_root(
    path: str | Path,
    tree_name: str = "BestLongTracks/TrackTuple",
    max_events: int | None = None,
) -> list[tuple[str, list[PrimaryVertex]]]:
    """Load primary vertices per event from a ROOT TTree.

    Returns a list of ``(event_id, pvs)`` tuples, one per TTree entry.
    """
    uproot, _ak = _require_uproot()
    tree = uproot.open(f"{path}:{tree_name}")

    pv_branches = [
        "PV_x", "PV_y", "PV_z", "PV_t",
        "PV_cov_0_0", "PV_cov_1_0", "PV_cov_1_1",
        "PV_cov_2_0", "PV_cov_2_1", "PV_cov_2_2",
        "PV_cov_3_3",
    ]
    event_branches = ["EventNumber", "RunNumber"]
    entry_stop = max_events if max_events is not None else None
    data = tree.arrays(
        expressions=pv_branches + event_branches,
        library="ak",
        entry_stop=entry_stop,
    )

    # Bulk convert awkward → Python lists
    py = {br: data[br].tolist() for br in pv_branches + event_branches}

    result: list[tuple[str, list[PrimaryVertex]]] = []
    for entry_idx in _progress(range(len(py["PV_x"])), desc="Loading PVs", total=len(py["PV_x"])):
        evt_num = int(py["EventNumber"][entry_idx])
        run_num = int(py["RunNumber"][entry_idx])
        event_id = f"run{run_num}_evt{evt_num}_idx{entry_idx}"
        ev_x = py["PV_x"][entry_idx]
        ev_y = py["PV_y"][entry_idx]
        ev_z = py["PV_z"][entry_idx]
        ev_t = py["PV_t"][entry_idx]
        ev_c00 = py["PV_cov_0_0"][entry_idx]
        ev_c10 = py["PV_cov_1_0"][entry_idx]
        ev_c11 = py["PV_cov_1_1"][entry_idx]
        ev_c20 = py["PV_cov_2_0"][entry_idx]
        ev_c21 = py["PV_cov_2_1"][entry_idx]
        ev_c22 = py["PV_cov_2_2"][entry_idx]
        ev_c33 = py["PV_cov_3_3"][entry_idx]
        pvs: list[PrimaryVertex] = []
        for pi in range(len(ev_x)):
            c00, c10, c11 = ev_c00[pi], ev_c10[pi], ev_c11[pi]
            c20, c21, c22 = ev_c20[pi], ev_c21[pi], ev_c22[pi]
            pvs.append(PrimaryVertex(
                pv_id=f"evt{entry_idx}_pv{pi}",
                x=ev_x[pi], y=ev_y[pi], z=ev_z[pi],
                cov3=((c00, c10, c20), (c10, c11, c21), (c20, c21, c22)),
                time=ev_t[pi],
                sigma_time=math.sqrt(max(ev_c33[pi], 0.0)),
            ))
        result.append((event_id, pvs))
    return result


def iter_events_root(
    path: str | Path,
    tree_name: str = "BestLongTracks/TrackTuple",
    max_events: int | None = None,
    track_type: str = "long",
    chunk_size: int = 100,
) -> Iterator[EventInput]:
    """Yield ``EventInput`` objects from a ROOT TTree, reading in chunks.

    Reads tracks and PVs in a single pass.  Only one chunk of events is
    held in memory at a time, so this works for arbitrarily large files.

    Parameters
    ----------
    chunk_size : int
        Number of TTree entries to read per chunk (default 100).
    """
    uproot, ak = _require_uproot()
    tree = uproot.open(f"{path}:{tree_name}")

    # -- all branches in a single read -----------------------------------
    track_branches = [
        "FirstMeasurement_x", "FirstMeasurement_y", "FirstMeasurement_z",
        "FirstMeasurement_tx", "FirstMeasurement_ty", "FirstMeasurement_qop",
    ]
    cov_branches = [
        f"FirstMeasurement_cov_{i}_{j}"
        for i in range(5) for j in range(i + 1)
    ]
    hit_branches = ["TVHits_z", "TVHits_t"]
    mc_branches = [
        "MC_truth", "MC_pid", "MC_key",
        "MC_px", "MC_py", "MC_pz", "MC_pe", "MC_charge",
        "MC_ovtx_x", "MC_ovtx_y", "MC_ovtx_z",
    ]
    mc_jagged_branches = ["MC_ancestor_pids", "MC_ancestor_keys"]
    pv_branches = [
        "PV_x", "PV_y", "PV_z", "PV_t",
        "PV_cov_0_0", "PV_cov_1_0", "PV_cov_1_1",
        "PV_cov_2_0", "PV_cov_2_1", "PV_cov_2_2",
        "PV_cov_3_3",
    ]
    event_branches = ["EventNumber", "RunNumber"]
    all_branches = (
        track_branches + cov_branches + hit_branches
        + mc_branches + mc_jagged_branches
        + pv_branches + event_branches
    )

    _cov_ij = [(i, j) for i in range(5) for j in range(i + 1)]
    _int_mc = {"MC_truth", "MC_pid", "MC_key"}

    import numpy as np

    entry_stop = max_events if max_events is not None else None
    global_idx = 0

    for chunk in tree.iterate(
        expressions=all_branches, library="ak",
        step_size=chunk_size, entry_stop=entry_stop,
    ):
        # Bulk convert chunk to Python lists (one C-level call per branch)
        py = {br: chunk[br].tolist() for br in all_branches}
        n_chunk = len(py["EventNumber"])

        for local_idx in range(n_chunk):
            entry_idx = global_idx + local_idx
            evt_num = int(py["EventNumber"][local_idx])
            run_num = int(py["RunNumber"][local_idx])
            event_id = f"run{run_num}_evt{evt_num}_idx{entry_idx}"

            # -- Build tracks (SoA → vectorized numpy → AoS TrackState) --
            qop_list = py["FirstMeasurement_qop"][local_idx]
            n_tracks = len(qop_list)

            if n_tracks == 0:
                tracks: list[TrackState] = []
            else:
                qop_np = np.array(qop_list)
                valid = qop_np != 0.0
                valid_idx = np.where(valid)[0]
                n_valid = len(valid_idx)

                if n_valid == 0:
                    tracks = []
                else:
                    # Vectorised scalar computation (numpy, all valid tracks at once)
                    qop_v = qop_np[valid]
                    p_mev = 1.0 / np.abs(qop_v)
                    charges = np.where(qop_v > 0, 1, -1)
                    x_np = np.array(py["FirstMeasurement_x"][local_idx])[valid]
                    y_np = np.array(py["FirstMeasurement_y"][local_idx])[valid]
                    z_np = np.array(py["FirstMeasurement_z"][local_idx])[valid]
                    tx_np = np.array(py["FirstMeasurement_tx"][local_idx])[valid]
                    ty_np = np.array(py["FirstMeasurement_ty"][local_idx])[valid]
                    p_gev = p_mev / 1000.0

                    # Vectorised t0 fitting (per-track numpy over hits)
                    slope = np.sqrt(1.0 + tx_np ** 2 + ty_np ** 2)
                    energy = np.sqrt(p_mev ** 2 + _PION_MASS_MEV ** 2)
                    beta_c = (p_mev / energy) * _C_LIGHT_MM_PER_NS

                    ev_hz = py["TVHits_z"][local_idx]
                    ev_ht = py["TVHits_t"][local_idx]
                    time_all = np.zeros(n_valid)
                    sigma_time_all = np.full(n_valid, 1e9)
                    for i, ti in enumerate(valid_idx):
                        hz = ev_hz[ti]
                        ht = ev_ht[ti]
                        if not hz:
                            continue
                        hz_np = np.array(hz)
                        ht_np = np.array(ht)
                        good = ~(np.isnan(hz_np) | np.isnan(ht_np))
                        hz_np, ht_np = hz_np[good], ht_np[good]
                        n_h = len(hz_np)
                        if n_h == 0:
                            continue
                        t0 = ht_np - (hz_np - z_np[i]) * slope[i] / beta_c[i]
                        time_all[i] = t0.mean()
                        if n_h > 1:
                            sigma_time_all[i] = max(t0.std() / math.sqrt(n_h), 1e-12)

                    # Covariance (Python lists, indexed)
                    ev_cov = {ij: py[f"FirstMeasurement_cov_{ij[0]}_{ij[1]}"][local_idx]
                              for ij in _cov_ij}
                    ev_mc = {k: py[k][local_idx] for k in mc_branches}
                    ev_mc_jag = {k: py[k][local_idx] for k in mc_jagged_branches}

                    # Construct TrackState objects
                    tracks = []
                    for i in range(n_valid):
                        ti = int(valid_idx[i])
                        cov_vals = {ij: ev_cov[ij][ti] for ij in _cov_ij}
                        cov4 = _build_sym_cov4(cov_vals)
                        metadata: dict[str, Any] = {}
                        for mc_key in mc_branches:
                            val = ev_mc[mc_key][ti]
                            metadata[mc_key.lower()] = int(val) if mc_key in _int_mc else val
                        for jb in mc_jagged_branches:
                            metadata[jb.lower()] = [int(v) for v in ev_mc_jag[jb][ti]]

                        track_id = f"evt{entry_idx}_{track_type}{ti}"
                        tracks.append(TrackState(
                            track_id=track_id,
                            z=float(z_np[i]), x=float(x_np[i]), y=float(y_np[i]),
                            tx=float(tx_np[i]), ty=float(ty_np[i]),
                            time=float(time_all[i]), cov4=cov4,
                            sigma_time=float(sigma_time_all[i]),
                            p=float(p_gev[i]), charge=int(charges[i]),
                            source_track_ids=(track_id,),
                            metadata=metadata,
                        ))

            # -- Build PVs --
            ev_pvx = py["PV_x"][local_idx]
            ev_pvy = py["PV_y"][local_idx]
            ev_pvz = py["PV_z"][local_idx]
            ev_pvt = py["PV_t"][local_idx]
            ev_c00 = py["PV_cov_0_0"][local_idx]
            ev_c10 = py["PV_cov_1_0"][local_idx]
            ev_c11 = py["PV_cov_1_1"][local_idx]
            ev_c20 = py["PV_cov_2_0"][local_idx]
            ev_c21 = py["PV_cov_2_1"][local_idx]
            ev_c22 = py["PV_cov_2_2"][local_idx]
            ev_c33 = py["PV_cov_3_3"][local_idx]
            pvs: list[PrimaryVertex] = []
            for pi in range(len(ev_pvx)):
                c00, c10, c11 = ev_c00[pi], ev_c10[pi], ev_c11[pi]
                c20, c21, c22 = ev_c20[pi], ev_c21[pi], ev_c22[pi]
                pvs.append(PrimaryVertex(
                    pv_id=f"evt{entry_idx}_pv{pi}",
                    x=ev_pvx[pi], y=ev_pvy[pi], z=ev_pvz[pi],
                    cov3=((c00, c10, c20), (c10, c11, c21), (c20, c21, c22)),
                    time=ev_pvt[pi],
                    sigma_time=math.sqrt(max(ev_c33[pi], 0.0)),
                ))

            yield EventInput(
                event_id=event_id,
                tracks=tuple(tracks),
                primary_vertices=tuple(pvs),
            )

        global_idx += n_chunk


def load_events_root(
    path: str | Path,
    tree_name: str = "BestLongTracks/TrackTuple",
    max_events: int | None = None,
    track_type: str = "long",
) -> list[EventInput]:
    """Load tracks and PVs from a ROOT TTree in a single call.

    Returns a list of ``EventInput`` objects with aligned tracks and PVs.
    For large files, prefer ``iter_events_root()`` which streams events
    and uses bounded memory.
    """
    return list(iter_events_root(path, tree_name, max_events=max_events, track_type=track_type))


# -- ROOT helpers --------------------------------------------------------


def _require_uproot():
    """Import uproot and awkward lazily."""
    try:
        import uproot  # type: ignore
        import awkward as ak  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "uproot and awkward are required to read ROOT files. "
            "Install them with: pip install 'track-combination-framework[root]'"
        ) from exc
    return uproot, ak


_C_LIGHT_MM_PER_NS = 299.792458
_PION_MASS_MEV = 139.57039


def _fit_track_t0(
    z_first: float,
    tx: float,
    ty: float,
    p: float,
    hit_z: Sequence[float],
    hit_t: Sequence[float],
    mass: float = _PION_MASS_MEV,
    c: float = _C_LIGHT_MM_PER_NS,
    weights: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Fit track t0 at ``z_first`` from hit times using time-of-flight correction.

    Each hit time is propagated back to ``z_first`` via
    ``t0_i = t_hit_i - path_i / (beta * c)`` where path uses the track slopes
    and beta is computed under the given mass hypothesis.

    Parameters
    ----------
    weights : optional
        Per-hit weights for the average. ``None`` uses equal weights.

    Returns ``(time, sigma_time)``.
    """
    if not hit_z or not hit_t:
        return 0.0, 1e9

    slope_factor = math.sqrt(1.0 + tx * tx + ty * ty)
    energy = math.sqrt(p * p + mass * mass)
    beta = p / energy if energy > 0.0 else 1.0
    beta_c = beta * c
    if beta_c <= 0.0:
        return 0.0, 1e9

    t0_values: list[float] = []
    w_values: list[float] = []
    for i, (hz, ht) in enumerate(zip(hit_z, hit_t)):
        if math.isnan(ht) or math.isnan(hz):
            continue
        path = (hz - z_first) * slope_factor
        t0 = ht - path / beta_c
        t0_values.append(t0)
        w_values.append(weights[i] if weights is not None else 1.0)

    n = len(t0_values)
    if n == 0:
        return 0.0, 1e9

    sum_w = sum(w_values)
    if sum_w <= 0.0:
        return 0.0, 1e9

    t0_mean = sum(w * t for w, t in zip(w_values, t0_values)) / sum_w

    if n == 1:
        return t0_mean, 1e9

    # Standard error of the (weighted) mean
    var = sum(w * (t - t0_mean) ** 2 for w, t in zip(w_values, t0_values)) / sum_w
    sigma = math.sqrt(var / n)
    return t0_mean, max(sigma, 1e-12)


def _build_sym_cov4(cov_vals: dict[tuple[int, int], float]) -> Matrix4x4:
    """Build symmetric 4x4 covariance from lower-triangular elements (indices 0..3)."""
    def _get(i: int, j: int) -> float:
        if i >= j:
            return cov_vals.get((i, j), 0.0)
        return cov_vals.get((j, i), 0.0)

    return (
        (_get(0, 0), _get(0, 1), _get(0, 2), _get(0, 3)),
        (_get(1, 0), _get(1, 1), _get(1, 2), _get(1, 3)),
        (_get(2, 0), _get(2, 1), _get(2, 2), _get(2, 3)),
        (_get(3, 0), _get(3, 1), _get(3, 2), _get(3, 3)),
    )


def _require_pandas():
    """Import pandas lazily and provide a clear installation hint on failure."""
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required to write output tables. Install pandas and pyarrow."
        ) from exc
    return pd


def _parse_track_item(item: Any, idx: int, context: str) -> TrackState:
    """Parse one track dictionary into a `TrackState`."""
    if not isinstance(item, dict):
        raise ValueError(f"Track entry at index {idx} in {context} must be an object.")
    source_ids_raw = item.get("source_track_ids")
    if source_ids_raw is None:
        source_ids = (str(item["track_id"]),)
    else:
        if not isinstance(source_ids_raw, list):
            raise ValueError("Track field 'source_track_ids' must be a list of strings.")
        source_ids = tuple(str(x) for x in source_ids_raw)
    state = item.get("state", item)
    if not isinstance(state, dict):
        raise ValueError(f"Track state at index {idx} in {context} must be an object.")
    if "time" in state:
        time_value = state["time"]
    elif "t" in item:
        time_value = item["t"]
    elif "time" in item:
        time_value = item["time"]
    else:
        raise ValueError(f"Track at index {idx} in {context} must define a time field.")
    return TrackState(
        track_id=str(item["track_id"]),
        z=float(item["z"]),
        x=float(state["x"]),
        y=float(state["y"]),
        tx=float(state["tx"]),
        ty=float(state["ty"]),
        time=float(time_value),
        cov4=_parse_cov4(item["cov4"]),
        sigma_time=float(item.get("sigma_time", item.get("sigma_t", 1.0))),
        p=float(item["p"]),
        charge=int(item.get("charge", 0)),
        has_rich1=bool(item.get("hasRICH1", False)),
        has_rich2=bool(item.get("hasRICH2", False)),
        rich_dll_pi=float(item.get("richDLL_pi", 0.0)),
        rich_dll_k=float(item.get("richDLL_k", 0.0)),
        rich_dll_p=float(item.get("richDLL_p", 0.0)),
        rich_dll_e=float(item.get("richDLL_e", 0.0)),
        has_calo=bool(item.get("hasCALO", False)),
        calo_dll_e=float(item.get("caloDLL_e", 0.0)),
        source_track_ids=source_ids,
    )


def _parse_primary_vertex_item(item: Any, idx: int, context: str) -> PrimaryVertex:
    """Parse one PV dictionary into a `PrimaryVertex`."""
    if not isinstance(item, dict):
        raise ValueError(f"Primary vertex at index {idx} in {context} must be an object.")
    return PrimaryVertex(
        pv_id=str(item.get("pv_id", f"pv{idx}")),
        x=float(item["x"]),
        y=float(item["y"]),
        z=float(item["z"]),
        cov3=_parse_cov3(item["cov3"]),
        time=float(item["time"]),
        sigma_time=float(item["sigma_time"]),
    )


def _parse_mass_entry(entry: Any) -> float | ParticleHypothesis:
    """Parse one mass-hypothesis entry."""
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, str):
        return particle_hypothesis_from_name(entry)
    if isinstance(entry, dict):
        # Accept shorthand aliases first, then explicit custom masses.
        if "pid" in entry:
            return particle_hypothesis_from_name(str(entry["pid"]))
        if "particle" in entry:
            return particle_hypothesis_from_name(str(entry["particle"]))
        if "mass" not in entry:
            raise ValueError("Mass hypothesis object must define 'mass', 'pid', or 'particle'.")
        mass = float(entry["mass"])
        name = str(entry.get("name", f"m={mass:g}"))
        pdg_id = entry.get("pdg_id")
        parsed_pdg = int(pdg_id) if pdg_id is not None else None
        return ParticleHypothesis(name=name, mass=mass, pdg_id=parsed_pdg)
    raise ValueError(
        f"Unsupported mass hypothesis entry {entry!r}. Use number, string, or object."
    )


def _extract_primary_vertices_payload(
    data: dict[str, Any],
    allow_object_fallback: bool = True,
) -> list[Any]:
    """Extract a PV list from an event/object payload."""
    pvs_data = data.get("primary_vertices", data.get("pvs"))
    if pvs_data is None:
        if not allow_object_fallback:
            raise ValueError("Event payload must contain 'primary_vertices' (or 'pvs') list.")
        pv_data = data.get("primary_vertex", data)
        if not isinstance(pv_data, dict):
            raise ValueError("Primary vertex JSON must contain 'primary_vertices' list or single object.")
        pvs_data = [pv_data]
    if not isinstance(pvs_data, list):
        raise ValueError("Primary vertex JSON key 'primary_vertices' must be a list.")
    return pvs_data


def _parse_cov4(value: Any):
    """Validate and convert a nested list into a 4x4 covariance tuple."""
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Track cov4 must be a 4x4 list.")
    rows: list[tuple[float, float, float, float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("Track cov4 must be a 4x4 list.")
        rows.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
    return (rows[0], rows[1], rows[2], rows[3])


def _parse_cov3(value: Any):
    """Validate and convert a nested list into a 3x3 covariance tuple."""
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("Primary vertex cov3 must be a 3x3 list.")
    rows: list[tuple[float, float, float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("Primary vertex cov3 must be a 3x3 list.")
        rows.append((float(row[0]), float(row[1]), float(row[2])))
    return (rows[0], rows[1], rows[2])


def _load_json(path: str | Path) -> dict[str, Any]:
    """Read and validate a JSON object document from disk."""
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"JSON document at {path} must be an object.")
    return data
