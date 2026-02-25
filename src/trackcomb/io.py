"""Input/output helpers for JSON inputs and tabular result export."""

from __future__ import annotations
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


import json
from pathlib import Path
from typing import Any

from .models import CombinationResult, EventInput, ParticleHypothesis, PrimaryVertex, TrackState
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
