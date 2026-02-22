"""Input/output helpers for JSON inputs and tabular result export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import CombinationResult, PrimaryVertex, TrackState


def load_tracks_json(path: str | Path) -> list[TrackState]:
    """Load track container JSON into `TrackState` objects."""
    data = _load_json(path)
    tracks_data = data.get("tracks")
    if not isinstance(tracks_data, list):
        raise ValueError("Input JSON must contain a list under key 'tracks'.")
    tracks: list[TrackState] = []
    for idx, item in enumerate(tracks_data):
        if not isinstance(item, dict):
            raise ValueError(f"Track entry at index {idx} must be an object.")
        source_ids_raw = item.get("source_track_ids")
        if source_ids_raw is None:
            source_ids = (str(item["track_id"]),)
        else:
            if not isinstance(source_ids_raw, list):
                raise ValueError("Track field 'source_track_ids' must be a list of strings.")
            source_ids = tuple(str(x) for x in source_ids_raw)
        tracks.append(
            TrackState(
                track_id=str(item["track_id"]),
                z=float(item["z"]),
                x=float(item["state"]["x"] if "state" in item else item["x"]),
                y=float(item["state"]["y"] if "state" in item else item["y"]),
                tx=float(item["state"]["tx"] if "state" in item else item["tx"]),
                ty=float(item["state"]["ty"] if "state" in item else item["ty"]),
                time=float(item["state"]["time"] if "state" in item else item.get("t", item["time"])),
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
        )
    return tracks


def load_primary_vertices_json(path: str | Path) -> list[PrimaryVertex]:
    """Load PV container JSON into `PrimaryVertex` objects.

    Supports both:
    - modern list format: `primary_vertices: [...]`
    - backward-compatible single PV object.
    """
    data = _load_json(path)
    pvs_data = data.get("primary_vertices")
    if pvs_data is None:
        # backward compatibility with single-PV JSON
        pv_data = data.get("primary_vertex", data)
        if not isinstance(pv_data, dict):
            raise ValueError("Primary vertex JSON must contain 'primary_vertices' list or single object.")
        pvs_data = [pv_data]
    if not isinstance(pvs_data, list):
        raise ValueError("Primary vertex JSON key 'primary_vertices' must be a list.")
    out: list[PrimaryVertex] = []
    for idx, pv_data in enumerate(pvs_data):
        if not isinstance(pv_data, dict):
            raise ValueError(f"Primary vertex at index {idx} must be an object.")
        out.append(
            PrimaryVertex(
                pv_id=str(pv_data.get("pv_id", f"pv{idx}")),
                x=float(pv_data["x"]),
                y=float(pv_data["y"]),
                z=float(pv_data["z"]),
                cov3=_parse_cov3(pv_data["cov3"]),
                time=float(pv_data["time"]),
                sigma_time=float(pv_data["sigma_time"]),
            )
        )
    return out


def load_mass_hypotheses_json(path: str | Path) -> list[tuple[float, ...]]:
    """Load mass-hypothesis sets from JSON."""
    data = _load_json(path)
    masses_data = data.get("mass_hypotheses")
    if not isinstance(masses_data, list):
        raise ValueError("Mass JSON must contain a list under key 'mass_hypotheses'.")
    parsed: list[tuple[float, ...]] = []
    for idx, hyp in enumerate(masses_data):
        if not isinstance(hyp, list):
            raise ValueError(f"Mass hypothesis at index {idx} must be a list.")
        parsed.append(tuple(float(x) for x in hyp))
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
            "track_ids": ",".join(res.track_ids),
            "source_track_ids": ",".join(res.source_track_ids),
            "masses": ",".join(str(x) for x in res.masses),
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
