"""End-to-end synthetic walkthrough for B -> J/psi(mu mu) K*(K pi) studies.

This script does three steps:
1. Generate a fake event sample with configurable signal fraction.
2. Build staged combiners (J/psi, K*, then B) on each event.
3. Write analysis tables (pandas DataFrame) for downstream plotting/studies.

Defaults are set to the requested scenario:
- 1000 events
- 20% events with truth B decay
- B candidate output window in 5000-6000 MeV (5.0-6.0 GeV)

Run from repository root:
    PYTHONPATH=src python3 examples/b_jpsi_kstar_fake_and_combine.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

from trackcomb import (
    CombinationCuts,
    EventInput,
    ParticleCombiner,
    ParticleHypothesis,
    PrimaryVertex,
    TrackPreselection,
    TrackState,
    combination_to_track_state,
    make_kaon,
    make_muon,
    make_pion,
)

MASS_MU = 0.1056583755
MASS_PI = 0.13957039
MASS_K = 0.493677
MASS_JPSI = 3.0969
MASS_KSTAR0 = 0.89555
MASS_B0 = 5.27965
WIDTH_KSTAR0 = 0.0473

C_MM_PER_NS = 299.792458
MEV_PER_GEV = 1000.0


@dataclass(frozen=True)
class TruthInfo:
    """Truth labels for one generated event."""

    has_signal: bool
    truth_b_track_ids: tuple[str, ...]
    truth_jpsi_track_ids: tuple[str, ...]
    truth_kstar_track_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    """Parse CLI options for fake-data generation and staged combining."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic B->J/psi K* sample and run staged combiners."
    )
    parser.add_argument("--n-events", type=int, default=1000, help="Number of events to generate.")
    parser.add_argument(
        "--signal-fraction",
        type=float,
        default=0.20,
        help="Fraction of events containing one truth B decay.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed for reproducibility.")
    parser.add_argument(
        "--out-events",
        default="examples/output_bjpsikstar_events.json",
        help="Output JSON with generated events.",
    )
    parser.add_argument(
        "--out-truth",
        default="examples/output_bjpsikstar_truth.json",
        help="Output JSON with event-level truth labels.",
    )
    parser.add_argument(
        "--out-candidates",
        default="examples/output_bjpsikstar_candidates.parquet",
        help="Output candidates table (.parquet/.csv/.pkl).",
    )
    return parser.parse_args()


def _require_pandas():
    """Import pandas with an actionable install hint."""
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required for this walkthrough. Install pandas and pyarrow."
        ) from exc
    return pd


def random_unit_vector(rng: Random) -> tuple[float, float, float]:
    """Sample an isotropic 3D unit vector."""
    cos_theta = rng.uniform(-1.0, 1.0)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = rng.uniform(0.0, 2.0 * math.pi)
    return sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta


def two_body_momentum(parent_mass: float, m1: float, m2: float) -> float:
    """Return daughter momentum magnitude in parent rest frame."""
    term = (parent_mass * parent_mass - (m1 + m2) * (m1 + m2)) * (
        parent_mass * parent_mass - (m1 - m2) * (m1 - m2)
    )
    if term <= 0.0:
        return 0.0
    return math.sqrt(term) / (2.0 * parent_mass)


def invariant_mass(p4: tuple[float, float, float, float]) -> float:
    """Compute invariant mass from `(E, px, py, pz)`."""
    e, px, py, pz = p4
    m2 = e * e - (px * px + py * py + pz * pz)
    return math.sqrt(m2) if m2 > 0.0 else 0.0


def lorentz_boost(
    p4: tuple[float, float, float, float],
    beta: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Boost a four-vector by beta vector."""
    e, px, py, pz = p4
    bx, by, bz = beta
    b2 = bx * bx + by * by + bz * bz
    if b2 <= 0.0:
        return p4
    gamma = 1.0 / math.sqrt(max(1e-16, 1.0 - b2))
    bp = bx * px + by * py + bz * pz
    gamma2 = (gamma - 1.0) / b2
    px_out = px + gamma2 * bp * bx + gamma * e * bx
    py_out = py + gamma2 * bp * by + gamma * e * by
    pz_out = pz + gamma2 * bp * bz + gamma * e * bz
    e_out = gamma * (e + bp)
    return e_out, px_out, py_out, pz_out


def decay_two_body(
    parent_p4: tuple[float, float, float, float],
    m1: float,
    m2: float,
    rng: Random,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Generate a two-body decay and return daughter four-vectors in lab frame."""
    parent_mass = invariant_mass(parent_p4)
    p = two_body_momentum(parent_mass, m1, m2)
    u = random_unit_vector(rng)
    e1 = math.sqrt(m1 * m1 + p * p)
    e2 = math.sqrt(m2 * m2 + p * p)
    d1_rest = (e1, p * u[0], p * u[1], p * u[2])
    d2_rest = (e2, -p * u[0], -p * u[1], -p * u[2])
    e_parent, px_parent, py_parent, pz_parent = parent_p4
    beta = (
        px_parent / e_parent,
        py_parent / e_parent,
        pz_parent / e_parent,
    )
    return lorentz_boost(d1_rest, beta), lorentz_boost(d2_rest, beta)


def sample_kstar_mass(rng: Random) -> float:
    """Sample a simple truncated K* mass around nominal value."""
    while True:
        m = rng.gauss(MASS_KSTAR0, WIDTH_KSTAR0 / 2.0)
        if MASS_K + MASS_PI + 0.01 <= m <= 1.15:
            return m


def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp scalar value to `[lo, hi]`."""
    return max(lo, min(hi, v))


def _species_pid(species: str, rng: Random) -> dict[str, float | bool]:
    """Return PID-like observables for generated species labels."""
    if species == "mu":
        return {
            "hasRICH1": bool(rng.random() < 0.1),
            "hasRICH2": bool(rng.random() < 0.1),
            "richDLL_pi": rng.gauss(-1.0, 1.0),
            "richDLL_k": rng.gauss(-2.0, 1.0),
            "richDLL_p": rng.gauss(-2.0, 1.0),
            "richDLL_e": rng.gauss(-1.5, 1.2),
            "hasCALO": True,
            "caloDLL_e": rng.gauss(-4.0, 1.5),
        }
    if species == "k":
        return {
            "hasRICH1": True,
            "hasRICH2": True,
            "richDLL_pi": rng.gauss(-1.0, 1.2),
            "richDLL_k": rng.gauss(5.0, 1.2),
            "richDLL_p": rng.gauss(0.2, 1.3),
            "richDLL_e": rng.gauss(-3.0, 1.3),
            "hasCALO": bool(rng.random() < 0.3),
            "caloDLL_e": rng.gauss(-3.0, 1.5),
        }
    if species == "pi":
        return {
            "hasRICH1": True,
            "hasRICH2": bool(rng.random() < 0.6),
            "richDLL_pi": rng.gauss(4.5, 1.1),
            "richDLL_k": rng.gauss(-1.2, 1.2),
            "richDLL_p": rng.gauss(-2.0, 1.5),
            "richDLL_e": rng.gauss(-1.0, 1.4),
            "hasCALO": bool(rng.random() < 0.4),
            "caloDLL_e": rng.gauss(-2.5, 1.8),
        }
    return {
        "hasRICH1": bool(rng.random() < 0.5),
        "hasRICH2": bool(rng.random() < 0.5),
        "richDLL_pi": rng.gauss(0.0, 3.0),
        "richDLL_k": rng.gauss(0.0, 3.0),
        "richDLL_p": rng.gauss(0.0, 3.0),
        "richDLL_e": rng.gauss(0.0, 3.0),
        "hasCALO": bool(rng.random() < 0.5),
        "caloDLL_e": rng.gauss(0.0, 4.0),
    }


def build_track_from_truth_particle(
    event_id: str,
    label: str,
    particle_p4: tuple[float, float, float, float],
    sv_xyz: tuple[float, float, float],
    sv_time: float,
    charge: int,
    species: str,
    rng: Random,
) -> TrackState:
    """Create a `TrackState` for one truth daughter from a generated decay."""
    _, px, py, pz = particle_p4
    p = max(0.5, math.sqrt(px * px + py * py + pz * pz) * rng.gauss(1.0, 0.01))
    safe_pz = pz if abs(pz) > 0.2 else (0.2 if pz >= 0.0 else -0.2)
    tx = _clamp(px / safe_pz + rng.gauss(0.0, 0.001), -0.6, 0.6)
    ty = _clamp(py / safe_pz + rng.gauss(0.0, 0.001), -0.6, 0.6)
    pid = _species_pid(species, rng)
    return TrackState(
        track_id=f"{event_id}_{label}",
        z=sv_xyz[2],
        x=sv_xyz[0] + rng.gauss(0.0, 0.02),
        y=sv_xyz[1] + rng.gauss(0.0, 0.02),
        tx=tx,
        ty=ty,
        time=sv_time + rng.gauss(0.0, 0.03),
        cov4=((0.01, 0.0, 0.0, 0.0), (0.0, 0.01, 0.0, 0.0), (0.0, 0.0, 0.0002, 0.0), (0.0, 0.0, 0.0, 0.0002)),
        sigma_time=0.04,
        p=p,
        charge=charge,
        has_rich1=bool(pid["hasRICH1"]),
        has_rich2=bool(pid["hasRICH2"]),
        rich_dll_pi=float(pid["richDLL_pi"]),
        rich_dll_k=float(pid["richDLL_k"]),
        rich_dll_p=float(pid["richDLL_p"]),
        rich_dll_e=float(pid["richDLL_e"]),
        has_calo=bool(pid["hasCALO"]),
        calo_dll_e=float(pid["caloDLL_e"]),
        source_track_ids=(f"{event_id}_{label}",),
    )


def build_background_track(
    event_id: str,
    index: int,
    pv: PrimaryVertex,
    rng: Random,
) -> TrackState:
    """Create one random background-like track."""
    species = rng.choice(("k", "pi", "x"))
    pid = _species_pid(species, rng)
    tx = _clamp(rng.gauss(0.0, 0.15), -0.8, 0.8)
    ty = _clamp(rng.gauss(0.0, 0.15), -0.8, 0.8)
    return TrackState(
        track_id=f"{event_id}_bg_{index:02d}",
        z=pv.z + rng.gauss(0.0, 2.5),
        x=pv.x + rng.gauss(0.0, 0.4),
        y=pv.y + rng.gauss(0.0, 0.4),
        tx=tx,
        ty=ty,
        time=pv.time + rng.gauss(0.0, 0.25),
        cov4=((0.03, 0.0, 0.0, 0.0), (0.0, 0.03, 0.0, 0.0), (0.0, 0.0, 0.001, 0.0), (0.0, 0.0, 0.0, 0.001)),
        sigma_time=0.08,
        p=rng.uniform(1.0, 45.0),
        charge=rng.choice((-1, 1)),
        has_rich1=bool(pid["hasRICH1"]),
        has_rich2=bool(pid["hasRICH2"]),
        rich_dll_pi=float(pid["richDLL_pi"]),
        rich_dll_k=float(pid["richDLL_k"]),
        rich_dll_p=float(pid["richDLL_p"]),
        rich_dll_e=float(pid["richDLL_e"]),
        has_calo=bool(pid["hasCALO"]),
        calo_dll_e=float(pid["caloDLL_e"]),
        source_track_ids=(f"{event_id}_bg_{index:02d}",),
    )


def sample_primary_vertex(event_id: str, rng: Random) -> PrimaryVertex:
    """Sample one primary vertex for a fake event."""
    return PrimaryVertex(
        pv_id=f"{event_id}_pv0",
        x=rng.gauss(0.0, 0.03),
        y=rng.gauss(0.0, 0.03),
        z=100.0 + rng.gauss(0.0, 0.5),
        cov3=((0.0025, 0.0, 0.0), (0.0, 0.0025, 0.0), (0.0, 0.0, 0.01)),
        time=rng.gauss(0.0, 0.05),
        sigma_time=0.04,
    )


def generate_signal_tracks(
    event_id: str,
    pv: PrimaryVertex,
    rng: Random,
) -> tuple[list[TrackState], TruthInfo]:
    """Generate one truth B -> J/psi(mu mu) K*(K pi) decay in an event."""
    pt_b = rng.uniform(3.0, 10.0)
    phi_b = rng.uniform(0.0, 2.0 * math.pi)
    pz_b = rng.uniform(20.0, 65.0)
    px_b = pt_b * math.cos(phi_b)
    py_b = pt_b * math.sin(phi_b)
    p_b = math.sqrt(px_b * px_b + py_b * py_b + pz_b * pz_b)
    e_b = math.sqrt(MASS_B0 * MASS_B0 + p_b * p_b)
    b_p4 = (e_b, px_b, py_b, pz_b)

    kstar_mass_true = sample_kstar_mass(rng)
    jpsi_p4, kstar_p4 = decay_two_body(b_p4, MASS_JPSI, kstar_mass_true, rng)
    mu_plus_p4, mu_minus_p4 = decay_two_body(jpsi_p4, MASS_MU, MASS_MU, rng)
    k_plus_p4, pi_minus_p4 = decay_two_body(kstar_p4, MASS_K, MASS_PI, rng)

    beta_b = p_b / e_b if e_b > 0.0 else 0.0
    gamma_b = 1.0 / math.sqrt(max(1e-16, 1.0 - beta_b * beta_b))
    c_tau_b_mm = 0.455
    mean_flight_mm = c_tau_b_mm * beta_b * gamma_b
    flight_mm = rng.expovariate(1.0 / max(0.2, mean_flight_mm))
    b_dir = (px_b / p_b, py_b / p_b, pz_b / p_b)
    sv_xyz = (
        pv.x + flight_mm * b_dir[0],
        pv.y + flight_mm * b_dir[1],
        pv.z + flight_mm * b_dir[2],
    )
    sv_time = pv.time + flight_mm / max(1e-6, beta_b * C_MM_PER_NS)

    tracks = [
        build_track_from_truth_particle(event_id, "sig_muplus", mu_plus_p4, sv_xyz, sv_time, +1, "mu", rng),
        build_track_from_truth_particle(event_id, "sig_muminus", mu_minus_p4, sv_xyz, sv_time, -1, "mu", rng),
        build_track_from_truth_particle(event_id, "sig_kplus", k_plus_p4, sv_xyz, sv_time, +1, "k", rng),
        build_track_from_truth_particle(event_id, "sig_piminus", pi_minus_p4, sv_xyz, sv_time, -1, "pi", rng),
    ]
    truth = TruthInfo(
        has_signal=True,
        truth_b_track_ids=tuple(t.track_id for t in tracks),
        truth_jpsi_track_ids=(tracks[0].track_id, tracks[1].track_id),
        truth_kstar_track_ids=(tracks[2].track_id, tracks[3].track_id),
    )
    return tracks, truth


def generate_events(
    n_events: int,
    signal_fraction: float,
    rng: Random,
) -> tuple[list[EventInput], dict[str, TruthInfo]]:
    """Generate event list and truth map with requested signal fraction."""
    n_signal = int(round(n_events * signal_fraction))
    n_signal = max(0, min(n_events, n_signal))
    signal_indices = set(rng.sample(range(n_events), n_signal))

    events: list[EventInput] = []
    truth_map: dict[str, TruthInfo] = {}
    for idx in range(n_events):
        event_id = f"evt{idx:04d}"
        pv = sample_primary_vertex(event_id, rng)

        tracks: list[TrackState] = []
        if idx in signal_indices:
            sig_tracks, truth = generate_signal_tracks(event_id, pv, rng)
            tracks.extend(sig_tracks)
            truth_map[event_id] = truth
            n_bg = rng.randint(2, 6)
        else:
            truth_map[event_id] = TruthInfo(False, (), (), ())
            n_bg = rng.randint(5, 10)

        tracks.extend(build_background_track(event_id, i, pv, rng) for i in range(n_bg))
        rng.shuffle(tracks)
        events.append(
            EventInput(
                event_id=event_id,
                tracks=tuple(tracks),
                primary_vertices=(pv,),
            )
        )
    return events, truth_map


def _track_to_json(track: TrackState) -> dict[str, Any]:
    """Serialize one `TrackState` into JSON-serializable dictionary."""
    return {
        "track_id": track.track_id,
        "z": track.z,
        "state": {
            "x": track.x,
            "y": track.y,
            "tx": track.tx,
            "ty": track.ty,
            "time": track.time,
        },
        "cov4": [list(row) for row in track.cov4],
        "sigma_time": track.sigma_time,
        "p": track.p,
        "charge": track.charge,
        "hasRICH1": track.has_rich1,
        "hasRICH2": track.has_rich2,
        "richDLL_pi": track.rich_dll_pi,
        "richDLL_k": track.rich_dll_k,
        "richDLL_p": track.rich_dll_p,
        "richDLL_e": track.rich_dll_e,
        "hasCALO": track.has_calo,
        "caloDLL_e": track.calo_dll_e,
        "source_track_ids": list(track.source_track_ids),
    }


def _pv_to_json(pv: PrimaryVertex) -> dict[str, Any]:
    """Serialize one `PrimaryVertex` into JSON-serializable dictionary."""
    return {
        "pv_id": pv.pv_id,
        "x": pv.x,
        "y": pv.y,
        "z": pv.z,
        "cov3": [list(row) for row in pv.cov3],
        "time": pv.time,
        "sigma_time": pv.sigma_time,
    }


def write_events_json(path: str, events: list[EventInput]) -> None:
    """Write generated events into framework `events` JSON schema."""
    data = {
        "events": [
            {
                "event_id": event.event_id,
                "tracks": [_track_to_json(t) for t in event.tracks],
                "primary_vertices": [_pv_to_json(pv) for pv in event.primary_vertices],
            }
            for event in events
        ]
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_truth_json(path: str, truth_map: dict[str, TruthInfo]) -> None:
    """Write event-level truth bookkeeping for downstream candidate labeling."""
    data = {
        event_id: {
            "has_signal": truth.has_signal,
            "truth_b_track_ids": list(truth.truth_b_track_ids),
            "truth_jpsi_track_ids": list(truth.truth_jpsi_track_ids),
            "truth_kstar_track_ids": list(truth.truth_kstar_track_ids),
        }
        for event_id, truth in truth_map.items()
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_composite_tracks(
    results: list,
    base_tracks: list[TrackState],
    prefix: str,
) -> list[TrackState]:
    """Convert accepted combinations into composite track-like objects."""
    track_map = {t.track_id: t for t in base_tracks}
    out: list[TrackState] = []
    for idx, res in enumerate(results):
        constituents = [track_map[tid] for tid in res.track_ids]
        out.append(combination_to_track_state(res, constituents, f"{prefix}_{idx}"))
    return out


def _is_truth_candidate(stage: str, source_track_ids: tuple[str, ...], truth: TruthInfo) -> bool:
    """Return truth label for one stage-specific candidate."""
    if not truth.has_signal:
        return False
    source = set(source_track_ids)
    if stage == "JPSI":
        return source == set(truth.truth_jpsi_track_ids)
    if stage == "KSTAR":
        return source == set(truth.truth_kstar_track_ids)
    if stage == "B":
        return source == set(truth.truth_b_track_ids)
    return False


def _candidate_row(
    result,
    stage: str,
    truth: TruthInfo,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one `CombinationResult` into a DataFrame row."""
    row: dict[str, Any] = {
        "event_id": result.event_id,
        "stage": stage,
        "candidate_mass_gev": result.candidate_p4.mass,
        "candidate_mass_mev": result.candidate_p4.mass * MEV_PER_GEV,
        "vertex_chi2": result.vertex_chi2,
        "vertex_time_chi2": result.vertex_time_chi2,
        "pair_time_chi2": result.pair_time_chi2,
        "pair_pt": result.pair_pt,
        "pair_eta": result.pair_eta,
        "charge_pattern": result.charge_pattern,
        "track_ids": ",".join(result.track_ids),
        "source_track_ids": ",".join(result.source_track_ids),
        "particle_hypotheses": ",".join(result.particle_hypotheses),
        "best_pv_id": result.best_pv_id,
        "has_signal_event": truth.has_signal,
        "is_truth": _is_truth_candidate(stage, result.source_track_ids, truth),
    }
    if extra:
        row.update(extra)
    return row


def run_staged_combiners(events: list[EventInput], truth_map: dict[str, TruthInfo]):
    """Run J/psi, K*, and B staged combinations for all events."""
    rows: list[dict[str, Any]] = []
    combiner = ParticleCombiner()
    jpsi_hyp = [[make_muon(), make_muon()]]
    kstar_hyp = [[make_kaon(), make_pion()], [make_pion(), make_kaon()]]
    b_hyp = [[ParticleHypothesis(name="J/psi", mass=MASS_JPSI), ParticleHypothesis(name="K*", mass=MASS_KSTAR0)]]

    for event in events:
        tracks = list(event.tracks)
        pvs = list(event.primary_vertices)
        truth = truth_map[event.event_id]

        jpsi_results = combiner.combine(
            tracks=tracks,
            primary_vertices=pvs,
            n_body=2,
            mass_hypotheses=jpsi_hyp,
            preselection=TrackPreselection(min_pt=0.3),
            cuts=CombinationCuts(
                allowed_charge_patterns=("+-", "-+"),
                max_doca=2.5,
                max_vertex_chi2=100.0,
                max_pair_time_chi2=40.0,
                min_mass=2.90,
                max_mass=3.30,
            ),
            event_id=event.event_id,
        )
        jpsi_results = sorted(jpsi_results, key=lambda r: abs(r.candidate_p4.mass - MASS_JPSI))[:12]
        for res in jpsi_results:
            rows.append(_candidate_row(res, "JPSI", truth))
        jpsi_tracks = _build_composite_tracks(jpsi_results, tracks, f"{event.event_id}_jpsi")

        kstar_results = combiner.combine(
            tracks=tracks,
            primary_vertices=pvs,
            n_body=2,
            mass_hypotheses=kstar_hyp,
            preselection=TrackPreselection(min_pt=0.3),
            cuts=CombinationCuts(
                allowed_charge_patterns=("+-", "-+"),
                max_doca=2.5,
                max_vertex_chi2=120.0,
                max_pair_time_chi2=45.0,
                min_mass=0.78,
                max_mass=1.02,
            ),
            event_id=event.event_id,
        )
        kstar_results = sorted(kstar_results, key=lambda r: abs(r.candidate_p4.mass - MASS_KSTAR0))[:20]
        for res in kstar_results:
            rows.append(_candidate_row(res, "KSTAR", truth))
        kstar_tracks = _build_composite_tracks(kstar_results, tracks, f"{event.event_id}_kstar")

        for jres, jtrack in zip(jpsi_results, jpsi_tracks, strict=True):
            for kres, ktrack in zip(kstar_results, kstar_tracks, strict=True):
                b_results = combiner.combine(
                    tracks=[jtrack, ktrack],
                    primary_vertices=pvs,
                    n_body=2,
                    mass_hypotheses=b_hyp,
                    cuts=CombinationCuts(
                        allowed_charge_patterns=("00",),
                        max_doca=2.5,
                        max_vertex_chi2=120.0,
                        max_pair_time_chi2=45.0,
                        min_mass=5.0,
                        max_mass=6.0,
                    ),
                    event_id=event.event_id,
                )
                for bres in b_results:
                    rows.append(
                        _candidate_row(
                            bres,
                            "B",
                            truth,
                            extra={
                                "jpsi_mass_mev": jres.candidate_p4.mass * MEV_PER_GEV,
                                "kstar_mass_mev": kres.candidate_p4.mass * MEV_PER_GEV,
                                "jpsi_is_truth": _is_truth_candidate("JPSI", jres.source_track_ids, truth),
                                "kstar_is_truth": _is_truth_candidate("KSTAR", kres.source_track_ids, truth),
                            },
                        )
                    )

    pd = _require_pandas()
    return pd.DataFrame(rows)


def write_dataframe(path: str, df) -> None:
    """Write DataFrame to parquet/csv/pickle according to file suffix."""
    out = Path(path)
    suffix = out.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(out, index=False)
    elif suffix == ".csv":
        df.to_csv(out, index=False)
    elif suffix in (".pkl", ".pickle"):
        df.to_pickle(out)
    else:
        raise ValueError("Use .parquet, .csv, or .pkl extension for output table.")


def summarize_dataframe(df) -> dict[str, Any]:
    """Build a compact summary for quick terminal inspection."""
    by_stage = df.groupby("stage").size().to_dict()
    truth_by_stage = df[df["is_truth"]].groupby("stage").size().to_dict()
    b_df = df[df["stage"] == "B"]
    b_window = b_df[(b_df["candidate_mass_mev"] >= 5000.0) & (b_df["candidate_mass_mev"] <= 6000.0)]
    return {
        "n_rows_total": int(len(df)),
        "n_rows_by_stage": {k: int(v) for k, v in by_stage.items()},
        "n_truth_rows_by_stage": {k: int(v) for k, v in truth_by_stage.items()},
        "b_rows_in_5000_6000_mev": int(len(b_window)),
        "b_truth_rows_in_5000_6000_mev": int(b_window["is_truth"].sum()) if len(b_window) else 0,
    }


def main() -> int:
    """Generate fake data, run staged combiners, and save DataFrame outputs."""
    args = parse_args()
    rng = Random(args.seed)

    events, truth_map = generate_events(args.n_events, args.signal_fraction, rng)
    write_events_json(args.out_events, events)
    write_truth_json(args.out_truth, truth_map)

    df = run_staged_combiners(events, truth_map)
    write_dataframe(args.out_candidates, df)

    summary = summarize_dataframe(df)
    summary_path = str(Path(args.out_candidates).with_suffix(".summary.json"))
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Synthetic walkthrough completed.")
    print(f"Events JSON: {args.out_events}")
    print(f"Truth JSON: {args.out_truth}")
    print(f"Candidates table: {args.out_candidates}")
    print(f"Summary JSON: {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
