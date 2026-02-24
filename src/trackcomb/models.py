"""Core data models used by the particle-combination framework.

This module defines:
- immutable physics objects (`TrackState`, `PrimaryVertex`, `LorentzVector`)
- event containers (`EventInput`)
- particle-mass assignment objects (`ParticleHypothesis`)
- combination outputs (`CombinationResult`)
- configurable filtering controls (`TrackPreselection`, `CombinationCuts`)
- helper iterator for n-body combinatorics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

Matrix2x2 = tuple[tuple[float, float], tuple[float, float]]
Matrix3x3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Matrix4x4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True)
class TrackState:
    """Single reconstructed track with kinematics, timing, covariance, and PID extras.

    The state is parameterized at a reference z plane as:
    `(x, y, tx=dx/dz, ty=dy/dz, time)`.
    `cov4` stores the covariance for `(x, y, tx, ty)`.
    """

    track_id: str
    z: float
    x: float
    y: float
    tx: float  # dx/dz
    ty: float  # dy/dz
    time: float
    cov4: Matrix4x4
    sigma_time: float
    p: float   # momentum magnitude
    charge: int = 0
    has_rich1: bool = False
    has_rich2: bool = False
    rich_dll_pi: float = 0.0
    rich_dll_k: float = 0.0
    rich_dll_p: float = 0.0
    rich_dll_e: float = 0.0
    has_calo: bool = False
    calo_dll_e: float = 0.0
    source_track_ids: tuple[str, ...] = ()

    def extrapolate(self, z_target: float) -> tuple[float, float]:
        """Linearly extrapolate x/y to a target z coordinate."""
        dz = z_target - self.z
        return self.x + self.tx * dz, self.y + self.ty * dz

    def extrapolate_xy_cov(self, z_target: float) -> tuple[tuple[float, float], Matrix2x2]:
        """Extrapolate x/y and propagate 2x2 covariance at target z."""
        dz = z_target - self.z
        x, y = self.extrapolate(z_target)
        c = self.cov4
        var_x = c[0][0] + 2.0 * dz * c[0][2] + (dz * dz) * c[2][2]
        var_y = c[1][1] + 2.0 * dz * c[1][3] + (dz * dz) * c[3][3]
        cov_xy = c[0][1] + dz * c[0][3] + dz * c[1][2] + (dz * dz) * c[2][3]
        return (x, y), ((var_x, cov_xy), (cov_xy, var_y))

    def direction(self) -> tuple[float, float, float]:
        """Return normalized 3D direction from slopes `(tx, ty, 1)`."""
        norm = (1.0 + self.tx * self.tx + self.ty * self.ty) ** 0.5
        return self.tx / norm, self.ty / norm, 1.0 / norm

    @property
    def pt(self) -> float:
        """Transverse momentum derived from momentum magnitude and direction."""
        d = self.direction()
        return self.p * (d[0] * d[0] + d[1] * d[1]) ** 0.5

    @property
    def eta(self) -> float:
        """Pseudorapidity computed from direction."""
        d = self.direction()
        pz = d[2]
        p = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) ** 0.5
        if p == abs(pz):
            return 1e9 if pz >= 0 else -1e9
        return 0.5 * math.log((p + pz) / (p - pz))


@dataclass(frozen=True)
class PrimaryVertex:
    """Primary-vertex hypothesis for one event."""

    pv_id: str
    x: float
    y: float
    z: float
    cov3: Matrix3x3
    time: float
    sigma_time: float


@dataclass(frozen=True)
class EventInput:
    """One event payload with its own track list and primary-vertex list."""

    event_id: str
    tracks: tuple[TrackState, ...]
    primary_vertices: tuple[PrimaryVertex, ...]


@dataclass(frozen=True)
class ParticleHypothesis:
    """Named particle hypothesis used to derive mass-dependent observables."""

    name: str
    mass: float
    pdg_id: int | None = None


@dataclass(frozen=True)
class LorentzVector:
    """Simple 4-vector with convenience properties and addition."""

    px: float
    py: float
    pz: float
    e: float

    def __add__(self, other: "LorentzVector") -> "LorentzVector":
        """Component-wise 4-vector addition."""
        return LorentzVector(
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz,
            self.e + other.e,
        )

    @property
    def p2(self) -> float:
        """Squared 3-momentum magnitude."""
        return self.px * self.px + self.py * self.py + self.pz * self.pz

    @property
    def mass2(self) -> float:
        """Invariant mass squared."""
        return self.e * self.e - self.p2

    @property
    def mass(self) -> float:
        """Invariant mass with signed handling for small negative mass2 values."""
        m2 = self.mass2
        return m2**0.5 if m2 >= 0.0 else -((-m2) ** 0.5)


@dataclass(frozen=True)
class CombinationResult:
    """One accepted n-body candidate with fitted vertex and observables."""

    track_ids: tuple[str, ...]
    masses: tuple[float, ...]
    particle_hypotheses: tuple[str, ...]
    vertex_xyz: tuple[float, float, float]
    vertex_cov_xyz: Matrix3x3
    vertex_time: float
    vertex_sigma_time: float
    vertices_xy: tuple[tuple[float, float], ...]
    candidate_p4: LorentzVector
    vertex_chi2: float
    vertex_time_chi2: float
    pair_time_chi2: float
    doca_pairs: dict[str, float]
    track_min_ip: dict[str, float]
    track_min_ip_chi2: dict[str, float]
    track_charges: dict[str, int]
    track_pid_info: dict[str, dict[str, float | bool]]
    charge_pattern: str
    total_charge: int
    pair_pt: float
    pair_eta: float
    source_track_ids: tuple[str, ...]
    event_id: str | None = None
    best_pv_id: str | None = None


@dataclass(frozen=True)
class TrackPreselection:
    """Track-level preselection applied before n-body combinatorics."""

    min_pt: float | None = None
    min_eta: float | None = None
    max_eta: float | None = None
    min_ip_to_any_pv: float | None = None


@dataclass(frozen=True)
class CombinationCuts:
    """Candidate-level cuts applied after building each n-body combination."""

    max_doca: float | None = None
    max_vertex_chi2: float | None = None
    max_vertex_time_chi2: float | None = None
    max_pair_time_chi2: float | None = None
    min_mass: float | None = None
    max_mass: float | None = None
    min_pair_pt: float | None = None
    max_pair_pt: float | None = None
    min_pair_eta: float | None = None
    max_pair_eta: float | None = None
    allowed_charge_patterns: tuple[str, ...] | None = None


def iter_n_body_combinations(
    tracks: Sequence[TrackState], n_body: int
) -> Iterable[tuple[TrackState, ...]]:
    """Yield track tuples for supported n-body values (2, 3, 4)."""
    if n_body not in (2, 3, 4):
        raise ValueError("Only 2-body, 3-body, and 4-body combinations are supported.")
    return combinations(tracks, n_body)
