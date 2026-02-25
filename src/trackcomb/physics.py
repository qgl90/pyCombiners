"""Physics/math helpers for building and filtering particle combinations."""

from __future__ import annotations
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import LorentzVector, Matrix3x3, PrimaryVertex, TrackState

C_LIGHT_MM_PER_NS = 299.792458


@dataclass(frozen=True)
class VertexFitResult:
    """Container for combined spatial/time vertex fit outputs."""

    vertex_xyz: tuple[float, float, float]
    vertex_time: float
    spatial_chi2: float
    time_chi2: float
    cov_xyz: Matrix3x3
    sigma_time: float


@dataclass(frozen=True)
class VertexTimeFitResult:
    """Container for mass-dependent vertex-time fit outputs."""

    vertex_time: float
    sigma_time: float
    chi2: float
    propagated_times: tuple[float, ...]


@dataclass(frozen=True)
class CompositePVAssociation:
    """Composite-to-PV association metrics for one primary-vertex candidate."""

    pv_id: str
    ip: float
    ip_chi2: float
    time_residual: float
    time_chi2: float
    flight_time: float


def track_to_lorentz(track: TrackState, mass: float) -> LorentzVector:
    """Convert a track plus mass hypothesis into a Lorentz 4-vector."""
    dx, dy, dz = track.direction()
    px = track.p * dx
    py = track.p * dy
    pz = track.p * dz
    energy = (track.p * track.p + mass * mass) ** 0.5
    return LorentzVector(px=px, py=py, pz=pz, e=energy)


def sum_lorentz(vectors: Iterable[LorentzVector]) -> LorentzVector:
    """Sum an iterable of Lorentz vectors."""
    total = LorentzVector(0.0, 0.0, 0.0, 0.0)
    for vec in vectors:
        total = total + vec
    return total


def pairwise_time_chi2(
    tracks: list[TrackState],
    masses: Sequence[float] | None = None,
    vertex_xyz: tuple[float, float, float] | None = None,
    speed_of_light: float = C_LIGHT_MM_PER_NS,
) -> float:
    """Average pairwise time chi2 over all unique track pairs.

    If `masses` and `vertex_xyz` are provided, each track time is first
    propagated to the vertex with a mass-dependent beta correction.
    """
    if len(tracks) <= 1:
        return 0.0
    if (masses is None) != (vertex_xyz is None):
        raise ValueError("Provide both masses and vertex_xyz, or neither.")
    if masses is not None and len(masses) != len(tracks):
        raise ValueError("Mass list length must match track multiplicity.")
    if masses is None:
        times = [t.time for t in tracks]
    else:
        assert vertex_xyz is not None
        times = [
            propagate_track_time_to_vertex(
                track=t,
                mass=mass,
                vertex_xyz=vertex_xyz,
                speed_of_light=speed_of_light,
            )
            for t, mass in zip(tracks, masses, strict=True)
        ]

    chi2 = 0.0
    n_pairs = 0
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            dt = times[i] - times[j]
            sigma2 = (
                tracks[i].sigma_time * tracks[i].sigma_time
                + tracks[j].sigma_time * tracks[j].sigma_time
            )
            if sigma2 <= 0.0:
                continue
            chi2 += (dt * dt) / sigma2
            n_pairs += 1
    return chi2 / n_pairs if n_pairs else 0.0


def fit_vertex_xyz_t(
    tracks: list[TrackState],
    masses: Sequence[float] | None = None,
    speed_of_light: float = C_LIGHT_MM_PER_NS,
) -> VertexFitResult:
    """
    Least-squares vertex in `(x,y,z)` plus mass-corrected time fit.

    Spatial coordinates are solved from line equations, while vertex time uses
    per-track times propagated to the fitted vertex via
    `beta = p / sqrt(p^2 + m^2)`. If `masses` is not provided, zero masses are
    used (ultra-relativistic approximation).
    """
    if masses is not None and len(masses) != len(tracks):
        raise ValueError("Mass list length must match track multiplicity.")
    mass_values = [0.0] * len(tracks) if masses is None else [float(m) for m in masses]

    if not tracks:
        time_fit = fit_vertex_time(
            tracks=[],
            masses=[],
            vertex_xyz=(0.0, 0.0, 0.0),
            speed_of_light=speed_of_light,
        )
        return VertexFitResult(
            vertex_xyz=(0.0, 0.0, 0.0),
            vertex_time=time_fit.vertex_time,
            spatial_chi2=0.0,
            time_chi2=time_fit.chi2,
            cov_xyz=((1e6, 0.0, 0.0), (0.0, 1e6, 0.0), (0.0, 0.0, 1e6)),
            sigma_time=time_fit.sigma_time,
        )
    if len(tracks) == 1:
        t = tracks[0]
        time_fit = fit_vertex_time(
            tracks=tracks,
            masses=mass_values,
            vertex_xyz=(t.x, t.y, t.z),
            speed_of_light=speed_of_light,
        )
        return VertexFitResult(
            vertex_xyz=(t.x, t.y, t.z),
            vertex_time=time_fit.vertex_time,
            spatial_chi2=0.0,
            time_chi2=time_fit.chi2,
            cov_xyz=((t.cov4[0][0], t.cov4[0][1], 0.0), (t.cov4[1][0], t.cov4[1][1], 0.0), (0.0, 0.0, 1e6)),
            sigma_time=time_fit.sigma_time,
        )

    # Normal equations for unknowns [x_v, y_v, z_v]
    ata = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    atb = [0.0, 0.0, 0.0]
    for t in tracks:
        rows = (
            ([1.0, 0.0, -t.tx], t.x - t.tx * t.z),
            ([0.0, 1.0, -t.ty], t.y - t.ty * t.z),
        )
        for a, b in rows:
            for i in range(3):
                atb[i] += a[i] * b
                for j in range(3):
                    ata[i][j] += a[i] * a[j]
    xyz = solve_3x3(ata, atb)
    cov_xyz = invert_3x3(ata)
    if xyz is None:
        xyz = (sum(t.x for t in tracks) / len(tracks), sum(t.y for t in tracks) / len(tracks), sum(t.z for t in tracks) / len(tracks))
    if cov_xyz is None:
        cov_xyz = ((1e6, 0.0, 0.0), (0.0, 1e6, 0.0), (0.0, 0.0, 1e6))
    x_v, y_v, z_v = xyz

    spatial_chi2 = 0.0
    for t in tracks:
        # Evaluate residuals in the transverse plane at the fitted z and weight
        # them with the propagated 2x2 measurement covariance.
        (x_t, y_t), cov_xy = t.extrapolate_xy_cov(z_v)
        inv = invert_2x2(cov_xy)
        dx = x_t - x_v
        dy = y_t - y_v
        if inv is None:
            spatial_chi2 += dx * dx + dy * dy
        else:
            spatial_chi2 += dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (
                inv[1][0] * dx + inv[1][1] * dy
            )

    time_fit = fit_vertex_time(
        tracks=tracks,
        masses=mass_values,
        vertex_xyz=(x_v, y_v, z_v),
        speed_of_light=speed_of_light,
    )

    return VertexFitResult(
        vertex_xyz=(x_v, y_v, z_v),
        vertex_time=time_fit.vertex_time,
        spatial_chi2=spatial_chi2,
        time_chi2=time_fit.chi2,
        cov_xyz=cov_xyz,
        sigma_time=time_fit.sigma_time,
    )


def fit_vertex_time(
    tracks: Sequence[TrackState],
    masses: Sequence[float],
    vertex_xyz: tuple[float, float, float],
    speed_of_light: float = C_LIGHT_MM_PER_NS,
) -> VertexTimeFitResult:
    """Fit candidate time from mass-corrected track times at the fitted vertex."""
    if not tracks:
        return VertexTimeFitResult(
            vertex_time=0.0,
            sigma_time=1e3,
            chi2=0.0,
            propagated_times=(),
        )
    if len(masses) != len(tracks):
        raise ValueError("Mass list length must match track multiplicity.")

    propagated = tuple(
        propagate_track_time_to_vertex(
            track=track,
            mass=float(mass),
            vertex_xyz=vertex_xyz,
            speed_of_light=speed_of_light,
        )
        for track, mass in zip(tracks, masses, strict=True)
    )

    sum_w = 0.0
    sum_wt = 0.0
    for t, t_prop in zip(tracks, propagated, strict=True):
        if t.sigma_time <= 0.0:
            continue
        w = 1.0 / (t.sigma_time * t.sigma_time)
        sum_w += w
        sum_wt += w * t_prop
    if sum_w > 0.0:
        t_v = sum_wt / sum_w
        sigma_t_v = (1.0 / sum_w) ** 0.5
    else:
        t_v = sum(propagated) / len(propagated)
        sigma_t_v = 1e3

    chi2 = 0.0
    for t, t_prop in zip(tracks, propagated, strict=True):
        if t.sigma_time <= 0.0:
            continue
        dt = t_prop - t_v
        chi2 += (dt * dt) / (t.sigma_time * t.sigma_time)
    return VertexTimeFitResult(
        vertex_time=t_v,
        sigma_time=sigma_t_v,
        chi2=chi2,
        propagated_times=propagated,
    )


def propagate_track_time_to_vertex(
    track: TrackState,
    mass: float,
    vertex_xyz: tuple[float, float, float],
    speed_of_light: float = C_LIGHT_MM_PER_NS,
) -> float:
    """Propagate a track time to a candidate vertex using a beta correction."""
    # The sign of the path length preserves whether the vertex lies downstream
    # or upstream of the track reference plane.
    path = signed_path_length_to_vertex(track, vertex_xyz)
    beta = track_beta(track, mass)
    if beta <= 0.0 or speed_of_light <= 0.0:
        return track.time
    return track.time + path / (beta * speed_of_light)


def signed_path_length_to_vertex(
    track: TrackState, vertex_xyz: tuple[float, float, float]
) -> float:
    """Return signed flight length from track reference state to a z-vertex."""
    dz = vertex_xyz[2] - track.z
    return dz * (1.0 + track.tx * track.tx + track.ty * track.ty) ** 0.5


def track_beta(track: TrackState, mass: float) -> float:
    """Compute beta from track momentum magnitude and mass hypothesis."""
    p = max(track.p, 0.0)
    m = abs(mass)
    energy = (p * p + m * m) ** 0.5
    if energy <= 0.0:
        return 0.0
    beta = p / energy
    if beta < 0.0:
        return 0.0
    if beta > 1.0:
        return 1.0
    return beta


def pairwise_doca(tracks: list[TrackState]) -> dict[str, float]:
    """Compute DOCA values for all track pairs in a combination."""
    out: dict[str, float] = {}
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            key = f"doca{i + 1}{j + 1}"
            out[key] = doca_between_tracks(tracks[i], tracks[j])
    return out


def doca_between_tracks(t1: TrackState, t2: TrackState) -> float:
    """Distance of closest approach between two 3D track lines."""
    p1 = (t1.x, t1.y, t1.z)
    p2 = (t2.x, t2.y, t2.z)
    u = t1.direction()
    v = t2.direction()
    w0 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

    a = dot3(u, u)
    b = dot3(u, v)
    c = dot3(v, v)
    d = dot3(u, w0)
    e = dot3(v, w0)
    den = a * c - b * b

    if abs(den) < 1e-12:
        diff = (w0[0] - d * u[0], w0[1] - d * u[1], w0[2] - d * u[2])
        return norm3(diff)

    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    diff = (
        w0[0] + s * u[0] - t * v[0],
        w0[1] + s * u[1] - t * v[1],
        w0[2] + s * u[2] - t * v[2],
    )
    return norm3(diff)


def impact_parameter_to_pv(track: TrackState, pv: PrimaryVertex) -> tuple[float, float]:
    """Compute 2D IP and IP chi2 of one track w.r.t. one primary vertex."""
    (x, y), cov_xy = track.extrapolate_xy_cov(pv.z)
    dx = x - pv.x
    dy = y - pv.y
    ip = math.sqrt(dx * dx + dy * dy)
    pv_cov_xy = ((pv.cov3[0][0], pv.cov3[0][1]), (pv.cov3[1][0], pv.cov3[1][1]))
    total_cov = (
        (cov_xy[0][0] + pv_cov_xy[0][0], cov_xy[0][1] + pv_cov_xy[0][1]),
        (cov_xy[1][0] + pv_cov_xy[1][0], cov_xy[1][1] + pv_cov_xy[1][1]),
    )
    inv = invert_2x2(total_cov)
    if inv is None:
        return ip, dx * dx + dy * dy
    chi2 = dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy)
    return ip, chi2


def min_impact_parameter_to_pvs(track: TrackState, pvs: list[PrimaryVertex]) -> tuple[float, float, str | None]:
    """Find minimum-IP PV association for one track over a PV list."""
    if not pvs:
        return 0.0, 0.0, None
    best_ip = float("inf")
    best_chi2 = float("inf")
    best_id: str | None = None
    for pv in pvs:
        ip, chi2 = impact_parameter_to_pv(track, pv)
        if ip < best_ip:
            best_ip = ip
            best_chi2 = chi2
            best_id = pv.pv_id
    return best_ip, best_chi2, best_id


def associate_composite_to_pvs(
    vertex_xyz: tuple[float, float, float],
    vertex_cov_xyz: Matrix3x3,
    vertex_time: float,
    vertex_sigma_time: float,
    candidate_p4: LorentzVector,
    pvs: Sequence[PrimaryVertex],
    speed_of_light: float = C_LIGHT_MM_PER_NS,
) -> list[CompositePVAssociation]:
    """Compute composite-to-PV IP/time compatibility metrics for all PVs."""
    out: list[CompositePVAssociation] = []
    for pv in pvs:
        ip, ip_chi2 = composite_impact_parameter_to_pv(
            vertex_xyz=vertex_xyz,
            vertex_cov_xyz=vertex_cov_xyz,
            candidate_p4=candidate_p4,
            pv=pv,
        )
        time_residual, time_chi2, flight_time = composite_time_agreement_to_pv(
            vertex_xyz=vertex_xyz,
            vertex_cov_xyz=vertex_cov_xyz,
            vertex_time=vertex_time,
            vertex_sigma_time=vertex_sigma_time,
            candidate_p4=candidate_p4,
            pv=pv,
            speed_of_light=speed_of_light,
        )
        out.append(
            CompositePVAssociation(
                pv_id=pv.pv_id,
                ip=ip,
                ip_chi2=ip_chi2,
                time_residual=time_residual,
                time_chi2=time_chi2,
                flight_time=flight_time,
            )
        )
    return out


def composite_impact_parameter_to_pv(
    vertex_xyz: tuple[float, float, float],
    vertex_cov_xyz: Matrix3x3,
    candidate_p4: LorentzVector,
    pv: PrimaryVertex,
) -> tuple[float, float]:
    """Compute 2D IP and IP chi2 of a composite candidate w.r.t. one PV."""
    x_v, y_v, z_v = vertex_xyz
    ux, uy, uz = _unit_candidate_direction(candidate_p4)
    if abs(uz) > 1e-12:
        tx = ux / uz
        ty = uy / uz
        dz = pv.z - z_v
        x = x_v + tx * dz
        y = y_v + ty * dz
        # Propagate only vertex-position covariance; slope uncertainty is not
        # available for composites at this stage.
        var_x = (
            vertex_cov_xyz[0][0]
            + tx * tx * vertex_cov_xyz[2][2]
            - 2.0 * tx * vertex_cov_xyz[0][2]
        )
        var_y = (
            vertex_cov_xyz[1][1]
            + ty * ty * vertex_cov_xyz[2][2]
            - 2.0 * ty * vertex_cov_xyz[1][2]
        )
        cov_xy = (
            vertex_cov_xyz[0][1]
            - ty * vertex_cov_xyz[0][2]
            - tx * vertex_cov_xyz[1][2]
            + tx * ty * vertex_cov_xyz[2][2]
        )
    else:
        x = x_v
        y = y_v
        var_x = vertex_cov_xyz[0][0]
        var_y = vertex_cov_xyz[1][1]
        cov_xy = vertex_cov_xyz[0][1]

    dx = x - pv.x
    dy = y - pv.y
    ip = math.sqrt(dx * dx + dy * dy)
    total_cov = (
        (
            max(var_x, 0.0) + pv.cov3[0][0],
            cov_xy + pv.cov3[0][1],
        ),
        (
            cov_xy + pv.cov3[1][0],
            max(var_y, 0.0) + pv.cov3[1][1],
        ),
    )
    inv = invert_2x2(total_cov)
    if inv is None:
        return ip, dx * dx + dy * dy
    chi2 = dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy)
    return ip, chi2


def composite_time_agreement_to_pv(
    vertex_xyz: tuple[float, float, float],
    vertex_cov_xyz: Matrix3x3,
    vertex_time: float,
    vertex_sigma_time: float,
    candidate_p4: LorentzVector,
    pv: PrimaryVertex,
    speed_of_light: float = C_LIGHT_MM_PER_NS,
) -> tuple[float, float, float]:
    """Return `(time_residual, time_chi2, flight_time)` for one composite-PV pair.

    The residual compares the PV measured time against the composite time
    propagated back from the composite vertex to the PV using
    `flight_time = L / (beta * c)` with `L` projected along candidate momentum.
    """
    direction = _unit_candidate_direction(candidate_p4)
    beta = _candidate_beta(candidate_p4)
    displacement = (
        vertex_xyz[0] - pv.x,
        vertex_xyz[1] - pv.y,
        vertex_xyz[2] - pv.z,
    )
    flight_length = dot3(displacement, direction)
    flight_time = 0.0
    sigma_flight2 = 0.0
    beta_c = beta * speed_of_light
    if beta_c > 0.0:
        flight_time = flight_length / beta_c
        combined_cov = _sum_cov3(vertex_cov_xyz, pv.cov3)
        sigma_length2 = _directional_variance(direction, combined_cov)
        sigma_flight2 = sigma_length2 / (beta_c * beta_c)
    t_at_pv = vertex_time - flight_time
    residual = t_at_pv - pv.time
    sigma2 = (
        max(vertex_sigma_time, 0.0) ** 2
        + max(pv.sigma_time, 0.0) ** 2
        + max(sigma_flight2, 0.0)
    )
    if sigma2 <= 0.0:
        return residual, residual * residual, flight_time
    return residual, (residual * residual) / sigma2, flight_time


def pair_kinematics(p4: LorentzVector) -> tuple[float, float]:
    """Return `(pt, eta)` from a candidate 4-vector."""
    pt = math.sqrt(p4.px * p4.px + p4.py * p4.py)
    p = math.sqrt(p4.px * p4.px + p4.py * p4.py + p4.pz * p4.pz)
    if p == abs(p4.pz):
        eta = 1e9 if p4.pz >= 0 else -1e9
    else:
        eta = 0.5 * math.log((p + p4.pz) / (p - p4.pz))
    return pt, eta


def invert_2x2(
    mat: tuple[tuple[float, float], tuple[float, float]]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Invert a 2x2 matrix. Return `None` if singular."""
    a, b = mat[0]
    c, d = mat[1]
    det = a * d - b * c
    if abs(det) < 1e-18:
        return None
    inv_det = 1.0 / det
    return ((d * inv_det, -b * inv_det), (-c * inv_det, a * inv_det))


def solve_3x3(a: list[list[float]], b: list[float]) -> tuple[float, float, float] | None:
    """Solve 3x3 linear system by Gaussian elimination with pivoting."""
    m = [row[:] + [rhs] for row, rhs in zip(a, b, strict=True)]
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        p = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= p
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]
    return m[0][3], m[1][3], m[2][3]


def invert_3x3(a: list[list[float]]) -> Matrix3x3 | None:
    """Invert 3x3 matrix by Gaussian elimination."""
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(3)] for i, row in enumerate(a)]
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        p = m[col][col]
        for j in range(col, 2 * n):
            m[col][j] /= p
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            for j in range(col, 2 * n):
                m[r][j] -= factor * m[col][j]
    return (
        (m[0][3], m[0][4], m[0][5]),
        (m[1][3], m[1][4], m[1][5]),
        (m[2][3], m[2][4], m[2][5]),
    )


def dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """3D dot product."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm3(a: tuple[float, float, float]) -> float:
    """Euclidean norm of a 3D vector."""
    return math.sqrt(dot3(a, a))


def _euclidean_spread(vertices_xy: list[tuple[float, float]], mean: tuple[float, float]) -> float:
    """Internal helper: unweighted squared spread in the XY plane."""
    chi2 = 0.0
    for x, y in vertices_xy:
        dx = x - mean[0]
        dy = y - mean[1]
        chi2 += dx * dx + dy * dy
    return chi2


def _unit_candidate_direction(candidate_p4: LorentzVector) -> tuple[float, float, float]:
    """Return unit momentum direction from candidate Lorentz vector."""
    norm = math.sqrt(candidate_p4.px * candidate_p4.px + candidate_p4.py * candidate_p4.py + candidate_p4.pz * candidate_p4.pz)
    if norm <= 1e-16:
        return (0.0, 0.0, 1.0)
    return (
        candidate_p4.px / norm,
        candidate_p4.py / norm,
        candidate_p4.pz / norm,
    )


def _candidate_beta(candidate_p4: LorentzVector) -> float:
    """Compute composite beta from candidate momentum and invariant mass."""
    p = math.sqrt(candidate_p4.p2)
    m = abs(candidate_p4.mass)
    energy = math.sqrt(p * p + m * m)
    if energy <= 0.0:
        return 0.0
    beta = p / energy
    if beta < 0.0:
        return 0.0
    if beta > 1.0:
        return 1.0
    return beta


def _sum_cov3(a: Matrix3x3, b: Matrix3x3) -> Matrix3x3:
    """Return element-wise sum of two 3x3 covariance matrices."""
    return (
        (a[0][0] + b[0][0], a[0][1] + b[0][1], a[0][2] + b[0][2]),
        (a[1][0] + b[1][0], a[1][1] + b[1][1], a[1][2] + b[1][2]),
        (a[2][0] + b[2][0], a[2][1] + b[2][1], a[2][2] + b[2][2]),
    )


def _directional_variance(direction: tuple[float, float, float], cov: Matrix3x3) -> float:
    """Project a 3x3 covariance matrix onto a unit 3D direction."""
    ux, uy, uz = direction
    var = (
        ux * ux * cov[0][0]
        + uy * uy * cov[1][1]
        + uz * uz * cov[2][2]
        + 2.0 * ux * uy * cov[0][1]
        + 2.0 * ux * uz * cov[0][2]
        + 2.0 * uy * uz * cov[1][2]
    )
    return max(var, 0.0)
