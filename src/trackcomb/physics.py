"""Physics/math helpers for building and filtering particle combinations."""

from __future__ import annotations

import math
from typing import Iterable

from .models import LorentzVector, PrimaryVertex, TrackState


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


def pairwise_time_chi2(tracks: list[TrackState]) -> float:
    """Average pairwise time chi2 over all unique track pairs."""
    if len(tracks) <= 1:
        return 0.0
    chi2 = 0.0
    n_pairs = 0
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            dt = tracks[i].time - tracks[j].time
            sigma2 = (
                tracks[i].sigma_time * tracks[i].sigma_time
                + tracks[j].sigma_time * tracks[j].sigma_time
            )
            if sigma2 <= 0.0:
                continue
            chi2 += (dt * dt) / sigma2
            n_pairs += 1
    return chi2 / n_pairs if n_pairs else 0.0


def fit_vertex_xyz_t(tracks: list[TrackState]) -> tuple[tuple[float, float, float], float, float, float]:
    """
    Least-squares vertex in (x,y,z) from line equations and weighted time average.
    Returns: (vertex_xyz, vertex_time, spatial_chi2, time_chi2)
    """
    if not tracks:
        return (0.0, 0.0, 0.0), 0.0, 0.0, 0.0
    if len(tracks) == 1:
        t = tracks[0]
        return (t.x, t.y, t.z), t.time, 0.0, 0.0

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
    if xyz is None:
        xyz = (sum(t.x for t in tracks) / len(tracks), sum(t.y for t in tracks) / len(tracks), sum(t.z for t in tracks) / len(tracks))
    x_v, y_v, z_v = xyz

    spatial_chi2 = 0.0
    for t in tracks:
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

    sum_w = 0.0
    sum_wt = 0.0
    for t in tracks:
        if t.sigma_time <= 0.0:
            continue
        w = 1.0 / (t.sigma_time * t.sigma_time)
        sum_w += w
        sum_wt += w * t.time
    t_v = sum_wt / sum_w if sum_w > 0.0 else sum(t.time for t in tracks) / len(tracks)
    time_chi2 = 0.0
    for t in tracks:
        if t.sigma_time <= 0.0:
            continue
        dt = t.time - t_v
        time_chi2 += (dt * dt) / (t.sigma_time * t.sigma_time)

    return (x_v, y_v, z_v), t_v, spatial_chi2, time_chi2


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
