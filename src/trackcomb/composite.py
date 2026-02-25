"""Helpers to treat combination outputs as track-like composite objects.

This enables hierarchical workflows where an intermediate resonance candidate
is reused as an input "track" for a higher-level combination.
"""

from __future__ import annotations
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


from typing import Sequence

from .models import CombinationResult, TrackState


def combination_to_track_state(
    result: CombinationResult,
    constituents: Sequence[TrackState],
    track_id: str,
) -> TrackState:
    """Convert one `CombinationResult` into a track-like `TrackState`.

    The composite state uses:
    - `(x, y, z, time)` from fitted combination vertex.
    - `(tx, ty)` from candidate momentum direction.
    - `p` from candidate momentum magnitude.
    - `charge` from summed constituent charge.
    - `cov4` from fitted vertex covariance + slope covariance approximation.
    - `sigma_time` from fitted combination time uncertainty.

    Notes:
    - Slope covariance is approximated by momentum-weighted constituent slope
      covariance contributions (sufficient for staged combiners, not a full fit).
    - PID fields are intentionally reset for composite objects.
    """
    px = result.candidate_p4.px
    py = result.candidate_p4.py
    pz = result.candidate_p4.pz
    p_mag = (px * px + py * py + pz * pz) ** 0.5
    if abs(pz) > 1e-12:
        tx = px / pz
        ty = py / pz
    else:
        tx = 0.0
        ty = 0.0

    w = _momentum_weights(constituents)
    var_tx = sum((wi * wi) * t.cov4[2][2] for wi, t in zip(w, constituents, strict=True))
    var_ty = sum((wi * wi) * t.cov4[3][3] for wi, t in zip(w, constituents, strict=True))
    cov_tx_ty = sum((wi * wi) * t.cov4[2][3] for wi, t in zip(w, constituents, strict=True))
    cov_x_tx = sum((wi * wi) * t.cov4[0][2] for wi, t in zip(w, constituents, strict=True))
    cov_y_ty = sum((wi * wi) * t.cov4[1][3] for wi, t in zip(w, constituents, strict=True))

    cov4 = (
        (result.vertex_cov_xyz[0][0], result.vertex_cov_xyz[0][1], cov_x_tx, 0.0),
        (result.vertex_cov_xyz[0][1], result.vertex_cov_xyz[1][1], 0.0, cov_y_ty),
        (cov_x_tx, 0.0, var_tx, cov_tx_ty),
        (0.0, cov_y_ty, cov_tx_ty, var_ty),
    )
    charge = result.total_charge

    return TrackState(
        track_id=track_id,
        z=result.vertex_xyz[2],
        x=result.vertex_xyz[0],
        y=result.vertex_xyz[1],
        tx=tx,
        ty=ty,
        time=result.vertex_time,
        cov4=cov4,
        sigma_time=result.vertex_sigma_time,
        p=p_mag,
        charge=charge,
        has_rich1=False,
        has_rich2=False,
        rich_dll_pi=0.0,
        rich_dll_k=0.0,
        rich_dll_p=0.0,
        rich_dll_e=0.0,
        has_calo=False,
        calo_dll_e=0.0,
        source_track_ids=result.source_track_ids,
    )


def _momentum_weights(tracks: Sequence[TrackState]) -> list[float]:
    """Return normalized positive momentum weights for covariance mixing."""
    if not tracks:
        return []
    total = sum(max(t.p, 0.0) for t in tracks)
    if total <= 0.0:
        return [1.0 / len(tracks)] * len(tracks)
    return [max(t.p, 0.0) / total for t in tracks]
