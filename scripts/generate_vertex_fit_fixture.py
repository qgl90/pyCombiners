#!/usr/bin/env python3
"""Generate vertex_fit_3d test fixture from real tracks."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np

from trackcomb.io import load_events

ROOT_FILE = (
    Path(__file__).resolve().parent.parent
    / "input"
    / "ntuple_minbias_1p5e32_1000evts.root"
)
TREE = "BestLongTracks/TrackTuple"
OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "input"
    / "test_vertex_fit_reference.npz"
)

COV_KEYS = [
    "cov_0_0",
    "cov_1_0",
    "cov_1_1",
    "cov_2_0",
    "cov_2_1",
    "cov_2_2",
    "cov_3_0",
    "cov_3_1",
    "cov_3_2",
    "cov_3_3",
]


def _per_candidate_vertex_fit(x, y, z, tx, ty, cov_fields):
    """Single-candidate vertex fit via np.linalg (reference implementation)."""
    n = len(x)  # number of daughters

    # Build ATA and ATb
    stx = np.sum(tx)
    sty = np.sum(ty)
    st2 = np.sum(tx**2 + ty**2)
    rx = x - tx * z
    ry = y - ty * z

    ATA = np.array(
        [
            [n, 0, -stx],
            [0, n, -sty],
            [-stx, -sty, st2],
        ]
    )
    ATb = np.array(
        [
            np.sum(rx),
            np.sum(ry),
            np.sum(-tx * rx - ty * ry),
        ]
    )

    vertex_xyz = np.linalg.solve(ATA, ATb)

    # Spatial chi2 and weighted vertex covariance
    z_v = vertex_xyz[2]
    x_v = vertex_xyz[0]
    y_v = vertex_xyz[1]

    dz = z_v - z
    x_ext = x + tx * dz
    y_ext = y + ty * dz
    dx = x_ext - x_v
    dy = y_ext - y_v

    spatial_chi2 = 0.0
    ATA_w = np.zeros((3, 3))
    for i in range(n):
        # Propagate track 2x2 covariance to vertex z
        var_x = (
            cov_fields["cov_0_0"][i]
            + 2.0 * dz[i] * cov_fields["cov_2_0"][i]
            + dz[i] ** 2 * cov_fields["cov_2_2"][i]
        )
        var_y = (
            cov_fields["cov_1_1"][i]
            + 2.0 * dz[i] * cov_fields["cov_3_1"][i]
            + dz[i] ** 2 * cov_fields["cov_3_3"][i]
        )
        cov_xy_val = (
            cov_fields["cov_1_0"][i]
            + dz[i] * cov_fields["cov_3_0"][i]
            + dz[i] * cov_fields["cov_2_1"][i]
            + dz[i] ** 2 * cov_fields["cov_3_2"][i]
        )

        C = np.array([[var_x, cov_xy_val], [cov_xy_val, var_y]])
        r = np.array([dx[i], dy[i]])
        det = np.linalg.det(C)
        if abs(det) < 1e-18:
            spatial_chi2 += r @ r
        else:
            W = np.linalg.inv(C)
            spatial_chi2 += r @ W @ r
            H = np.array([[1, 0, -tx[i]], [0, 1, -ty[i]]])
            ATA_w += H.T @ W @ H

    cov_xyz = np.linalg.inv(ATA_w)
    return vertex_xyz, spatial_chi2, cov_xyz


def main():
    if not ROOT_FILE.exists():
        raise FileNotFoundError(f"ROOT file not found: {ROOT_FILE}")

    tracks, _, _ = load_events(str(ROOT_FILE), TREE, max_events=1)

    # Get tracks from event 0
    fields = {}
    for key in ["x", "y", "z", "tx", "ty"] + COV_KEYS:
        fields[key] = ak.to_numpy(tracks[key][0])  # event 0, shape (n_tracks,)

    n_tracks = len(fields["x"])

    # Pick 2 pairs with sufficient opening angle
    pairs = []
    for i in range(n_tracks):
        if len(pairs) >= 2:
            break
        for j in range(i + 1, n_tracks):
            dtx = fields["tx"][i] - fields["tx"][j]
            dty = fields["ty"][i] - fields["ty"][j]
            opening = np.sqrt(dtx**2 + dty**2)
            if opening > 0.05:
                pairs.append((i, j))
                break

    if len(pairs) < 2:
        raise RuntimeError("Could not find 2 suitable track pairs")

    print(f"Selected track pairs: {pairs}")

    # Build (2, 2) arrays for the 2 candidates
    indices = np.array(pairs)  # (2, 2)
    data = {}
    for key in ["x", "y", "z", "tx", "ty"] + COV_KEYS:
        data[key] = fields[key][indices]  # (2, 2)

    # Compute per-candidate reference
    ref_xyz = np.zeros((2, 3))
    ref_chi2 = np.zeros(2)
    ref_cov = np.zeros((2, 3, 3))

    for ci in range(2):
        cov_f = {k: data[k][ci] for k in COV_KEYS}
        xyz, chi2, cov = _per_candidate_vertex_fit(
            data["x"][ci],
            data["y"][ci],
            data["z"][ci],
            data["tx"][ci],
            data["ty"][ci],
            cov_f,
        )
        ref_xyz[ci] = xyz
        ref_chi2[ci] = chi2
        ref_cov[ci] = cov

    # Save
    save_dict = {}
    for key in ["x", "y", "z", "tx", "ty"] + COV_KEYS:
        save_dict[key] = data[key]
    save_dict["vertex_xyz"] = ref_xyz
    save_dict["spatial_chi2"] = ref_chi2
    save_dict["vertex_cov"] = ref_cov

    np.savez(OUT_PATH, **save_dict)
    print(f"Saved fixture to {OUT_PATH}")
    print(f"  vertex_xyz = {ref_xyz}")
    print(f"  spatial_chi2 = {ref_chi2}")


if __name__ == "__main__":
    main()
