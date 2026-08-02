#!/usr/bin/env python3
"""2D relative photon energy residual vs true energy.

One point per photon (lead cluster), after a single global scale
calibration — any structure is genuine PicoCal non-linearity/spread.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from trackcomb import make_figure  # noqa: E402
from trackcomb.fits import gauss_fit  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="(E_reco - E_true)/E_true vs E_true scatter for photons"
    )
    parser.add_argument(
        "--input",
        default="public/calo_study/reconstruction/bgamma_matches.parquet",
    )
    parser.add_argument("--out-dir", default="public/calo_study/analysis")
    parser.add_argument("--min-e", type=float, default=500.0)
    parser.add_argument(
        "--min-pt",
        type=float,
        default=0.0,
        help="reco pT cut [GeV, calibrated] against beam-hole photons; "
        "adds a _pt<val> suffix to the output file",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    lead = df[
        (df["cluster_rank"] == 0)
        & (df["gamma_e_true"] > args.min_e)
        & df["gamma_is_direct_bs"]
    ]

    # single global scale: Gaussian core mean of e/E over the sample
    # (fitted before any pT cut — the scale is a detector constant)
    c_global, _, _, _ = gauss_fit(
        lead["cluster_e"].values / lead["gamma_e_true"].values
    )
    if args.min_pt > 0:
        r = np.hypot(lead["cluster_x"], lead["cluster_y"])
        sin_t = r / np.hypot(r, lead["cluster_z"])
        pt_reco = lead["cluster_e"] / c_global * sin_t / 1000  # GeV
        lead = lead[pt_reco > args.min_pt]
    e_true = lead["gamma_e_true"].values / 1000  # GeV
    e_reco = lead["cluster_e"].values / c_global / 1000  # GeV
    residual = (e_reco - e_true) / e_true
    print(f"photons: {len(lead)}, global calibration C = {c_global:.2f}")

    src_tag = r"$B_s^0 \to \phi\gamma$ photons"
    if args.min_pt > 0:
        src_tag += rf", $p_T^{{\rm reco}} > {args.min_pt:g}$ GeV"
    fig, ax = make_figure(figsize=(16, 12))
    ax.scatter(
        e_true,
        residual,
        s=14,
        alpha=0.35,
        color="steelblue",
        edgecolors="none",
        label=f"one point per photon ({len(lead)})",
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax.axhline(
        0.5,
        color="grey",
        linestyle=":",
        linewidth=1.5,
        label=r"$\pm 50\%\,E_\gamma^{\rm true}$",
    )
    ax.axhline(-0.5, color="grey", linestyle=":", linewidth=1.5)
    ax.set_xlabel(r"true $E_\gamma$ [GeV]")
    ax.set_ylabel(
        r"$(E_\gamma^{\rm reco} - E_\gamma^{\rm true})/E_\gamma^{\rm true}$"
    )
    ax.set_xlim(0, e_true.max() * 1.05)
    ax.legend(loc="upper right")
    ax.set_title(f"PicoCal photon energy residual ({src_tag})")

    fig.tight_layout()
    tag = f"_pt{args.min_pt:g}".replace(".", "p") if args.min_pt > 0 else ""
    out = out_dir / f"gamma_energy_residual_2d_direct_bs_rel{tag}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
