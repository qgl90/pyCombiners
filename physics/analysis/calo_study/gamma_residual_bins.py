#!/usr/bin/env python3
"""Per-energy-bin distributions of the relative photon energy residual.

Plain mean/std per true-energy bin (no fitting) plus a summary plot of
the trends vs energy.
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

E_BIN_EDGES_GEV = [5, 15, 30, 60, 120, 240, 500]


def main():
    parser = argparse.ArgumentParser(
        description="Relative residual distributions in energy bins"
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
        "adds a _pt<val> suffix to the output files",
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

    # global scale fitted before any pT cut — it is a detector constant
    c_global, _, _, _ = gauss_fit(
        lead["cluster_e"].values / lead["gamma_e_true"].values
    )
    if args.min_pt > 0:
        r = np.hypot(lead["cluster_x"], lead["cluster_y"])
        sin_t = r / np.hypot(r, lead["cluster_z"])
        pt_reco = lead["cluster_e"] / c_global * sin_t / 1000  # GeV
        lead = lead[pt_reco > args.min_pt]
    e_true = lead["gamma_e_true"].values / 1000  # GeV
    residual = (lead["cluster_e"].values / c_global / 1000 - e_true) / e_true
    print(f"photons: {len(residual)}, global calibration C = {c_global:.2f}")

    src_tag = r"$B_s^0 \to \phi\gamma$ photons"
    if args.min_pt > 0:
        src_tag += rf", $p_T^{{\rm reco}} > {args.min_pt:g}$ GeV"
    n_bins = len(E_BIN_EDGES_GEV) - 1
    fig, axes = make_figure(nrows=2, ncols=3, figsize=(24, 14))
    axes = axes.flatten()

    summary = {k: [] for k in ("center", "n", "mean", "std")}
    print(f"\n{'E bin [GeV]':>14s} {'n':>6s} {'mean':>8s} {'std':>8s}")
    for i in range(n_bins):
        lo, hi = E_BIN_EDGES_GEV[i], E_BIN_EDGES_GEV[i + 1]
        m = (e_true >= lo) & (e_true < hi)
        ax = axes[i]
        vals = residual[m]
        mean = float(np.mean(vals)) if len(vals) else float("nan")
        std = float(np.std(vals)) if len(vals) else float("nan")
        print(f"{f'{lo}-{hi}':>14s} {len(vals):6d} {mean:8.3f} {std:8.3f}")
        summary["center"].append(float(np.median(e_true[m])))
        summary["n"].append(len(vals))
        summary["mean"].append(mean)
        summary["std"].append(std)

        ax.hist(
            vals,
            bins=50,
            range=(-1.5, 3.5),
            histtype="stepfilled",
            alpha=0.7,
            color="steelblue",
        )
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(
            f"$E_\\gamma^{{\\rm true}}$ {lo}-{hi} GeV  (n={len(vals)})\n"
            f"mean = {mean:.3f},  std = {std:.3f}",
            fontsize=22,
        )
        ax.set_xlabel(
            r"$(E_\gamma^{\rm reco} - E_\gamma^{\rm true})/E_\gamma^{\rm true}$",
            fontsize=20,
        )
        ax.tick_params(labelsize=18)

    fig.suptitle(
        f"Photon energy residual per energy bin ({src_tag})", fontsize=28
    )
    fig.tight_layout()
    tag = f"_pt{args.min_pt:g}".replace(".", "p") if args.min_pt > 0 else ""
    out = out_dir / f"gamma_residual_bins_direct_bs{tag}.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")

    # ---- summary: mean and std vs energy, single axis in percent ----------
    c = np.array(summary["center"])
    n = np.array(summary["n"], dtype=float)
    mean = np.array(summary["mean"])
    std = np.array(summary["std"])
    fig2, ax2 = make_figure(figsize=(16, 12))
    ax2.errorbar(
        c,
        mean * 100,
        yerr=std / np.sqrt(np.maximum(n, 1)) * 100,
        fmt="o-",
        color="steelblue",
        markersize=10,
        label="mean (bias)",
    )
    ax2.errorbar(
        c,
        std * 100,
        yerr=std / np.sqrt(2 * np.maximum(n - 1, 1)) * 100,
        fmt="s--",
        color="darkorange",
        markersize=10,
        label="std deviation",
    )
    ax2.axhline(0.0, color="grey", linestyle=":", linewidth=1)
    ax2.set_xlim(E_BIN_EDGES_GEV[0], E_BIN_EDGES_GEV[-1])
    ax2.set_xlabel(r"true $E_\gamma$ [GeV]")
    ax2.set_ylabel(
        r"$(E_\gamma^{\rm reco} - E_\gamma^{\rm true})/E_\gamma^{\rm true}$ [%]"
    )
    ax2.set_xscale("log")
    ax2.legend(loc="upper right")
    ax2.set_title(f"Photon energy residual: mean and std ({src_tag})")
    fig2.tight_layout()
    out2 = out_dir / f"gamma_residual_bins_summary_direct_bs{tag}.png"
    fig2.savefig(out2, dpi=150)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
