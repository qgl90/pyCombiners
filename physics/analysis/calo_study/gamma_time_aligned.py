#!/usr/bin/env python3
"""Aligned weighted-mean photon time vs cluster ET.

Section-aligned front/back weighted mean, DSCB fit per ET bin: summary
(mean, resolution) plus the per-bin fits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from trackcomb import make_figure  # noqa: E402
from trackcomb.physics import c_light  # noqa: E402
from trackcomb.fits import dscb, dscb_fit  # noqa: E402

# alignment constants [ns] per (area, section), see picocal_time.py
BIAS = {
    "front": {
        1: 1.264,
        2: 1.261,
        3: 1.258,
        4: 1.255,
        5: 1.281,
        6: 1.276,
        7: 1.341,
    },
    "back": {
        1: 0.951,
        2: 0.946,
        3: 0.946,
        4: 0.948,
        5: 1.092,
        6: 1.094,
        7: 1.259,
    },
}
# single-section aligned resolutions [ps] -> weights
SIGMA_F, SIGMA_B = 50.0, 36.0


def main():
    parser = argparse.ArgumentParser(
        description="Aligned weighted-mean photon time vs ET"
    )
    parser.add_argument(
        "--input",
        default="public/calo_study/reconstruction/bgamma_matches.parquet",
    )
    parser.add_argument("--out-dir", default="public/calo_study/analysis")
    parser.add_argument("--min-e", type=float, default=500.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    lead = df[
        (df["cluster_rank"] == 0)
        & df["gamma_is_direct_bs"]
        & (df["gamma_e_true"] > args.min_e)
        & (df["cluster_front_e"] > 0)
        & (df["cluster_back_e"] > 0)
    ]
    area = (lead["cluster_seed_cellid"].values.astype(np.int64) >> 24) & 0x7

    def aligned(sec):
        d = np.sqrt(
            (lead[f"cluster_{sec}_x"] - lead["gamma_ovtx_x"]) ** 2
            + (lead[f"cluster_{sec}_y"] - lead["gamma_ovtx_y"]) ** 2
            + (lead[f"cluster_{sec}_z"] - lead["gamma_ovtx_z"]) ** 2
        )
        dt = (
            lead[f"cluster_{sec}_time"]
            - (lead["gamma_ovtx_time"] + d / c_light)
        ).values
        return dt - np.array([BIAS[sec][a] for a in area])

    dt_f = aligned("front")
    dt_b = aligned("back")
    w_f, w_b = 1 / SIGMA_F**2, 1 / SIGMA_B**2
    dt_w = (w_f * dt_f + w_b * dt_b) / (w_f + w_b)
    print(f"same-sample clusters (both sections): {len(lead)}")

    # DSCB fit per raw-cluster-ET bin
    r_xy = np.hypot(lead["cluster_x"].values, lead["cluster_y"].values)
    sin_t = r_xy / np.hypot(r_xy, lead["cluster_z"].values)
    et = lead["cluster_e"].values * sin_t / 1000  # GeV
    edges = np.geomspace(0.5, 20, 9)
    centers = np.sqrt(edges[:-1] * edges[1:])
    lo, hi = -0.4, 0.4
    bins = 80
    mus = np.full(len(centers), np.nan)
    mu_errs = np.full(len(centers), np.nan)
    sigmas = np.full(len(centers), np.nan)
    print(
        f"\n{'ET bin [GeV]':>14s} {'n':>6s} {'mu [ps]':>7s} "
        f"{'sigma [ps]':>10s} {'chi2/ndf':>8s}"
    )
    fig_b, axes_b = make_figure(nrows=2, ncols=4, figsize=(32, 14))
    axes_b = axes_b.flatten()
    for i, (elo, ehi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (et >= elo) & (et < ehi) & np.isfinite(dt_w)
        ax = axes_b[i]
        if m.sum() < 200:
            ax.set_visible(False)
            continue
        popt, chi2 = dscb_fit(dt_w[m], bins=bins, rng=(lo, hi))
        mus[i], sigmas[i] = popt[1], popt[2]
        mu_errs[i] = popt[2] / np.sqrt(m.sum())
        print(
            f"{f'{elo:.2g}-{ehi:.2g}':>14s} {int(m.sum()):6d} "
            f"{mus[i] * 1000:7.0f} {sigmas[i] * 1000:10.0f} {chi2:8.1f}"
        )
        ax.hist(
            dt_w[m],
            bins=bins,
            range=(lo, hi),
            histtype="stepfilled",
            alpha=0.7,
            color="steelblue",
        )
        xs = np.linspace(lo, hi, 800)
        ax.plot(xs, dscb(xs, *popt), color="red", linewidth=2)
        ax.set_title(
            f"$E_T$ {elo:.2g}-{ehi:.2g} GeV  (n={int(m.sum())})\n"
            rf"$\mu$ = {mus[i] * 1000:.0f} ps, "
            rf"$\sigma$ = {sigmas[i] * 1000:.0f} ps",
            fontsize=20,
        )
        ax.set_xlabel("weighted-mean time residual [ns]", fontsize=16)
        ax.tick_params(labelsize=14)
    for ax in axes_b[len(centers) :]:
        ax.set_visible(False)
    fig_b.suptitle(
        r"PicoCal weighted-mean photon time: DSCB fits per $E_T$ bin "
        r"($B_s^0 \to \phi\gamma$ photons)",
        fontsize=28,
    )
    fig_b.tight_layout()
    out_b = out_dir / "gamma_time_aligned_vs_et_bins.png"
    fig_b.savefig(out_b, dpi=130)
    print(f"Saved {out_b}")

    fig, ax = make_figure(figsize=(16, 12))
    ax.errorbar(
        centers,
        mus * 1000,
        yerr=mu_errs * 1000,
        fmt="o-",
        color="steelblue",
        markersize=10,
        label="mean",
    )
    ax.plot(
        centers,
        sigmas * 1000,
        "s--",
        color="darkorange",
        markersize=10,
        label="resolution",
    )
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"raw cluster $E_T$ [GeV]")
    ax.set_ylabel(r"weighted-mean time residual [ps]")
    ax.legend(loc="upper right")
    ax.set_title(
        r"PicoCal weighted-mean photon time vs $E_T$ "
        r"($B_s^0 \to \phi\gamma$ photons)"
    )
    fig.tight_layout()
    out = out_dir / "gamma_time_aligned_vs_et.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
