#!/usr/bin/env python3
"""PicoCal default-time resolution with Bs -> phi gamma signal photons.

dt = t_cluster - (ovtx_t + TOF): triple-Gaussian global fit plus a
per-region EMG decomposition printing the (area x section) calibration
table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, "src")
from trackcomb import make_figure  # noqa: E402
from trackcomb.physics import c_light  # noqa: E402
from trackcomb.fits import (  # noqa: E402
    emg_left,
    emg_left_fit,
    gauss,
    gauss_fit,
)


def collect(input_path, min_e):
    """dt of the lead cluster per direct-Bs photon (combined time)."""
    df = pd.read_parquet(input_path)
    lead = df[
        (df["cluster_rank"] == 0)
        & df["gamma_is_direct_bs"]
        & (df["gamma_e_true"] > min_e)
    ]
    d = np.sqrt(
        (lead["cluster_x"] - lead["gamma_ovtx_x"]) ** 2
        + (lead["cluster_y"] - lead["gamma_ovtx_y"]) ** 2
        + (lead["cluster_z"] - lead["gamma_ovtx_z"]) ** 2
    )
    dts = (
        lead["cluster_time"] - (lead["gamma_ovtx_time"] + d / c_light)
    ).values
    return dts[np.isfinite(dts)]


def by_area(input_path, min_e, out_dir):
    """Default-time dt per calo region (cellID area bits 24-26).

    Within one region the front and back clocks are each a single
    reference, so the dt splits into two clean components — fitted
    separately using the seed cellID FB bit (0 = front, 1 = back).
    """
    df = pd.read_parquet(input_path)
    lead = df[
        (df["cluster_rank"] == 0)
        & df["gamma_is_direct_bs"]
        & (df["gamma_e_true"] > min_e)
    ]
    d = np.sqrt(
        (lead["cluster_x"] - lead["gamma_ovtx_x"]) ** 2
        + (lead["cluster_y"] - lead["gamma_ovtx_y"]) ** 2
        + (lead["cluster_z"] - lead["gamma_ovtx_z"]) ** 2
    )
    dt = (
        lead["cluster_time"] - (lead["gamma_ovtx_time"] + d / c_light)
    ).values
    cid = lead["cluster_seed_cellid"].values.astype(np.int64)
    area = (cid >> 24) & 0x7
    is_back = (cid & 0x1) == 1
    ok = np.isfinite(dt)

    areas = sorted(int(a) for a in np.unique(area[ok]))
    rng, bins = (0.7, 1.6), 60
    ncols = 4
    nrows = int(np.ceil(len(areas) / ncols))
    fig, axes = make_figure(
        nrows=nrows, ncols=ncols, figsize=(30, 8 * nrows), sharex=True
    )
    axes = np.atleast_2d(axes)
    table = []
    for i, a in enumerate(areas):
        ax = axes[i // ncols][i % ncols]
        m = ok & (area == a)
        ax.hist(
            dt[m],
            bins=bins,
            range=rng,
            histtype="stepfilled",
            alpha=0.7,
            color="steelblue",
            zorder=0,
        )
        xs = np.linspace(*rng, 500)
        for name, sel, color in (
            ("front", m & ~is_back, "red"),
            ("back", m & is_back, "darkorange"),
        ):
            v = dt[sel]
            if len(v) < 20:
                continue
            try:
                amp, mu, s, tau, chi2 = emg_left_fit(v)
                ax.plot(
                    xs,
                    emg_left(xs, amp, mu, s, tau),
                    color=color,
                    linewidth=2,
                    label=(
                        rf"{name}: mean={(mu - tau) * 1000:.0f} ps, "
                        rf"resolution={s * 1000:.0f} ps"
                    ),
                )
            except RuntimeError:
                mu, s, _, _ = gauss_fit(v)
                tau, chi2 = np.nan, np.nan
                counts, edges = np.histogram(v, bins=bins, range=rng)
                amp = counts.max()
                ax.plot(
                    xs,
                    gauss(xs, amp, mu, s),
                    color=color,
                    linewidth=2,
                    linestyle="--",
                    label=(
                        rf"{name} (gauss): mean={mu * 1000:.0f} ps, "
                        rf"resolution={s * 1000:.0f} ps"
                    ),
                )
            table.append(
                {
                    "area": a,
                    "section": name,
                    "n": int(len(v)),
                    "frac": len(v) / max(m.sum(), 1),
                    "bias": (mu - tau if np.isfinite(tau) else mu) * 1000,
                    "mu": mu * 1000,
                    "sigma": s * 1000,
                    "tau": tau * 1000,
                    "chi2": chi2,
                }
            )
        ax.set_title(f"area {a}  (n={m.sum()})", fontsize=18)
        ax.legend(loc="upper left", fontsize=14)
        ax.tick_params(labelsize=14)
        if i // ncols == nrows - 1:
            ax.set_xlabel(
                r"$t_{\rm cluster}$ - ($t_{\rm ovtx}$ + $d/c$) [ns]",
                fontsize=15,
            )
    for j in range(len(areas), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(
        r"PicoCal default-time $dt$ per region "
        r"(cellID area bits; seed FB bit splits front/back)",
        fontsize=26,
    )
    fig.tight_layout()
    out = out_dir / "gamma_time_resolution_default_by_area.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")

    # bias = EMG effective mean (mu - tau) = the alignment constant
    print(
        "\nPicoCal per-(area x section) time calibration table [ps]:"
        f"\n{'area':>4s} {'section':>7s} {'n':>6s} {'frac':>5s} "
        f"{'bias':>6s} {'mu':>6s} {'sigma':>6s} {'tau':>5s} {'chi2/ndf':>8s}"
    )
    for r in table:
        print(
            f"{r['area']:4d} {r['section']:>7s} {r['n']:6d} "
            f"{r['frac']:5.0%} {r['bias']:6.0f} {r['mu']:6.0f} "
            f"{r['sigma']:6.0f} {r['tau']:5.0f} {r['chi2']:8.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Cluster default-time resolution"
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

    dts = collect(args.input, args.min_e)
    print(f"signal photons (lead cluster): {len(dts)}")

    # front-seeded / back-seeded / mixed populations sit at different
    # time references -> fit the sum of three Gaussians
    bins = 80
    fig, ax = make_figure(figsize=(16, 12))
    med = float(np.median(dts))
    lo, hi = med - 0.6, med + 0.5
    counts, edges = np.histogram(dts, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])

    def triple(x, a1, m1, s1, a2, m2, s2, a3, m3, s3):
        return (
            gauss(x, a1, m1, s1) + gauss(x, a2, m2, s2) + gauss(x, a3, m3, s3)
        )

    m0 = float(centers[np.argmax(counts)])
    p0, lo_b, hi_b = [], [], []
    for dm in (-0.33, -0.18, 0.0):
        a0 = float(counts[np.argmin(np.abs(centers - (m0 + dm)))])
        p0 += [max(a0, 1.0), m0 + dm, 0.05]
        lo_b += [0.0, lo, 0.01]
        hi_b += [np.inf, hi, 0.2]
    popt, _ = curve_fit(
        triple, centers, counts, p0=p0, bounds=(lo_b, hi_b), maxfev=20000
    )
    peaks = sorted([popt[i : i + 3] for i in (0, 3, 6)], key=lambda p: p[1])
    norm = sum(a * s for a, _, s in peaks)
    xs = np.linspace(lo, hi, 500)
    ax.plot(
        xs,
        triple(xs, *np.concatenate(peaks)),
        color="red",
        linewidth=2,
        label="triple-Gaussian fit",
    )
    print(f"\n{'peak':>4s} {'mu [ps]':>8s} {'sigma [ps]':>10s} {'frac':>6s}")
    for i, ((a, m, s), color) in enumerate(
        zip(peaks, ("darkorange", "seagreen", "purple"))
    ):
        frac = a * s / norm
        print(f"{i + 1:4d} {m * 1000:8.0f} {s * 1000:10.0f} {frac:6.1%}")
        ax.plot(
            xs,
            gauss(xs, a, m, s),
            color=color,
            linewidth=2,
            linestyle="--",
            label=(
                rf"$\mu$ = {m * 1000:.0f} ps, "
                rf"$\sigma$ = {s * 1000:.0f} ps ({frac:.0%})"
            ),
        )

    ax.hist(
        dts,
        bins=bins,
        range=(lo, hi),
        histtype="stepfilled",
        alpha=0.7,
        color="steelblue",
        label=f"signal photons ({len(dts)})",
        zorder=0,
    )
    ax.set_xlabel(r"$t_{\rm cluster}$ - ($t_{\rm ovtx}$ + $d/c$) [ns]")
    ax.set_ylabel(f"photons / {1000 * (hi - lo) / bins:.0f} ps")
    ax.legend(loc="upper right")
    ax.set_title(
        r"PicoCal default-time resolution ($B_s^0 \to \phi\gamma$ photons)"
    )
    fig.tight_layout()
    out = out_dir / "gamma_time_resolution_default.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    by_area(args.input, args.min_e, out_dir)


if __name__ == "__main__":
    main()
