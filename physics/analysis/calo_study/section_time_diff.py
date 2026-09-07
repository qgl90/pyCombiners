#!/usr/bin/env python3
"""Same-shower time difference between the two PicoCal sections.

(t_back - t_front) - (d_back - d_front)/c per lead cluster with energy
in both sections; a negative core proves the sections sit on different
time references.
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
from trackcomb.fits import (  # noqa: E402
    emg_left,
    emg_left_fit,
    gauss,
    gauss_fit,
)

XY_EDGES = np.linspace(-2500, 2500, 8)  # 7x7 cells for the grid of hists
MAP_EDGES = np.linspace(-2500, 2500, 26)  # 200 mm cells for the mean map
MAP_MIN_N = 10


def main():
    parser = argparse.ArgumentParser(
        description="Same-shower back-front section time difference"
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

    def dist(sec):
        return np.sqrt(
            (lead[f"cluster_{sec}_x"] - lead["gamma_ovtx_x"]) ** 2
            + (lead[f"cluster_{sec}_y"] - lead["gamma_ovtx_y"]) ** 2
            + (lead[f"cluster_{sec}_z"] - lead["gamma_ovtx_z"]) ** 2
        )

    diff = (
        lead["cluster_back_time"]
        - lead["cluster_front_time"]
        - (dist("back") - dist("front")) / c_light
    ).values
    diff = diff[np.isfinite(diff)]

    mu, sigma, _, _ = gauss_fit(diff)
    print(f"same-shower clusters (both sections): {len(diff)}")
    print(f"core mu = {mu * 1000:.0f} ps, sigma = {sigma * 1000:.0f} ps")

    # ---- same difference in XY bins of the calo face -----------------------
    x = lead["cluster_x"].values
    y = lead["cluster_y"].values
    all_diff = (
        lead["cluster_back_time"]
        - lead["cluster_front_time"]
        - (dist("back") - dist("front")) / c_light
    ).values
    inside = (
        (np.abs(x) < XY_EDGES[-1])
        & (np.abs(y) < XY_EDGES[-1])
        & np.isfinite(all_diff)
    )
    n_cells = len(XY_EDGES) - 1
    fig2, axes = make_figure(
        nrows=n_cells, ncols=n_cells, figsize=(30, 26), sharex=True
    )
    rng = (-0.7, 0.3)
    print(f"\n{'x [mm]':>12s} {'y [mm]':>12s} {'n':>6s} {'median [ps]':>11s}")
    for iy in range(n_cells):
        for ix in range(n_cells):
            ax = axes[n_cells - 1 - iy][ix]
            m = (
                inside
                & (x >= XY_EDGES[ix])
                & (x < XY_EDGES[ix + 1])
                & (y >= XY_EDGES[iy])
                & (y < XY_EDGES[iy + 1])
            )
            n = int(m.sum())
            med = np.median(all_diff[m]) * 1000 if n else np.nan
            ax.axvline(0, color="grey", linestyle="--", linewidth=1.2)
            if n:
                ax.hist(
                    all_diff[m],
                    bins=50,
                    range=rng,
                    histtype="stepfilled",
                    alpha=0.7,
                    color="steelblue",
                )
                ax.axvline(
                    med / 1000, color="red", linestyle=":", linewidth=1.5
                )
            print(
                f"{f'{XY_EDGES[ix]:.0f}..{XY_EDGES[ix + 1]:.0f}':>12s} "
                f"{f'{XY_EDGES[iy]:.0f}..{XY_EDGES[iy + 1]:.0f}':>12s} "
                f"{n:6d} {med:11.0f}"
            )
            ax.set_title(
                f"x {XY_EDGES[ix] / 1000:g}..{XY_EDGES[ix + 1] / 1000:g}, "
                f"y {XY_EDGES[iy] / 1000:g}..{XY_EDGES[iy + 1] / 1000:g} m"
                f"  (n={n}, med={med:.0f} ps)",
                fontsize=11,
            )
            ax.tick_params(labelsize=11)
            if iy == 0:
                ax.set_xlabel(
                    r"$(t_{\rm back} - t_{\rm front})$"
                    r" - $\Delta d/c$ [ns]",
                    fontsize=12,
                )
    fig2.suptitle(
        r"PicoCal same-shower section time difference across the calo face "
        r"(dashed = causal limit, red dotted = cell median)",
        fontsize=26,
    )
    fig2.tight_layout()
    out2 = out_dir / "gamma_section_time_diff_xy.png"
    fig2.savefig(out2, dpi=120)
    print(f"Saved {out2}")

    # ---- 2D map: per-cell mean of the difference ---------------------------
    sums, _, _ = np.histogram2d(
        x[inside],
        y[inside],
        bins=[MAP_EDGES, MAP_EDGES],
        weights=all_diff[inside],
    )
    counts, _, _ = np.histogram2d(
        x[inside], y[inside], bins=[MAP_EDGES, MAP_EDGES]
    )
    mean_map = np.where(
        counts >= MAP_MIN_N, sums / np.maximum(counts, 1), np.nan
    )
    fig3, ax3 = make_figure(figsize=(17, 14))
    pc = ax3.pcolormesh(
        MAP_EDGES,
        MAP_EDGES,
        mean_map.T * 1000,  # histogram2d: first axis = x -> transpose
        cmap="viridis",
        vmin=-400,
        vmax=0,
    )
    cb = fig3.colorbar(pc, ax=ax3, pad=0.02)
    cb.set_label(r"mean $(t_{\rm back} - t_{\rm front}) - \Delta d/c$ [ps]")
    ax3.set_xlabel(r"cluster $x$ [mm]")
    ax3.set_ylabel(r"cluster $y$ [mm]")
    ax3.set_aspect("equal")
    ax3.set_title(
        rf"PicoCal section time difference map "
        rf"(200 mm cells, n $\geq$ {MAP_MIN_N})"
    )
    fig3.tight_layout()
    out3 = out_dir / "gamma_section_time_diff_map.png"
    fig3.savefig(out3, dpi=150)
    print(f"Saved {out3}")

    # ---- per-area (cellID bits 24-26): one offset per region ---------------
    area = (lead["cluster_seed_cellid"].values.astype(np.int64) >> 24) & 0x7
    areas = sorted(int(a) for a in np.unique(area))
    ncols = 4
    nrows = int(np.ceil(len(areas) / ncols))
    fig4, axes4 = make_figure(
        nrows=nrows, ncols=ncols, figsize=(30, 8 * nrows)
    )
    axes4 = np.atleast_2d(axes4)
    print(
        f"\n{'area':>4s} {'n':>6s} {'mean [ps]':>9s} {'sigma [ps]':>10s} "
        f"{'tau [ps]':>8s} {'chi2/ndf':>8s}"
    )
    for i, a in enumerate(areas):
        ax = axes4[i // ncols][i % ncols]
        v = all_diff[np.isfinite(all_diff) & (area == a)]
        med0 = float(np.median(v))
        rng4 = (med0 - 0.45, med0 + 0.3)
        counts_a, _ = np.histogram(v, bins=50, range=rng4)
        ax.hist(
            v,
            bins=50,
            range=rng4,
            histtype="stepfilled",
            alpha=0.7,
            color="steelblue",
        )
        xs4 = np.linspace(*rng4, 400)
        try:
            amp_a, mu_a, s_a, tau_a, chi2 = emg_left_fit(v, rng=rng4)
            ax.plot(
                xs4,
                emg_left(xs4, amp_a, mu_a, s_a, tau_a),
                color="red",
                linewidth=2,
            )
            print(
                f"{a:4d} {len(v):6d} {(mu_a - tau_a) * 1000:9.0f} "
                f"{s_a * 1000:10.0f} {tau_a * 1000:8.0f} {chi2:8.1f}"
            )
            title_fit = (
                rf"$\sigma$={s_a * 1000:.0f}, $\tau$={tau_a * 1000:.0f} ps"
            )
        except RuntimeError:
            mu_a, s_a, _, _ = gauss_fit(v)
            ax.plot(
                xs4,
                gauss(xs4, counts_a.max(), mu_a, s_a),
                color="red",
                linewidth=2,
                linestyle="--",
            )
            print(f"{a:4d} {len(v):6d}  EMG fit failed, gauss fallback")
            title_fit = rf"gauss $\sigma$={s_a * 1000:.0f} ps"
        ax.axvline(0, color="grey", linestyle="--", linewidth=1.2)
        ax.set_title(
            f"area {a}  (n={len(v)}, {title_fit})",
            fontsize=18,
        )
        ax.tick_params(labelsize=14)
        if i // ncols == nrows - 1:
            ax.set_xlabel(
                r"$(t_{\rm back} - t_{\rm front})$ - $\Delta d/c$ [ns]",
                fontsize=15,
            )
    for j in range(len(areas), nrows * ncols):
        axes4[j // ncols][j % ncols].set_visible(False)
    fig4.suptitle(
        r"PicoCal same-shower section time difference per region "
        r"(cellID area bits; dashed = causal limit)",
        fontsize=26,
    )
    fig4.tight_layout()
    out4 = out_dir / "gamma_section_time_diff_by_area.png"
    fig4.savefig(out4, dpi=120)
    print(f"Saved {out4}")


if __name__ == "__main__":
    main()
