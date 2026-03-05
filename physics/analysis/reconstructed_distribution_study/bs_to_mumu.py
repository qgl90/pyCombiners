#!/usr/bin/env python3
"""Bs -> mu+mu- distribution analysis: load Parquet, plot signal vs background."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Bs -> mu+mu- distribution analysis (from Parquet)",
    )
    parser.add_argument("--dist", required=True, help="Path to dist.parquet")
    parser.add_argument("--summary", required=True, help="Path to dist_summary.json")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots")
    parser.add_argument("--lumi", default="", help="Luminosity label for plot titles")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.read_parquet(args.dist)
    summary = json.loads(Path(args.summary).read_text())
    n_true = summary["n_true_decays"]

    sig = df[df["is_signal"]].copy()
    bkg = df[~df["is_signal"]].copy()
    n_sig = len(sig)
    n_bkg = len(bkg)

    print(f"\n{'=' * 60}")
    print(f"True Bs->mumu in data:     {n_true}")
    print(f"Reconstructed (signal):    {n_sig}")
    print(f"Reconstructed (bkg):       {n_bkg}")
    print(
        f"Efficiency:                {n_sig}/{n_true} = "
        f"{n_sig / max(n_true, 1) * 100:.1f}%"
    )
    print(
        f"Purity:                    {n_sig}/{n_sig + n_bkg} = "
        f"{n_sig / max(n_sig + n_bkg, 1) * 100:.1f}%"
    )
    print(f"{'=' * 60}")

    def _build_obs_arrays(subset):
        obs = {}
        obs["mass"] = subset["mass"].values
        obs["vtx_chi2"] = subset["vertex_chi2"].values
        obs["time_chi2"] = subset["pair_time_chi2"].values
        obs["doca"] = (
            subset["max_doca"].values
            if "max_doca" in subset.columns
            else np.zeros(len(subset))
        )
        obs["dira"] = subset["dira"].fillna(0.0).values
        obs["B_pt"] = subset["pt"].values
        obs["B_ip"] = subset["composite_ip"].fillna(0.0).values
        obs["B_ip_chi2"] = subset["composite_ip_chi2"].fillna(0.0).values
        daughter_pt_cols = sorted(
            [
                c
                for c in subset.columns
                if c.endswith("_pt") and (c.startswith("daughter"))
            ]
        )
        if daughter_pt_cols:
            daughter_pts = subset[daughter_pt_cols].values
            obs["min_track_pt"] = np.nanmin(daughter_pts, axis=1)
            obs["max_track_pt"] = np.nanmax(daughter_pts, axis=1)
        else:
            obs["min_track_pt"] = np.zeros(len(subset))
            obs["max_track_pt"] = np.zeros(len(subset))
        daughter_ip_cols = sorted(
            [
                c
                for c in subset.columns
                if c.endswith("_min_ip") and (c.startswith("daughter"))
            ]
        )
        if daughter_ip_cols:
            daughter_ips = subset[daughter_ip_cols].values
            obs["min_track_ip"] = np.nanmin(daughter_ips, axis=1)
            obs["max_track_ip"] = np.nanmax(daughter_ips, axis=1)
        else:
            obs["min_track_ip"] = np.zeros(len(subset))
            obs["max_track_ip"] = np.zeros(len(subset))
        return obs

    sig_obs = _build_obs_arrays(sig)
    bkg_obs = _build_obs_arrays(bkg)

    if n_sig > 0:
        print(f"\nSignal percentiles:")
        print(
            f"{'observable':<20}  {'1%':>8}  {'5%':>8}  {'median':>8}  "
            f"{'95%':>8}  {'99%':>8}"
        )
        print("-" * 68)
        for name in sig_obs:
            vals = sig_obs[name]
            p1, p5, p50, p95, p99 = np.percentile(vals, [1, 5, 50, 95, 99])
            print(
                f"{name:<20}  {p1:>8.4f}  {p5:>8.4f}  {p50:>8.4f}  "
                f"{p95:>8.4f}  {p99:>8.4f}"
            )

    # ---- Plots ----
    import matplotlib.pyplot as plt
    from trackcomb.plot import make_figure

    plot_configs = [
        ("mass", r"$m(\mu\mu)$ [GeV]", (4.5, 6.0), 50),
        ("vtx_chi2", r"Vertex $\chi^2$", (0, 25), 50),
        ("time_chi2", r"Pair time $\chi^2$", (0, 15), 50),
        ("doca", r"DOCA [mm]", (0, 0.5), 50),
        ("dira", r"DIRA", (0.9998, 1.0), 50),
        ("B_pt", r"$p_T(B_s^0)$ [GeV]", (0, 15), 50),
        ("B_ip", r"$B_s^0$ IP [mm]", (0, 0.2), 50),
        ("B_ip_chi2", r"$B_s^0$ IP $\chi^2$", (0, 0.006), 50),
        ("min_track_ip", r"min track IP [mm]", (0, 2), 50),
        ("max_track_ip", r"max track IP [mm]", (0, 5), 50),
        ("min_track_pt", r"min track $p_T$ [GeV]", (0, 5), 50),
        ("max_track_pt", r"max track $p_T$ [GeV]", (0, 10), 50),
    ]

    n_vars = len(plot_configs)
    ncols = min(n_vars, 3)
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = make_figure(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    axes = list(axes.flatten())

    for ax, (name, xlabel, xrange, nbins) in zip(axes, plot_configs):
        sig_v = sig_obs[name]
        bkg_v = bkg_obs[name]
        bins = np.linspace(xrange[0], xrange[1], nbins + 1)
        if len(sig_v) > 0:
            ax.hist(
                sig_v,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.5,
                color="steelblue",
                label=f"Signal ({len(sig_v)})",
            )
        if len(bkg_v) > 0:
            ax.hist(
                bkg_v,
                bins=bins,
                density=True,
                histtype="step",
                color="red",
                linewidth=1.5,
                label=f"Bkg ({len(bkg_v)})",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalised")
        ax.set_xlim(xrange)
        if name == "dira":
            ax.tick_params(axis="x", rotation=30)
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    for ax in axes[n_vars:]:
        ax.set_visible(False)

    lumi_tag = f" [{args.lumi}]" if args.lumi else ""
    fig.suptitle(f"Bs signal vs background (normalised){lumi_tag}")
    fig.tight_layout()
    fig.savefig(out_dir / "bs_observables.png", dpi=150)
    print(f"\nSaved {out_dir / 'bs_observables.png'}")


if __name__ == "__main__":
    main()
