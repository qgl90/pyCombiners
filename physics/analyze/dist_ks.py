#!/usr/bin/env python3
"""Ks -> pi+pi- distribution analysis: load Parquet, plot signal vs background."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Ks -> pi+pi- distribution analysis (from Parquet)",
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

    print(f"\n{'='*60}")
    print(f"True Ks->pipi in data:     {n_true}")
    print(f"Reconstructed (signal):    {n_sig}")
    print(f"Reconstructed (bkg):       {n_bkg}")
    print(f"Efficiency:                {n_sig}/{n_true} = "
          f"{n_sig/max(n_true,1)*100:.1f}%")
    print(f"Purity:                    {n_sig}/{n_sig+n_bkg} = "
          f"{n_sig/max(n_sig+n_bkg,1)*100:.1f}%")
    print(f"{'='*60}")

    # Build observable extractor functions on DataFrame rows
    doca_cols = [c for c in df.columns if c.startswith("doca_")]
    ip_cols = sorted([c for c in df.columns if c.startswith("ip_")])

    def _build_obs_arrays(subset):
        """Build dict of observable name -> numpy array for a DataFrame subset."""
        obs = {}
        obs["mass"] = subset["candidate_mass"].values
        obs["vtx_chi2"] = subset["vertex_chi2"].values
        obs["time_chi2"] = subset["pair_time_chi2"].values
        obs["doca"] = subset[doca_cols].max(axis=1).values if doca_cols else np.zeros(len(subset))
        obs["dira"] = subset["dira"].fillna(0.0).values
        obs["Ks_pt"] = subset["pair_pt"].values
        obs["Ks_ip"] = subset["composite_min_ip"].fillna(0.0).values
        obs["Ks_ip_chi2"] = subset["composite_min_ip_chi2"].fillna(0.0).values
        if len(ip_cols) >= 2:
            ip_vals = subset[ip_cols].values
            obs["min_track_ip"] = np.nanmin(ip_vals, axis=1)
            obs["max_track_ip"] = np.nanmax(ip_vals, axis=1)
        else:
            obs["min_track_ip"] = np.zeros(len(subset))
            obs["max_track_ip"] = np.zeros(len(subset))
        obs["min_track_pt"] = subset["min_track_pt"].values
        obs["max_track_pt"] = subset["max_track_pt"].values
        return obs

    sig_obs = _build_obs_arrays(sig)
    bkg_obs = _build_obs_arrays(bkg)

    # Signal percentiles
    if n_sig > 0:
        print(f"\nSignal percentiles:")
        print(f"{'observable':<20}  {'1%':>8}  {'5%':>8}  {'median':>8}  "
              f"{'95%':>8}  {'99%':>8}")
        print("-" * 68)
        for name in sig_obs:
            vals = sig_obs[name]
            p1, p5, p50, p95, p99 = np.percentile(vals, [1, 5, 50, 95, 99])
            print(f"{name:<20}  {p1:>8.4f}  {p5:>8.4f}  {p50:>8.4f}  "
                  f"{p95:>8.4f}  {p99:>8.4f}")

    # ---- Plots: normalised signal vs background ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_configs = [
        ("mass",          r"$m(\pi\pi)$ [GeV]",        (0.47, 0.52),  60),
        ("vtx_chi2",      r"Vertex $\chi^2$",          (0, 25),       50),
        ("time_chi2",     r"Pair time $\chi^2$",       (0, 15),       50),
        ("doca",          r"DOCA [mm]",                (0, 1),        50),
        ("dira",          r"DIRA",                     (0.9999, 1.0), 50),
        ("Ks_pt",         r"$p_T(K_S^0)$ [GeV]",      (0, 5),        50),
        ("Ks_ip",         r"$K_S^0$ IP [mm]",          (0, 2),        50),
        ("Ks_ip_chi2",    r"$K_S^0$ IP $\chi^2$",     (0, 2),        50),
        ("min_track_ip",  r"min track IP [mm]",        (0, 30),       50),
        ("max_track_ip",  r"max track IP [mm]",        (0, 50),       50),
        ("min_track_pt",  r"min track $p_T$ [GeV]",    (0, 2),        50),
        ("max_track_pt",  r"max track $p_T$ [GeV]",    (0, 3),        50),
    ]

    n_vars = len(plot_configs)
    ncols = min(n_vars, 3)
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = list(axes.flatten())

    for ax, (name, xlabel, xrange, nbins) in zip(axes, plot_configs):
        sig_v = sig_obs[name]
        bkg_v = bkg_obs[name]
        bins = np.linspace(xrange[0], xrange[1], nbins + 1)
        if len(sig_v) > 0:
            ax.hist(sig_v, bins=bins, density=True, histtype="stepfilled",
                    alpha=0.5, color="steelblue", label=f"Signal ({len(sig_v)})")
        if len(bkg_v) > 0:
            ax.hist(bkg_v, bins=bins, density=True, histtype="step",
                    color="red", linewidth=1.5, label=f"Bkg ({len(bkg_v)})")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Normalised", fontsize=11)
        ax.set_xlim(xrange)
        if name == "dira":
            ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    for ax in axes[n_vars:]:
        ax.set_visible(False)

    lumi_tag = f" [{args.lumi}]" if args.lumi else ""
    fig.suptitle(f"Ks signal vs background (normalised){lumi_tag}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "ks_observables.png", dpi=150)
    print(f"\nSaved {out_dir / 'ks_observables.png'}")


if __name__ == "__main__":
    main()
