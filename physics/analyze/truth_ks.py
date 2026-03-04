#!/usr/bin/env python3
"""Ks -> pi+pi- truth study analysis: load cheated Parquet, plot signal distributions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Ks -> pi+pi- truth analysis (from Parquet)",
    )
    parser.add_argument("--cheated", required=True, help="Path to cheated.parquet")
    parser.add_argument("--summary", required=True, help="Path to cheated_summary.json")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots")
    parser.add_argument("--lumi", default="", help="Luminosity label for plot titles")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.read_parquet(args.cheated)
    summary = json.loads(Path(args.summary).read_text())
    n_true = summary["n_true_decays"]
    n_events = summary["n_events"]

    # Filter signal only
    sig = df[df["is_signal"]].copy()
    n_sig = len(sig)

    print(f"\n{'='*60}")
    print(f"True Ks->pipi in data:     {n_true}")
    print(f"Cheated reco (signal):     {n_sig}")
    print(f"Cheated efficiency:        {n_sig}/{n_true} = "
          f"{n_sig/max(n_true,1)*100:.1f}%")
    print(f"{'='*60}")

    if n_sig == 0:
        print("No signal candidates found, exiting.")
        return

    # Derive observables from DataFrame columns
    def _get_doca(row):
        doca_cols = [c for c in df.columns if c.startswith("doca_")]
        if not doca_cols:
            return 0.0
        return max(row[c] for c in doca_cols if not np.isnan(row[c]))

    def _get_track_ips(row):
        ip_cols = [c for c in df.columns if c.startswith("ip_")]
        return [row[c] for c in ip_cols if not np.isnan(row[c])]

    # Build numpy arrays for each observable
    obs_values = {}
    obs_values["mass"] = sig["candidate_mass"].values
    obs_values["vertex_chi2"] = sig["vertex_chi2"].values
    obs_values["pair_time_chi2"] = sig["pair_time_chi2"].values

    # DOCA
    doca_cols = [c for c in sig.columns if c.startswith("doca_")]
    if doca_cols:
        obs_values["doca"] = sig[doca_cols].max(axis=1).values
    else:
        obs_values["doca"] = np.zeros(n_sig)

    obs_values["dira"] = sig["dira"].fillna(0.0).values
    obs_values["Ks_pt"] = sig["pair_pt"].values
    obs_values["Ks_ip"] = sig["composite_min_ip"].fillna(0.0).values
    obs_values["Ks_ip_chi2"] = sig["composite_min_ip_chi2"].fillna(0.0).values

    # Track IPs
    ip_cols = sorted([c for c in sig.columns if c.startswith("ip_")])
    if len(ip_cols) >= 2:
        ip_vals = sig[ip_cols].values
        obs_values["min_track_ip"] = np.nanmin(ip_vals, axis=1)
        obs_values["max_track_ip"] = np.nanmax(ip_vals, axis=1)
    else:
        obs_values["min_track_ip"] = np.zeros(n_sig)
        obs_values["max_track_ip"] = np.zeros(n_sig)

    obs_values["min_track_pt"] = sig["min_track_pt"].values
    obs_values["max_track_pt"] = sig["max_track_pt"].values

    # Print percentiles
    print(f"\n{'observable':<20}  {'min':>8}  {'1%':>8}  {'5%':>8}  {'median':>8}  "
          f"{'95%':>8}  {'99%':>8}  {'max':>8}")
    print("-" * 94)
    for name in obs_values:
        vals = obs_values[name]
        p1, p5, p50, p95, p99 = np.percentile(vals, [1, 5, 50, 95, 99])
        print(f"{name:<20}  {vals.min():>8.4f}  {p1:>8.4f}  {p5:>8.4f}  {p50:>8.4f}  "
              f"{p95:>8.4f}  {p99:>8.4f}  {vals.max():>8.4f}")

    # ---- Plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_configs = [
        ("mass",           r"$m(\pi^+\pi^-)$ [GeV]",    (0.47, 0.52), 60),
        ("vertex_chi2",    r"Vertex $\chi^2$",           (0, 25),      50),
        ("pair_time_chi2", r"Pair time $\chi^2$",        (0, 15),      50),
        ("doca",           r"DOCA [mm]",                 (0, 1),       50),
        ("dira",           r"DIRA",                      (0.9999, 1.0), 50),
        ("Ks_pt",          r"$p_T(K_S^0)$ [GeV]",       (0, 5),       50),
        ("Ks_ip",          r"$K_S^0$ IP [mm]",           (0, 2),       50),
        ("Ks_ip_chi2",     r"$K_S^0$ IP $\chi^2$",      (0, 2),       50),
        ("min_track_ip",   r"min track IP [mm]",         (0, 30),      50),
        ("max_track_ip",   r"max track IP [mm]",         (0, 50),      50),
        ("min_track_pt",   r"min track $p_T$ [GeV]",     (0, 2),       50),
        ("max_track_pt",   r"max track $p_T$ [GeV]",     (0, 3),       50),
    ]

    n_vars = len(plot_configs)
    ncols = min(n_vars, 3)
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = list(axes.flatten())

    for ax, (name, xlabel, xrange, nbins) in zip(axes, plot_configs):
        vals = obs_values[name]
        ax.hist(vals, bins=nbins, range=xrange,
                histtype="stepfilled", alpha=0.7, color="steelblue",
                label=f"Signal ({len(vals)})")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Candidates", fontsize=11)
        p1, p5, p95, p99 = np.percentile(vals, [1, 5, 95, 99])
        ax.axvline(p1, color="orange", ls=":", lw=1, alpha=0.7,
                   label=f"1%: {p1:.3f}")
        ax.axvline(p5, color="red", ls="--", lw=1, alpha=0.7,
                   label=f"5%: {p5:.3f}")
        ax.axvline(p95, color="red", ls="--", lw=1, alpha=0.7,
                   label=f"95%: {p95:.3f}")
        ax.axvline(p99, color="orange", ls=":", lw=1, alpha=0.7,
                   label=f"99%: {p99:.3f}")
        ax.set_xlim(xrange)
        if name == "dira":
            ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)

    for ax in axes[n_vars:]:
        ax.set_visible(False)

    lumi_tag = f" [{args.lumi}]" if args.lumi else ""
    fig.suptitle(
        f"$K_S^0 \\to \\pi^+\\pi^-$ cheated signal{lumi_tag} "
        f"({n_sig} candidates, {n_events} events)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ks_signal_distributions.png", dpi=150)
    print(f"\nSaved {out_dir / 'ks_signal_distributions.png'}")


if __name__ == "__main__":
    main()
