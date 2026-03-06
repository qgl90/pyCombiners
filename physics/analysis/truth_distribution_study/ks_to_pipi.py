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

    print(f"\n{'=' * 60}")
    print(f"True Ks->pipi in data:     {n_true}")
    print(f"Cheated reco (signal):     {n_sig}")
    print(
        f"Cheated efficiency:        {n_sig}/{n_true} = "
        f"{n_sig / max(n_true, 1) * 100:.1f}%"
    )
    print(f"{'=' * 60}")

    if n_sig == 0:
        print("No signal candidates found, exiting.")
        return

    # Build numpy arrays for each observable
    obs_values = {}
    obs_values["mass"] = sig["mass"].values
    obs_values["vertex_chi2"] = sig["vertex_chi2"].values
    obs_values["pair_time_chi2"] = sig["pair_time_chi2"].values
    obs_values["doca"] = (
        sig["max_doca"].values if "max_doca" in sig.columns else np.zeros(n_sig)
    )

    obs_values["dira"] = sig["dira"].fillna(0.0).values
    obs_values["Ks_pt"] = sig["pt"].values
    obs_values["Ks_ip"] = sig["composite_ip"].fillna(0.0).values
    obs_values["Ks_ip_chi2"] = sig["composite_ip_chi2"].fillna(0.0).values

    # Track-level observables from daughter columns
    daughter_pt_cols = sorted(
        [c for c in sig.columns if c.endswith("_pt") and (c.startswith("daughter"))]
    )
    if daughter_pt_cols:
        daughter_pts = sig[daughter_pt_cols].values
        obs_values["min_track_pt"] = np.nanmin(daughter_pts, axis=1)
        obs_values["max_track_pt"] = np.nanmax(daughter_pts, axis=1)
    else:
        obs_values["min_track_pt"] = np.zeros(n_sig)
        obs_values["max_track_pt"] = np.zeros(n_sig)

    daughter_ip_cols = sorted(
        [c for c in sig.columns if c.endswith("_min_ip") and (c.startswith("daughter"))]
    )
    if daughter_ip_cols:
        daughter_ips = sig[daughter_ip_cols].values
        obs_values["min_track_ip"] = np.nanmin(daughter_ips, axis=1)
        obs_values["max_track_ip"] = np.nanmax(daughter_ips, axis=1)
    else:
        obs_values["min_track_ip"] = np.zeros(n_sig)
        obs_values["max_track_ip"] = np.zeros(n_sig)

    # Print percentiles
    print(
        f"\n{'observable':<20}  {'min':>8}  {'1%':>8}  {'5%':>8}  {'median':>8}  "
        f"{'95%':>8}  {'99%':>8}  {'max':>8}"
    )
    print("-" * 94)
    for name in obs_values:
        vals = obs_values[name]
        p1, p5, p50, p95, p99 = np.percentile(vals, [1, 5, 50, 95, 99])
        print(
            f"{name:<20}  {vals.min():>8.4f}  {p1:>8.4f}  {p5:>8.4f}  {p50:>8.4f}  "
            f"{p95:>8.4f}  {p99:>8.4f}  {vals.max():>8.4f}"
        )

    # ---- Plots ----
    from trackcomb.plot import make_figure

    plot_configs = [
        ("mass", r"$m(\pi^+\pi^-)$ [MeV]", (470, 520), 60),
        ("vertex_chi2", r"Vertex $\chi^2$", (0, 25), 50),
        ("pair_time_chi2", r"Pair time $\chi^2$", (0, 15), 50),
        ("doca", r"DOCA [mm]", (0, 1), 50),
        ("dira", r"DIRA", (0.9999, 1.0), 50),
        ("Ks_pt", r"$p_T(K_S^0)$ [MeV]", (0, 5000), 50),
        ("Ks_ip", r"$K_S^0$ IP [mm]", (0, 2), 50),
        ("Ks_ip_chi2", r"$K_S^0$ IP $\chi^2$", (0, 2), 50),
        ("min_track_ip", r"min track IP [mm]", (0, 30), 50),
        ("max_track_ip", r"max track IP [mm]", (0, 50), 50),
        ("min_track_pt", r"min track $p_T$ [MeV]", (0, 2000), 50),
        ("max_track_pt", r"max track $p_T$ [MeV]", (0, 3000), 50),
    ]

    n_vars = len(plot_configs)
    ncols = min(n_vars, 3)
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = make_figure(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    axes = list(axes.flatten())

    for ax, (name, xlabel, xrange, nbins) in zip(axes, plot_configs):
        vals = obs_values[name]
        ax.hist(
            vals,
            bins=nbins,
            range=xrange,
            histtype="stepfilled",
            alpha=0.7,
            color="steelblue",
            label=f"Signal ({len(vals)})",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Candidates")
        p1, p5, p95, p99 = np.percentile(vals, [1, 5, 95, 99])
        ax.axvline(p1, color="orange", ls=":", lw=1, alpha=0.7, label=f"1%: {p1:.3f}")
        ax.axvline(p5, color="red", ls="--", lw=1, alpha=0.7, label=f"5%: {p5:.3f}")
        ax.axvline(p95, color="red", ls="--", lw=1, alpha=0.7, label=f"95%: {p95:.3f}")
        ax.axvline(
            p99, color="orange", ls=":", lw=1, alpha=0.7, label=f"99%: {p99:.3f}"
        )
        ax.set_xlim(xrange)
        if name == "dira":
            ax.tick_params(axis="x", rotation=30)
        ax.legend()

    for ax in axes[n_vars:]:
        ax.set_visible(False)

    lumi_tag = f" [{args.lumi}]" if args.lumi else ""
    fig.suptitle(
        f"$K_S^0 \\to \\pi^+\\pi^-$ cheated signal{lumi_tag} "
        f"({n_sig} candidates, {n_events} events)",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ks_signal_distributions.png", dpi=150)
    print(f"\nSaved {out_dir / 'ks_signal_distributions.png'}")


if __name__ == "__main__":
    main()
