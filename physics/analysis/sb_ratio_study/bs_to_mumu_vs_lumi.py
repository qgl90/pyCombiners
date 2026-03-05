#!/usr/bin/env python3
"""Bs -> mu+mu- signal-to-background ratio comparison across luminosities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# Luminosity label -> numerical value in units of 1e34 cm^-2 s^-1
LUMI_VALUES = {
    "run3": 0.2,
    "1p0e34": 1.0,
    "1p3e34": 1.3,
    "1p5e34": 1.5,
}


def _lumi_to_value(label):
    """Convert luminosity label to numerical value (1e34 units)."""
    if label in LUMI_VALUES:
        return LUMI_VALUES[label]
    # Try to parse from label pattern like "XpYeZ"
    import re

    m = re.match(r"(\d+)p(\d+)e(\d+)", label)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}") * 10 ** (int(m.group(3)) - 34)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Bs -> mu+mu- S/B ratio vs luminosity",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="lumi=path pairs, e.g. 1p5e34=full.parquet 1p3e34=full.parquet",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    # Parse lumi=path pairs
    lumis = []
    frames = []
    for item in args.inputs:
        lumi, path = item.split("=", 1)
        lumis.append(lumi)
        frames.append(pd.read_parquet(path))

    eta_bins = np.linspace(2, 5, 13)
    centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])

    # Compute S/B per lumi (overall and per eta bin)
    overall_sb = []
    per_eta_sb = []
    for df in frames:
        sig = df[df["is_signal"]]
        bkg = df[~df["is_signal"]]
        ns, nb = len(sig), len(bkg)
        overall_sb.append(ns / max(nb, 1))

        n_sig, _ = np.histogram(sig["eta"].values, bins=eta_bins)
        n_bkg, _ = np.histogram(bkg["eta"].values, bins=eta_bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_eta_sb.append(np.where(n_bkg > 0, n_sig / n_bkg, np.nan))

    # Print summary
    print(f"\n{'lumi':>10}  {'S':>6}  {'B':>6}  {'S/B':>10}")
    print("-" * 38)
    for lumi, df, sb in zip(lumis, frames, overall_sb):
        ns = df["is_signal"].sum()
        nb = (~df["is_signal"]).sum()
        print(f"{lumi:>10}  {ns:6d}  {nb:6d}  {sb:10.4f}")

    import matplotlib.pyplot as plt
    from trackcomb.plot import make_figure

    # ---- Plot 1: overall S/B vs lumi ----
    lumi_vals = [_lumi_to_value(l) for l in lumis]
    # Sort by luminosity value
    order = np.argsort(lumi_vals)
    x_sorted = [lumi_vals[i] for i in order]
    sb_sorted = [overall_sb[i] for i in order]
    labels_sorted = [lumis[i] for i in order]

    fig1, ax1 = make_figure(figsize=(16, 12))
    bar_width = 0.06
    ax1.bar(
        x_sorted,
        sb_sorted,
        width=bar_width,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xticks(x_sorted)
    ax1.set_xticklabels([f"{l}\n({v})" for l, v in zip(labels_sorted, x_sorted)])
    ax1.set_xlabel(r"Luminosity [$\times 10^{34}$ cm$^{-2}$s$^{-1}$]")
    ax1.set_ylabel("S / B")
    ax1.set_title(r"$B_s^0 \to \mu^+\mu^-$ overall S/B vs luminosity")
    for xv, v in zip(x_sorted, sb_sorted):
        ax1.text(xv, v, f"{v:.4f}", ha="center", va="bottom")
    fig1.tight_layout()
    fig1.savefig(out_dir / "bs_sb_vs_lumi.png", dpi=150)
    print(f"\nSaved {out_dir / 'bs_sb_vs_lumi.png'}")

    # ---- Plot 2: S/B vs eta, one curve per lumi ----
    fig2, ax2 = make_figure(figsize=(16, 12))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(lumis), 3)))
    for i, (lumi, sb_eta) in enumerate(zip(lumis, per_eta_sb)):
        mask = ~np.isnan(sb_eta)
        ax2.plot(
            centers[mask], sb_eta[mask], "o-", color=colors[i], label=lumi, markersize=5
        )
    ax2.set_xlabel(r"$\eta(B_s^0)$")
    ax2.set_ylabel("S / B")
    ax2.set_title(r"$B_s^0 \to \mu^+\mu^-$ S/B vs $\eta$ by luminosity")
    ax2.legend()
    ax2.set_xlim(eta_bins[0], eta_bins[-1])
    fig2.tight_layout()
    fig2.savefig(out_dir / "bs_sb_vs_eta_by_lumi.png", dpi=150)
    print(f"Saved {out_dir / 'bs_sb_vs_eta_by_lumi.png'}")


if __name__ == "__main__":
    main()
