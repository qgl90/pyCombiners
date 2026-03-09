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
        return float(f"{m.group(1)}.{m.group(2)}") * 10 ** (
            int(m.group(3)) - 34
        )
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
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for plots"
    )
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

    # Compute overall S/B per lumi
    overall_sb = []
    for df in frames:
        sig = df[df["is_signal"]]
        bkg = df[~df["is_signal"]]
        ns, nb = len(sig), len(bkg)
        overall_sb.append(ns / max(nb, 1))

    # Print summary
    print(f"\n{'lumi':>10}  {'S':>6}  {'B':>6}  {'S/B':>10}")
    print("-" * 38)
    for lumi, df, sb in zip(lumis, frames, overall_sb):
        ns = df["is_signal"].sum()
        nb = (~df["is_signal"]).sum()
        print(f"{lumi:>10}  {ns:6d}  {nb:6d}  {sb:10.4f}")

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
    ax1.set_xticklabels(
        [f"{l}\n({v})" for l, v in zip(labels_sorted, x_sorted)]
    )
    ax1.set_xlabel(r"Luminosity [$\times 10^{34}$ cm$^{-2}$s$^{-1}$]")
    ax1.set_ylabel("S / B")
    ax1.set_title(r"$B_s^0 \to \mu^+\mu^-$ overall S/B vs luminosity")
    for xv, v in zip(x_sorted, sb_sorted):
        ax1.text(xv, v, f"{v:.4f}", ha="center", va="bottom")
    fig1.tight_layout()
    fig1.savefig(out_dir / "sb_ratio_vs_lumi.png", dpi=150)
    print(f"\nSaved {out_dir / 'sb_ratio_vs_lumi.png'}")

    # ---- Plot 2: mass distribution per lumi (signal vs background) ----
    n_lumis = len(lumis)
    ncols = min(n_lumis, 2)
    nrows = (n_lumis + ncols - 1) // ncols
    fig3, axes3 = make_figure(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    axes3 = np.atleast_1d(axes3).flatten()

    # Consistent mass range across all lumis
    all_mass = pd.concat([df["mass"] for df in frames])
    mass_range = (all_mass.min(), all_mass.max())
    mass_bins = 40

    for i in order:
        ax = axes3[list(order).index(i)]
        df = frames[i]
        lumi = lumis[i]
        sig = df[df["is_signal"]]
        bkg = df[~df["is_signal"]]
        ns, nb = len(sig), len(bkg)
        sb = ns / max(nb, 1)

        ax.hist(
            bkg["mass"].values,
            bins=mass_bins,
            range=mass_range,
            histtype="stepfilled",
            alpha=0.5,
            color="salmon",
            label=f"Bkg ({nb})",
        )
        ax.hist(
            sig["mass"].values,
            bins=mass_bins,
            range=mass_range,
            histtype="stepfilled",
            alpha=0.7,
            color="steelblue",
            label=f"Sig ({ns})",
        )
        ax.set_title(f"{lumi}  (S/B={sb:.4f})")
        ax.set_xlabel(r"$m(\mu^+\mu^-)$ [MeV]")
        ax.set_ylabel("Candidates")
        ax.legend()

    for i in range(n_lumis, len(axes3)):
        axes3[i].set_visible(False)

    fig3.suptitle(r"$B_s^0 \to \mu^+\mu^-$ mass distribution by luminosity")
    fig3.tight_layout()
    fig3.savefig(out_dir / "mass_vs_lumi.png", dpi=150)
    print(f"Saved {out_dir / 'mass_vs_lumi.png'}")


if __name__ == "__main__":
    main()
