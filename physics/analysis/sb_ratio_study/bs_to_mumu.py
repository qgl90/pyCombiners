#!/usr/bin/env python3
"""Bs -> mu+mu- signal-to-background ratio vs eta."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Bs -> mu+mu- S/B ratio vs eta",
    )
    parser.add_argument("--full", required=True, help="Path to full.parquet")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots")
    parser.add_argument("--lumi", default="", help="Luminosity label for plot titles")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    full = pd.read_parquet(args.full)

    sig = full[full["is_signal"]]
    bkg = full[~full["is_signal"]]

    eta_bins = np.linspace(2, 5, 13)

    n_sig, _ = np.histogram(sig["eta"].values, bins=eta_bins)
    n_bkg, _ = np.histogram(bkg["eta"].values, bins=eta_bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        sb_ratio = np.where(n_bkg > 0, n_sig / n_bkg, np.nan)

    centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])
    mask = n_bkg > 0

    # Print summary
    total_sig = len(sig)
    total_bkg = len(bkg)
    total_sb = total_sig / max(total_bkg, 1)
    print(f"\nSignal: {total_sig}, Background: {total_bkg}, S/B = {total_sb:.4f}")
    print(f"\n{'eta bin':>12}  {'S':>6}  {'B':>6}  {'S/B':>10}")
    print("-" * 40)
    for i in range(len(centers)):
        sb_str = f"{sb_ratio[i]:.4f}" if mask[i] else "N/A"
        print(
            f"  {eta_bins[i]:.2f}-{eta_bins[i + 1]:.2f}  {n_sig[i]:6d}  {n_bkg[i]:6d}  {sb_str:>10}"
        )

    # Plot
    from trackcomb.plot import make_figure

    fig, ax = make_figure(figsize=(16, 12))
    ax.bar(
        centers[mask],
        sb_ratio[mask],
        width=eta_bins[1] - eta_bins[0],
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel(r"$\eta(B_s^0)$")
    ax.set_ylabel("S / B")

    lumi_tag = f" [{args.lumi}]" if args.lumi else ""
    ax.set_title(
        f"$B_s^0 \\to \\mu^+\\mu^-$ S/B vs $\\eta${lumi_tag}"
        f"  (S={total_sig}, B={total_bkg})",
    )
    ax.set_xlim(eta_bins[0], eta_bins[-1])
    fig.tight_layout()
    fig.savefig(out_dir / "bs_sb_vs_eta.png", dpi=150)
    print(f"\nSaved {out_dir / 'bs_sb_vs_eta.png'}")

    # ---- Mass distribution per eta bin ----
    n_eta_bins = len(eta_bins) - 1
    ncols = 4
    nrows = (n_eta_bins + ncols - 1) // ncols
    fig_m, axes_m = make_figure(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows))
    axes_m = axes_m.flatten()

    sig_eta = sig["eta"].values
    bkg_eta = bkg["eta"].values
    sig_mass = sig["mass"].values
    bkg_mass = bkg["mass"].values
    mass_range = (full["mass"].min(), full["mass"].max())
    mass_bins = 30

    for i in range(n_eta_bins):
        ax_i = axes_m[i]
        lo, hi = eta_bins[i], eta_bins[i + 1]
        s_mask = (sig_eta >= lo) & (sig_eta < hi)
        b_mask = (bkg_eta >= lo) & (bkg_eta < hi)
        ns, nb = s_mask.sum(), b_mask.sum()

        ax_i.hist(
            bkg_mass[b_mask],
            bins=mass_bins,
            range=mass_range,
            histtype="stepfilled",
            alpha=0.5,
            color="salmon",
            label=f"Bkg ({nb})",
        )
        ax_i.hist(
            sig_mass[s_mask],
            bins=mass_bins,
            range=mass_range,
            histtype="stepfilled",
            alpha=0.7,
            color="steelblue",
            label=f"Sig ({ns})",
        )
        sb_str = f"S/B={ns / nb:.3f}" if nb > 0 else "B=0"
        ax_i.set_title(f"$\\eta \\in [{lo:.2f}, {hi:.2f})$  {sb_str}")
        ax_i.legend()
        ax_i.set_xlabel(r"$m(\mu^+\mu^-)$ [MeV]")
        ax_i.set_ylabel("Candidates")

    for i in range(n_eta_bins, len(axes_m)):
        axes_m[i].set_visible(False)

    fig_m.suptitle(
        f"$B_s^0 \\to \\mu^+\\mu^-$ mass per $\\eta$ bin{lumi_tag}",
    )
    fig_m.tight_layout()
    fig_m.savefig(out_dir / "bs_mass_vs_eta.png", dpi=150)
    print(f"Saved {out_dir / 'bs_mass_vs_eta.png'}")


if __name__ == "__main__":
    main()
