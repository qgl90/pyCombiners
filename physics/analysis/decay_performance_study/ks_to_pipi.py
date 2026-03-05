#!/usr/bin/env python3
"""Ks -> pi+pi- performance analysis: load Parquet, plot efficiency and mass."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PDG_KS_MASS = 0.497611  # GeV


def main():
    parser = argparse.ArgumentParser(
        description="Ks -> pi+pi- performance analysis (from Parquet)",
    )
    parser.add_argument("--cheated", required=True, help="Path to cheated.parquet")
    parser.add_argument("--full", required=True, help="Path to full.parquet")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots")
    parser.add_argument("--lumi", default="", help="Luminosity label for plot titles")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    cheated = pd.read_parquet(args.cheated)
    full = pd.read_parquet(args.full)

    # Denominator: truth-matched cheated candidates (all events, no common-event filter)
    reco = cheated[cheated["is_signal"]].copy()
    reco["p"] = np.sqrt(reco["px"] ** 2 + reco["py"] ** 2 + reco["pz"] ** 2)

    # Selection already applied in reconstruction
    sel = full
    found = sel[sel["is_signal"]].copy()
    found["p"] = np.sqrt(found["px"] ** 2 + found["py"] ** 2 + found["pz"] ** 2)
    bkg = sel[~sel["is_signal"]]

    n_reco = len(reco)
    n_found = len(found)
    n_bkg = len(bkg)
    eff = n_found / max(n_reco, 1) * 100
    pur = n_found / max(n_found + n_bkg, 1) * 100

    print(f"\nReconstructible: {n_reco}  Found: {n_found}  Bkg: {n_bkg}")
    print(f"Efficiency: {n_found}/{n_reco} = {eff:.1f}%")
    print(f"Purity: {n_found}/{n_found + n_bkg} = {pur:.1f}%")

    # ---- Efficiency vs kinematics ----
    from trackcomb.plot import make_figure

    variables = [
        ("pt", r"$p_T(K_S^0)$ [GeV]", np.linspace(0, 5, 11)),
        ("eta", r"$\eta(K_S^0)$", np.linspace(2, 5, 13)),
        ("p", r"$p(K_S^0)$ [GeV]", np.linspace(0, 50, 11)),
        ("vertex_z", r"Vertex $z$ [mm]", np.linspace(-200, 800, 11)),
    ]

    fig, axes = make_figure(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (key, xlabel, bins) in zip(axes, variables):
        num, _ = np.histogram(found[key].values, bins=bins)
        den, _ = np.histogram(reco[key].values, bins=bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff_bin = np.where(den > 0, num / den, np.nan)
            err = np.where(den > 0, np.sqrt(eff_bin * (1 - eff_bin) / den), np.nan)
        centers = 0.5 * (bins[:-1] + bins[1:])
        mask = den > 0
        ax.errorbar(
            centers[mask],
            eff_bin[mask],
            yerr=err[mask],
            fmt="o",
            markersize=5,
            capsize=3,
            color="black",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Efficiency")
        ax.set_ylim(-0.05, 1.15)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.5)

    lumi_tag = f" [{args.lumi}]" if args.lumi else ""
    fig.suptitle(
        f"$K_S^0 \\to \\pi^+\\pi^-$ efficiency{lumi_tag} "
        f"({n_found}/{n_reco} = {eff:.0f}%)",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ks_eff_vs_kinematics.png", dpi=150)
    print(f"Saved {out_dir / 'ks_eff_vs_kinematics.png'}")

    # ---- Mass distribution ----
    fig_m, ax_m = make_figure(figsize=(16, 12))
    ax_m.hist(
        sel["mass"].values,
        bins=50,
        range=(0.47, 0.52),
        histtype="stepfilled",
        alpha=0.7,
        color="steelblue",
        label=f"All candidates ({len(sel)})",
    )
    ax_m.axvline(
        PDG_KS_MASS, color="red", linestyle="--", linewidth=1, label=r"PDG $m(K_S^0)$"
    )
    ax_m.set_xlabel(r"$m(\pi^+\pi^-)$ [GeV]")
    ax_m.set_ylabel("Candidates")
    ax_m.set_title(
        f"$K_S^0 \\to \\pi^+\\pi^-$ mass{lumi_tag}",
    )
    ax_m.legend()
    fig_m.tight_layout()
    fig_m.savefig(out_dir / "ks_mass.png", dpi=150)
    print(f"Saved {out_dir / 'ks_mass.png'}")


if __name__ == "__main__":
    main()
