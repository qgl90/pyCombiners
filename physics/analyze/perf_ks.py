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

    # Restrict to shared events (cheated/full may use different max_events)
    common_events = set(cheated["event_id"]) & set(full["event_id"])
    cheated = cheated[cheated["event_id"].isin(common_events)]
    full = full[full["event_id"].isin(common_events)]

    # Denominator: truth-matched cheated candidates
    reco = cheated[cheated["is_signal"]].copy()
    # Candidate momentum from px, py, pz
    reco["p"] = np.sqrt(reco["px"]**2 + reco["py"]**2 + reco["pz"]**2)

    # Numerator: truth-matched full candidates (no additional post-cuts for Ks)
    found = full[full["is_signal"]].copy()
    found["p"] = np.sqrt(found["px"]**2 + found["py"]**2 + found["pz"]**2)

    bkg = full[~full["is_signal"]]

    n_reco = len(reco)
    n_found = len(found)
    n_bkg = len(bkg)
    eff = n_found / max(n_reco, 1) * 100
    pur = n_found / max(n_found + n_bkg, 1) * 100

    print(f"\nReconstructible: {n_reco}  Found: {n_found}  Bkg: {n_bkg}")
    print(f"Efficiency: {n_found}/{n_reco} = {eff:.1f}%")
    print(f"Purity: {n_found}/{n_found+n_bkg} = {pur:.1f}%")

    # ---- Efficiency vs kinematics ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variables = [
        ("pair_pt",  r"$p_T(K_S^0)$ [GeV]",  np.linspace(0, 5, 11)),
        ("pair_eta", r"$\eta(K_S^0)$",        np.linspace(2, 5, 13)),
        ("p",        r"$p(K_S^0)$ [GeV]",     np.linspace(0, 50, 11)),
        ("vertex_z", r"Vertex $z$ [mm]",       np.linspace(-200, 800, 11)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (key, xlabel, bins) in zip(axes, variables):
        num, _ = np.histogram(found[key].values, bins=bins)
        den, _ = np.histogram(reco[key].values, bins=bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff_bin = np.where(den > 0, num / den, np.nan)
            err = np.where(den > 0,
                           np.sqrt(eff_bin * (1 - eff_bin) / den),
                           np.nan)
        centers = 0.5 * (bins[:-1] + bins[1:])
        mask = den > 0
        ax.errorbar(centers[mask], eff_bin[mask], yerr=err[mask],
                    fmt="o", markersize=5, capsize=3, color="black")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Efficiency", fontsize=12)
        ax.set_ylim(-0.05, 1.15)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.5)

    lumi_tag = f" [{args.lumi}]" if args.lumi else ""
    fig.suptitle(
        f"$K_S^0 \\to \\pi^+\\pi^-$ efficiency{lumi_tag} "
        f"({n_found}/{n_reco} = {eff:.0f}%)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ks_eff_vs_kinematics.png", dpi=150)
    print(f"Saved {out_dir / 'ks_eff_vs_kinematics.png'}")

    # ---- Mass distribution ----
    fig_m, ax_m = plt.subplots(figsize=(8, 5))
    ax_m.hist(full["candidate_mass"].values, bins=50, range=(0.47, 0.52),
              histtype="stepfilled", alpha=0.7, color="steelblue",
              label=f"All candidates ({len(full)})")
    ax_m.axvline(PDG_KS_MASS, color="red", linestyle="--", linewidth=1,
                 label=r"PDG $m(K_S^0)$")
    ax_m.set_xlabel(r"$m(\pi^+\pi^-)$ [GeV]", fontsize=13)
    ax_m.set_ylabel("Candidates", fontsize=13)
    ax_m.set_title(
        f"$K_S^0 \\to \\pi^+\\pi^-$ mass{lumi_tag}",
        fontsize=14,
    )
    ax_m.legend(fontsize=11)
    fig_m.tight_layout()
    fig_m.savefig(out_dir / "ks_mass.png", dpi=150)
    print(f"Saved {out_dir / 'ks_mass.png'}")


if __name__ == "__main__":
    main()
