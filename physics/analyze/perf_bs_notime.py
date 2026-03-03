#!/usr/bin/env python3
"""Bs -> mu+mu- (NO TIMING) performance analysis: load Parquet, apply post-cuts, plot."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PDG_BS_MASS = 5.36688  # GeV


def main():
    parser = argparse.ArgumentParser(
        description="Bs -> mu+mu- performance analysis - NO TIMING (from Parquet)",
    )
    parser.add_argument("--cheated", required=True, help="Path to cheated.parquet")
    parser.add_argument("--full", required=True,
                        help="Path to full_notime.parquet")
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
    reco["p"] = np.sqrt(reco["px"]**2 + reco["py"]**2 + reco["pz"]**2)

    # ---- Post-combination cuts (same as perf_bs, minus timing) ----
    sel = full.copy()
    sel = sel[(sel["dira"].isna()) | (sel["dira"] >= 0.9995)]
    sel = sel[(sel["composite_min_ip"].isna()) | (sel["composite_min_ip"] < 0.1)]
    sel = sel[(sel["composite_min_ip_chi2"].isna()) | (sel["composite_min_ip_chi2"] < 0.01)]
    sel = sel[sel["pair_pt"] >= 1.0]
    sel = sel[sel["max_track_pt"] >= 2.0]

    found = sel[sel["is_signal"]].copy()
    found["p"] = np.sqrt(found["px"]**2 + found["py"]**2 + found["pz"]**2)
    bkg = sel[~sel["is_signal"]]

    n_reco = len(reco)
    n_found = len(found)
    n_bkg = len(bkg)
    eff = n_found / max(n_reco, 1) * 100
    pur = n_found / max(n_found + n_bkg, 1) * 100

    print(f"\n{'='*60}")
    print(f"[NO TIMING]")
    print(f"Reconstructible (cheated): {n_reco}")
    print(f"Found (full selection):    {n_found}")
    print(f"Background:                {n_bkg}")
    print(f"Efficiency: {n_found}/{n_reco} = {eff:.1f}%")
    print(f"Purity:     {n_found}/{n_found+n_bkg} = {pur:.1f}%")
    print(f"{'='*60}")

    # ---- Efficiency vs kinematics ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variables = [
        ("pair_pt",  r"$p_T(B_s^0)$ [GeV]",  np.linspace(0, 15, 16)),
        ("pair_eta", r"$\eta(B_s^0)$",        np.linspace(2, 5, 13)),
        ("p",        r"$p(B_s^0)$ [GeV]",     np.linspace(0, 200, 11)),
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
        f"$B_s^0 \\to \\mu^+\\mu^-$ efficiency (NO TIMING){lumi_tag} "
        f"({n_found}/{n_reco} = {eff:.0f}%)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "bs_eff_vs_kinematics.png", dpi=150)
    print(f"\nSaved {out_dir / 'bs_eff_vs_kinematics.png'}")

    # ---- Mass distribution ----
    fig_m, ax_m = plt.subplots(figsize=(8, 5))
    ax_m.hist(sel["candidate_mass"].values, bins=50, range=(4.7, 6.0),
              histtype="stepfilled", alpha=0.7, color="steelblue",
              label=f"All candidates ({len(sel)})")
    ax_m.axvline(PDG_BS_MASS, color="red", linestyle="--", linewidth=1,
                 label=r"PDG $m(B_s^0)$")
    ax_m.set_xlabel(r"$m(\mu^+\mu^-)$ [GeV]", fontsize=13)
    ax_m.set_ylabel("Candidates", fontsize=13)
    ax_m.set_title(
        f"$B_s^0 \\to \\mu^+\\mu^-$ mass (NO TIMING){lumi_tag}",
        fontsize=14,
    )
    ax_m.legend(fontsize=11)
    fig_m.tight_layout()
    fig_m.savefig(out_dir / "bs_mass.png", dpi=150)
    print(f"Saved {out_dir / 'bs_mass.png'}")


if __name__ == "__main__":
    main()
