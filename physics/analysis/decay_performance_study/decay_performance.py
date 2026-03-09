#!/usr/bin/env python3
"""Unified decay performance analysis: efficiency vs kinematics + mass distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CHANNELS = {
    "ks_to_pipi": {
        "decay_latex": r"K_S^0 \to \pi^+\pi^-",
        "mass_latex": r"m(\pi^+\pi^-)",
        "pdg_mass": 497.611,
        "mass_range": (470, 520),
        "prefix": "ks",
        "variables": [
            ("pt", r"$p_T$ [MeV]", (0, 5000, 11)),
            ("eta", r"$\eta$", (2, 5, 13)),
            ("p", r"$p$ [MeV]", (0, 50000, 11)),
            ("vertex_z", r"Vertex $z$ [mm]", (-200, 800, 11)),
        ],
    },
    "bs_to_mumu": {
        "decay_latex": r"B_s^0 \to \mu^+\mu^-",
        "mass_latex": r"m(\mu^+\mu^-)",
        "pdg_mass": 5366.88,
        "mass_range": (4700, 6000),
        "prefix": "bs",
        "variables": [
            ("pt", r"$p_T$ [MeV]", (0, 15000, 16)),
            ("eta", r"$\eta$", (2, 5, 13)),
            ("p", r"$p$ [MeV]", (0, 200000, 11)),
            ("vertex_z", r"Vertex $z$ [mm]", (-200, 800, 11)),
        ],
    },
    "bs_to_jpsiphi": {
        "decay_latex": r"B_s^0 \to J/\psi(\mu\mu)\,\phi(KK)",
        "mass_latex": r"m(J/\psi\,\phi)",
        "pdg_mass": 5366.88,
        "mass_range": (5100, 5600),
        "prefix": "bs",
        "variables": [
            ("pt", r"$p_T$ [MeV]", (0, 20000, 16)),
            ("eta", r"$\eta$", (2, 5, 13)),
            ("p", r"$p$ [MeV]", (0, 300000, 11)),
            ("vertex_z", r"Vertex $z$ [mm]", (-200, 800, 11)),
        ],
        "intermediates": [
            {
                "column": "daughter0_mass",
                "name": "jpsi",
                "latex": r"J/\psi",
                "mass_latex": r"m(\mu^+\mu^-)",
                "pdg_mass": 3096.9,
                "mass_range": (2800, 3400),
            },
            {
                "column": "daughter1_mass",
                "name": "phi",
                "latex": r"\phi",
                "mass_latex": r"m(K^+K^-)",
                "pdg_mass": 1019.461,
                "mass_range": (990, 1060),
            },
        ],
    },
}


def main():
    parser = argparse.ArgumentParser(description="Decay performance analysis")
    parser.add_argument(
        "--cheated", required=True, help="Path to cheated parquet"
    )
    parser.add_argument(
        "--full", required=True, help="Path to full reco parquet"
    )
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for plots"
    )
    parser.add_argument(
        "--channel", required=True, choices=CHANNELS, help="Decay channel"
    )
    parser.add_argument(
        "--lumi", default="", help="Luminosity label for plot titles"
    )
    parser.add_argument(
        "--tag", default="", help="Extra tag in plot titles (e.g. 'NO TIMING')"
    )
    args = parser.parse_args()

    ch = CHANNELS[args.channel]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    cheated = pd.read_parquet(args.cheated)
    full = pd.read_parquet(args.full)

    reco = cheated[cheated["is_signal"]].copy()
    reco["p"] = np.sqrt(reco["px"] ** 2 + reco["py"] ** 2 + reco["pz"] ** 2)

    sel = full
    found = sel[sel["is_signal"]].copy()
    found["p"] = np.sqrt(
        found["px"] ** 2 + found["py"] ** 2 + found["pz"] ** 2
    )
    bkg = sel[~sel["is_signal"]]

    n_reco = len(reco)
    n_found = len(found)
    n_bkg = len(bkg)
    eff = n_found / max(n_reco, 1) * 100
    pur = n_found / max(n_found + n_bkg, 1) * 100

    tag_str = f" ({args.tag})" if args.tag else ""
    print(f"\n{'=' * 60}")
    print(f"Channel: {args.channel}{tag_str}")
    print(f"Reconstructible (cheated): {n_reco}")
    print(f"Found (full selection):    {n_found}")
    print(f"Background:                {n_bkg}")
    print(f"Efficiency: {n_found}/{n_reco} = {eff:.1f}%")
    print(f"Purity:     {n_found}/{n_found + n_bkg} = {pur:.1f}%")
    print(f"{'=' * 60}")

    # ---- Efficiency vs kinematics ----
    from trackcomb.plot import make_figure

    lumi_tag = f" [{args.lumi}]" if args.lumi else ""
    title_suffix = f" {args.tag}" if args.tag else ""

    fig, axes = make_figure(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (key, xlabel, (lo, hi, nbins)) in zip(axes, ch["variables"]):
        bins = np.linspace(lo, hi, nbins)
        num, _ = np.histogram(found[key].values, bins=bins)
        den, _ = np.histogram(reco[key].values, bins=bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff_bin = np.where(den > 0, num / den, np.nan)
            err = np.where(
                den > 0, np.sqrt(eff_bin * (1 - eff_bin) / den), np.nan
            )
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

    fig.suptitle(
        f"${ch['decay_latex']}$ efficiency{title_suffix}{lumi_tag} "
        f"({n_found}/{n_reco} = {eff:.0f}%)",
    )
    fig.tight_layout()
    prefix = ch["prefix"]
    fig.savefig(out_dir / f"{prefix}_eff_vs_kinematics.png", dpi=150)
    print(f"Saved {out_dir / f'{prefix}_eff_vs_kinematics.png'}")

    # ---- Mass distribution ----
    fig_m, ax_m = make_figure(figsize=(16, 12))
    ax_m.hist(
        sel["mass"].values,
        bins=50,
        range=ch["mass_range"],
        histtype="stepfilled",
        alpha=0.7,
        color="steelblue",
        label=f"All candidates ({len(sel)})",
    )
    ax_m.axvline(
        ch["pdg_mass"],
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"PDG mass",
    )
    ax_m.set_xlabel(f"${ch['mass_latex']}$ [MeV]")
    ax_m.set_ylabel("Candidates")
    ax_m.set_title(f"${ch['decay_latex']}$ mass{title_suffix}{lumi_tag}")
    ax_m.legend()
    fig_m.tight_layout()
    fig_m.savefig(out_dir / f"{prefix}_mass.png", dpi=150)
    print(f"Saved {out_dir / f'{prefix}_mass.png'}")

    # ---- Intermediate resonance mass distributions ----
    for inter in ch.get("intermediates", []):
        col = inter["column"]
        if col not in sel.columns:
            continue
        fig_i, ax_i = make_figure(figsize=(16, 12))
        ax_i.hist(
            sel[col].values,
            bins=50,
            range=inter["mass_range"],
            histtype="stepfilled",
            alpha=0.7,
            color="steelblue",
            label=f"All candidates ({len(sel)})",
        )
        ax_i.axvline(
            inter["pdg_mass"],
            color="red",
            linestyle="--",
            linewidth=1,
            label="PDG mass",
        )
        ax_i.set_xlabel(f"${inter['mass_latex']}$ [MeV]")
        ax_i.set_ylabel("Candidates")
        ax_i.set_title(f"${inter['latex']}$ mass{title_suffix}{lumi_tag}")
        ax_i.legend()
        fig_i.tight_layout()
        fname = f"{prefix}_{inter['name']}_mass.png"
        fig_i.savefig(out_dir / fname, dpi=150)
        print(f"Saved {out_dir / fname}")


if __name__ == "__main__":
    main()
