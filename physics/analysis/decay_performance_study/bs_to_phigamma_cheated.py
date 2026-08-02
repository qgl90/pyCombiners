#!/usr/bin/env python3
"""Bs -> phi gamma cheated mass plots.

phi and Bs mass from the cheated parquet, plus the raw-cluster-energy
Bs mass (lead cluster vs sum of the split clusters).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from trackcomb import make_figure  # noqa: E402
from trackcomb.fits import dscb, dscb_fit, gauss, gauss_fit  # noqa: E402

M_BS = 5366.9
M_PHI = 1019.461


def rebuild_mass(df, e_gamma):
    px = df["phi_px"].values + e_gamma * df["gamma_ux"].values
    py = df["phi_py"].values + e_gamma * df["gamma_uy"].values
    pz = df["phi_pz"].values + e_gamma * df["gamma_uz"].values
    e = df["phi_energy"].values + e_gamma
    m2 = e**2 - (px**2 + py**2 + pz**2)
    return np.sqrt(np.maximum(m2, 0.0))


def main():
    parser = argparse.ArgumentParser(
        description="Bs -> phi gamma cheated mass plots"
    )
    parser.add_argument(
        "--input",
        default="public/bs_to_phigamma/reconstruction/cheated.parquet",
    )
    parser.add_argument(
        "--out-dir",
        default="public/bs_to_phigamma/analysis",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    print(f"cheated candidates: {len(df)}")

    # phi mass
    mu, sigma, _, _ = gauss_fit(df["phi_mass"].values)
    print(f"phi mass: mu = {mu:.1f} MeV, core sigma = {sigma:.1f} MeV")
    rng, bins = (990, 1050), 60
    fig, ax = make_figure(figsize=(16, 12))
    counts, edges = np.histogram(df["phi_mass"], bins=bins, range=rng)
    centers = 0.5 * (edges[:-1] + edges[1:])
    core = np.abs(centers - mu) < 2 * sigma
    amp = counts[core].max() if core.any() else counts.max()
    ax.hist(
        df["phi_mass"],
        bins=bins,
        range=rng,
        histtype="stepfilled",
        alpha=0.7,
        color="steelblue",
    )
    xs = np.linspace(*rng, 500)
    ax.plot(
        xs,
        gauss(xs, amp, mu, sigma),
        color="red",
        linewidth=2,
        label=rf"core fit: $\mu$={mu:.1f}, $\sigma$={sigma:.1f} MeV",
    )
    ax.axvline(M_PHI, color="grey", linestyle="--", linewidth=1.5)
    ax.set_xlabel(r"$m(K^+K^-)$ [MeV]")
    ax.set_ylabel("candidates / 1 MeV")
    ax.legend(loc="upper right", fontsize=24)
    fig.tight_layout()
    out = out_dir / "phi_mass.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    # Bs mass with the cheated photon energy
    mass = df["mass"].values
    med = float(np.median(mass))
    half = 0.5 * (np.percentile(mass, 84) - np.percentile(mass, 16))
    lo, hi = med - 8 * half, med + 8 * half
    popt, chi2 = dscb_fit(mass, bins=60, rng=(lo, hi))
    print(
        f"Bs mass: mu = {popt[1]:.0f} MeV, sigma = {popt[2]:.0f} MeV, "
        f"chi2/ndf = {chi2:.1f}"
    )
    fig, ax = make_figure(figsize=(16, 12))
    ax.hist(
        mass,
        bins=60,
        range=(lo, hi),
        histtype="stepfilled",
        alpha=0.7,
        color="steelblue",
    )
    xs = np.linspace(lo, hi, 500)
    ax.plot(
        xs,
        dscb(xs, *popt),
        color="red",
        linewidth=2,
        label=rf"DSCB: $\mu$={popt[1]:.0f}, $\sigma$={popt[2]:.0f} MeV",
    )
    ax.axvline(M_BS, color="grey", linestyle="--", linewidth=1.5)
    ax.set_xlabel(r"$m(K^+K^-\gamma)$ [MeV]")
    ax.set_ylabel(f"candidates / {(hi - lo) / 60:.0f} MeV")
    ax.legend(loc="upper right", fontsize=24)
    fig.tight_layout()
    out = out_dir / "bs_mass.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    # Bs mass with the raw cluster energy: lead vs summed clusters
    m_lead = rebuild_mass(df, df["cluster_e_raw"].values)
    m_sum = rebuild_mass(df, df["sum_matched_e_raw"].values)
    for name, mm in [("lead", m_lead), ("sum", m_sum)]:
        print(
            f"raw-E mass ({name}): median = {np.median(mm):.0f} MeV, "
            f"frac m<4500 = {(mm < 4500).mean():.1%}"
        )
    rng, bins = (3500, 7500), 60
    fig, ax = make_figure(figsize=(16, 12))
    ax.hist(
        m_lead,
        bins=bins,
        range=rng,
        histtype="stepfilled",
        alpha=0.5,
        color="steelblue",
        label=r"$E_\gamma$ = lead cluster $e$",
    )
    ax.hist(
        m_sum,
        bins=bins,
        range=rng,
        histtype="step",
        linewidth=2.5,
        color="crimson",
        label=r"$E_\gamma = \sum$ matched clusters",
    )
    ax.axvline(M_BS, color="grey", linestyle="--", linewidth=1.5)
    ax.set_xlabel(r"$m(K^+K^-\gamma)$ [MeV]")
    ax.set_ylabel("candidates / 67 MeV")
    ax.legend(loc="upper right", fontsize=26)
    fig.tight_layout()
    out = out_dir / "mass_rawE.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
