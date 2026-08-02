#!/usr/bin/env python3
"""PicoCal shower splitting for the Bs -> phi gamma photon.

One truth photon is often matched to several clusters (the shower is
split by the clusterizer). Profiles vs true energy: cluster
multiplicity, lead-satellite distance and lead-cluster energy fraction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from trackcomb import make_figure  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="PicoCal shower splitting profiles"
    )
    parser.add_argument(
        "--input",
        default="public/calo_study/reconstruction/bgamma_matches.parquet",
    )
    parser.add_argument("--out-dir", default="public/calo_study/analysis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    lead = df[(df["gamma_is_direct_bs"]) & (df["cluster_rank"] == 0)]
    n = lead["n_matched_clusters"].values
    print(f"direct-Bs photons with >=1 matched cluster: {len(lead)}")
    print(f"mean multiplicity = {n.mean():.2f}")
    print(f"split fraction (n >= 2) = {(n >= 2).mean():.1%}")

    # mean multiplicity vs true photon energy
    e_true = lead["gamma_e_true"].values / 1e3  # GeV
    edges = np.geomspace(2, 300, 16)
    centers = np.sqrt(edges[:-1] * edges[1:])
    means, stds = np.full(len(centers), np.nan), np.full(len(centers), np.nan)
    print(
        f"\n{'E range [GeV]':>15s} {'photons':>8s} {'mean(n)':>8s} {'std':>5s}"
    )
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        s = (e_true >= lo) & (e_true < hi)
        if s.sum() < 20:
            continue
        means[i], stds[i] = n[s].mean(), n[s].std()
        print(
            f"{lo:6.1f} - {hi:6.1f} {s.sum():8d} {means[i]:8.2f} "
            f"{stds[i]:5.2f}"
        )

    fig, ax = make_figure(figsize=(16, 12))
    ax.errorbar(
        centers,
        means,
        yerr=stds,
        fmt="o",
        color="steelblue",
        markersize=9,
        capsize=4,
        linewidth=1.8,
    )
    ax.axhline(1, color="grey", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"true photon energy $E_\gamma$ [GeV]")
    ax.set_ylabel(r"matched clusters per photon (mean $\pm$ std)")
    ax.set_ylim(0, None)
    ax.set_title(
        rf"$B_s^0 \to \phi\gamma$ photon cluster multiplicity vs energy "
        rf"({len(lead)} photons)"
    )
    fig.tight_layout()
    out = out_dir / "cluster_splitting_profile.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    # mean lead-satellite distance vs true photon energy
    key = ["run_number", "event_number", "gamma_key"]
    sat = df[(df["gamma_is_direct_bs"]) & (df["cluster_rank"] > 0)]
    pairs = sat[key + ["cluster_x", "cluster_y"]].merge(
        lead[key + ["cluster_x", "cluster_y", "gamma_e_true"]],
        on=key,
        suffixes=("", "_lead"),
    )
    dist = np.hypot(
        pairs["cluster_x"] - pairs["cluster_x_lead"],
        pairs["cluster_y"] - pairs["cluster_y_lead"],
    ).values
    e_pair = pairs["gamma_e_true"].values / 1e3
    print(f"\nsatellite clusters: {len(pairs)}")
    d_mean = np.full(len(centers), np.nan)
    d_std = np.full(len(centers), np.nan)
    print(
        f"{'E range [GeV]':>15s} {'sats':>6s} {'mean(d) [mm]':>12s} {'std':>5s}"
    )
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        s = (e_pair >= lo) & (e_pair < hi)
        if s.sum() < 20:
            continue
        d_mean[i], d_std[i] = dist[s].mean(), dist[s].std()
        print(
            f"{lo:6.1f} - {hi:6.1f} {s.sum():6d} {d_mean[i]:12.0f} "
            f"{d_std[i]:5.0f}"
        )

    fig, ax = make_figure(figsize=(16, 12))
    ax.errorbar(
        centers,
        d_mean,
        yerr=d_std,
        fmt="o",
        color="steelblue",
        markersize=9,
        capsize=4,
        linewidth=1.8,
    )
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"true photon energy $E_\gamma$ [GeV]")
    ax.set_ylabel(r"lead-satellite distance [mm] (mean $\pm$ std)")
    ax.set_ylim(0, None)
    ax.set_title(
        rf"$B_s^0 \to \phi\gamma$ satellite cluster distance vs energy "
        rf"({len(pairs)} satellites)"
    )
    fig.tight_layout()
    out = out_dir / "cluster_splitting_distance.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    # lead-cluster energy fraction vs true photon energy (split photons)
    frac = (lead["cluster_e"] / lead["sum_matched_e"]).values
    f_split = np.full(len(centers), np.nan)
    f_std = np.full(len(centers), np.nan)
    n_split_tot = int((n >= 2).sum())
    print(
        f"\n{'E range [GeV]':>15s} {'split':>6s} {'mean frac':>9s} {'std':>5s}"
    )
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sp = (e_true >= lo) & (e_true < hi) & (n >= 2)
        if sp.sum() < 20:
            continue
        f_split[i], f_std[i] = frac[sp].mean(), frac[sp].std()
        print(
            f"{lo:6.1f} - {hi:6.1f} {sp.sum():6d} {f_split[i]:9.3f} "
            f"{f_std[i]:5.3f}"
        )

    fig, ax = make_figure(figsize=(16, 12))
    ax.errorbar(
        centers,
        f_split,
        yerr=f_std,
        fmt="o",
        color="steelblue",
        markersize=9,
        capsize=4,
        linewidth=1.8,
    )
    ax.axhline(1, color="grey", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"true photon energy $E_\gamma$ [GeV]")
    ax.set_ylabel(
        r"$E_{\mathrm{lead}} / \sum E_{\mathrm{matched}}$ (mean $\pm$ std)"
    )
    ax.set_ylim(0, 1.15)
    ax.set_title(
        rf"$B_s^0 \to \phi\gamma$ lead-cluster energy fraction, "
        rf"split photons only ({n_split_tot} photons)"
    )
    fig.tight_layout()
    out = out_dir / "cluster_splitting_lead_fraction.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
