#!/usr/bin/env python3
"""Track-to-PV association truth analysis: load Parquet, generate plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Track-to-PV association analysis (from Parquet)",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Reconstruct output dir with pv_stats.parquet + pv_assoc.parquet",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for plots")
    parser.add_argument("--lumi", default="", help="Luminosity label for plot titles")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.read_parquet(input_dir / "pv_stats.parquet")
    df_assoc = pd.read_parquet(input_dir / "pv_assoc.parquet")
    summary = json.loads((input_dir / "summary.json").read_text())

    n_matched = len(df)
    n_events = summary["n_events"]

    print(f"\n{'=' * 60}")
    print(f"Total tracks:              {summary['n_tracks_total']}")
    print(f"Truth tracks:              {summary['n_truth']}")
    print(f"Matched to true PV:        {n_matched}")
    print(f"No true PV found:          {summary['n_no_true_pv']}")
    print(f"{'=' * 60}")

    if n_matched == 0:
        print("No matched tracks, exiting.")
        return

    # Print percentiles
    observables = [
        ("IP [mm]", df["ip"].values),
        ("IP chi2", df["ip_chi2"].values),
        ("dt (raw) [ns]", df["dt_raw"].values),
        ("dt (flight corr) [ns]", df["dt_corrected"].values),
        ("track pT [GeV]", df["track_pt"].values),
        ("track eta", df["track_eta"].values),
    ]

    print(
        f"\n{'observable':<25}  {'1%':>8}  {'5%':>8}  {'median':>8}  "
        f"{'95%':>8}  {'99%':>8}"
    )
    print("-" * 78)
    for name, vals in observables:
        p1, p5, p50, p95, p99 = np.percentile(vals, [1, 5, 50, 95, 99])
        print(
            f"{name:<25}  {p1:>8.4f}  {p5:>8.4f}  {p50:>8.4f}  {p95:>8.4f}  {p99:>8.4f}"
        )

    # ---- Plot 1: Track-to-PV observables ----
    from trackcomb.plot import make_figure

    plot_configs = [
        ("IP [mm]", df["ip"].values, r"IP [mm]", (0, 0.5), 60),
        ("IP chi2", df["ip_chi2"].values, r"IP $\chi^2$", (0, 10), 60),
        (
            "dt (raw) [ns]",
            df["dt_raw"].values,
            r"$t_{track} - t_{PV}$ [ns]",
            (-0.5, 0.5),
            60,
        ),
        (
            "dt (flight corr) [ns]",
            df["dt_corrected"].values,
            r"$t_{track} - t_{flight} - t_{PV}$ [ns]",
            (-0.5, 0.5),
            60,
        ),
        ("track pT [GeV]", df["track_pt"].values, r"track $p_T$ [GeV]", (0, 5), 50),
        ("track eta", df["track_eta"].values, r"track $\eta$", (2, 5.5), 50),
    ]

    n_vars = len(plot_configs)
    ncols = min(n_vars, 3)
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = make_figure(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    axes = list(axes.flatten())

    for ax, (name, vals, xlabel, xrange, nbins) in zip(axes, plot_configs):
        ax.hist(
            vals,
            bins=nbins,
            range=xrange,
            histtype="stepfilled",
            alpha=0.7,
            color="steelblue",
            label=f"Tracks ({len(vals)})",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Tracks")
        p5, p95 = np.percentile(vals, [5, 95])
        ax.axvline(p5, color="red", ls="--", lw=1, alpha=0.7, label=f"5%: {p5:.4f}")
        ax.axvline(p95, color="red", ls="--", lw=1, alpha=0.7, label=f"95%: {p95:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for ax in axes[n_vars:]:
        ax.set_visible(False)

    lumi_tag = f" [{args.lumi}]" if args.lumi else ""
    fig.suptitle(
        f"Track-to-true-PV association{lumi_tag} ({n_matched} tracks, {n_events} events)",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "track_pv_association.png", dpi=150)
    print(f"\nSaved {out_dir / 'track_pv_association.png'}")

    # ---- Plot 2: PV association efficiency (vectorized) ----
    group_key = ["event_id", "track_id"]
    df_assoc["abs_dt"] = df_assoc["dt_corrected"].abs()

    # IP-only baseline: for each track, find row with min IP
    idx_min_ip = df_assoc.groupby(group_key)["ip"].idxmin()
    best_ip_rows = df_assoc.loc[idx_min_ip]
    n_total = len(best_ip_rows)
    n_correct_ip = int(best_ip_rows["is_true_pv"].sum())

    eff_ip = n_correct_ip / max(n_total, 1) * 100
    print(f"\nBaseline (min IP only):    {n_correct_ip}/{n_total} = {eff_ip:.2f}%")

    dt_thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    print(
        f"\n{'dt_cut [ns]':<15} {'passed':>8} {'correct':>8} "
        f"{'eff [%]':>10} {'assoc_eff [%]':>14}"
    )
    print("-" * 60)

    eff_results = []
    for dt_cut in dt_thresholds:
        filtered = df_assoc[df_assoc["abs_dt"] < dt_cut]
        if filtered.empty:
            eff_results.append((dt_cut, 0, 0, 0.0, 0.0))
            print(f"{dt_cut:<15.3f} {0:>8} {0:>8} {0.0:>10.2f} {0.0:>14.2f}")
            continue
        idx_min = filtered.groupby(group_key)["ip"].idxmin()
        best_rows = filtered.loc[idx_min]
        n_passed = len(best_rows)
        n_correct = int(best_rows["is_true_pv"].sum())
        eff = n_correct / max(n_total, 1) * 100
        assoc_eff = n_correct / max(n_passed, 1) * 100
        eff_results.append((dt_cut, n_passed, n_correct, eff, assoc_eff))
        print(
            f"{dt_cut:<15.3f} {n_passed:>8} {n_correct:>8} "
            f"{eff:>10.2f} {assoc_eff:>14.2f}"
        )

    fig2, (ax1, ax2) = make_figure(1, 2, figsize=(18, 9))

    cuts = [r[0] for r in eff_results]
    effs = [r[3] for r in eff_results]
    assoc_effs = [r[4] for r in eff_results]
    passed_frac = [r[1] / max(n_total, 1) * 100 for r in eff_results]

    ax1.plot(cuts, effs, "o-", color="steelblue", label="Correct assoc. eff.")
    ax1.plot(
        cuts,
        passed_frac,
        "s--",
        color="orange",
        label="Tracks with $\\geq 1$ PV passing",
    )
    ax1.axhline(
        eff_ip, color="red", ls=":", lw=1.5, label=f"IP-only baseline: {eff_ip:.1f}%"
    )
    ax1.set_xlabel(r"|$\Delta t_{corrected}$| threshold [ns]")
    ax1.set_ylabel("Fraction [%]")
    ax1.set_xscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Overall efficiency")

    ax2.plot(cuts, assoc_effs, "o-", color="steelblue", label="Correct / associated")
    ax2.axhline(
        eff_ip, color="red", ls=":", lw=1.5, label=f"IP-only baseline: {eff_ip:.1f}%"
    )
    ax2.set_xlabel(r"|$\Delta t_{corrected}$| threshold [ns]")
    ax2.set_ylabel("Association purity [%]")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Among associated tracks")

    fig2.suptitle(
        f"PV association: min IP + time cut{lumi_tag} "
        f"({n_total} tracks, {n_events} events)",
    )
    fig2.tight_layout()
    fig2.savefig(out_dir / "pv_assoc_efficiency.png", dpi=150)
    print(f"Saved {out_dir / 'pv_assoc_efficiency.png'}")


if __name__ == "__main__":
    main()
