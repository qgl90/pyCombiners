#!/usr/bin/env python3
"""Track-to-PV association analysis: PV association efficiency study."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Track-to-PV association efficiency analysis",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Reconstruction output dir with pv_assoc.parquet",
    )
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for plots"
    )
    parser.add_argument(
        "--lumi", default="", help="Luminosity label for plot titles"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df_assoc = pd.read_parquet(input_dir / "pv_assoc.parquet")
    if df_assoc.empty:
        print("No association data, exiting.")
        return

    # PV association efficiency study
    from trackcomb.plot import make_figure

    group_key = ["event_id", "track_id"]
    df_assoc["abs_dt"] = df_assoc["dt_corrected"].abs()

    # IP-only baseline: for each track, find row with min IP
    idx_min_ip = df_assoc.groupby(group_key)["ip"].idxmin()
    best_ip_rows = df_assoc.loc[idx_min_ip]
    n_total = len(best_ip_rows)
    n_correct_ip = int(best_ip_rows["is_true_pv"].sum())

    eff_ip = n_correct_ip / max(n_total, 1) * 100
    print(
        f"\nBaseline (min IP only):    {n_correct_ip}/{n_total} = {eff_ip:.2f}%"
    )

    dt_thresholds = [
        0.01,
        0.02,
        0.03,
        0.05,
        0.07,
        0.1,
        0.15,
        0.2,
        0.3,
        0.5,
        1.0,
    ]
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

    fig, ax = make_figure(1, 1, figsize=(16, 9))

    cuts = [r[0] for r in eff_results]
    effs = [r[3] for r in eff_results]
    passed_frac = [r[1] / max(n_total, 1) * 100 for r in eff_results]

    ax.plot(cuts, effs, "o-", color="steelblue", label="Correct assoc. eff.")
    ax.plot(
        cuts,
        passed_frac,
        "s--",
        color="orange",
        label="Tracks with $\\geq 1$ PV passing",
    )
    ax.axhline(
        eff_ip,
        color="red",
        ls=":",
        lw=1.5,
        label=f"IP-only baseline: {eff_ip:.1f}%",
    )
    ax.set_xlabel(r"|$\Delta t_{corrected}$| threshold [ns]")
    ax.set_ylabel("Fraction [%]")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    lumi_tag = f" [{args.lumi}]" if args.lumi else ""
    fig.suptitle(
        f"PV association: min IP + time cut{lumi_tag} ({n_total} tracks)",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "track_pv_timing_scan.png", dpi=150)
    print(f"\nSaved {out_dir / 'track_pv_timing_scan.png'}")


if __name__ == "__main__":
    main()
