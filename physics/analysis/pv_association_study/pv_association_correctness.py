#!/usr/bin/env python3
"""Bs -> mu+mu- PV association correctness across reconstruction modes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trackcomb.plot import make_figure


def main():
    parser = argparse.ArgumentParser(
        description="Bs -> mu+mu- PV association study",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing full.parquet, full_notime.parquet, full_pvtag.parquet",
    )
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for plots"
    )
    parser.add_argument(
        "--lumi", default="", help="Luminosity label for plot titles"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = {
        "full": "Full (with timing)",
        "full_notime": "Full (no timing)",
        "full_pvtag": "PV-tag",
    }
    colors = {
        "full": "steelblue",
        "full_notime": "darkorange",
        "full_pvtag": "forestgreen",
    }

    results = {}
    for mode, label in modes.items():
        path = data_dir / f"{mode}.parquet"
        if not path.exists():
            print(f"Skipping {mode}: {path} not found")
            continue
        df = pd.read_parquet(path)
        sig = df[df["is_signal"]]
        if len(sig) == 0:
            print(f"Skipping {mode}: no signal candidates")
            continue

        correct = sig["best_pv_mc_key"].values == sig["mc_pv_key"].values
        n_sig = len(sig)
        n_correct = int(correct.sum())
        pct = n_correct / n_sig * 100

        results[mode] = {
            "label": label,
            "n_sig": n_sig,
            "n_correct": n_correct,
            "pct": pct,
        }
        print(f"{label}: {n_correct}/{n_sig} = {pct:.1f}% correct PV")

    if not results:
        print("No results to plot")
        return

    lumi_tag = f" [{args.lumi}]" if args.lumi else ""

    fig, ax = make_figure(figsize=(8, 6))
    labels = [r["label"] for r in results.values()]
    pcts = [r["pct"] for r in results.values()]
    bar_colors = [colors[m] for m in results]
    bars = ax.bar(labels, pcts, color=bar_colors, width=0.5)
    for bar, r in zip(bars, results.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{r['pct']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=14,
        )
    ax.set_ylabel("Correct PV association [%]")
    ax.set_ylim(90, 102)
    ax.tick_params(axis="x", rotation=15)
    ax.set_title(rf"$B_s^0 \to \mu^+\mu^-$ PV association{lumi_tag}")
    fig.tight_layout()
    fig.savefig(out_dir / "pv_association_correctness.png", dpi=150)
    print(f"\nSaved {out_dir / 'pv_association_correctness.png'}")


if __name__ == "__main__":
    main()
