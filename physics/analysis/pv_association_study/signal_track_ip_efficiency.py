#!/usr/bin/env python3
"""Evaluate signal-track IP-cut efficiency versus timing preselection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Signal-track IP efficiency versus dt/dt-chi2 selection"
    )
    parser.add_argument("--input", required=True, help="scan Parquet file")
    parser.add_argument("--min-ip-min", type=float)
    parser.add_argument("--min-ip-max", type=float)
    parser.add_argument("--second-ip-min", type=float)
    parser.add_argument("--second-ip-max", type=float)
    parser.add_argument(
        "--out",
        default="public/pv_association/ip_efficiency.parquet",
        help="output Parquet file",
    )
    args = parser.parse_args()

    cut_values = (
        args.min_ip_min,
        args.min_ip_max,
        args.second_ip_min,
        args.second_ip_max,
    )
    if all(value is None for value in cut_values):
        parser.error("provide at least one min-IP or second-IP cut")

    df = pd.read_parquet(args.input)
    if df.empty:
        print("No signal-track rows found.")
        return

    selected = np.ones(len(df), dtype=bool)
    if args.min_ip_min is not None:
        selected &= df["min_ip"].to_numpy() >= args.min_ip_min
    if args.min_ip_max is not None:
        selected &= df["min_ip"].to_numpy() <= args.min_ip_max
    if args.second_ip_min is not None:
        selected &= df["second_min_ip"].to_numpy() >= args.second_ip_min
    if args.second_ip_max is not None:
        selected &= df["second_min_ip"].to_numpy() <= args.second_ip_max
    df = df.assign(selected=selected)

    rows = []
    for (metric, threshold), group in df.groupby(
        ["timing_metric", "timing_threshold"], sort=True
    ):
        n_total = len(group)
        n_with_pv = int((group["n_pvs_considered"] >= 1).sum())
        n_with_second = int((group["n_pvs_considered"] >= 2).sum())
        n_selected = int(group["selected"].sum())
        rows.append(
            {
                "timing_metric": metric,
                "timing_threshold": threshold,
                "n_signal_tracks": n_total,
                "n_zero_pvs": n_total - n_with_pv,
                "fraction_with_pv": n_with_pv / max(n_total, 1),
                "fraction_with_second_pv": n_with_second / max(n_total, 1),
                "n_selected": n_selected,
                "efficiency": n_selected / max(n_total, 1),
            }
        )

    result = pd.DataFrame(rows)
    print(
        result.to_string(
            index=False, float_format=lambda value: f"{value:.6g}"
        )
    )
    out = Path(args.out)
    if out.suffix != ".parquet":
        parser.error("--out must have a .parquet extension")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
