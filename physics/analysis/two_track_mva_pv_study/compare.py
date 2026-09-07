#!/usr/bin/env python3
"""Compare TwoTrackMVA PV-scan summaries from several labelled samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs=2,
        action="append",
        metavar=("LABEL", "PARQUET"),
        default=[],
        help="repeat for inputs using the global --pv-policy",
    )
    parser.add_argument(
        "--series",
        nargs=3,
        action="append",
        metavar=("LABEL", "PARQUET", "PV_POLICY"),
        default=[],
        help="repeat to compare inputs with a different PV policy per series",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--track-sample", default="all_tracks")
    parser.add_argument("--pv-policy", default="common_on_time")
    parser.add_argument(
        "--charge",
        choices=("all", "opposite_sign", "same_sign"),
        default="all",
    )
    args = parser.parse_args()
    if not args.input and not args.series:
        parser.error("provide at least one --input or --series")

    series = [(label, path, args.pv_policy) for label, path in args.input] + [
        tuple(specification) for specification in args.series
    ]
    frames = []
    print("TwoTrackMVA comparison series:")
    for label, path, policy in series:
        frame = pd.read_parquet(path)
        selected = frame.loc[
            (frame["track_sample"] == args.track_sample)
            & (frame["pv_policy"] == policy)
            & (frame["charge_selection"] == args.charge)
        ].copy()
        if selected.empty:
            raise ValueError(
                f"series {label!r} has no rows for track_sample="
                f"{args.track_sample}, pv_policy={policy}, charge={args.charge}"
            )
        selected["comparison_label"] = label
        selected["comparison_pv_policy"] = policy
        selected["comparison_source"] = path
        frames.append(selected)
        print(f"  {label}: {path}; pv_policy={policy}")
    combined = pd.concat(frames, ignore_index=True)

    metrics = (
        (
            "pv_selected_event_fraction",
            "Events with >=1 selected PV / input events",
        ),
        (
            "mean_unique_pvs_per_selected_event",
            "Mean distinct PVs / selected event",
        ),
        (
            "signal_pv_retention_efficiency",
            "Signal-PV retention / truth-evaluable input events",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for (label, policy), group in combined.groupby(
        ["comparison_label", "comparison_pv_policy"], sort=False
    ):
        group = group.sort_values("mva_cut")
        for axis, (metric, ylabel) in zip(axes, metrics):
            axis.plot(
                group["mva_cut"],
                group[metric],
                marker="o",
                markersize=3,
                label=f"{label} ({policy})",
            )
            axis.set_xlabel("TwoTrackMVA cut")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle(
        f"{args.track_sample}; per-series PV policy; "
        f"{args.charge.replace('_', ' ')}"
    )
    figure.tight_layout()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    combined.to_parquet(output.with_suffix(".parquet"), index=False)
    print(f"Saved {output} and {output.with_suffix('.parquet')}")


if __name__ == "__main__":
    main()
