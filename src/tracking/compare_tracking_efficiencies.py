#!/usr/bin/env python3
"""Compare tracking efficiencies and ghost rates across Parquet samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trackcomb.plot import make_figure
from tracking.tracking_efficiencies import (
    COMMON_TAGS,
    KINEMATIC_LABELS,
    TRACK_TYPES,
    _performance_tables,
    _selection,
    _set_track_type_tags,
)


VARIABLES = ("pt", "eta", "p", "phi")


def _integrated_summary(frame, label, track_type, tags):
    _set_track_type_tags(frame)
    reconstructible = frame[frame["row_type"] == "reconstructible"]
    long_tracks = frame[frame["row_type"] == "long"]
    efficiency_denominator = int(
        _selection(reconstructible, track_type, tags).sum()
    )
    efficiency_numerator = int(
        (
            long_tracks["truth_matched"]
            & long_tracks["is_unique_truth_match"]
            & _selection(long_tracks, track_type, tags)
        ).sum()
    )
    ghost_numerator = int((~long_tracks["truth_matched"]).sum())
    ghost_denominator = len(long_tracks)
    return {
        "sample_label": label,
        "track_type": track_type,
        "selection_tags": ",".join(tags),
        "efficiency_numerator": efficiency_numerator,
        "efficiency_denominator": efficiency_denominator,
        "efficiency": efficiency_numerator / max(efficiency_denominator, 1),
        "ghost_numerator": ghost_numerator,
        "ghost_denominator": ghost_denominator,
        "ghost_rate": ghost_numerator / max(ghost_denominator, 1),
    }


def _plot_comparison(
    tables,
    value,
    uncertainty,
    ylabel,
    title,
    output,
):
    import matplotlib.pyplot as plt

    denominator_field = {
        "efficiency": "efficiency_denominator",
        "fake_rate": "fake_denominator",
    }[value]
    fig, axes = make_figure(1, len(VARIABLES), figsize=(28, 6))
    for axis, variable in zip(axes, VARIABLES):
        distribution_axis = axis.twinx()
        for label, table in tables.items():
            points = table[table["variable"] == variable]
            centers = 0.5 * (points["bin_low"] + points["bin_high"])
            distribution_edges = np.linspace(
                points["bin_low"].iloc[0],
                points["bin_high"].iloc[-1],
                100,
            )
            axis.errorbar(
                centers,
                100.0 * points[value],
                yerr=100.0 * points[uncertainty],
                fmt="o-",
                markersize=4,
                capsize=2,
                label=label,
            )
            distribution_axis.stairs(
                points[denominator_field],
                distribution_edges,
                fill=True,
                color="gray",
                alpha=0.08,
            )
        axis.set_xlabel(KINEMATIC_LABELS[variable])
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 105.0)
        axis.grid(True, alpha=0.3)
        axis.legend()
        distribution_axis.set_ylabel("Denominator entries / bin", color="0.4")
        distribution_axis.tick_params(axis="y", colors="0.4")
        distribution_axis.set_ylim(bottom=0.0)
        distribution_axis.set_zorder(0)
        axis.set_zorder(1)
        axis.patch.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare(inputs, out_dir, track_types, tags):
    """Build comparison tables and plots from labeled particle Parquets."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {label: pd.read_parquet(path) for label, path in inputs}

    binned_parts = []
    summary_rows = []
    per_type_tables = {}
    for track_type in track_types:
        labeled_tables = {}
        for label, frame in frames.items():
            table = _performance_tables(frame.copy(), track_type, tags)
            table["sample_label"] = label
            binned_parts.append(table)
            labeled_tables[label] = table
            summary_rows.append(
                _integrated_summary(frame.copy(), label, track_type, tags)
            )
        per_type_tables[track_type] = labeled_tables

    binned = pd.concat(binned_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    binned_path = out_dir / "tracking_comparison_binned.parquet"
    summary_path = out_dir / "tracking_comparison_summary.parquet"
    binned.to_parquet(binned_path, index=False)
    summary.to_parquet(summary_path, index=False)

    tag_label = " & ".join(tags) if tags else "inclusive"
    for track_type, labeled_tables in per_type_tables.items():
        _plot_comparison(
            labeled_tables,
            "efficiency",
            "efficiency_uncertainty",
            "Tracking efficiency [%]",
            f"Tracking efficiency: {track_type} & {tag_label}",
            out_dir / f"tracking_efficiency_comparison_{track_type}.png",
        )

    # Ghost rate is defined on the reconstructed Long collection and is
    # therefore independent of the MC reconstructibility category.
    ghost_tables = {
        label: per_type_tables[track_types[0]][label] for label in frames
    }
    _plot_comparison(
        ghost_tables,
        "fake_rate",
        "fake_rate_uncertainty",
        "Ghost rate [%]",
        "Reconstructed Long-track ghost rate",
        out_dir / "tracking_ghost_rate_comparison.png",
    )

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved {summary_path}")
    print(f"Saved {binned_path}")
    for track_type in track_types:
        print(
            f"Saved {out_dir / f'tracking_efficiency_comparison_{track_type}.png'}"
        )
    print(f"Saved {out_dir / 'tracking_ghost_rate_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare tracking performance across labeled Parquets"
    )
    parser.add_argument(
        "--input",
        nargs=2,
        action="append",
        required=True,
        metavar=("LABEL", "PARQUET"),
        help="repeat for every sample",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--track-types",
        nargs="+",
        choices=TRACK_TYPES,
        default=list(TRACK_TYPES),
    )
    parser.add_argument(
        "--selection-tags",
        nargs="*",
        choices=COMMON_TAGS,
        default=["from_signal"],
        help="additional ANDed truth tags; pass with no values for inclusive",
    )
    args = parser.parse_args()

    labels = [label for label, _ in args.input]
    if len(labels) != len(set(labels)):
        parser.error("--input labels must be unique")
    compare(args.input, args.out_dir, args.track_types, args.selection_tags)


if __name__ == "__main__":
    main()
