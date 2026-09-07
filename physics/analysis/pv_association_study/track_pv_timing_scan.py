#!/usr/bin/env python3
"""Plot track-to-PV timing scans from reconstruction Parquet output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NO_TIMING_METRIC = "none"


def _no_cut_position(thresholds):
    """Place the no-cut point one bounded visual step beyond the scan."""
    values = np.sort(np.asarray(thresholds, dtype=float))
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) > 1:
        log_step = np.median(np.diff(np.log(values)))
        factor = np.exp(np.clip(log_step, np.log(1.3), np.log(3.0)))
    else:
        factor = 2.0
    return values[-1] * factor


def _label_suffix(label):
    """Return a filename suffix without leaving a trailing underscore."""
    return f"_{label}" if label else ""


def _add_no_cut_endpoint(axis, line, x_last, y_last, x_no_cut, y_no_cut):
    """Connect a finite scan to its categorical no-cut endpoint."""
    color = line.get_color()
    axis.plot(
        [x_last, x_no_cut],
        [y_last, y_no_cut],
        linestyle=":",
        color=color,
        linewidth=1.5,
    )
    axis.plot(
        x_no_cut,
        y_no_cut,
        marker="*",
        markersize=12,
        color=color,
    )


def _label_no_cut_endpoint(axis, x_no_cut):
    axis.axvline(x_no_cut, color="0.5", linestyle=":", linewidth=1)
    axis.annotate(
        "No cut",
        xy=(x_no_cut, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-4, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=10,
        rotation=90,
    )
    axis.margins(x=0.05)


def _scan_summary(df):
    """Summarize one-row-per-signal-track timing scan points."""
    required = {
        "timing_metric",
        "timing_threshold",
        "n_pvs_considered",
        "has_true_pv",
        "true_pv_on_time",
        "best_is_true_pv",
        "run_number",
        "event_number",
        "n_tracks_event",
        "n_pvs_event",
        "mean_tracks_on_time_per_pv_event",
        "min_ip_all_pvs_index",
        "true_pv_index",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"scan Parquet is missing columns: {missing}")

    rows = []
    grouped = df.groupby(
        ["timing_metric", "timing_threshold"], sort=True, observed=True
    )
    for (metric, threshold), group in grouped:
        has_true = group["has_true_pv"].astype(bool)
        with_pv = group["n_pvs_considered"] >= 1
        with_second = group["n_pvs_considered"] >= 2
        true_on_time = group["true_pv_on_time"].astype(bool)
        best_is_true = group["best_is_true_pv"].astype(bool)

        # Event quantities are repeated for every signal track. Deduplicate
        # events before forming the PV-weighted reverse multiplicity.
        event_rows = group.drop_duplicates(["run_number", "event_number"])
        total_event_pvs = float(event_rows["n_pvs_event"].sum())
        n_on_time_relations = float(
            (
                event_rows["mean_tracks_on_time_per_pv_event"]
                * event_rows["n_pvs_event"]
            ).sum()
        )
        n_no_time_relations = float(
            (event_rows["n_tracks_event"] * event_rows["n_pvs_event"]).sum()
        )

        n_tracks = len(group)
        n_truth_matched = int(has_true.sum())
        n_with_pv_truth = int((with_pv & has_true).sum())
        n_true_on_time = int(true_on_time.sum())
        n_best_true = int(best_is_true.sum())
        best_all_is_true = has_true & (
            group["min_ip_all_pvs_index"] == group["true_pv_index"]
        )
        rows.append(
            {
                "timing_metric": metric,
                "timing_threshold": float(threshold),
                "n_signal_tracks": n_tracks,
                "n_truth_matched_tracks": n_truth_matched,
                "n_zero_pvs": int((~with_pv).sum()),
                "fraction_with_pv": float(with_pv.mean()),
                "fraction_with_second_pv": float(with_second.mean()),
                "true_pv_retention": n_true_on_time / max(n_truth_matched, 1),
                "best_pv_correct_efficiency": n_best_true
                / max(n_truth_matched, 1),
                "best_pv_correct_purity": n_best_true
                / max(n_with_pv_truth, 1),
                "mean_selected_pvs_per_signal_track": float(
                    group["n_pvs_considered"].mean()
                ),
                "mean_on_time_tracks_per_pv": n_on_time_relations
                / max(total_event_pvs, 1.0),
                "no_time_mean_pvs_per_signal_track": float(
                    group["n_pvs_event"].mean()
                ),
                "no_time_mean_tracks_per_pv": n_no_time_relations
                / max(total_event_pvs, 1.0),
                "no_time_best_pv_correct_efficiency": float(
                    best_all_is_true.sum() / max(n_truth_matched, 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_scan(summary, out_path, lumi=""):
    from trackcomb.plot import make_figure

    preferred_order = ["dt", "dt_chi2"]
    present = list(summary["timing_metric"].drop_duplicates())
    metrics = [m for m in preferred_order if m in present]
    metrics += [
        m for m in present if m not in metrics and m != NO_TIMING_METRIC
    ]
    no_cut_rows = summary[summary["timing_metric"] == NO_TIMING_METRIC]
    no_cut = no_cut_rows.iloc[0] if len(no_cut_rows) else None

    fig, axes = make_figure(
        len(metrics), 3, squeeze=False, figsize=(30, 8 * len(metrics))
    )
    labels = {
        "dt": r"$|\Delta t_{corrected}|$ threshold [ns]",
        "dt_chi2": r"$\Delta t^2/\sigma_{\Delta t}^2$ threshold",
    }

    for row, metric in enumerate(metrics):
        points = summary[summary["timing_metric"] == metric].sort_values(
            "timing_threshold"
        )
        x = points["timing_threshold"]
        ax_eff, ax_pvs, ax_tracks = axes[row]

        efficiency_curves = (
            ("fraction_with_pv", "At least one PV", "o-"),
            ("fraction_with_second_pv", "At least two PVs", "s--"),
            ("true_pv_retention", "True PV retained", "^-"),
            (
                "best_pv_correct_efficiency",
                "Minimum-IP PV is true",
                "D-.",
            ),
        )
        for column, label, style in efficiency_curves:
            values = 100.0 * points[column]
            (line,) = ax_eff.plot(x, values, style, label=label)
            if no_cut is not None:
                _add_no_cut_endpoint(
                    ax_eff,
                    line,
                    x.iloc[-1],
                    values.iloc[-1],
                    _no_cut_position(x),
                    100.0 * no_cut[column],
                )
        if no_cut is None:
            no_time_efficiency = (
                100.0 * points["no_time_best_pv_correct_efficiency"].iloc[0]
            )
            ax_eff.axhline(
                no_time_efficiency,
                color="black",
                linestyle=":",
                linewidth=2,
                label=(
                    "Minimum-IP PV is true, no time cut "
                    f"({no_time_efficiency:.1f}%)"
                ),
            )

        ax_eff.set_ylabel("Signal-track fraction [%]")
        ax_eff.set_ylim(0.0, 105.0)
        ax_eff.legend(fontsize=11)
        ax_eff.grid(True, alpha=0.3)

        (pv_line,) = ax_pvs.plot(
            x,
            points["mean_selected_pvs_per_signal_track"],
            "o-",
            label="With timing selection",
        )
        if no_cut is not None:
            _add_no_cut_endpoint(
                ax_pvs,
                pv_line,
                x.iloc[-1],
                points["mean_selected_pvs_per_signal_track"].iloc[-1],
                _no_cut_position(x),
                no_cut["mean_selected_pvs_per_signal_track"],
            )
        else:
            no_time_pvs = points["no_time_mean_pvs_per_signal_track"].iloc[0]
            ax_pvs.axhline(
                no_time_pvs,
                color="black",
                linestyle=":",
                linewidth=2,
                label=f"No time cut ({no_time_pvs:.2f})",
            )
        ax_pvs.set_ylabel("Mean selected PVs / signal track")
        ax_pvs.legend(fontsize=11)
        ax_pvs.grid(True, alpha=0.3)

        (track_line,) = ax_tracks.plot(
            x,
            points["mean_on_time_tracks_per_pv"],
            "o-",
            label="With timing selection",
        )
        if no_cut is not None:
            _add_no_cut_endpoint(
                ax_tracks,
                track_line,
                x.iloc[-1],
                points["mean_on_time_tracks_per_pv"].iloc[-1],
                _no_cut_position(x),
                no_cut["mean_on_time_tracks_per_pv"],
            )
        else:
            no_time_tracks = points["no_time_mean_tracks_per_pv"].iloc[0]
            ax_tracks.axhline(
                no_time_tracks,
                color="black",
                linestyle=":",
                linewidth=2,
                label=f"No time cut ({no_time_tracks:.2f})",
            )
        ax_tracks.set_ylabel("Mean on-time tracks / PV")
        ax_tracks.legend(fontsize=11)
        ax_tracks.grid(True, alpha=0.3)

        xlabel = labels.get(metric, f"{metric} threshold")
        panel_titles = (
            "Signal-track efficiency",
            "PV multiplicity per signal track",
            "Track multiplicity per PV",
        )
        for axis, title in zip(axes[row], panel_titles):
            axis.set_xlabel(xlabel)
            if np.all(x > 0):
                axis.set_xscale("log")
            if no_cut is not None:
                _label_no_cut_endpoint(axis, _no_cut_position(x))
            n_points = len(points) + int(no_cut is not None)
            axis.set_title(f"{title}: {metric} scan ({n_points} points)")

    n_tracks = int(summary["n_signal_tracks"].max())
    lumi_tag = f" [{lumi}]" if lumi else ""
    fig.suptitle(
        f"Signal-track to PV timing scan{lumi_tag} ({n_tracks} tracks)",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def _plot_legacy_pairs(df, out_path, lumi=""):
    """Keep support for the older one-row-per-track-PV input."""
    from trackcomb.plot import make_figure

    group_key = ["event_id", "track_id"]
    df = df.assign(abs_dt=df["dt_corrected"].abs())
    best_all = df.loc[df.groupby(group_key)["ip"].idxmin()]
    n_total = len(best_all)
    baseline = 100.0 * best_all["is_true_pv"].mean()
    thresholds = np.array(
        [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    )
    efficiencies, available = [], []
    for threshold in thresholds:
        selected = df[df["abs_dt"] <= threshold]
        best = selected.loc[selected.groupby(group_key)["ip"].idxmin()]
        efficiencies.append(100.0 * best["is_true_pv"].sum() / max(n_total, 1))
        available.append(100.0 * len(best) / max(n_total, 1))

    fig, ax = make_figure(1, 1, figsize=(16, 9))
    ax.plot(thresholds, efficiencies, "o-", label="Correct association")
    ax.plot(thresholds, available, "s--", label="At least one PV")
    ax.axhline(baseline, color="red", ls=":", label="All-PV baseline")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\Delta t_{corrected}|$ threshold [ns]")
    ax.set_ylabel("Signal-track fraction [%]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    lumi_tag = f" [{lumi}]" if lumi else ""
    fig.suptitle(f"Legacy track-PV timing scan{lumi_tag}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def _resolve_input(args):
    if args.input:
        return Path(args.input)
    input_dir = Path(args.input_dir)
    scan = input_dir / "signal_track_pv_scan.parquet"
    return scan if scan.exists() else input_dir / "pv_assoc.parquet"


def main():
    parser = argparse.ArgumentParser(
        description="Plot signal-track to PV dt and dt-chi2 scans"
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="scan Parquet file")
    inputs.add_argument(
        "--input-dir",
        help="legacy directory containing signal_track_pv_scan.parquet or pv_assoc.parquet",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="output directory for plots and summary",
    )
    parser.add_argument("--lumi", default="", help="label for plot titles")
    parser.add_argument("--label", default="", help="label for plot saving")

    args = parser.parse_args()

    input_path = _resolve_input(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(input_path)
    if df.empty:
        print("No association data, exiting.")
        return

    suffix = _label_suffix(args.label)
    plot_path = out_dir / f"track_pv_timing_scan{suffix}.png"
    if {"timing_metric", "timing_threshold"}.issubset(df.columns):
        summary = _scan_summary(df)
        summary_path = (
            out_dir / f"track_pv_timing_scan_summary{suffix}.parquet"
        )
        summary.to_parquet(summary_path, index=False)
        print(
            summary.to_string(index=False, float_format=lambda x: f"{x:.6g}")
        )
        _plot_scan(summary, plot_path, args.lumi)
        print(f"\nSaved {summary_path}")
    else:
        _plot_legacy_pairs(df, plot_path, args.lumi)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
