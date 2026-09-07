#!/usr/bin/env python3
"""Scan TwoTrackMVA thresholds and plot candidate/PV-selection performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

NOMINAL_MVA_CUT = 0.9569
DEFAULT_THRESHOLDS = np.linspace(0.80, 0.99, 20)
POLICIES = ("all_pairs", "common_on_time", "same_best_pv")


def _stored_values(frame: pd.DataFrame, column: str) -> str:
    """Format distinct non-null values stored in a Parquet column."""
    if column not in frame:
        return "<not stored in this Parquet>"
    values = frame[column].dropna().drop_duplicates().tolist()
    if not values:
        return "None"
    return ", ".join(str(value) for value in values)


def print_analysis_setup(frame: pd.DataFrame, args) -> None:
    """Print producer settings and the exact downstream scan definitions."""
    thresholds = np.unique(
        np.append(np.asarray(args.thresholds, dtype=float), NOMINAL_MVA_CUT)
    )
    event_rows = frame.loc[~frame["is_candidate"].astype(bool)]
    candidate_rows = frame.loc[frame["is_candidate"].astype(bool)]

    print("TwoTrackMVA producer configuration stored in the Parquet:")
    print(f"  sample_label: {_stored_values(frame, 'sample_label')}")
    for field in (
        "max_dt_chi2",
        "require_common_pv_on_time",
        "max_vertex_chi2",
        "max_vertex_time_chi2",
        "model_path",
        "track_samples",
    ):
        print(f"  {field}: {_stored_values(frame, f'study_{field}')}")
    n_input_events = len(
        event_rows[["run_number", "event_number"]].drop_duplicates()
    )
    print(f"  input events: {n_input_events}")
    print(f"  event x track-sample rows: {len(event_rows)}")
    print(f"  pre-MVA candidate rows: {len(candidate_rows)}")

    print("TwoTrackMVA reconstruction logic:")
    print(
        "  tracks: pion hypothesis; pt >= 200 MeV; "
        "timed minIP >= 0.06 mm; chi2/ndof <= 10"
    )
    print(
        "  pair prefit: maxDOCA <= 1 mm; sum daughter pt >= 400 MeV; "
        "pair pt >= 1000 MeV"
    )
    print(
        "  daughter PV compatibility: common-on-time and same-best metadata; "
        "producer prefilter follows the stored configuration"
    )
    print(
        "  vertex: spatial chi2 and maxDOCA <= 0.2 mm; z >= -330 mm; "
        "optional time chi2"
    )
    print(
        "  composite PV: apply dt_chi2 mask, then choose minimum "
        "transverse-IP PV"
    )
    print(
        "  PV-dependent values: IPchi2, DIRA, FDchi2, flight eta, "
        "and mcor use that best PV"
    )
    print(
        "  final: 2 <= flight eta <= 5; mcor >= 1000 MeV; "
        "daughter minIPchi2 >= 4"
    )
    print("         daughter pt >= 200 MeV; composite minIPchi2 <= 16")
    print(
        "  MVA features: log(FDchi2), sum daughter pt [GeV], "
        "vertex chi2, log(min daughter IPchi2)"
    )

    print("Analysis configuration and definitions:")
    print(f"  input: {args.input}")
    print(f"  scan output: {args.out}")
    print(f"  plot directory: {args.plot_dir}")
    print(f"  thresholds: {thresholds.tolist()}")
    print(f"  PV policies scanned: {list(POLICIES)}")
    print("  charge categories scanned: ['all', 'opposite_sign', 'same_sign']")
    print(f"  charge category displayed in plots: {args.charge}")
    print(
        f"  PV policy displayed in raw multiplicity plots: {args.plot_pv_policy}"
    )
    print(f"  1D PV-multiplicity working point: {args.multiplicity_cut}")
    print("  selected event: >=1 passing candidate with a valid best-PV index")
    print(
        "  PV multiplicity: unique passing-candidate best-PV indices, "
        "conditional on selection"
    )
    print(
        "  signal-PV retention efficiency: P(signal PV in selected best-PV "
        "set | signal PV identifiable in the input event)",
    )
    print(
        "  signal-PV correctness: P(signal PV in selected best-PV set | "
        "selected, truth-evaluable); this conditional diagnostic need not "
        "be monotonic",
        flush=True,
    )


def _event_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["run_number"].astype(str)
        + ":"
        + frame["event_number"].astype(str)
    )


def _decode_indices(value) -> set[int]:
    if pd.isna(value):
        return set()
    if isinstance(value, str):
        return {int(index) for index in json.loads(value)}
    return {int(index) for index in value}


def _policy_mask(candidates: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "all_pairs":
        return pd.Series(True, index=candidates.index)
    if policy == "common_on_time":
        return (
            candidates["daughters_share_pv_on_time"].fillna(False).astype(bool)
        )
    if policy == "same_best_pv":
        return candidates["daughters_share_pv_on_time"].fillna(False).astype(
            bool
        ) & candidates["daughters_same_best_pv"].fillna(False).astype(bool)
    raise ValueError(f"unknown PV policy: {policy}")


def _charge_mask(candidates: pd.DataFrame, charge: str) -> pd.Series:
    same_sign = candidates["same_sign"].fillna(False).astype(bool)
    if charge == "all":
        return pd.Series(True, index=candidates.index)
    if charge == "opposite_sign":
        return ~same_sign
    if charge == "same_sign":
        return same_sign
    raise ValueError(f"unknown charge selection: {charge}")


def build_scan(
    frame: pd.DataFrame,
    thresholds=DEFAULT_THRESHOLDS,
    policies=POLICIES,
    charges=("all", "opposite_sign", "same_sign"),
) -> pd.DataFrame:
    """Summarize rates using explicit event rows as the denominator."""
    event_rows = frame.loc[~frame["is_candidate"].astype(bool)].copy()
    candidates = frame.loc[frame["is_candidate"].astype(bool)].copy()
    candidate_defaults = {
        "mva_response": pd.Series(dtype=float),
        "best_pv_index": pd.Series(dtype=int),
        "daughters_share_pv_on_time": pd.Series(dtype=bool),
        "daughters_same_best_pv": pd.Series(dtype=bool),
        "same_sign": pd.Series(dtype=bool),
    }
    for column, default in candidate_defaults.items():
        if column not in candidates:
            candidates[column] = default
    event_rows["event_key"] = _event_key(event_rows)
    candidates["event_key"] = _event_key(candidates)
    thresholds = np.unique(
        np.append(np.asarray(thresholds, dtype=float), NOMINAL_MVA_CUT)
    )
    rows = []

    for track_sample, events in event_rows.groupby("track_sample", sort=False):
        sample_candidates = candidates.loc[
            candidates["track_sample"] == track_sample
        ]
        n_events = len(events)
        signal_by_event = {
            key: _decode_indices(value)
            for key, value in zip(
                events["event_key"], events["signal_pv_indices_json"]
            )
        }
        n_signal_pvs = sum(
            len(indices) for indices in signal_by_event.values()
        )
        n_signal_evaluable_events = sum(
            bool(indices) for indices in signal_by_event.values()
        )

        for policy in policies:
            policy_candidates = sample_candidates.loc[
                _policy_mask(sample_candidates, policy)
            ]
            for charge in charges:
                selected_charge = policy_candidates.loc[
                    _charge_mask(policy_candidates, charge)
                ]
                for threshold in thresholds:
                    selected = selected_charge.loc[
                        selected_charge["mva_response"] >= threshold
                    ]
                    selected_pvs = (
                        selected.loc[selected["best_pv_index"] >= 0]
                        .groupby("event_key")["best_pv_index"]
                        .agg(lambda values: {int(value) for value in values})
                        .to_dict()
                    )
                    n_selected_pvs = sum(
                        len(value) for value in selected_pvs.values()
                    )
                    n_pv_selected_events = len(selected_pvs)
                    n_signal_selected = sum(
                        len(selected_pvs.get(key, set()) & signal_pvs)
                        for key, signal_pvs in signal_by_event.items()
                    )
                    selected_events_with_signal_truth = {
                        key
                        for key in selected_pvs
                        if signal_by_event.get(key, set())
                    }
                    signal_pv_found_events = sum(
                        bool(selected_pvs[key] & signal_by_event[key])
                        for key in selected_events_with_signal_truth
                    )
                    n_signal_evaluable_selected_events = len(
                        selected_events_with_signal_truth
                    )
                    rows.append(
                        {
                            "sample_label": events["sample_label"].iloc[0],
                            "track_sample": track_sample,
                            "pv_policy": policy,
                            "charge_selection": charge,
                            "mva_cut": threshold,
                            "n_events": n_events,
                            "n_candidates": len(selected),
                            "candidates_per_event": len(selected) / n_events,
                            "n_pv_selected_events": n_pv_selected_events,
                            "pv_selected_event_fraction": n_pv_selected_events
                            / n_events,
                            # Backward-compatible alias: after best-PV
                            # association an accepted event has >=1 selected PV.
                            "accepted_event_fraction": n_pv_selected_events
                            / n_events,
                            "n_selected_event_pvs": n_selected_pvs,
                            "mean_selected_pvs_per_event": n_selected_pvs
                            / n_events,
                            "mean_unique_pvs_per_selected_event": (
                                n_selected_pvs / n_pv_selected_events
                                if n_pv_selected_events
                                else np.nan
                            ),
                            "n_signal_event_pvs": n_signal_pvs,
                            "n_signal_evaluable_events": (
                                n_signal_evaluable_events
                            ),
                            "n_signal_event_pvs_selected": n_signal_selected,
                            "n_signal_evaluable_selected_events": (
                                n_signal_evaluable_selected_events
                            ),
                            "n_selected_events_without_signal_pv_truth": (
                                n_pv_selected_events
                                - n_signal_evaluable_selected_events
                            ),
                            "n_selected_events_signal_pv_found": (
                                signal_pv_found_events
                            ),
                            "signal_pv_efficiency_given_selected_event": (
                                signal_pv_found_events
                                / n_signal_evaluable_selected_events
                                if n_signal_evaluable_selected_events
                                else np.nan
                            ),
                            "signal_pv_correctness_given_selected_event": (
                                signal_pv_found_events
                                / n_signal_evaluable_selected_events
                                if n_signal_evaluable_selected_events
                                else np.nan
                            ),
                            "signal_pv_retention_efficiency": (
                                signal_pv_found_events
                                / n_signal_evaluable_events
                                if n_signal_evaluable_events
                                else np.nan
                            ),
                            "signal_pv_efficiency": (
                                n_signal_selected / n_signal_pvs
                                if n_signal_pvs
                                else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def plot_scan(scan: pd.DataFrame, output: Path, charge: str = "all") -> None:
    view = scan.loc[scan["charge_selection"] == charge]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
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
    styles = {
        "all_pairs": ":",
        "common_on_time": "-",
        "same_best_pv": "--",
    }
    for (track_sample, policy), group in view.groupby(
        ["track_sample", "pv_policy"]
    ):
        label = f"{track_sample}; {policy}"
        group = group.sort_values("mva_cut")
        for axis, (metric, ylabel) in zip(axes, metrics):
            axis.plot(
                group["mva_cut"],
                group[metric],
                linestyle=styles[policy],
                marker="o",
                markersize=3,
                label=label,
            )
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    for axis in axes:
        axis.axvline(NOMINAL_MVA_CUT, color="black", alpha=0.45, linestyle=":")
        axis.set_xlabel("TwoTrackMVA cut")
    axes[0].legend(fontsize=8)
    figure.suptitle(f"TwoTrackMVA PV study ({charge.replace('_', ' ')})")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_signal_pv_retention(
    scan: pd.DataFrame, output: Path, charge: str = "all"
) -> None:
    """Plot event-level signal-PV retention with a fixed denominator."""
    view = scan.loc[scan["charge_selection"] == charge]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    styles = {
        "all_pairs": ":",
        "common_on_time": "-",
        "same_best_pv": "--",
    }
    drew_curve = False
    for (track_sample, policy), group in view.groupby(
        ["track_sample", "pv_policy"]
    ):
        group = group.sort_values("mva_cut")
        valid = group["signal_pv_retention_efficiency"].notna()
        if not valid.any():
            continue
        group = group.loc[valid]
        denominator = int(group["n_signal_evaluable_events"].iloc[0])
        axis.plot(
            group["mva_cut"],
            group["signal_pv_retention_efficiency"],
            linestyle=styles[policy],
            marker="o",
            markersize=3,
            label=f"{track_sample}; {policy} (N={denominator})",
        )
        drew_curve = True
    axis.axvline(NOMINAL_MVA_CUT, color="black", alpha=0.45, linestyle=":")
    axis.set(
        xlabel="TwoTrackMVA cut",
        ylabel="Signal-PV retention efficiency",
        title=(
            "Signal PV contained in the selected unique-PV set; "
            f"{charge.replace('_', ' ')}"
        ),
        ylim=(0.0, 1.05),
    )
    axis.grid(alpha=0.25)
    if drew_curve:
        axis.legend(fontsize=8)
    else:
        axis.text(
            0.5,
            0.5,
            "No events with an identifiable signal PV",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_baseline(
    scan: pd.DataFrame, output: Path, charge: str = "all"
) -> None:
    nominal = scan.loc[
        np.isclose(scan["mva_cut"], NOMINAL_MVA_CUT)
        & (scan["charge_selection"] == charge)
        & (scan["pv_policy"] == "common_on_time")
    ].set_index("track_sample")
    metrics = (
        ("pv_selected_event_fraction", "Events with >=1 selected PV / events"),
        (
            "mean_unique_pvs_per_selected_event",
            "Mean distinct PVs / selected event",
        ),
        (
            "signal_pv_retention_efficiency",
            "Signal-PV retention / truth-evaluable input events",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    samples = [
        sample
        for sample in ("all_tracks", "truth_matched")
        if sample in nominal.index
    ]
    for axis, (metric, ylabel) in zip(axes, metrics):
        axis.bar(
            samples,
            nominal.loc[samples, metric],
            color=("tab:blue", "tab:orange"),
        )
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"Nominal cut {NOMINAL_MVA_CUT:g}: fake-track prefilter effect"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_candidate_diagnostics(
    frame: pd.DataFrame, output: Path, charge: str = "all"
) -> None:
    """Plot the MVA spectrum and nominal distinct-PV multiplicity."""
    events = frame.loc[~frame["is_candidate"].astype(bool)].copy()
    candidates = frame.loc[frame["is_candidate"].astype(bool)].copy()
    for column, dtype in {
        "mva_response": float,
        "best_pv_index": int,
        "daughters_share_pv_on_time": bool,
        "same_sign": bool,
    }.items():
        if column not in candidates:
            candidates[column] = pd.Series(dtype=dtype)
    events["event_key"] = _event_key(events)
    candidates["event_key"] = _event_key(candidates)
    candidates = candidates.loc[
        _policy_mask(candidates, "common_on_time")
        & _charge_mask(candidates, charge)
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for track_sample, group in candidates.groupby("track_sample", sort=False):
        axes[0].hist(
            group["mva_response"],
            bins=np.linspace(0.0, 1.0, 51),
            histtype="step",
            density=True,
            label=track_sample,
        )
    axes[0].axvline(NOMINAL_MVA_CUT, color="black", alpha=0.45, linestyle=":")
    axes[0].set(xlabel="TwoTrackMVA response", ylabel="Normalized candidates")
    axes[0].legend()

    for track_sample, _ in events.groupby("track_sample", sort=False):
        selected = candidates.loc[
            (candidates["track_sample"] == track_sample)
            & (candidates["mva_response"] >= NOMINAL_MVA_CUT)
            & (candidates["best_pv_index"] >= 0)
        ]
        multiplicity = (
            selected.groupby("event_key")["best_pv_index"].nunique().to_numpy()
        )
        if len(multiplicity) == 0:
            continue
        max_value = max(int(multiplicity.max(initial=0)), 1)
        axes[1].hist(
            multiplicity,
            bins=np.arange(0.5, max_value + 1.5),
            histtype="step",
            density=True,
            label=track_sample,
        )
    axes[1].set(
        xlabel="Distinct best PVs per selected event",
        ylabel="Fraction of selected events",
    )
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(
        f"Nominal candidate diagnostics ({charge.replace('_', ' ')})"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_pv_multiplicity_rows(
    frame: pd.DataFrame,
    thresholds,
    policy: str = "common_on_time",
    charge: str = "all",
) -> pd.DataFrame:
    """Return one row per selected event and MVA working point."""
    events = frame.loc[~frame["is_candidate"].astype(bool)].copy()
    candidates = frame.loc[frame["is_candidate"].astype(bool)].copy()
    events["event_key"] = _event_key(events)
    candidates["event_key"] = _event_key(candidates)
    candidates = candidates.loc[
        _policy_mask(candidates, policy) & _charge_mask(candidates, charge)
    ]
    thresholds = np.unique(
        np.append(np.asarray(thresholds, dtype=float), NOMINAL_MVA_CUT)
    )
    rows = []

    for track_sample, sample_events in events.groupby(
        "track_sample", sort=False
    ):
        event_n_pvs = sample_events.set_index("event_key")[
            "n_pvs_event"
        ].to_dict()
        sample_candidates = candidates.loc[
            candidates["track_sample"] == track_sample
        ]
        for threshold in thresholds:
            selected = sample_candidates.loc[
                (sample_candidates["mva_response"] >= threshold)
                & (sample_candidates["best_pv_index"] >= 0)
            ]
            multiplicities = selected.groupby("event_key")[
                "best_pv_index"
            ].nunique()
            for event_key, n_selected_pvs in multiplicities.items():
                rows.append(
                    {
                        "track_sample": track_sample,
                        "mva_cut": threshold,
                        "event_key": event_key,
                        "n_selected_pvs": int(n_selected_pvs),
                        "n_pvs_event": int(event_n_pvs[event_key]),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=(
            "track_sample",
            "mva_cut",
            "event_key",
            "n_selected_pvs",
            "n_pvs_event",
        ),
    )


def plot_raw_pv_multiplicity(
    frame: pd.DataFrame,
    thresholds,
    output: Path,
    policy: str = "common_on_time",
    charge: str = "all",
) -> pd.DataFrame:
    """Plot raw selected-PV counts versus working point and total event PVs."""
    multiplicity = build_pv_multiplicity_rows(
        frame, thresholds, policy=policy, charge=charge
    )
    threshold_values = np.unique(
        np.append(np.asarray(thresholds, dtype=float), NOMINAL_MVA_CUT)
    )
    samples = (
        frame.loc[~frame["is_candidate"].astype(bool), "track_sample"]
        .drop_duplicates()
        .tolist()
    )
    figure, axes = plt.subplots(
        nrows=len(samples),
        ncols=3,
        figsize=(16, 4.5 * len(samples)),
        squeeze=False,
    )

    for row, track_sample in enumerate(samples):
        sample = multiplicity.loc[multiplicity["track_sample"] == track_sample]
        axis_wp, axis_2d, axis_all = axes[row]
        if sample.empty:
            for axis in axes[row]:
                axis.text(
                    0.5, 0.5, "No selected events", ha="center", va="center"
                )
                axis.set_axis_off()
            continue

        max_selected = int(sample["n_selected_pvs"].max())
        counts = np.zeros((max_selected, len(threshold_values)), dtype=int)
        for column, threshold in enumerate(threshold_values):
            at_threshold = sample.loc[
                np.isclose(sample["mva_cut"], threshold), "n_selected_pvs"
            ].value_counts()
            for n_selected, count in at_threshold.items():
                counts[int(n_selected) - 1, column] = int(count)
        image = axis_wp.imshow(
            np.ma.masked_equal(counts, 0),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            norm=LogNorm(vmin=1, vmax=max(int(counts.max()), 1)),
        )
        tick_step = max(1, len(threshold_values) // 8)
        tick_indices = np.arange(0, len(threshold_values), tick_step)
        axis_wp.set_xticks(tick_indices)
        axis_wp.set_xticklabels(
            [f"{threshold_values[index]:.4g}" for index in tick_indices],
            rotation=45,
            ha="right",
        )
        axis_wp.set_yticks(np.arange(max_selected))
        axis_wp.set_yticklabels(np.arange(1, max_selected + 1))
        axis_wp.set(
            xlabel="TwoTrackMVA cut",
            ylabel="Unique selected PVs / selected event",
            title=f"{track_sample}: raw selected-event counts",
        )
        figure.colorbar(image, ax=axis_wp, label="Selected events")

        nominal = sample.loc[np.isclose(sample["mva_cut"], NOMINAL_MVA_CUT)]
        if nominal.empty:
            for axis in (axis_2d, axis_all):
                axis.text(
                    0.5,
                    0.5,
                    "No nominal selected events",
                    ha="center",
                    va="center",
                )
            continue
        x_max = int(nominal["n_pvs_event"].max())
        y_max = int(nominal["n_selected_pvs"].max())
        hist = axis_2d.hist2d(
            nominal["n_pvs_event"],
            nominal["n_selected_pvs"],
            bins=(
                np.arange(-0.5, x_max + 1.5),
                np.arange(0.5, y_max + 1.5),
            ),
            cmin=1,
            norm=LogNorm(),
        )
        axis_2d.set(
            xlabel="All reconstructed PVs in event",
            ylabel="Unique selected PVs",
            title=f"Nominal cut {NOMINAL_MVA_CUT:g}: raw 2D counts",
        )
        figure.colorbar(hist[3], ax=axis_2d, label="Selected events")

        multiplicity_max = max(x_max, y_max)
        multiplicity_bins = np.arange(-0.5, multiplicity_max + 1.5)
        axis_all.hist(
            nominal["n_pvs_event"],
            bins=multiplicity_bins,
            color="0.5",
            alpha=0.45,
            label="All reconstructed PVs",
        )
        axis_all.hist(
            nominal["n_selected_pvs"],
            bins=multiplicity_bins,
            histtype="step",
            linewidth=2,
            color="tab:blue",
            label="PVs selected by TwoTrackMVA candidates",
        )
        axis_all.set(
            xlabel="PV multiplicity in selected event",
            ylabel="Selected events",
            title=(
                f"Nominal cut {NOMINAL_MVA_CUT:g}: all versus selected PVs"
            ),
        )
        axis_all.legend()
        for axis in axes[row]:
            axis.grid(alpha=0.15)

    figure.suptitle(
        f"Selected-PV multiplicity; policy={policy}; "
        f"charge={charge.replace('_', ' ')}"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return multiplicity


def build_processed_event_pv_multiplicity(
    frame: pd.DataFrame,
    mva_cut: float,
    policy: str = "common_on_time",
    charge: str = "all",
) -> pd.DataFrame:
    """Count unique selected best PVs for every processed event, including zero."""
    events = frame.loc[~frame["is_candidate"].astype(bool)].copy()
    candidates = frame.loc[frame["is_candidate"].astype(bool)].copy()
    events["event_key"] = _event_key(events)
    candidates["event_key"] = _event_key(candidates)
    candidates = candidates.loc[
        _policy_mask(candidates, policy)
        & _charge_mask(candidates, charge)
        & (candidates["mva_response"] >= mva_cut)
        & (candidates["best_pv_index"] >= 0)
    ]
    rows = []
    for track_sample, sample_events in events.groupby(
        "track_sample", sort=False
    ):
        selected_counts = (
            candidates.loc[candidates["track_sample"] == track_sample]
            .groupby("event_key")["best_pv_index"]
            .nunique()
            .to_dict()
        )
        for event in sample_events.itertuples(index=False):
            rows.append(
                {
                    "track_sample": track_sample,
                    "mva_cut": mva_cut,
                    "event_key": event.event_key,
                    "n_selected_pvs": int(
                        selected_counts.get(event.event_key, 0)
                    ),
                    "n_pvs_event": int(event.n_pvs_event),
                }
            )
    return pd.DataFrame(rows)


def plot_processed_event_pv_multiplicity(
    frame: pd.DataFrame,
    mva_cut: float,
    output: Path,
    policy: str = "common_on_time",
    charge: str = "all",
) -> pd.DataFrame:
    """Plot the raw 1D unique-PV distribution over all processed events."""
    multiplicity = build_processed_event_pv_multiplicity(
        frame, mva_cut, policy=policy, charge=charge
    )
    samples = multiplicity["track_sample"].drop_duplicates().tolist()
    maximum = max(int(multiplicity["n_selected_pvs"].max()), 1)
    values = np.arange(maximum + 1)
    width = 0.8 / max(len(samples), 1)

    figure, axis = plt.subplots(figsize=(8, 5))
    for index, track_sample in enumerate(samples):
        sample = multiplicity.loc[
            multiplicity["track_sample"] == track_sample, "n_selected_pvs"
        ]
        counts = sample.value_counts().reindex(values, fill_value=0)
        offset = (index - (len(samples) - 1) / 2) * width
        selected_fraction = (sample > 0).mean()
        axis.bar(
            values + offset,
            counts.to_numpy(),
            width=width,
            alpha=0.75,
            label=(
                f"{track_sample} "
                f"(fraction with >=1 PV={selected_fraction:.3g})"
            ),
        )
    axis.set_xticks(values)
    axis.set(
        xlabel="Number of unique PVs selected in the event",
        ylabel="Processed events",
        title=(
            f"TwoTrackMVA cut >= {mva_cut:g}; policy={policy}; "
            f"charge={charge.replace('_', ' ')}"
        ),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return multiplicity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="candidate/event Parquet"
    )
    parser.add_argument("--out", required=True, help="scan-summary Parquet")
    parser.add_argument("--plot-dir", required=True)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS
    )
    parser.add_argument(
        "--charge",
        choices=("all", "opposite_sign", "same_sign"),
        default="all",
    )
    parser.add_argument(
        "--plot-pv-policy",
        choices=POLICIES,
        default="common_on_time",
        help="daughter-PV policy used in the raw multiplicity plots",
    )
    parser.add_argument(
        "--multiplicity-cut",
        type=float,
        default=NOMINAL_MVA_CUT,
        help="MVA working point for the 1D per-processed-event PV distribution",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    print_analysis_setup(frame, args)
    scan = build_scan(frame, thresholds=args.thresholds)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    scan.to_parquet(output, index=False)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_scan(scan, plot_dir / "mva_threshold_scan.png", args.charge)
    plot_signal_pv_retention(
        scan,
        plot_dir / "signal_pv_retention_efficiency.png",
        args.charge,
    )
    plot_baseline(
        scan, plot_dir / "ghost_track_effect_nominal.png", args.charge
    )
    plot_candidate_diagnostics(
        frame, plot_dir / "candidate_diagnostics_nominal.png", args.charge
    )
    multiplicity = plot_raw_pv_multiplicity(
        frame,
        args.thresholds,
        plot_dir / "raw_selected_pv_multiplicity.png",
        policy=args.plot_pv_policy,
        charge=args.charge,
    )
    multiplicity.to_parquet(
        output.with_name(f"{output.stem}_pv_multiplicity.parquet"), index=False
    )
    processed = plot_processed_event_pv_multiplicity(
        frame,
        args.multiplicity_cut,
        plot_dir / "unique_selected_pvs_per_processed_event.png",
        policy=args.plot_pv_policy,
        charge=args.charge,
    )
    processed.to_parquet(
        output.with_name(f"{output.stem}_processed_event_pvs.parquet"),
        index=False,
    )
    print(f"Saved {output} and plots in {plot_dir}")


if __name__ == "__main__":
    main()
