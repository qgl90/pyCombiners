#!/usr/bin/env python3
"""Produce and plot matched reconstructed-PV residuals versus PV ndof."""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from trackcomb import load_event_info, load_pvs, run_reconstruction
from tracking.tracking_efficiencies import _gaussian_core_fit


DEFAULT_NDOF_EDGES = np.array(
    [0, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 500, 1000],
    dtype=float,
)

RESIDUALS = {
    "x": ("delta_x", "PV x residual [mm]"),
    "y": ("delta_y", "PV y residual [mm]"),
    "z": ("delta_z", "PV z residual [mm]"),
    "time": ("delta_time", "PV time residual [ns]"),
}

PULLS = {
    "x": ("pull_x", "PV x pull"),
    "y": ("pull_y", "PV y pull"),
    "z": ("pull_z", "PV z pull"),
    "time": ("pull_time", "PV time pull"),
}

_COVARIANCE_DIAGONAL = {
    "x": "cov_0_0",
    "y": "cov_1_1",
    "z": "cov_2_2",
    "time": "cov_3_3",
}


def _flat(values):
    return np.asarray(ak.flatten(values))


def _repeat_event(values, counts):
    return np.repeat(np.asarray(values), counts)


def _add_pull_columns(frame):
    """Add PV errors and pulls from the reconstructed covariance diagonal."""
    for variable, covariance_field in _COVARIANCE_DIAGONAL.items():
        variance = frame[covariance_field].to_numpy(dtype=float)
        error = np.sqrt(
            np.where(
                np.isfinite(variance) & (variance > 0.0), variance, np.nan
            )
        )
        frame[f"sigma_{variable}"] = error
        frame[f"pull_{variable}"] = frame[f"delta_{variable}"] / error
    return frame


def _pv_frame(pvs, event_info):
    """Flatten aligned PVState/PVMC entries and retain valid MC matches."""
    state_counts = ak.to_numpy(ak.num(pvs["x"]))
    mc_counts = ak.to_numpy(ak.num(pvs["mc_key"]))
    if not np.array_equal(state_counts, mc_counts):
        raise ValueError(
            "PVState and PVMC multiplicities differ within an event"
        )

    n_rows = int(state_counts.sum())
    frame = pd.DataFrame(
        {
            "run_number": _repeat_event(
                event_info["run_number"], state_counts
            ),
            "event_number": _repeat_event(
                event_info["event_number"], state_counts
            ),
            "pv_index": _flat(pvs["pv_index"]),
            "mc_key": _flat(pvs["mc_key"]),
            "x": _flat(pvs["x"]),
            "y": _flat(pvs["y"]),
            "z": _flat(pvs["z"]),
            "time": _flat(pvs["time"]),
            "mc_x": _flat(pvs["mc_x"]),
            "mc_y": _flat(pvs["mc_y"]),
            "mc_z": _flat(pvs["mc_z"]),
            "mc_time": _flat(pvs["mc_time"]),
            "ndof": _flat(pvs["ndof"]),
            "chi2ndof": _flat(pvs["chi2ndof"]),
            **{
                f"cov_{i}_{j}": _flat(pvs[f"cov_{i}_{j}"])
                for i in range(4)
                for j in range(i + 1)
            },
        }
    )
    if len(frame) != n_rows:
        raise ValueError(
            "PV flattening did not preserve the aligned entry count"
        )

    frame = frame[frame["mc_key"] != -1].copy()
    frame["delta_x"] = frame["x"] - frame["mc_x"]
    frame["delta_y"] = frame["y"] - frame["mc_y"]
    frame["delta_z"] = frame["z"] - frame["mc_z"]
    frame["delta_time"] = frame["time"] - frame["mc_time"]
    return _add_pull_columns(frame).reset_index(drop=True)


def reconstruction(chunk):
    """Return one dataframe row per reconstructed PV with a valid MC match."""
    return _pv_frame(load_pvs(chunk, mc=True), load_event_info(chunk))


def _fit_table(
    frame,
    ndof_edges,
    min_entries=30,
    clip_sigma=3.0,
    definitions=RESIDUALS,
    quantity="residual",
):
    rows = []
    ndof = frame["ndof"].to_numpy(dtype=float)
    for variable, (field, unit_label) in definitions.items():
        residual = frame[field].to_numpy(dtype=float)
        for index, (low, high) in enumerate(
            zip(ndof_edges[:-1], ndof_edges[1:])
        ):
            in_bin = (ndof >= low) & (
                (ndof <= high)
                if index == len(ndof_edges) - 2
                else (ndof < high)
            )
            values = residual[in_bin & np.isfinite(residual)]
            fit = _gaussian_core_fit(
                values,
                min_entries=min_entries,
                clip_sigma=clip_sigma,
            )
            rows.append(
                {
                    "variable": variable,
                    "quantity": quantity,
                    "value_field": field,
                    "unit": (
                        ""
                        if quantity == "pull"
                        else ("ns" if variable == "time" else "mm")
                    ),
                    "ndof_low": low,
                    "ndof_high": high,
                    "n_pvs": len(values),
                    "n_fit": fit["n_fit"],
                    "n_rejected": len(values) - fit["n_fit"],
                    "core_fraction": fit["core_fraction"],
                    "fit_status": fit["fit_status"],
                    "clip_sigma": clip_sigma,
                    "seed_center": fit["seed_center"],
                    "seed_sigma": fit["seed_sigma"],
                    "seed_fit_low": fit["seed_fit_low"],
                    "seed_fit_high": fit["seed_fit_high"],
                    "fit_low": fit["fit_low"],
                    "fit_high": fit["fit_high"],
                    "bias": fit["mean"],
                    "bias_uncertainty": fit["mean_error"],
                    "resolution": fit["sigma"],
                    "resolution_uncertainty": fit["sigma_error"],
                    "axis_label": unit_label,
                }
            )
    return pd.DataFrame(rows)


def _global_pull_table(frame, min_entries=30, clip_sigma=3.0):
    rows = []
    for variable, (field, _) in PULLS.items():
        values = frame[field].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        fit = _gaussian_core_fit(
            values,
            min_entries=min_entries,
            clip_sigma=clip_sigma,
        )
        rows.append(
            {
                "variable": variable,
                "n_pvs": len(values),
                "n_fit": fit["n_fit"],
                "n_rejected": len(values) - fit["n_fit"],
                "core_fraction": fit["core_fraction"],
                "fit_status": fit["fit_status"],
                "clip_sigma": clip_sigma,
                "seed_fit_low": fit["seed_fit_low"],
                "seed_fit_high": fit["seed_fit_high"],
                "fit_low": fit["fit_low"],
                "fit_high": fit["fit_high"],
                "pull_mean": fit["mean"],
                "pull_mean_uncertainty": fit["mean_error"],
                "pull_width": fit["sigma"],
                "pull_width_uncertainty": fit["sigma_error"],
            }
        )
    return pd.DataFrame(rows)


def _central_limits(values, quantile=0.995):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return -1.0, 1.0
    limit = float(np.quantile(np.abs(values), quantile))
    return (-limit, limit) if limit > 0.0 else (-1.0, 1.0)


def _plot_residuals(frame, output, label):
    import matplotlib.pyplot as plt

    from trackcomb.plot import make_figure

    fig, axes = make_figure(1, 4, figsize=(28, 6))
    for axis, (variable, (field, ylabel)) in zip(axes, RESIDUALS.items()):
        finite = np.isfinite(frame["ndof"]) & np.isfinite(frame[field])
        x = frame.loc[finite, "ndof"]
        y = frame.loc[finite, field]
        axis.hexbin(x, y, gridsize=60, mincnt=1, bins="log", cmap="viridis")
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xlabel("PV ndof")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*_central_limits(y))
        axis.grid(True, alpha=0.2)
        axis.set_title(variable)
    fig.suptitle(f"{label}: matched PVState - PVMC residuals")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_global_pulls(frame, global_fits, output, label):
    import matplotlib.pyplot as plt

    from trackcomb.plot import make_figure

    fig, axes = make_figure(1, 4, figsize=(28, 6))
    for axis, (variable, (field, xlabel)) in zip(axes, PULLS.items()):
        values = frame[field].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        fit = global_fits[global_fits["variable"] == variable].iloc[0]
        _, edges, _ = axis.hist(
            values,
            bins=60,
            range=(-5.0, 5.0),
            histtype="step",
            color="black",
            label="PVs",
        )
        x = np.linspace(-5.0, 5.0, 500)
        bin_width = edges[1] - edges[0]
        standard = (
            len(values)
            * bin_width
            * np.exp(-0.5 * x**2)
            / np.sqrt(2.0 * np.pi)
        )
        axis.plot(x, standard, color="0.4", linestyle="--", label="N(0, 1)")
        if fit["fit_status"] == "fitted":
            gaussian = (
                fit["n_fit"]
                * bin_width
                * np.exp(
                    -0.5 * ((x - fit["pull_mean"]) / fit["pull_width"]) ** 2
                )
                / (np.sqrt(2.0 * np.pi) * fit["pull_width"])
            )
            axis.plot(x, gaussian, color="tab:red", label="Gaussian core")
        axis.axvline(0.0, color="black", linewidth=1, alpha=0.5)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("PVs")
        axis.set_xlim(-5.0, 5.0)
        axis.text(
            0.03,
            0.96,
            f"mean = {fit['pull_mean']:.3g}\nwidth = {fit['pull_width']:.3g}\n"
            f"N = {fit['n_pvs']:.0f}, Nfit = {fit['n_fit']:.0f}",
            transform=axis.transAxes,
            va="top",
            fontsize=10,
        )
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize=9)
    fig.suptitle(f"{label}: PV covariance pull calibration")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_bias_resolution(table, output, label, pulls=False):
    import matplotlib.pyplot as plt

    from trackcomb.plot import make_figure

    fig, axes = make_figure(2, 4, figsize=(28, 11), sharex="col")
    definitions = PULLS if pulls else RESIDUALS
    for column, (variable, _) in enumerate(definitions.items()):
        unit = "ns" if variable == "time" else "mm"
        symbol = "t" if variable == "time" else variable
        points = table[table["variable"] == variable]
        centers = 0.5 * (points["ndof_low"] + points["ndof_high"])
        widths = 0.5 * (points["ndof_high"] - points["ndof_low"])
        axes[0, column].errorbar(
            centers,
            points["resolution"],
            xerr=widths,
            yerr=points["resolution_uncertainty"],
            fmt="o",
            capsize=2,
        )
        if pulls:
            axes[0, column].axhline(
                1.0, color="black", linestyle="--", linewidth=1
            )
            axes[0, column].set_ylabel(
                rf"Pull width $\sigma(P_{{{symbol}}})$", fontsize=10
            )
        else:
            axes[0, column].set_ylabel(
                rf"Resolution $\sigma(\Delta {symbol})$ [{unit}]",
                fontsize=10,
            )
        axes[0, column].set_ylim(bottom=0.0)
        axes[1, column].errorbar(
            centers,
            points["bias"],
            xerr=widths,
            yerr=points["bias_uncertainty"],
            fmt="o",
            capsize=2,
        )
        axes[1, column].axhline(0.0, color="black", linewidth=1)
        axes[1, column].set_xlabel("PV ndof", fontsize=10)
        if pulls:
            axes[1, column].set_ylabel(
                rf"Pull mean $\mu(P_{{{symbol}}})$", fontsize=10
            )
        else:
            axes[1, column].set_ylabel(
                rf"Bias $\mu(\Delta {symbol})$ [{unit}]", fontsize=10
            )
        for axis in axes[:, column]:
            axis.grid(True, alpha=0.3)
            axis.set_xscale("symlog", linthresh=10.0)
            axis.tick_params(axis="both", labelsize=8)
    title = (
        "Gaussian PV pull mean and width versus ndof"
        if pulls
        else "Gaussian PV bias and resolution versus ndof"
    )
    fig.suptitle(f"{label}: {title}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_fit_checks(
    frame,
    table,
    variable,
    output,
    label,
    definitions=RESIDUALS,
    quantity="residual",
):
    import matplotlib.pyplot as plt

    field, axis_label = definitions[variable]
    points = table[table["variable"] == variable].reset_index(drop=True)
    n_columns = 3
    n_rows = int(np.ceil(len(points) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 3.8 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    ndof = frame["ndof"].to_numpy(dtype=float)
    residual = frame[field].to_numpy(dtype=float)

    for index, (axis, (_, point)) in enumerate(zip(axes, points.iterrows())):
        selected = (ndof >= point["ndof_low"]) & (
            (ndof <= point["ndof_high"])
            if index == len(points) - 1
            else (ndof < point["ndof_high"])
        )
        sample = residual[selected & np.isfinite(residual)]
        limits = np.asarray(
            [
                point["seed_fit_low"],
                point["seed_fit_high"],
                point["fit_low"],
                point["fit_high"],
            ]
        )
        limits = limits[np.isfinite(limits)]
        if len(limits) >= 2:
            low, high = limits.min(), limits.max()
        else:
            low, high = _central_limits(sample, quantile=0.95)
        span = high - low
        center = 0.5 * (low + high)
        if span <= max(abs(center) * 1e-6, 1e-12):
            span = max(abs(center) * 0.2, 1e-3)
            low, high = center - 0.5 * span, center + 0.5 * span
        display = (low - 0.15 * span, high + 0.15 * span)
        visible = sample[(sample >= display[0]) & (sample <= display[1])]
        n_bins = min(40, max(8, int(2.0 * np.sqrt(len(visible)))))
        _, edges, _ = axis.hist(
            visible, bins=n_bins, range=display, histtype="step", color="black"
        )
        if np.isfinite(point["seed_fit_low"]):
            axis.axvline(point["seed_fit_low"], color="0.55", linestyle="--")
            axis.axvline(point["seed_fit_high"], color="0.55", linestyle="--")
        if point["fit_status"] == "fitted":
            axis.axvspan(
                point["fit_low"],
                point["fit_high"],
                color="tab:blue",
                alpha=0.1,
            )
            x = np.linspace(point["fit_low"], point["fit_high"], 300)
            gaussian = (
                point["n_fit"]
                * (edges[1] - edges[0])
                * np.exp(
                    -0.5 * ((x - point["bias"]) / point["resolution"]) ** 2
                )
                / (np.sqrt(2.0 * np.pi) * point["resolution"])
            )
            axis.plot(x, gaussian, color="tab:red")
        axis.set_title(
            f"{point['ndof_low']:.0f} <= ndof < {point['ndof_high']:.0f}",
            fontsize=11,
        )
        axis.set_xlabel(axis_label, fontsize=9)
        axis.set_ylabel("PVs", fontsize=9)
        axis.tick_params(axis="both", labelsize=8)
        axis.text(
            0.03,
            0.96,
            f"{point['fit_status']}\nN={point['n_pvs']:.0f}, Nfit={point['n_fit']:.0f}\n"
            f"mean={point['bias']:.3g}\nsigma={point['resolution']:.3g}",
            transform=axis.transAxes,
            va="top",
            fontsize=7,
        )
        axis.grid(True, alpha=0.2)
    for axis in axes[len(points) :]:
        axis.set_visible(False)
    fig.suptitle(
        f"{label}: Gaussian {quantity} fit checks for PV {variable}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    dataframe_path,
    plot_dir,
    label=None,
    ndof_edges=DEFAULT_NDOF_EDGES,
    min_entries=30,
    clip_sigma=3.0,
):
    """Read matched PVs and write residual, fit, and diagnostic products."""
    frame = pd.read_parquet(dataframe_path)
    frame = _add_pull_columns(frame)
    ndof_edges = np.asarray(ndof_edges, dtype=float)
    if len(ndof_edges) < 2 or np.any(np.diff(ndof_edges) <= 0.0):
        raise ValueError("ndof edges must be strictly increasing")
    residual_table = _fit_table(
        frame, ndof_edges, min_entries, clip_sigma, RESIDUALS, "residual"
    )
    pull_table = _fit_table(
        frame, ndof_edges, min_entries, clip_sigma, PULLS, "pull"
    )
    global_pull_table = _global_pull_table(frame, min_entries, clip_sigma)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    label = label or Path(dataframe_path).stem.removeprefix("pv_residuals_")
    suffix = f"_{label}" if label else ""
    residual_table["sample_label"] = label
    pull_table["sample_label"] = label
    global_pull_table["sample_label"] = label

    residual_table_path = plot_dir / f"pv_resolution_fits{suffix}.parquet"
    pull_table_path = plot_dir / f"pv_pull_fits{suffix}.parquet"
    global_pull_path = plot_dir / f"pv_pull_global_fits{suffix}.parquet"
    residual_path = plot_dir / f"pv_residuals_vs_ndof{suffix}.png"
    resolution_path = plot_dir / f"pv_bias_resolution_vs_ndof{suffix}.png"
    pull_path = plot_dir / f"pv_pulls{suffix}.png"
    pull_ndof_path = plot_dir / f"pv_pull_mean_width_vs_ndof{suffix}.png"
    residual_table.to_parquet(residual_table_path, index=False)
    pull_table.to_parquet(pull_table_path, index=False)
    global_pull_table.to_parquet(global_pull_path, index=False)
    _plot_residuals(frame, residual_path, label)
    _plot_bias_resolution(residual_table, resolution_path, label)
    _plot_global_pulls(frame, global_pull_table, pull_path, label)
    _plot_bias_resolution(pull_table, pull_ndof_path, label, pulls=True)
    print(f"Saved {residual_table_path}")
    print(f"Saved {pull_table_path}")
    print(f"Saved {global_pull_path}")
    print(f"Saved {residual_path}")
    print(f"Saved {resolution_path}")
    print(f"Saved {pull_path}")
    print(f"Saved {pull_ndof_path}")
    for variable in RESIDUALS:
        output = plot_dir / f"pv_gaussian_fit_checks_{variable}{suffix}.png"
        _plot_fit_checks(frame, residual_table, variable, output, label)
        print(f"Saved {output}")
        pull_output = (
            plot_dir / f"pv_pull_gaussian_fit_checks_{variable}{suffix}.png"
        )
        _plot_fit_checks(
            frame,
            pull_table,
            variable,
            pull_output,
            label,
            PULLS,
            "pull",
        )
        print(f"Saved {pull_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Produce matched-PV residual Parquet and resolution plots"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="ROOT path; wildcards allowed")
    source.add_argument("--dataframe", help="existing PV residual Parquet")
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out",
        default="public/pv_resolution/pv_residuals.parquet",
        help="output dataframe when reading ROOT input",
    )
    parser.add_argument("--plot-dir", default="public/pv_resolution")
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--ndof-edges",
        nargs="+",
        type=float,
        default=DEFAULT_NDOF_EDGES.tolist(),
    )
    parser.add_argument("--min-fit-entries", type=int, default=30)
    parser.add_argument("--fit-sigma", type=float, default=3.0)
    args = parser.parse_args()

    if args.dataframe:
        dataframe_path = Path(args.dataframe)
    else:
        dataframe_path = Path(args.out)
        run_reconstruction(
            reconstruction,
            input_data=args.input,
            out=dataframe_path,
            max_events=args.max_events or None,
            chunk_size=args.chunk_size,
            workers=args.workers,
            print_throughput=True,
        )
        print(f"Saved {dataframe_path}")

    make_plots(
        dataframe_path,
        args.plot_dir,
        args.label,
        args.ndof_edges,
        args.min_fit_entries,
        args.fit_sigma,
    )


if __name__ == "__main__":
    main()
