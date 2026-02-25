"""Plot/study utility for synthetic B -> J/psi K* combiner output tables.

This script expects the candidate table produced by:
    examples/b_jpsi_kstar_fake_and_combine.py

It performs pandas-based studies and writes summary plots for composite masses.
"""

from __future__ import annotations
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


import argparse
import json
from pathlib import Path


def _require_pandas():
    """Import pandas with an actionable install hint."""
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required. Install with: pip install pandas pyarrow"
        ) from exc
    return pd


def load_table(path: str):
    """Load candidate table from parquet/csv/pickle."""
    pd = _require_pandas()
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in (".pkl", ".pickle"):
        return pd.read_pickle(p)
    raise ValueError("Supported formats: .parquet, .csv, .pkl")


def parse_args() -> argparse.Namespace:
    """Parse CLI options for mass-plot and summary studies."""
    parser = argparse.ArgumentParser(description="Study synthetic B->J/psi K* combiner output.")
    parser.add_argument(
        "--input",
        default="examples/output_bjpsikstar_candidates.parquet",
        help="Input candidates table (.parquet/.csv/.pkl).",
    )
    parser.add_argument(
        "--out-dir",
        default="examples/output_bjpsikstar_study",
        help="Output directory for plots and JSON summaries.",
    )
    parser.add_argument("--b-min-mev", type=float, default=5000.0, help="Lower bound of B study window (MeV).")
    parser.add_argument("--b-max-mev", type=float, default=6000.0, help="Upper bound of B study window (MeV).")
    parser.add_argument("--bins", type=int, default=120, help="Histogram bins.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def summarize(df, b_min_mev: float, b_max_mev: float) -> dict[str, object]:
    """Compute stage-wise and B-window summary metrics from candidate DataFrame."""
    stage_counts = df.groupby("stage").size().to_dict()
    stage_truth_counts = df[df["is_truth"]].groupby("stage").size().to_dict()

    b_df = df[df["stage"] == "B"].copy()
    b_window_df = b_df[
        (b_df["candidate_mass_mev"] >= b_min_mev)
        & (b_df["candidate_mass_mev"] <= b_max_mev)
    ].copy()
    n_b_window = int(len(b_window_df))
    n_b_truth_window = int(b_window_df["is_truth"].sum()) if n_b_window else 0
    b_purity = float(n_b_truth_window / n_b_window) if n_b_window else 0.0

    peak_df = b_window_df[
        (b_window_df["candidate_mass_mev"] >= 5250.0)
        & (b_window_df["candidate_mass_mev"] <= 5310.0)
    ]
    n_peak = int(len(peak_df))
    n_peak_truth = int(peak_df["is_truth"].sum()) if n_peak else 0
    peak_purity = float(n_peak_truth / n_peak) if n_peak else 0.0

    return {
        "n_rows_total": int(len(df)),
        "n_rows_by_stage": {k: int(v) for k, v in stage_counts.items()},
        "n_truth_rows_by_stage": {k: int(v) for k, v in stage_truth_counts.items()},
        "b_window_mev": [b_min_mev, b_max_mev],
        "b_rows_in_window": n_b_window,
        "b_truth_rows_in_window": n_b_truth_window,
        "b_window_purity": b_purity,
        "b_peak_mev": [5250.0, 5310.0],
        "b_peak_rows": n_peak,
        "b_peak_truth_rows": n_peak_truth,
        "b_peak_purity": peak_purity,
    }


def maybe_plot(df, out_dir: Path, b_min_mev: float, b_max_mev: float, bins: int) -> None:
    """Generate stage mass plots and a B-mass-vs-vertex-chi2 scatter plot."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping plot generation.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    jpsi = df[df["stage"] == "JPSI"]
    kstar = df[df["stage"] == "KSTAR"]
    b_df = df[df["stage"] == "B"]

    # J/psi mass
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(jpsi["candidate_mass_mev"], bins=bins, histtype="step", linewidth=1.5, label="all")
    ax.hist(
        jpsi[jpsi["is_truth"]]["candidate_mass_mev"],
        bins=bins,
        histtype="step",
        linewidth=1.5,
        label="truth",
    )
    ax.set_xlabel("m(mu mu) [MeV]")
    ax.set_ylabel("Candidates")
    ax.set_title("J/psi candidate mass")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "jpsi_mass_mev.png", dpi=120)
    plt.close(fig)

    # K* mass
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(kstar["candidate_mass_mev"], bins=bins, histtype="step", linewidth=1.5, label="all")
    ax.hist(
        kstar[kstar["is_truth"]]["candidate_mass_mev"],
        bins=bins,
        histtype="step",
        linewidth=1.5,
        label="truth",
    )
    ax.set_xlabel("m(K pi) [MeV]")
    ax.set_ylabel("Candidates")
    ax.set_title("K* candidate mass")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "kstar_mass_mev.png", dpi=120)
    plt.close(fig)

    # B mass in requested range
    fig, ax = plt.subplots(figsize=(8, 5))
    b_in_window = b_df[
        (b_df["candidate_mass_mev"] >= b_min_mev) & (b_df["candidate_mass_mev"] <= b_max_mev)
    ]
    ax.hist(
        b_in_window["candidate_mass_mev"],
        bins=bins,
        histtype="step",
        linewidth=1.5,
        label="all in window",
    )
    ax.hist(
        b_in_window[b_in_window["is_truth"]]["candidate_mass_mev"],
        bins=bins,
        histtype="step",
        linewidth=1.5,
        label="truth in window",
    )
    ax.axvline(b_min_mev, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(b_max_mev, color="black", linestyle="--", linewidth=1.2, label="B window")
    ax.set_xlabel("m(B) [MeV]")
    ax.set_ylabel("Candidates")
    ax.set_title("B candidate mass (5000-6000 MeV study window)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "b_mass_mev_window.png", dpi=120)
    plt.close(fig)

    # B mass vs vertex chi2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        b_in_window["candidate_mass_mev"],
        b_in_window["vertex_chi2"],
        s=8,
        alpha=0.45,
        label="all in window",
    )
    truth_b = b_in_window[b_in_window["is_truth"]]
    if not truth_b.empty:
        ax.scatter(
            truth_b["candidate_mass_mev"],
            truth_b["vertex_chi2"],
            s=10,
            alpha=0.65,
            label="truth in window",
        )
    ax.set_xlabel("m(B) [MeV]")
    ax.set_ylabel("vertex_chi2")
    ax.set_title("B candidate mass vs vertex chi2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "b_mass_vs_vertex_chi2.png", dpi=120)
    plt.close(fig)


def main() -> int:
    """Load candidates, run pandas studies, and save summary/plots."""
    args = parse_args()
    df = load_table(args.input)

    summary = summarize(df, args.b_min_mev, args.b_max_mev)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Stage counts:")
    print(df.groupby("stage").size().to_string())
    print("\nTruth counts by stage:")
    print(df[df["is_truth"]].groupby("stage").size().to_string())
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote summary JSON: {summary_path}")

    if not args.no_plot:
        maybe_plot(df, out_dir, args.b_min_mev, args.b_max_mev, args.bins)
        print(f"Wrote plots into: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
