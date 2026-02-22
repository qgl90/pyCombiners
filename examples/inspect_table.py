"""Utility script to inspect/plot table outputs produced by the combiner."""

from __future__ import annotations

import argparse
from pathlib import Path


def _require_pandas():
    """Import pandas with an actionable error if not installed."""
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required. Install with: pip install pandas pyarrow"
        ) from exc
    return pd


def load_table(path: str):
    """Load table data from parquet/csv/pickle into a pandas DataFrame."""
    pd = _require_pandas()
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in (".pkl", ".pickle"):
        return pd.read_pickle(p)
    raise ValueError("Supported input formats: .parquet, .csv, .pkl")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for interactive inspection and optional quick plotting."""
    parser = argparse.ArgumentParser(description="Inspect combiner output table.")
    parser.add_argument("--input", required=True, help="Path to .parquet/.csv/.pkl output.")
    parser.add_argument("--head", type=int, default=10, help="Rows to print.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create a quick pair_pt vs candidate_mass scatter plot (png).",
    )
    args = parser.parse_args(argv)

    df = load_table(args.input)
    print(df.head(args.head).to_string(index=False))
    print(f"\nRows={len(df)}  Columns={len(df.columns)}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ModuleNotFoundError:
            print("matplotlib not installed; skipping plot.")
            return 0
        out = Path(args.input).with_suffix(".png")
        ax = df.plot.scatter(x="pair_pt", y="candidate_mass", alpha=0.6)
        ax.set_title("pair_pt vs candidate_mass")
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        print(f"Saved plot: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
