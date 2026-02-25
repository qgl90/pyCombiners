"""Offline mass-peak study utility for combiner output tables.

This script is intended for quick signal/background studies after running the
particle combiner over many events and writing a parquet/csv/pickle table.
"""

from __future__ import annotations
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChannelConfig:
    """Channel-specific defaults for mass-window and sideband studies."""

    name: str
    mass_center: float
    signal_half_window: float
    sideband_inner: float
    sideband_outer: float
    query: str


CHANNELS: dict[str, ChannelConfig] = {
    "ks_pipi": ChannelConfig(
        name="ks_pipi",
        mass_center=0.497611,
        signal_half_window=0.015,
        sideband_inner=0.030,
        sideband_outer=0.080,
        query="(charge_pattern == '+-' or charge_pattern == '-+')",
    ),
    "dplus_kpipi": ChannelConfig(
        name="dplus_kpipi",
        mass_center=1.86966,
        signal_half_window=0.030,
        sideband_inner=0.060,
        sideband_outer=0.150,
        query="(charge_pattern == '-++' or charge_pattern == '+--')",
    ),
}


def _require_pandas():
    """Import pandas with a clear install hint on failure."""
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required. Install with: pip install pandas pyarrow"
        ) from exc
    return pd


def load_table(path: str):
    """Load combiner output table from parquet/csv/pickle."""
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


def apply_channel_selection(df, channel: ChannelConfig, extra_query: str | None):
    """Apply default channel query and optional user query."""
    out = df.query(channel.query) if channel.query else df
    if extra_query:
        out = out.query(extra_query)
    return out


def estimate_signal_background(
    masses,
    m0: float,
    w_sig: float,
    w_sb_in: float,
    w_sb_out: float,
) -> dict[str, float]:
    """Estimate S/B from a signal window and two symmetric sidebands.

    Background in the signal window is estimated from sideband density assuming
    approximately flat local background near the peak.
    """
    sig_lo = m0 - w_sig
    sig_hi = m0 + w_sig
    sb_l_lo = m0 - w_sb_out
    sb_l_hi = m0 - w_sb_in
    sb_r_lo = m0 + w_sb_in
    sb_r_hi = m0 + w_sb_out

    n_sig_window = float(((masses >= sig_lo) & (masses <= sig_hi)).sum())
    n_sb_left = float(((masses >= sb_l_lo) & (masses <= sb_l_hi)).sum())
    n_sb_right = float(((masses >= sb_r_lo) & (masses <= sb_r_hi)).sum())
    n_sb_total = n_sb_left + n_sb_right

    sideband_total_width = 2.0 * (w_sb_out - w_sb_in)
    signal_width = 2.0 * w_sig
    bkg_density = n_sb_total / sideband_total_width if sideband_total_width > 0 else 0.0
    bkg_in_signal = bkg_density * signal_width
    signal_est = n_sig_window - bkg_in_signal
    s_over_b = signal_est / bkg_in_signal if bkg_in_signal > 0 else 0.0
    significance = signal_est / ((signal_est + bkg_in_signal) ** 0.5) if (signal_est + bkg_in_signal) > 0 else 0.0

    return {
        "n_sig_window": n_sig_window,
        "n_sideband_left": n_sb_left,
        "n_sideband_right": n_sb_right,
        "n_sideband_total": n_sb_total,
        "bkg_in_signal_est": bkg_in_signal,
        "signal_est": signal_est,
        "s_over_b": s_over_b,
        "s_over_sqrt_s_plus_b": significance,
    }


def maybe_plot(
    masses,
    m0: float,
    w_sig: float,
    w_sb_in: float,
    w_sb_out: float,
    out_png: Path,
    bins: int,
) -> None:
    """Render a mass histogram with signal/sideband window lines."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(masses, bins=bins, histtype="step", linewidth=1.4)
    ax.axvline(m0 - w_sig, color="green", linestyle="--", linewidth=1.2)
    ax.axvline(m0 + w_sig, color="green", linestyle="--", linewidth=1.2, label="signal window")
    ax.axvline(m0 - w_sb_in, color="orange", linestyle=":", linewidth=1.2)
    ax.axvline(m0 - w_sb_out, color="orange", linestyle=":", linewidth=1.2)
    ax.axvline(m0 + w_sb_in, color="orange", linestyle=":", linewidth=1.2)
    ax.axvline(m0 + w_sb_out, color="orange", linestyle=":", linewidth=1.2, label="sidebands")
    ax.set_xlabel("candidate_mass [GeV]")
    ax.set_ylabel("Candidates")
    ax.set_title("Offline mass peak study")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run channel selection and sideband-based S/B estimation."""
    parser = argparse.ArgumentParser(description="Offline peak study from combiner table output.")
    parser.add_argument("--input", required=True, help="Input table (.parquet/.csv/.pkl).")
    parser.add_argument(
        "--channel",
        default="ks_pipi",
        choices=sorted(CHANNELS.keys()),
        help="Physics channel preset.",
    )
    parser.add_argument("--query", default=None, help="Additional pandas query to refine selections.")
    parser.add_argument("--mass-center", type=float, default=None, help="Override peak mass center.")
    parser.add_argument("--signal-half-window", type=float, default=None, help="Override signal half-window.")
    parser.add_argument("--sideband-inner", type=float, default=None, help="Override inner sideband offset.")
    parser.add_argument("--sideband-outer", type=float, default=None, help="Override outer sideband offset.")
    parser.add_argument("--bins", type=int, default=120, help="Histogram bins if plotting.")
    parser.add_argument("--plot", action="store_true", help="Write a histogram PNG next to the input.")
    parser.add_argument("--out-json", default=None, help="Optional JSON summary output path.")
    args = parser.parse_args(argv)

    df = load_table(args.input)
    ch = CHANNELS[args.channel]
    selected = apply_channel_selection(df, ch, args.query)
    masses = selected["candidate_mass"]

    m0 = ch.mass_center if args.mass_center is None else args.mass_center
    w_sig = ch.signal_half_window if args.signal_half_window is None else args.signal_half_window
    w_sb_in = ch.sideband_inner if args.sideband_inner is None else args.sideband_inner
    w_sb_out = ch.sideband_outer if args.sideband_outer is None else args.sideband_outer

    stats = estimate_signal_background(masses, m0, w_sig, w_sb_in, w_sb_out)
    summary: dict[str, Any] = {
        "channel": args.channel,
        "n_total_rows": int(len(df)),
        "n_selected_rows": int(len(selected)),
        "mass_center": m0,
        "signal_half_window": w_sig,
        "sideband_inner": w_sb_in,
        "sideband_outer": w_sb_out,
        **stats,
    }

    print(json.dumps(summary, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary: {args.out_json}")

    if args.plot:
        out_png = Path(args.input).with_suffix(f".{args.channel}.png")
        maybe_plot(masses, m0, w_sig, w_sb_in, w_sb_out, out_png, args.bins)
        print(f"Wrote plot: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
