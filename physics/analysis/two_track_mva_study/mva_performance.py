#!/usr/bin/env python3
"""TwoTrackMVA performance: MVA response distribution and tagged PVs per event."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


B_PDGIDS = {511, 521, 531, 541, 5122, 5132, 5232, 5332}


def is_from_b(mc_pid):
    """True if mc_pid is a b-hadron."""
    return np.isin(np.abs(mc_pid), list(B_PDGIDS))


def main():
    parser = argparse.ArgumentParser(
        description="TwoTrackMVA performance analysis"
    )
    parser.add_argument("--input", required=True, help="Path to mva.parquet")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--channel", required=True, help="Decay channel label")
    parser.add_argument("--lumi", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.read_parquet(args.input)

    if len(df) == 0:
        for name in (
            "mva_response_distribution.png",
            "tagged_pvs_per_event.png",
        ):
            (out_dir / name).touch()
        return

    import matplotlib.pyplot as plt
    from trackcomb.plot import make_figure

    lumi_label = f" [{args.lumi}]" if args.lumi else ""

    # Classify candidates
    has_ancestor = df["mc_truth"].values == 1
    from_b = has_ancestor & is_from_b(df["mc_pid"].values)
    from_signal = from_b & (df["mc_fromsignal"].values == 1)
    bkg = ~from_b

    n_from_b = int(from_b.sum())
    n_signal = int(from_signal.sum())
    n_bkg = int(bkg.sum())

    # --- Plot 1: MVA response distribution ---
    fig, ax = make_figure(figsize=(16, 12))
    bins = np.linspace(0.0, 1.0, 51)
    if n_signal > 0:
        ax.hist(
            df.loc[from_signal, "mva_response"],
            bins=bins,
            alpha=0.7,
            label=f"From signal ({n_signal})",
            color="C2",
        )
    if n_from_b > 0:
        ax.hist(
            df.loc[from_b, "mva_response"],
            bins=bins,
            alpha=0.5,
            label=f"From b-hadron ({n_from_b})",
            color="C0",
        )
    if n_bkg > 0:
        ax.hist(
            df.loc[bkg, "mva_response"],
            bins=bins,
            alpha=0.5,
            label=f"Background ({n_bkg})",
            color="C1",
        )
    nominal_cut = 0.9569
    ax.axvline(
        nominal_cut,
        color="red",
        ls="--",
        lw=1.2,
        label=f"Nominal cut ({nominal_cut})",
    )
    ax.set_yscale("log")
    ax.set_xlabel("MVA response")
    ax.set_ylabel("Candidates")
    ax.set_title(f"TwoTrackMVA response ({args.channel}){lumi_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mva_response_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'mva_response_distribution.png'}")

    # --- Plot 2: Tagged PVs per event ---
    mva = df["mva_response"].values
    mva_pass = mva > nominal_cut
    df_pass = df.loc[mva_pass]

    if len(df_pass) > 0 and "best_pv_index" in df.columns:
        unique_pvs = df_pass.groupby("event_number")["best_pv_index"].nunique()
        n_events_with_sv = len(unique_pvs)
        max_npv = int(unique_pvs.max())

        fig, ax = make_figure(figsize=(16, 12))
        bins = np.arange(0.5, max_npv + 1.5, 1)
        ax.hist(unique_pvs.values, bins=bins, color="C0", edgecolor="black")
        ax.set_xlabel("Tagged PVs per event")
        ax.set_ylabel("Events")
        ax.set_xticks(range(1, max_npv + 1))
        ax.set_title(
            f"Tagged PVs per event (MVA > {nominal_cut}, {args.channel}){lumi_label}\n"
            f"(mean={unique_pvs.mean():.2f}, {n_events_with_sv} events)"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "tagged_pvs_per_event.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / 'tagged_pvs_per_event.png'}")
    else:
        (out_dir / "tagged_pvs_per_event.png").touch()


if __name__ == "__main__":
    main()
