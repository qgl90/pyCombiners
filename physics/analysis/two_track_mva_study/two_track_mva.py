#!/usr/bin/env python3
"""TwoTrackMVA study: MVA response distributions, mass, and key observables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


B_PDGIDS = {511, 521, 531, 541, 5122, 5132, 5232, 5332}


def is_from_b(mc_pid):
    """True if mc_pid is a b-hadron."""
    return np.isin(np.abs(mc_pid), list(B_PDGIDS))


def main():
    parser = argparse.ArgumentParser(description="TwoTrackMVA analysis")
    parser.add_argument("--input", required=True, help="Path to mva.parquet")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lumi", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.read_parquet(args.input)

    if len(df) == 0:
        print("No candidates, skipping plots.")
        # Create empty placeholder files so snakemake is satisfied
        for name in ("mva_response.png", "mass.png", "observables.png"):
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

    n_total = len(df)
    n_from_b = int(from_b.sum())
    n_signal = int(from_signal.sum())
    n_bkg = int(bkg.sum())
    print(f"Total candidates: {n_total}")
    print(f"  From b-hadron (mc_truth=1 & |mc_pid| in B): {n_from_b}")
    print(f"    From signal: {n_signal}")
    print(f"  Background: {n_bkg}")

    # --- Plot 1: MVA response ---
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
    ax.axvline(0.9569, color="red", ls="--", lw=1.2, label="Nominal cut (0.957)")
    ax.set_yscale("log")
    ax.set_xlabel("MVA response")
    ax.set_ylabel("Candidates")
    ax.set_title(f"TwoTrackMVA response{lumi_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mva_response.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Mass distribution ---
    fig, ax = make_figure(figsize=(16, 12))
    bins = np.linspace(0, 10, 51)
    if n_signal > 0:
        ax.hist(
            df.loc[from_signal, "mass"],
            bins=bins,
            alpha=0.7,
            label=f"From signal ({n_signal})",
            color="C2",
        )
    if n_from_b > 0:
        ax.hist(
            df.loc[from_b, "mass"],
            bins=bins,
            alpha=0.5,
            label=f"From b-hadron ({n_from_b})",
            color="C0",
        )
    if n_bkg > 0:
        ax.hist(
            df.loc[bkg, "mass"],
            bins=bins,
            alpha=0.5,
            label=f"Background ({n_bkg})",
            color="C1",
        )
    ax.set_xlabel("Mass [MeV]")
    ax.set_ylabel("Candidates")
    ax.set_title(f"Invariant mass{lumi_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mass.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: Key observable distributions ---
    observables = [
        ("fdchi2", "log(fdchi2)", lambda x: np.log(np.maximum(x, 1e-10))),
        ("vertex_chi2", "Vertex chi2", None),
        ("composite_ip_chi2", "Composite IP chi2", None),
        ("mcor", "Corrected mass [MeV]", None),
        ("flight_eta", "Flight eta", None),
        ("pt", "SV pT [MeV]", None),
    ]

    fig, axes = make_figure(2, 3, figsize=(24, 14))
    for ax, (field, xlabel, transform) in zip(axes.flat, observables):
        vals = df[field].values.copy()
        if transform is not None:
            vals = transform(vals)
        lo, hi = np.nanpercentile(vals, [1, 99])
        bins = np.linspace(lo, hi, 41)
        if n_signal > 0:
            ax.hist(
                vals[from_signal], bins=bins, alpha=0.7, label="From signal", color="C2"
            )
        if n_from_b > 0:
            ax.hist(
                vals[from_b], bins=bins, alpha=0.5, label="From b-hadron", color="C0"
            )
        if n_bkg > 0:
            ax.hist(vals[bkg], bins=bins, alpha=0.5, label="Background", color="C1")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Candidates")
        ax.legend()
    fig.suptitle(f"TwoTrackMVA observables{lumi_label}")
    fig.tight_layout()
    fig.savefig(out_dir / "observables.png", dpi=150)
    plt.close(fig)

    # --- Efficiency at nominal cut ---
    nominal_cut = 0.9569
    mva = df["mva_response"].values
    b_pass = int((from_b & (mva > nominal_cut)).sum())
    bkg_pass = int((bkg & (mva > nominal_cut)).sum())
    b_eff = b_pass / n_from_b if n_from_b > 0 else 0.0
    sb_ratio = b_pass / bkg_pass if bkg_pass > 0 else float("inf")

    print(f"\n--- Nominal MVA cut = {nominal_cut} ---")
    print(f"  From-B passing:      {b_pass} / {n_from_b}  (eff = {b_eff:.4f})")
    print(f"  Background passing:  {bkg_pass} / {n_bkg}")
    print(f"  S/B ratio:           {sb_ratio:.4f}")

    # --- Unique PVs passing MVA cut per event ---
    mva_pass = mva > nominal_cut
    df_pass = df.loc[mva_pass]
    if len(df_pass) > 0 and "best_pv_index" in df.columns:
        unique_pvs = df_pass.groupby("event_number")["best_pv_index"].nunique()
        n_events_with_sv = len(unique_pvs)
        print(f"\n--- Unique PVs with passing SVs (per event) ---")
        print(f"  Events with >= 1 passing SV: {n_events_with_sv}")
        print(
            f"  Unique PVs per event: "
            f"mean={unique_pvs.mean():.2f}, "
            f"median={unique_pvs.median():.1f}, "
            f"max={unique_pvs.max()}"
        )
        for npv in range(1, min(unique_pvs.max() + 1, 11)):
            count = int((unique_pvs == npv).sum())
            print(
                f"    {npv} PV(s): {count} events "
                f"({100 * count / n_events_with_sv:.1f}%)"
            )

        # Plot
        max_npv = int(unique_pvs.max())
        fig, ax = make_figure(figsize=(16, 12))
        bins = np.arange(0.5, max_npv + 1.5, 1)
        ax.hist(unique_pvs.values, bins=bins, color="C0", edgecolor="black")
        ax.set_xlabel("Unique PVs per event")
        ax.set_ylabel("Events")
        ax.set_xticks(range(1, max_npv + 1))
        ax.set_title(
            f"Unique PVs with MVA > {nominal_cut}{lumi_label}\n"
            f"(mean={unique_pvs.mean():.2f}, "
            f"{n_events_with_sv} events)"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "unique_pvs.png", dpi=150)
        plt.close(fig)
    else:
        print("\n  (best_pv_index not available, skipping unique PV count)")

    print(f"\nPlots saved to {out_dir}")


if __name__ == "__main__":
    main()
