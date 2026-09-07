#!/usr/bin/env python3
"""Unified decay performance analysis: efficiency vs kinematics + mass distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CHANNELS = {
    "ks_to_pipi": {
        "decay_latex": r"K_S^0 \to \pi^+\pi^-",
        "mass_latex": r"m(\pi^+\pi^-)",
        "pdg_mass": 497.611,
        "mass_range": (470, 520),
        "prefix": "ks",
        "variables": [
            ("pt", r"$p_T$ [MeV]", (0, 5000, 11)),
            ("eta", r"$\eta$", (2, 5, 13)),
            ("p", r"$p$ [MeV]", (0, 50000, 11)),
            ("vertex_z", r"Vertex $z$ [mm]", (-200, 800, 11)),
        ],
    },
    "bs_to_mumu": {
        "decay_latex": r"B_s^0 \to \mu^+\mu^-",
        "mass_latex": r"m(\mu^+\mu^-)",
        "pdg_mass": 5366.88,
        "mass_range": (4700, 6000),
        "prefix": "bs",
        "variables": [
            ("pt", r"$p_T$ [MeV]", (0, 15000, 16)),
            ("eta", r"$\eta$", (2, 5, 13)),
            ("p", r"$p$ [MeV]", (0, 200000, 11)),
            ("vertex_z", r"Vertex $z$ [mm]", (-200, 800, 11)),
        ],
    },
    "bs_to_jpsiphi": {
        "decay_latex": r"B_s^0 \to J/\psi(\mu\mu)\,\phi(KK)",
        "mass_latex": r"m(J/\psi\,\phi)",
        "pdg_mass": 5366.88,
        "mass_range": (4000, 6500),
        "prefix": "bs",
        "variables": [
            ("pt", r"$p_T$ [MeV]", (0, 20000, 16)),
            ("eta", r"$\eta$", (2, 5, 13)),
            ("p", r"$p$ [MeV]", (0, 300000, 11)),
            ("vertex_z", r"Vertex $z$ [mm]", (-200, 800, 11)),
        ],
        "intermediates": [
            {
                "column": "daughter0_mass",
                "name": "jpsi",
                "latex": r"J/\psi",
                "mass_latex": r"m(\mu^+\mu^-)",
                "pdg_mass": 3096.9,
                "mass_range": (0, 5500),
            },
            {
                "column": "daughter1_mass",
                "name": "phi",
                "latex": r"\phi",
                "mass_latex": r"m(K^+K^-)",
                "pdg_mass": 1019.461,
                "mass_range": (0, 1100),
            },
        ],
    },
}


SIGNAL_CUTFLOW_STAGES = (
    ("has_four_signal_long_tracks", "Four truth-matched signal Long tracks"),
    ("has_cheated_signal_candidate", "Cheated Bs candidate reconstructed"),
    ("passes_signal_muon_selection", "Both signal muons selected"),
    (
        "passes_four_signal_track_selection",
        "All four signal tracks selected",
    ),
    ("passes_signal_jpsi_selection", "Signal J/psi selected"),
    ("passes_signal_phi_selection", "Signal phi selected"),
    ("passes_signal_bs_selection", "Signal Bs selected"),
)

BKG_CATEGORY_LABELS = {
    -1: "Undefined",
    0: "Signal",
    10: "QuasiSignal",
    20: "FullyRecoPhysBkg",
    30: "Reflection",
    40: "PartRecoPhysBkg",
    50: "LowMassBkg",
    60: "Ghost",
    63: "Clone",
    66: "Hierarchy",
    70: "FromPV",
    80: "AllFromSamePV",
    100: "FromDifferentPV",
    110: "bbar",
    120: "ccbar",
    130: "uds",
    1000: "LastGlobal",
}


def bkgcat_group(code):
    """Return the presentation-level official background-category group."""
    code = int(code)
    if code in (0, 10, 50):
        return "signal-like (0,10,50)"
    if code in (20, 30, 40):
        return "physics background (20,30,40)"
    if code >= 60:
        return "combinatorial / fake (>=60)"
    return "undefined / other"


def build_signal_cutflow(frame):
    """Summarize cumulative event-level signal selection stages."""
    import pandas as pd

    denominator = int(frame[SIGNAL_CUTFLOW_STAGES[0][0]].astype(bool).sum())
    previous = denominator
    rows = []
    for column, label in SIGNAL_CUTFLOW_STAGES:
        selected = int(frame[column].astype(bool).sum())
        rows.append(
            {
                "stage": column,
                "stage_label": label,
                "selected_signal_events": selected,
                "previous_stage_signal_events": previous,
                "relative_efficiency": selected / previous
                if previous
                else np.nan,
                "cumulative_efficiency": (
                    selected / denominator if denominator else np.nan
                ),
            }
        )
        previous = selected
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Decay performance analysis")
    parser.add_argument(
        "--cheated", required=True, help="Path to cheated parquet"
    )
    parser.add_argument(
        "--cheated-selected",
        default=None,
        help="truth-guided candidate Parquet with the full selection applied",
    )
    parser.add_argument(
        "--full", required=True, help="Path to full reco parquet"
    )
    parser.add_argument(
        "--cutflow",
        default=None,
        help="optional event-level signal-only cut-flow Parquet",
    )
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for plots"
    )
    parser.add_argument(
        "--channel", required=True, choices=CHANNELS, help="Decay channel"
    )
    parser.add_argument(
        "--lumi", default="", help="Luminosity label for plot titles"
    )
    parser.add_argument(
        "--tag", default="", help="Extra tag in plot titles (e.g. 'NO TIMING')"
    )
    args = parser.parse_args()

    ch = CHANNELS[args.channel]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    cheated = pd.read_parquet(args.cheated)
    cheated_selected = (
        pd.read_parquet(args.cheated_selected)
        if args.cheated_selected
        else None
    )
    full = pd.read_parquet(args.full)
    cutflow_frame = pd.read_parquet(args.cutflow) if args.cutflow else None

    signal_key = ["run_number", "event_number", "mc_key"]
    event_key = ["run_number", "event_number"]

    reco = cheated[cheated["is_signal"]].drop_duplicates(signal_key).copy()
    reco["p"] = np.sqrt(reco["px"] ** 2 + reco["py"] ** 2 + reco["pz"] ** 2)

    sel = full
    full_signal_candidates = sel[sel["is_signal"]].copy()
    found_all = full_signal_candidates.drop_duplicates(signal_key).copy()
    reconstructible_keys = pd.MultiIndex.from_frame(reco[signal_key])
    found_keys = pd.MultiIndex.from_frame(found_all[signal_key])
    found_in_baseline = found_keys.isin(reconstructible_keys)
    n_found_outside_baseline = int((~found_in_baseline).sum())
    found = found_all.loc[found_in_baseline].copy()
    found["p"] = np.sqrt(
        found["px"] ** 2 + found["py"] ** 2 + found["pz"] ** 2
    )
    bkg = sel[~sel["is_signal"]]

    cheated_selected_signal = None
    selected_truth_guided = None
    n_selected_truth_guided = 0
    n_selected_truth_guided_events = 0
    if cheated_selected is not None:
        cheated_selected_signal = cheated_selected[
            cheated_selected["is_signal"]
        ].drop_duplicates(signal_key)
        selected_keys = pd.MultiIndex.from_frame(
            cheated_selected_signal[signal_key]
        )
        selected_truth_guided = cheated_selected_signal.loc[
            selected_keys.isin(reconstructible_keys)
        ].copy()
        n_selected_truth_guided = len(selected_truth_guided)
        n_selected_truth_guided_events = len(
            selected_truth_guided[event_key].drop_duplicates()
        )

    n_events_processed = (
        len(cutflow_frame[event_key].drop_duplicates())
        if cutflow_frame is not None
        else np.nan
    )

    n_reco = len(reco)
    n_full_signal_total = len(found_all)
    n_full_signal_candidates = len(full_signal_candidates)
    n_found = len(found)
    n_bkg = len(bkg)
    eff = n_found / max(n_reco, 1) * 100
    pur = n_full_signal_candidates / max(len(sel), 1) * 100
    eff_fraction = n_found / max(n_reco, 1)
    eff_uncertainty = (
        np.sqrt(eff_fraction * (1.0 - eff_fraction) / n_reco)
        if n_reco
        else np.nan
    )
    n_reco_events = len(reco[event_key].drop_duplicates())
    n_found_events = len(found[event_key].drop_duplicates())
    event_eff_fraction = n_found_events / max(n_reco_events, 1)
    event_eff_uncertainty = (
        np.sqrt(
            event_eff_fraction * (1.0 - event_eff_fraction) / n_reco_events
        )
        if n_reco_events
        else np.nan
    )

    prefix = ch["prefix"]
    summary = pd.DataFrame(
        [
            {
                "channel": args.channel,
                "luminosity_label": args.lumi,
                "tag": args.tag,
                "signal_definition": "bkgcat <= 10",
                "denominator_definition": (
                    "unique truth-assisted reconstructible signal candidates"
                ),
                "reconstructible_signal_decays": n_reco,
                "truth_guided_selected_signal_decays": n_selected_truth_guided,
                "truth_guided_signal_selection_efficiency": (
                    n_selected_truth_guided / n_reco if n_reco else np.nan
                ),
                "selected_signal_decays": n_found,
                "full_signal_decays_total": n_full_signal_total,
                "full_signal_candidates_made": n_full_signal_candidates,
                "signal_selection_efficiency": eff_fraction,
                "signal_selection_efficiency_uncertainty": eff_uncertainty,
                "reconstructible_signal_events": n_reco_events,
                "truth_guided_selected_signal_events": (
                    n_selected_truth_guided_events
                ),
                "truth_guided_event_selection_efficiency": (
                    n_selected_truth_guided_events / n_reco_events
                    if n_reco_events
                    else np.nan
                ),
                "selected_signal_events": n_found_events,
                "event_signal_selection_efficiency": event_eff_fraction,
                "event_signal_selection_efficiency_uncertainty": (
                    event_eff_uncertainty
                ),
                "selected_signal_decays_outside_baseline": (
                    n_found_outside_baseline
                ),
                "background_candidates": n_bkg,
                "events_processed": n_events_processed,
                "cheated_candidates_made": len(cheated),
                "cheated_selected_candidates_made": (
                    len(cheated_selected)
                    if cheated_selected is not None
                    else np.nan
                ),
                "full_candidates_made": len(full),
                "selected_sample_purity": pur / 100.0,
            }
        ]
    )
    summary_parquet = out_dir / f"{prefix}_selection_efficiency.parquet"
    summary_csv = out_dir / f"{prefix}_selection_efficiency.csv"
    summary.to_parquet(summary_parquet, index=False)
    summary.to_csv(summary_csv, index=False)

    tag_str = f" ({args.tag})" if args.tag else ""
    print(f"\n{'=' * 60}")
    print(f"Channel: {args.channel}{tag_str}")
    print(f"Reconstructible (cheated): {n_reco}")
    if cheated_selected is not None:
        print(
            "Selected (cheated + cuts): "
            f"{n_selected_truth_guided}/{n_reco} = "
            f"{100.0 * n_selected_truth_guided / max(n_reco, 1):.1f}%"
        )
    print(f"Found (full selection):    {n_found}")
    if np.isfinite(n_events_processed):
        print(f"Events processed:          {int(n_events_processed)}")
    print(
        "Selected signal events:     "
        f"{n_found_events}/{n_reco_events} = "
        f"{100.0 * event_eff_fraction:.1f}%"
    )
    print(f"Background:                {n_bkg}")
    print(f"Efficiency: {n_found}/{n_reco} = {eff:.1f}%")
    print(
        "Purity:     "
        f"{n_full_signal_candidates}/{len(sel)} candidates = {pur:.1f}%"
    )
    print(f"Selection table: {summary_parquet} and {summary_csv}")
    if n_found_outside_baseline:
        print(
            "WARNING: full selection contains "
            f"{n_found_outside_baseline} signal decays absent from the "
            "cheated denominator"
        )
    print(f"{'=' * 60}")

    # ---- Efficiency vs kinematics ----
    from trackcomb.plot import make_figure

    if cutflow_frame is not None:
        signal_cutflow = build_signal_cutflow(cutflow_frame)
        cutflow_parquet = out_dir / f"{prefix}_selection_cutflow.parquet"
        cutflow_csv = out_dir / f"{prefix}_selection_cutflow.csv"
        signal_cutflow.to_parquet(cutflow_parquet, index=False)
        signal_cutflow.to_csv(cutflow_csv, index=False)

        fig_c, ax_c = make_figure(figsize=(16, 10))
        positions = np.arange(len(signal_cutflow))
        efficiencies = 100.0 * signal_cutflow["cumulative_efficiency"]
        ax_c.barh(positions, efficiencies, color="steelblue", alpha=0.8)
        ax_c.set_yticks(positions)
        ax_c.set_yticklabels(signal_cutflow["stage_label"])
        ax_c.invert_yaxis()
        ax_c.set_xlim(0.0, 105.0)
        ax_c.set_xlabel("Cumulative signal-event efficiency [%]")
        ax_c.set_title(f"${ch['decay_latex']}$ signal-only selection cut flow")
        for position, efficiency, count in zip(
            positions,
            efficiencies,
            signal_cutflow["selected_signal_events"],
        ):
            ax_c.text(
                min(efficiency + 1.0, 101.0),
                position,
                f"{int(count)} ({efficiency:.1f}%)",
                va="center",
            )
        fig_c.tight_layout()
        cutflow_plot = out_dir / f"{prefix}_selection_cutflow.png"
        fig_c.savefig(cutflow_plot, dpi=150)
        print(
            f"Signal cut flow: {cutflow_parquet}, {cutflow_csv}, "
            f"and {cutflow_plot}"
        )

    lumi_tag = f" [{args.lumi}]" if args.lumi else ""
    title_suffix = f" {args.tag}" if args.tag else ""

    fig, axes = make_figure(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (key, xlabel, (lo, hi, nbins)) in zip(axes, ch["variables"]):
        bins = np.linspace(lo, hi, nbins)
        num, _ = np.histogram(found[key].values, bins=bins)
        den, _ = np.histogram(reco[key].values, bins=bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff_bin = np.where(den > 0, num / den, np.nan)
            err = np.where(
                den > 0, np.sqrt(eff_bin * (1 - eff_bin) / den), np.nan
            )
        centers = 0.5 * (bins[:-1] + bins[1:])
        mask = den > 0
        ax.errorbar(
            centers[mask],
            eff_bin[mask],
            yerr=err[mask],
            fmt="o",
            markersize=5,
            capsize=3,
            color="black",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Efficiency")
        ax.set_ylim(-0.05, 1.15)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.5)

    fig.suptitle(
        f"${ch['decay_latex']}$ efficiency{title_suffix}{lumi_tag} "
        f"({n_found}/{n_reco} = {eff:.0f}%)",
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_eff_vs_kinematics.png", dpi=150)
    print(f"Saved {out_dir / f'{prefix}_eff_vs_kinematics.png'}")

    event_text = (
        f"Events processed: {int(n_events_processed):,}"
        if np.isfinite(n_events_processed)
        else "Events processed: unavailable"
    )

    def _plot_candidate_comparison(
        column,
        mass_range,
        xlabel,
        title,
        output,
        pdg_mass,
    ):
        """Compare raw candidate yields using unselected cheated as y reference."""
        fig_mass, ax_mass = make_figure(figsize=(16, 12))
        bin_edges = np.linspace(mass_range[0], mass_range[1], 51)
        cheated_values = cheated[column].dropna().values
        cheated_counts, _ = np.histogram(cheated_values, bins=bin_edges)

        ax_mass.hist(
            cheated_values,
            bins=bin_edges,
            histtype="stepfilled",
            alpha=0.3,
            color="grey",
            edgecolor="black",
            linewidth=1.5,
            label=f"Cheated, no selection ({len(cheated):,} candidates)",
        )
        if cheated_selected is not None:
            ax_mass.hist(
                cheated_selected[column].dropna().values,
                bins=bin_edges,
                histtype="step",
                color="darkorange",
                linewidth=2.0,
                label=(
                    "Cheated + HLT2-like selection "
                    f"({len(cheated_selected):,} candidates)"
                ),
            )
        ax_mass.hist(
            full[column].dropna().values,
            bins=bin_edges,
            histtype="step",
            color="steelblue",
            linewidth=2.0,
            label=(
                "Full combinatorics "
                f"({len(full):,} candidates: "
                f"{n_full_signal_candidates:,} signal, {n_bkg:,} background)"
            ),
        )
        ax_mass.axvline(
            pdg_mass,
            color="red",
            linestyle="--",
            linewidth=1,
            label="PDG mass",
        )
        # Deliberately keep a common absolute-yield scale anchored to the
        # no-selection truth-guided reconstruction, as requested.
        cheated_peak = int(cheated_counts.max()) if len(cheated_counts) else 0
        ax_mass.set_ylim(0.0, max(1.0, 1.18 * cheated_peak))
        ax_mass.text(
            0.98,
            0.97,
            event_text,
            transform=ax_mass.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax_mass.set_xlabel(xlabel)
        ax_mass.set_ylabel("Candidates / bin")
        ax_mass.set_title(f"{title}{title_suffix}{lumi_tag}")
        ax_mass.legend(loc="upper left")
        fig_mass.tight_layout()
        fig_mass.savefig(output, dpi=150)
        print(f"Saved {output}")

    # ---- Raw candidate mass comparison ----
    _plot_candidate_comparison(
        "mass",
        ch["mass_range"],
        f"${ch['mass_latex']}$ [MeV]",
        f"${ch['decay_latex']}$ candidate mass",
        out_dir / f"{prefix}_mass.png",
        ch["pdg_mass"],
    )

    # ---- Full-reconstruction background-category plots ----
    bkg_counts = (
        full["bkgcat"]
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("bkgcat")
    )
    bkg_summary = bkg_counts.rename("candidates").reset_index()
    bkg_summary["category"] = bkg_summary["bkgcat"].map(
        lambda code: BKG_CATEGORY_LABELS.get(code, "Unknown")
    )
    bkg_summary["group"] = bkg_summary["bkgcat"].map(bkgcat_group)
    bkg_summary["fraction"] = bkg_summary["candidates"] / max(len(full), 1)
    bkg_summary["events_processed"] = n_events_processed
    bkg_summary.to_parquet(
        out_dir / f"{prefix}_bkgcat_yields.parquet", index=False
    )
    bkg_summary.to_csv(out_dir / f"{prefix}_bkgcat_yields.csv", index=False)

    fig_b, ax_b = make_figure(figsize=(16, 10))
    positions = np.arange(len(bkg_summary))
    bars = ax_b.bar(
        positions, bkg_summary["candidates"], color="steelblue", alpha=0.8
    )
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(
        [
            f"{code}\n{label}"
            for code, label in zip(
                bkg_summary["bkgcat"], bkg_summary["category"]
            )
        ],
        rotation=30,
        ha="right",
    )
    ax_b.set_xlabel("Assigned background category")
    ax_b.set_ylabel("Full-reconstruction candidates")
    ax_b.set_title(
        f"${ch['decay_latex']}$ full-reconstruction background categories"
        f"{title_suffix}{lumi_tag}"
    )
    ax_b.bar_label(
        bars, labels=[f"{value:,}" for value in bkg_summary["candidates"]]
    )
    ax_b.text(
        0.98,
        0.97,
        event_text,
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig_b.tight_layout()
    bkgcat_plot = out_dir / f"{prefix}_bkgcat.png"
    fig_b.savefig(bkgcat_plot, dpi=150)
    print(f"Saved {bkgcat_plot}")

    def _plot_full_mass_by_bkgcat(
        column, mass_range, xlabel, title, output, pdg_mass
    ):
        """Plot raw full-reconstruction candidate masses split by bkgcat."""
        fig_cat, ax_cat = make_figure(figsize=(16, 12))
        bin_edges = np.linspace(mass_range[0], mass_range[1], 51)
        for row in bkg_summary.itertuples(index=False):
            category_values = full.loc[
                full["bkgcat"].astype(int) == row.bkgcat, column
            ].dropna()
            ax_cat.hist(
                category_values.values,
                bins=bin_edges,
                histtype="step",
                linewidth=1.8,
                label=f"{row.bkgcat}: {row.category} ({row.candidates:,})",
            )
        ax_cat.axvline(
            pdg_mass,
            color="black",
            linestyle="--",
            linewidth=1,
            label="PDG mass",
        )
        ax_cat.text(
            0.98,
            0.97,
            event_text,
            transform=ax_cat.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax_cat.set_xlabel(xlabel)
        ax_cat.set_ylabel("Full-reconstruction candidates / bin")
        ax_cat.set_title(f"{title}{title_suffix}{lumi_tag}")
        ax_cat.legend(loc="upper left", fontsize="small")
        fig_cat.tight_layout()
        fig_cat.savefig(output, dpi=150)
        print(f"Saved {output}")

    _plot_full_mass_by_bkgcat(
        "mass",
        ch["mass_range"],
        f"${ch['mass_latex']}$ [MeV]",
        f"${ch['decay_latex']}$ full-reconstruction mass by bkgcat",
        out_dir / f"{prefix}_mass_by_bkgcat.png",
        ch["pdg_mass"],
    )

    fig_group, ax_group = make_figure(figsize=(16, 12))
    group_styles = {
        "signal-like (0,10,50)": "darkorange",
        "physics background (20,30,40)": "tab:green",
        "combinatorial / fake (>=60)": "steelblue",
        "undefined / other": "grey",
    }
    mass_edges = np.linspace(ch["mass_range"][0], ch["mass_range"][1], 51)
    full_groups = full["bkgcat"].map(bkgcat_group)
    for group_name in group_styles:
        values = full.loc[full_groups == group_name, "mass"].dropna()
        if values.empty:
            continue
        ax_group.hist(
            values.values,
            bins=mass_edges,
            histtype="step",
            linewidth=2.0,
            color=group_styles[group_name],
            label=f"{group_name} ({len(values):,})",
        )
    ax_group.axvline(
        ch["pdg_mass"],
        color="black",
        linestyle="--",
        linewidth=1,
        label="PDG mass",
    )
    ax_group.text(
        0.98,
        0.97,
        event_text,
        transform=ax_group.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax_group.set_xlabel(f"${ch['mass_latex']}$ [MeV]")
    ax_group.set_ylabel("Full-reconstruction candidates / bin")
    ax_group.set_title(
        f"${ch['decay_latex']}$ full-reconstruction mass by bkgcat group"
        f"{title_suffix}{lumi_tag}"
    )
    ax_group.legend(loc="upper left")
    fig_group.tight_layout()
    grouped_mass_plot = out_dir / f"{prefix}_mass_by_bkgcat_group.png"
    fig_group.savefig(grouped_mass_plot, dpi=150)
    print(f"Saved {grouped_mass_plot}")

    # ---- Mass resolution reference (channel-specific) ----
    ref = ch.get("resolution_reference")
    if ref and ref["column"] in reco.columns:
        fig_r, ax_r = make_figure(figsize=(16, 12))

        def _half68(x):
            x = x[np.isfinite(x)]
            return 0.5 * (np.percentile(x, 84) - np.percentile(x, 16))

        m_reco = reco["mass"].values
        m_ref = reco[ref["column"]].values
        ax_r.hist(
            m_reco,
            bins=50,
            range=ch["mass_range"],
            histtype="stepfilled",
            alpha=0.6,
            color="steelblue",
            label=f"Cheated, reco ($\\sigma_{{68}}$ = {_half68(m_reco):.0f} MeV)",
        )
        ax_r.hist(
            m_ref[np.isfinite(m_ref)],
            bins=50,
            range=ch["mass_range"],
            histtype="stepfilled",
            alpha=0.6,
            color="darkorange",
            label=f"{ref['label']} ($\\sigma_{{68}}$ = {_half68(m_ref):.0f} MeV)",
        )
        ax_r.axvline(
            ch["pdg_mass"],
            color="red",
            linestyle="--",
            linewidth=1,
            label="PDG mass",
        )
        ax_r.set_xlabel(f"${ch['mass_latex']}$ [MeV]")
        ax_r.set_ylabel("Candidates")
        ax_r.set_title(
            f"${ch['decay_latex']}$ mass resolution{title_suffix}{lumi_tag}"
        )
        ax_r.legend()
        fig_r.tight_layout()
        fig_r.savefig(out_dir / f"{prefix}_mass_resolution.png", dpi=150)
        print(f"Saved {out_dir / f'{prefix}_mass_resolution.png'}")

    # ---- Intermediate resonance mass distributions ----
    for inter in ch.get("intermediates", []):
        col = inter["column"]
        if col not in sel.columns:
            continue
        fname = f"{prefix}_{inter['name']}_mass.png"
        _plot_candidate_comparison(
            col,
            inter["mass_range"],
            f"${inter['mass_latex']}$ [MeV]",
            f"${inter['latex']}$ mass from final $B_s^0$ candidates",
            out_dir / fname,
            inter["pdg_mass"],
        )
        _plot_full_mass_by_bkgcat(
            col,
            inter["mass_range"],
            f"${inter['mass_latex']}$ [MeV]",
            f"${inter['latex']}$ mass in full $B_s^0$ candidates by bkgcat",
            out_dir / f"{prefix}_{inter['name']}_mass_by_bkgcat.png",
            inter["pdg_mass"],
        )


if __name__ == "__main__":
    main()
