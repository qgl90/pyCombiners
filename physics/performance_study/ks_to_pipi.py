#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from trackcomb import (
    CombinationCuts,
    TrackPreselection,
    combine,
    iter_events_root,
    make_decay,
    truth_match,
)


def main():
    parser = argparse.ArgumentParser(description="Ks -> pi+pi- performance study")
    parser.add_argument("--input", required=True, help="ROOT file path")
    parser.add_argument("--tree", default="BestLongTracks/TrackTuple")
    parser.add_argument("--max-events", type=int, default=10)
    parser.add_argument("--out-dir", default="physics/performance_study/plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = iter_events_root(args.input, args.tree, max_events=args.max_events)

    # Cheated: truth filter, no kinematic cuts
    cheated_decay = make_decay(
        ["pi", "pi"],
        cuts=CombinationCuts(allowed_charge_patterns=("+-", "-+")),
    )

    def ks_track_filter(t):
        pids = t.metadata.get("mc_ancestor_pids", [])
        return any(abs(p) == 310 for p in pids)

    # Full: kinematic selection
    full_decay = make_decay(
        ["pi", "pi"],
        preselection=TrackPreselection(min_pt=0.05, min_ip_to_any_pv=0.1),
        cuts=CombinationCuts(
            min_mass=0.47, max_mass=0.52,
            max_doca=0.2, max_vertex_chi2=50.0,
            max_pair_time_chi2=15.0,
            allowed_charge_patterns=("+-", "-+"),
        ),
    )

    # Collect per-candidate kinematics for efficiency binning
    reco_vals: dict[str, list[float]] = {k: [] for k in ["pt", "eta", "p", "vz"]}
    found_vals: dict[str, list[float]] = {k: [] for k in ["pt", "eta", "p", "vz"]}

    n_reco_all, n_found_all, n_bkg_all = 0, 0, 0
    all_masses: list[float] = []

    for event in tqdm(events, desc="Processing", total=args.max_events):
        tracks = list(event.tracks)
        pvs = list(event.primary_vertices)

        # Cheated
        cheated_results = combine(
            cheated_decay, tracks, pvs,
            event_id=event.event_id,
            track_filter=ks_track_filter,
        )
        for c in cheated_results:
            if truth_match(c, tracks, 310, [211, 211]):
                n_reco_all += 1
                reco_vals["pt"].append(c.pair_pt)
                reco_vals["eta"].append(c.pair_eta)
                reco_vals["p"].append(c.candidate_p4.p2 ** 0.5)
                reco_vals["vz"].append(c.vertex_xyz[2])

        # Full
        full_results = combine(full_decay, tracks, pvs, event_id=event.event_id)
        for c in full_results:
            all_masses.append(c.candidate_p4.mass)
            if truth_match(c, tracks, 310, [211, 211]):
                n_found_all += 1
                found_vals["pt"].append(c.pair_pt)
                found_vals["eta"].append(c.pair_eta)
                found_vals["p"].append(c.candidate_p4.p2 ** 0.5)
                found_vals["vz"].append(c.vertex_xyz[2])
            else:
                n_bkg_all += 1

    eff = n_found_all / max(n_reco_all, 1) * 100
    pur = n_found_all / max(n_found_all + n_bkg_all, 1) * 100
    print(f"\nReconstructible: {n_reco_all}  Found: {n_found_all}  Bkg: {n_bkg_all}")
    print(f"Efficiency: {n_found_all}/{n_reco_all} = {eff:.1f}%")
    print(f"Purity: {n_found_all}/{n_found_all+n_bkg_all} = {pur:.1f}%")

    # -----------------------------------------------------------------
    # Efficiency vs kinematics plots
    # -----------------------------------------------------------------
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        variables = [
            ("pt",  r"$p_T(K_S^0)$ [GeV]",  np.linspace(0, 5, 11)),
            ("eta", r"$\eta(K_S^0)$",        np.linspace(2, 5, 13)),
            ("p",   r"$p(K_S^0)$ [GeV]",     np.linspace(0, 50, 11)),
            ("vz",  r"Vertex $z$ [mm]",       np.linspace(-200, 800, 11)),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        for ax, (key, xlabel, bins) in zip(axes, variables):
            num, _ = np.histogram(found_vals[key], bins=bins)
            den, _ = np.histogram(reco_vals[key], bins=bins)
            with np.errstate(divide="ignore", invalid="ignore"):
                eff_bin = np.where(den > 0, num / den, np.nan)
                # Binomial uncertainty
                err = np.where(den > 0,
                               np.sqrt(eff_bin * (1 - eff_bin) / den),
                               np.nan)
            centers = 0.5 * (bins[:-1] + bins[1:])
            mask = den > 0
            ax.errorbar(centers[mask], eff_bin[mask], yerr=err[mask],
                        fmt="o", markersize=5, capsize=3, color="black")
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("Efficiency", fontsize=12)
            ax.set_ylim(-0.05, 1.15)
            ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.5)

        fig.suptitle(
            f"$K_S^0 \\to \\pi^+\\pi^-$ efficiency "
            f"({args.max_events} events, {n_found_all}/{n_reco_all} = {eff:.0f}%)",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "ks_eff_vs_kinematics.png", dpi=150)
        print(f"Saved {out_dir / 'ks_eff_vs_kinematics.png'}")

        # Mass distribution (all candidates, no signal/background split)
        fig_m, ax_m = plt.subplots(figsize=(8, 5))
        ax_m.hist(all_masses, bins=50, range=(0.47, 0.52),
                  histtype="stepfilled", alpha=0.7, color="steelblue",
                  label=f"All candidates ({len(all_masses)})")
        ax_m.axvline(0.497611, color="red", linestyle="--", linewidth=1,
                     label=r"PDG $m(K_S^0)$")
        ax_m.set_xlabel(r"$m(\pi^+\pi^-)$ [GeV]", fontsize=13)
        ax_m.set_ylabel("Candidates", fontsize=13)
        ax_m.set_title(
            f"$K_S^0 \\to \\pi^+\\pi^-$ mass ({args.max_events} events)",
            fontsize=14,
        )
        ax_m.legend(fontsize=11)
        fig_m.tight_layout()
        fig_m.savefig(out_dir / "ks_mass.png", dpi=150)
        print(f"Saved {out_dir / 'ks_mass.png'}")
    except ImportError:
        print("matplotlib/numpy not available, skipping plots.")


if __name__ == "__main__":
    main()
