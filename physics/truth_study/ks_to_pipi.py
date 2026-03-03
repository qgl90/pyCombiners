#!/usr/bin/env python3
"""Cheated Ks -> pi+ pi- study: signal-only distributions.

Uses truth-filtered tracks (Ks ancestry) so there is no combinatorial
background.  This is fast and lets you see the signal shapes to define
reasonable cut ranges before running the full distribution study.

Usage:
    micromamba activate run5
    PYTHONPATH=src python3 physics/truth_study/ks_pipi.py \
        --input input/ntuple_full_1000.root \
        --max-events 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from trackcomb import (
    CombinationCuts,
    combine,
    iter_events_root,
    make_decay,
    truth_match,
)
from trackcomb.truth import get_true_decay_groups

KS_PDG = 310
PION_PDG = 211
PDG_KS_MASS = 0.497611  # GeV


def main():
    parser = argparse.ArgumentParser(
        description="Cheated Ks -> pi+pi- distributions (signal only)",
    )
    parser.add_argument("--input", required=True, help="ROOT file path")
    parser.add_argument("--tree", default="BestLongTracks/TrackTuple")
    parser.add_argument("--max-events", type=int, default=200)
    parser.add_argument("--out-dir", default="physics/truth_study/plots_ks")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = iter_events_root(args.input, args.tree, max_events=args.max_events)

    # Cheated decay: no kinematic cuts, just opposite charge
    cheated_decay = make_decay(
        ["pi", "pi"],
        cuts=CombinationCuts(allowed_charge_patterns=("+-", "-+")),
    )

    def ks_track_filter(t):
        pids = t.metadata.get("mc_ancestor_pids", [])
        return any(abs(p) == KS_PDG for p in pids)

    sig_cands = []
    sig_tracks = []  # per-candidate track lists (for track-level observables)
    n_true_total = 0

    for event in tqdm(events, desc="Processing", total=args.max_events):
        tracks = list(event.tracks)
        pvs = list(event.primary_vertices)
        track_map = {t.track_id: t for t in tracks}

        n_true_total += len(get_true_decay_groups(
            tracks, mother_pdg=KS_PDG, daughter_pdgs=[PION_PDG, PION_PDG],
        ))

        candidates = combine(
            cheated_decay, tracks, pvs,
            event_id=event.event_id,
            track_filter=ks_track_filter,
        )
        for c in candidates:
            if truth_match(c, tracks, mother_pdg=KS_PDG,
                           daughter_pdgs=[PION_PDG, PION_PDG]):
                sig_cands.append(c)
                sig_tracks.append([track_map[tid] for tid in c.source_track_ids])

    n_sig = len(sig_cands)
    print(f"\n{'='*60}")
    print(f"True Ks->pipi in data:     {n_true_total}")
    print(f"Cheated reco (signal):     {n_sig}")
    print(f"Cheated efficiency:        {n_sig}/{n_true_total} = "
          f"{n_sig/max(n_true_total,1)*100:.1f}%")
    print(f"{'='*60}")

    if not sig_cands:
        print("No signal candidates found, exiting.")
        return

    # -----------------------------------------------------------------
    # Print observable ranges (percentiles) to help define cuts
    # -----------------------------------------------------------------
    def _get_doca(c):
        return max(c.doca_pairs.values()) if c.doca_pairs else 0.0

    def _get_min_ip(c):
        return min(c.track_min_ip.values()) if c.track_min_ip else 0.0

    observables = [
        ("mass",           lambda c, _: c.candidate_p4.mass),
        ("vertex_chi2",    lambda c, _: c.vertex_chi2),
        ("pair_time_chi2", lambda c, _: c.pair_time_chi2),
        ("doca",           lambda c, _: _get_doca(c)),
        ("pair_pt",        lambda c, _: c.pair_pt),
        ("pair_eta",       lambda c, _: c.pair_eta),
        ("min_track_ip",   lambda c, _: _get_min_ip(c)),
        ("min_track_pt",   lambda _, trks: min(t.pt for t in trks)),
    ]

    print(f"\n{'observable':<20}  {'min':>8}  {'1%':>8}  {'5%':>8}  {'median':>8}  "
          f"{'95%':>8}  {'99%':>8}  {'max':>8}")
    print("-" * 94)
    obs_values = {}
    for name, fn in observables:
        vals = np.array([fn(c, trks) for c, trks in zip(sig_cands, sig_tracks)])
        obs_values[name] = vals
        p1, p5, p50, p95, p99 = np.percentile(vals, [1, 5, 50, 95, 99])
        print(f"{name:<20}  {vals.min():>8.4f}  {p1:>8.4f}  {p5:>8.4f}  {p50:>8.4f}  "
              f"{p95:>8.4f}  {p99:>8.4f}  {vals.max():>8.4f}")

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_configs = [
            ("mass",           r"$m(\pi^+\pi^-)$ [GeV]",  None,      60),
            ("vertex_chi2",    r"Vertex $\chi^2$",         (0, 25),   50),
            ("pair_time_chi2", r"Pair time $\chi^2$",      (0, 15),   50),
            ("doca",           r"DOCA [mm]",               (0, 1),    50),
            ("pair_pt",        r"$p_T(K_S^0)$ [GeV]",     (0, 5),    50),
            ("pair_eta",       r"$\eta(K_S^0)$",           (2, 5.5),  50),
            ("min_track_ip",   r"min track IP [mm]",       (0, 5),    50),
            ("min_track_pt",   r"min track $p_T$ [GeV]",   (0, 3),    50),
        ]

        n_vars = len(plot_configs)
        ncols = min(n_vars, 3)
        nrows = (n_vars + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
        axes = list(axes.flatten())

        for ax, (name, xlabel, xrange, nbins) in zip(axes, plot_configs):
            vals = obs_values[name]
            if xrange is None:
                margin = 0.01 * max(abs(vals.min()), abs(vals.max()), 1e-6)
                xrange = (vals.min() - margin, vals.max() + margin)
            ax.hist(vals, bins=nbins, range=xrange,
                    histtype="stepfilled", alpha=0.7, color="steelblue",
                    label=f"Signal ({len(vals)})")
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("Candidates", fontsize=11)
            p1, p5, p95, p99 = np.percentile(vals, [1, 5, 95, 99])
            ax.axvline(p1, color="orange", ls=":", lw=1, alpha=0.7,
                       label=f"1%: {p1:.3f}")
            ax.axvline(p5, color="red", ls="--", lw=1, alpha=0.7,
                       label=f"5%: {p5:.3f}")
            ax.axvline(p95, color="red", ls="--", lw=1, alpha=0.7,
                       label=f"95%: {p95:.3f}")
            ax.axvline(p99, color="orange", ls=":", lw=1, alpha=0.7,
                       label=f"99%: {p99:.3f}")
            ax.legend(fontsize=8)

        for ax in axes[n_vars:]:
            ax.set_visible(False)

        fig.suptitle(
            f"$K_S^0 \\to \\pi^+\\pi^-$ cheated signal "
            f"({n_sig} candidates, {args.max_events} events)",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "ks_signal_distributions.png", dpi=150)
        print(f"\nSaved {out_dir / 'ks_signal_distributions.png'}")
    except ImportError:
        print("matplotlib/numpy not available, skipping plots.")


if __name__ == "__main__":
    main()
