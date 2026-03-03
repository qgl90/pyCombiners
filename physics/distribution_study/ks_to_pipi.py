#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from trackcomb import (
    CombinationCuts,
    TrackPreselection,
    combine,
    count_true_decays,
    iter_events_root,
    make_decay,
    truth_match,
)
from trackcomb.plot import plot_distributions


def main():
    parser = argparse.ArgumentParser(description="Ks -> pi+pi- distribution study")
    parser.add_argument("--input", required=True, help="ROOT file path")
    parser.add_argument("--tree", default="BestLongTracks/TrackTuple")
    parser.add_argument("--max-events", type=int, default=10)
    parser.add_argument("--out-dir", default="physics/distribution_study/plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = iter_events_root(args.input, args.tree, max_events=args.max_events)

    # -----------------------------------------------------------------
    # 1. Reconstruct with loose cuts to collect candidates
    # -----------------------------------------------------------------
    ks_decay = make_decay(
        ["pi", "pi"],
        preselection=TrackPreselection(min_pt=0.06, min_ip_to_any_pv=0.08),
        cuts=CombinationCuts(
            min_mass=0.47,
            max_mass=0.52,
            max_doca=0.3,
            max_vertex_chi2=10.0,
            max_pair_time_chi2=10.0,
            allowed_charge_patterns=("+-", "-+"),
        ),
    )

    sig_cands, bkg_cands = [], []
    cand_min_track_pt: dict[int, float] = {}  # id(candidate) -> min track pt
    n_true_total = 0

    for event in tqdm(events, desc="Processing", total=args.max_events):
        tracks = list(event.tracks)
        pvs = list(event.primary_vertices)
        track_map = {t.track_id: t for t in tracks}
        n_true_total += count_true_decays(
            tracks, mother_pdg=310, daughter_pdgs=[211, 211],
        )

        candidates = combine(ks_decay, tracks, pvs, event_id=event.event_id)
        for c in candidates:
            cand_min_track_pt[id(c)] = min(
                track_map[tid].pt for tid in c.source_track_ids
            )
            if truth_match(c, tracks, mother_pdg=310, daughter_pdgs=[211, 211]):
                sig_cands.append(c)
            else:
                bkg_cands.append(c)

    n_sig = len(sig_cands)
    n_bkg = len(bkg_cands)
    print(f"\n{'='*60}")
    print(f"True Ks->pipi in data:     {n_true_total}")
    print(f"Reconstructed (signal):    {n_sig}")
    print(f"Reconstructed (bkg):       {n_bkg}")
    print(f"Efficiency:                {n_sig}/{n_true_total} = "
          f"{n_sig/max(n_true_total,1)*100:.1f}%")
    print(f"Purity:                    {n_sig}/{n_sig+n_bkg} = "
          f"{n_sig/max(n_sig+n_bkg,1)*100:.1f}%")
    print(f"{'='*60}")

    # -----------------------------------------------------------------
    # 2. Signal percentiles
    # -----------------------------------------------------------------
    obs = [
        ("mass",          lambda c: c.candidate_p4.mass,
         r"$m(\pi\pi)$ [GeV]", (0.47, 0.52), 50),
        ("vtx_chi2",      lambda c: c.vertex_chi2,
         r"Vertex $\chi^2$",   (0, 10),      50),
        ("time_chi2",     lambda c: c.pair_time_chi2,
         r"Pair time $\chi^2$", (0, 10),       45),
        ("doca",          lambda c: max(c.doca_pairs.values()) if c.doca_pairs else 0,
         r"DOCA [mm]",          (0, 0.5),       40),
        ("pair_pt",       lambda c: c.pair_pt,
         r"$p_T$ [GeV]",       (0, 2),       50),
        ("min_ip",        lambda c: min(c.track_min_ip.values()) if c.track_min_ip else 0,
         r"min track IP [mm]", (0, 2),        50),
        ("min_track_pt",  lambda c: cand_min_track_pt[id(c)],
         r"min track $p_T$ [GeV]", (0, 2),    50),
    ]

    if sig_cands:
        import numpy as np
        print(f"\nSignal percentiles:")
        print(f"{'observable':<20}  {'1%':>8}  {'5%':>8}  {'median':>8}  "
              f"{'95%':>8}  {'99%':>8}")
        print("-" * 68)
        for name, fn, *_ in obs:
            vals = np.array([fn(c) for c in sig_cands])
            p1, p5, p50, p95, p99 = np.percentile(vals, [1, 5, 50, 95, 99])
            print(f"{name:<20}  {p1:>8.4f}  {p5:>8.4f}  {p50:>8.4f}  "
                  f"{p95:>8.4f}  {p99:>8.4f}")

    # -----------------------------------------------------------------
    # 3. Plots: normalised signal vs background distributions
    # -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")

        plot_distributions(
            sig_cands, bkg_cands, obs,
            title="Ks signal vs background (normalised)",
            out_path=out_dir / "ks_observables.png",
        )
        print(f"\nSaved {out_dir / 'ks_observables.png'}")
    except ImportError:
        print("matplotlib not available, skipping plots.")


if __name__ == "__main__":
    main()
