"""Example custom callback: write a tiny summary of best timing candidate."""

from __future__ import annotations
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


import json
from pathlib import Path


def process(results, context):
    """Select best candidate by pair-time chi2 and save a summary JSON."""
    best = min(results, key=lambda r: r.pair_time_chi2, default=None)
    summary = {
        "n_results": len(results),
        "best_pair_time_chi2": None if best is None else best.pair_time_chi2,
        "best_track_ids": None if best is None else list(best.track_ids),
        "best_candidate_mass": None if best is None else best.candidate_p4.mass,
    }
    out_path = Path(context["output_path"]).with_name("summary.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Custom analysis summary written to {out_path}")
