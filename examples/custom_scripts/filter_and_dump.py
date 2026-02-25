"""Example custom callback: apply tighter cuts and dump selected candidates."""

from __future__ import annotations
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


import json
from pathlib import Path


def process(results, context):
    """Filter by quality + mass window and write a compact JSON report."""
    selected = [
        r
        for r in results
        if r.pair_time_chi2 < 2.0 and r.vertex_chi2 < 2.0 and 0.2 < r.candidate_p4.mass < 2.0
    ]
    payload = {
        "n_selected": len(selected),
        "selected": [
            {
                "track_ids": list(r.track_ids),
                "mass": r.candidate_p4.mass,
                "pair_time_chi2": r.pair_time_chi2,
                "vertex_chi2": r.vertex_chi2,
            }
            for r in selected
        ],
    }
    out = Path(context["output_path"]).with_name("selected_candidates.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
