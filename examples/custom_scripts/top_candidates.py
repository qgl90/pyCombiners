"""Example custom callback: rank candidates and persist top-N summary."""

from __future__ import annotations

import json
from pathlib import Path


def process(results, context):
    """Sort by timing/vertex quality and save top candidates."""
    ranked = sorted(results, key=lambda r: (r.pair_time_chi2, r.vertex_chi2))
    top3 = ranked[:3]
    payload = {
        "n_total": len(results),
        "top_candidates": [
            {
                "track_ids": list(r.track_ids),
                "masses": list(r.masses),
                "mass": r.candidate_p4.mass,
                "pair_time_chi2": r.pair_time_chi2,
                "vertex_chi2": r.vertex_chi2,
                "pair_pt": r.pair_pt,
                "pair_eta": r.pair_eta,
            }
            for r in top3
        ],
    }
    out = Path(context["output_path"]).with_name("top_candidates.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
