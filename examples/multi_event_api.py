"""Multi-event API example using named particle-hypothesis builders.

Run from repository root without installation:
    PYTHONPATH=src python examples/multi_event_api.py
"""

from __future__ import annotations
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


from pathlib import Path

from trackcomb import ParticleCombiner, make_kaon, make_pion
from trackcomb.io import load_events_json, write_results_table


def main() -> int:
    """Load events, run 2-body combinations, and write a parquet table."""
    events = load_events_json("examples/events.json")
    mass_hypotheses = [
        [make_pion(), make_pion()],
        [make_kaon(), make_pion()],
    ]
    results = ParticleCombiner().combine_events(
        events=events,
        n_body=2,
        mass_hypotheses=mass_hypotheses,
    )
    out_path = Path("examples/multi_event_output.parquet")
    write_results_table(out_path, results)
    print(f"Wrote {len(results)} candidates to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
