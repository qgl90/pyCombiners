"""Template script to build custom decay-channel combiners step by step.

This script is designed for adaptation:
1. Map your raw input schema into `EventInput`.
2. Define channel-specific hypotheses/cuts.
3. Run direct or staged combinations.
4. Write analysis-ready output table.

Run without package installation:
    PYTHONPATH=src python3 examples/new_decay_channel_template.py \
      --input-events examples/events.json \
      --output examples/custom_channel_output.parquet \
      --channel dplus_kpipi
"""

from __future__ import annotations
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


import argparse
import json
from pathlib import Path
from typing import Any

from trackcomb import (
    CombinationCuts,
    EventInput,
    ParticleCombiner,
    ParticleHypothesis,
    TrackPreselection,
    TrackState,
    combination_to_track_state,
    make_kaon,
    make_muon,
    make_pion,
)
from trackcomb.io import load_events_json, write_results_table

MASS_JPSI = 3.0969


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the template workflow."""
    parser = argparse.ArgumentParser(description="Template for custom decay-channel studies.")
    parser.add_argument(
        "--input-events",
        required=True,
        help=(
            "Input JSON file. It can be the framework `events` format or your own "
            "raw format if you implement `map_raw_event_to_event_input`."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output table path (.parquet/.csv/.pkl).",
    )
    parser.add_argument(
        "--channel",
        default="dplus_kpipi",
        choices=("dplus_kpipi", "b_to_jpsi_k_staged"),
        help="Reference channel workflow to run.",
    )
    parser.add_argument(
        "--raw-schema",
        action="store_true",
        help=(
            "Use this if input is not already in framework events format. "
            "You must adapt `map_raw_event_to_event_input` for your schema."
        ),
    )
    return parser.parse_args()


def load_events(path: str, raw_schema: bool) -> list[EventInput]:
    """Load events either directly or through a custom schema adapter."""
    if not raw_schema:
        return load_events_json(path)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_events = payload.get("events", payload)
    if not isinstance(raw_events, list):
        raise ValueError("Raw input must contain an events list.")
    return [map_raw_event_to_event_input(evt) for evt in raw_events]


def map_raw_event_to_event_input(raw_event: dict[str, Any]) -> EventInput:
    """Map one raw event object into framework-native `EventInput`.

    Adapt this function to your own input schema. The sample expects:
    - `event_id` (or fallback id)
    - `tracks` list
    - `primary_vertices` list
    with field names matching framework examples.
    """
    from trackcomb import PrimaryVertex, TrackState

    event_id = str(raw_event.get("event_id", "evt_unknown"))

    tracks: list[TrackState] = []
    for idx, trk in enumerate(raw_event.get("tracks", [])):
        state = trk.get("state", trk)
        cov4 = tuple(tuple(float(v) for v in row) for row in trk["cov4"])
        tracks.append(
            TrackState(
                track_id=str(trk.get("track_id", f"trk{idx}")),
                z=float(trk["z"]),
                x=float(state["x"]),
                y=float(state["y"]),
                tx=float(state["tx"]),
                ty=float(state["ty"]),
                time=float(state.get("time", trk["time"])),
                cov4=cov4,  # type: ignore[arg-type]
                sigma_time=float(trk.get("sigma_time", 1.0)),
                p=float(trk["p"]),
                charge=int(trk.get("charge", 0)),
                has_rich1=bool(trk.get("hasRICH1", False)),
                has_rich2=bool(trk.get("hasRICH2", False)),
                rich_dll_pi=float(trk.get("richDLL_pi", 0.0)),
                rich_dll_k=float(trk.get("richDLL_k", 0.0)),
                rich_dll_p=float(trk.get("richDLL_p", 0.0)),
                rich_dll_e=float(trk.get("richDLL_e", 0.0)),
                has_calo=bool(trk.get("hasCALO", False)),
                calo_dll_e=float(trk.get("caloDLL_e", 0.0)),
                source_track_ids=(str(trk.get("track_id", f"trk{idx}")),),
            )
        )

    pvs: list[PrimaryVertex] = []
    for idx, pv in enumerate(raw_event.get("primary_vertices", [])):
        cov3 = tuple(tuple(float(v) for v in row) for row in pv["cov3"])
        pvs.append(
            PrimaryVertex(
                pv_id=str(pv.get("pv_id", f"pv{idx}")),
                x=float(pv["x"]),
                y=float(pv["y"]),
                z=float(pv["z"]),
                cov3=cov3,  # type: ignore[arg-type]
                time=float(pv["time"]),
                sigma_time=float(pv["sigma_time"]),
            )
        )

    return EventInput(event_id=event_id, tracks=tuple(tracks), primary_vertices=tuple(pvs))


def run_dplus_kpipi(events: list[EventInput]) -> list:
    """Example direct 3-body channel: D+ -> K- pi+ pi+ (+ charge conjugate)."""
    combiner = ParticleCombiner()
    return combiner.combine_events(
        events=events,
        n_body=3,
        mass_hypotheses=[[make_kaon(), make_pion(), make_pion()]],
        preselection=TrackPreselection(min_pt=0.5, min_ip_to_any_pv=0.05),
        cuts=CombinationCuts(
            allowed_charge_patterns=("-++", "+--"),
            max_doca=1.2,
            max_vertex_chi2=80.0,
            max_pair_time_chi2=25.0,
            min_mass=1.80,
            max_mass=1.94,
        ),
    )


def run_b_to_jpsi_k_staged(events: list[EventInput]) -> list:
    """Example staged channel: B -> J/psi(mu mu) K."""
    combiner = ParticleCombiner()
    final_candidates = []

    for event in events:
        tracks = list(event.tracks)
        pvs = list(event.primary_vertices)

        jpsi_results = combiner.combine(
            tracks=tracks,
            primary_vertices=pvs,
            n_body=2,
            mass_hypotheses=[[make_muon(), make_muon()]],
            cuts=CombinationCuts(
                allowed_charge_patterns=("+-", "-+"),
                max_doca=1.5,
                min_mass=2.9,
                max_mass=3.3,
            ),
            event_id=event.event_id,
        )

        track_map = {t.track_id: t for t in tracks}
        jpsi_tracks: list[TrackState] = []
        for idx, cand in enumerate(jpsi_results):
            constituents = [track_map[tid] for tid in cand.track_ids]
            jpsi_tracks.append(
                combination_to_track_state(
                    result=cand,
                    constituents=constituents,
                    track_id=f"{event.event_id}_jpsi_{idx}",
                )
            )

        kaon_tracks = [t for t in tracks if t.rich_dll_k > t.rich_dll_pi]
        for jpsi in jpsi_tracks:
            for kaon in kaon_tracks:
                final_candidates.extend(
                    combiner.combine(
                        tracks=[jpsi, kaon],
                        primary_vertices=pvs,
                        n_body=2,
                        mass_hypotheses=[
                            [ParticleHypothesis(name="J/psi", mass=MASS_JPSI), make_kaon()]
                        ],
                        cuts=CombinationCuts(max_doca=1.5, min_mass=4.9, max_mass=5.7),
                        event_id=event.event_id,
                    )
                )

    return final_candidates


def main() -> int:
    """Run the selected channel workflow and write one output table."""
    args = parse_args()
    events = load_events(args.input_events, raw_schema=args.raw_schema)

    if args.channel == "dplus_kpipi":
        results = run_dplus_kpipi(events)
    else:
        results = run_b_to_jpsi_k_staged(events)

    write_results_table(args.output, results)
    print(
        f"Wrote {len(results)} candidates for channel '{args.channel}' to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
