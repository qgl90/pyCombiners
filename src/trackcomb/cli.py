"""Command-line interface for running particle combinations on event inputs."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from .combiner import ParticleCombiner
from .io import (
    load_mass_hypotheses_json,
    load_primary_vertices_json,
    load_tracks_json,
    write_results_table,
)
from .models import CombinationCuts, CombinationResult, TrackPreselection


def build_parser() -> argparse.ArgumentParser:
    """Define and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="track-combiner",
        description="Build n-body track combinations with Lorentz vectors and time-chi2 filtering.",
    )
    parser.add_argument("--tracks", required=True, help="Input JSON with key 'tracks'.")
    parser.add_argument(
        "--primary-vertices",
        required=True,
        help="Input JSON for primary vertices (list with x,y,z,cov3,time,sigma_time).",
    )
    parser.add_argument(
        "--masses",
        required=True,
        help="Input JSON with key 'mass_hypotheses' (list of mass arrays).",
    )
    parser.add_argument(
        "--n-body",
        required=True,
        type=int,
        choices=[2, 3, 4],
        help="Combination multiplicity.",
    )
    parser.add_argument(
        "--max-vertex-chi2",
        type=float,
        default=None,
        help="Optional cut on 3D vertex fit chi2.",
    )
    parser.add_argument(
        "--max-vertex-time-chi2",
        type=float,
        default=None,
        help="Optional cut on vertex-time chi2.",
    )
    parser.add_argument(
        "--max-pair-time-chi2",
        type=float,
        default=None,
        help="Optional cut on pairwise time chi2.",
    )
    parser.add_argument("--max-doca", type=float, default=None, help="Require all pair DOCAs below this value.")
    parser.add_argument("--min-mass", type=float, default=None, help="Minimum candidate invariant mass.")
    parser.add_argument("--max-mass", type=float, default=None, help="Maximum candidate invariant mass.")
    parser.add_argument("--min-pair-pt", type=float, default=None, help="Minimum candidate transverse momentum.")
    parser.add_argument("--max-pair-pt", type=float, default=None, help="Maximum candidate transverse momentum.")
    parser.add_argument("--min-pair-eta", type=float, default=None, help="Minimum candidate pseudorapidity.")
    parser.add_argument("--max-pair-eta", type=float, default=None, help="Maximum candidate pseudorapidity.")
    parser.add_argument(
        "--allowed-charge-patterns",
        type=str,
        default=None,
        help="Comma-separated allowed charge patterns by track order (e.g. ++,--,+-,-+ or ++-,+--).",
    )
    parser.add_argument("--min-track-pt", type=float, default=None, help="Track preselection: minimum pT.")
    parser.add_argument("--min-track-eta", type=float, default=None, help="Track preselection: minimum eta.")
    parser.add_argument("--max-track-eta", type=float, default=None, help="Track preselection: maximum eta.")
    parser.add_argument(
        "--min-track-ip-to-any-pv",
        type=float,
        default=None,
        help="Track preselection: minimum IP wrt all PVs in event.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output table file for combinations (.parquet, .csv, .pkl).",
    )
    parser.add_argument(
        "--custom-script",
        default=None,
        help="Path to Python file with process(results, context) function.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: load inputs, run combiner, write table, optional custom hook."""
    args = build_parser().parse_args(argv)
    tracks = load_tracks_json(args.tracks)
    primary_vertices = load_primary_vertices_json(args.primary_vertices)
    mass_hypotheses = load_mass_hypotheses_json(args.masses)
    preselection = TrackPreselection(
        min_pt=args.min_track_pt,
        min_eta=args.min_track_eta,
        max_eta=args.max_track_eta,
        min_ip_to_any_pv=args.min_track_ip_to_any_pv,
    )
    cuts = CombinationCuts(
        max_doca=args.max_doca,
        max_vertex_chi2=args.max_vertex_chi2,
        max_vertex_time_chi2=args.max_vertex_time_chi2,
        max_pair_time_chi2=args.max_pair_time_chi2,
        min_mass=args.min_mass,
        max_mass=args.max_mass,
        min_pair_pt=args.min_pair_pt,
        max_pair_pt=args.max_pair_pt,
        min_pair_eta=args.min_pair_eta,
        max_pair_eta=args.max_pair_eta,
        allowed_charge_patterns=None
        if args.allowed_charge_patterns is None
        else tuple(x.strip() for x in args.allowed_charge_patterns.split(",") if x.strip()),
    )

    combiner = ParticleCombiner()
    results = combiner.combine(
        tracks=tracks,
        primary_vertices=primary_vertices,
        n_body=args.n_body,
        mass_hypotheses=mass_hypotheses,
        preselection=preselection,
        cuts=cuts,
    )
    write_results_table(args.out, results)

    if args.custom_script:
        run_custom_script(
            script_path=args.custom_script,
            results=results,
            context={
                "tracks_path": args.tracks,
                "primary_vertices_path": args.primary_vertices,
                "masses_path": args.masses,
                "n_body": args.n_body,
                "preselection": preselection,
                "cuts": cuts,
                "output_path": args.out,
            },
        )
    return 0


def run_custom_script(
    script_path: str, results: list[CombinationResult], context: dict[str, Any]
) -> None:
    """Execute user-supplied post-processing callback `process(results, context)`."""
    module = _load_module(script_path)
    process = getattr(module, "process", None)
    if process is None or not callable(process):
        raise ValueError(
            f"Custom script {script_path} must define callable process(results, context)."
        )
    process(results, context)


def _load_module(script_path: str):
    """Import a Python module from an arbitrary file path."""
    path = Path(script_path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import custom script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
