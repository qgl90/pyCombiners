#!/usr/bin/env python3
"""Produce a reusable Long-track PID dataframe and performance plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
from pid.performance import PIDPerformance

from trackcomb import (
    candidates_to_dataframe,
    counters,
    load_event_info,
    load_tracks,
    run_reconstruction,
)


def reconstruction(chunk):
    """Load only Long-track truth, kinematics, and RICH quantities."""
    tracks = load_tracks(chunk, hits=())
    event_info = load_event_info(chunk)
    df = candidates_to_dataframe(tracks)
    counts = ak.to_numpy(ak.num(tracks["x"]))
    df["run_number"] = np.repeat(event_info["run_number"], counts)
    df["event_number"] = np.repeat(event_info["event_number"], counts)
    counters("Long tracks for PID").add(len(df))
    return df


def main():
    parser = argparse.ArgumentParser(
        description="PID performance curve maker",
    )
    parser.add_argument(
        "--input", required=True, help="ROOT file path (wildcards allowed)"
    )
    parser.add_argument("--max-events", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        default="public/pid_performance/reconstruction",
    )
    parser.add_argument(
        "--out-tag",
        default="pid_performance",
    )
    parser.add_argument(
        "--out-file",
        default=None,
        help="override the output track Parquet path",
    )

    args = parser.parse_args()
    args.max_events = args.max_events or None
    out_dir = Path(args.out_dir)
    out_file = (
        Path(args.out_file)
        if args.out_file
        else out_dir / f"{args.out_tag}.parquet"
    )
    run_reconstruction(
        reconstruction,
        input_data=args.input,
        out=out_file,
        max_events=args.max_events,
        chunk_size=args.chunk_size,
        workers=args.workers,
        print_throughput=True,
    )
    print(f"\nSaved {out_file}")

    kpi_performance = PIDPerformance.from_parquet(
        out_file,
        out_dir=out_dir,
        out_tag=args.out_tag,
        signal_id="K+",
        background_id="pi+",
        dll_field="rich_dll_kaon",
    )
    kpi_performance.run_all(
        do_roc=True,
        do_kinematics=True,
        do_maps=True,
        roc_vars=("eta", "p", "pt"),
        kin_vars=("eta", "p", "pt"),
        targets=(
            0.01,
            0.03,
            0.05,
            0.10,
            0.15,
        ),  # 1% eff , 3% eff , 5 % , 10 % , 15 %
    )


if __name__ == "__main__":
    main()
