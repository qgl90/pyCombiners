#!/usr/bin/env python3
"""Build a reusable candidate/event Parquet for TwoTrackMVA PV studies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from trackcomb import (
    all_in_tree,
    apply_cuts,
    apply_mask,
    combine,
    composite_pv_association,
    counters,
    cut_max,
    cut_max_ip_chi2,
    cut_min,
    cut_min_ip,
    cut_min_ip_chi2,
    cut_range,
    get_daughter,
    load_event_info,
    load_pvs,
    load_tracks,
    onnx_models,
    run_reconstruction,
    set_tracks_pid,
    sum_in_tree,
    tracks_pv_association,
)


@dataclass(frozen=True)
class StudyConfig:
    max_dt_chi2: float | None = 16.0
    require_common_pv_on_time: bool = True
    max_vertex_chi2: float = 20.0
    max_vertex_time_chi2: float | None = 16.0
    model_path: str = "models/two_track_mva.onnx"
    label: str = "sample"
    track_samples: tuple[str, ...] = ("all_tracks", "truth_matched")


DEFAULT_CONFIG = StudyConfig()


def _signal_pv_indices(tracks, pvs):
    """Reconstructed PV indices matching any truth-matched signal track PV."""
    signal = tracks["mc_truth"] & tracks["mc_fromsignal"]
    result = []
    for track_keys, pv_keys in zip(
        ak.to_list(tracks["mc_pv_key"][signal]), ak.to_list(pvs["mc_key"])
    ):
        wanted = {int(key) for key in track_keys if int(key) >= 0}
        result.append(
            [
                index
                for index, key in enumerate(pv_keys)
                if int(key) >= 0 and int(key) in wanted
            ]
        )
    return result


def _event_rows(
    event_info,
    n_tracks_input,
    n_tracks_preselected,
    n_pvs,
    signal_pvs,
    track_sample,
    config,
):
    rows = pd.DataFrame(
        {
            "run_number": event_info["run_number"],
            "event_number": event_info["event_number"],
            "track_sample": track_sample,
            "is_candidate": False,
            "n_tracks_input_event": n_tracks_input,
            "n_tracks_preselected_event": n_tracks_preselected,
            "n_pvs_event": n_pvs,
            "signal_pv_indices_json": [json.dumps(v) for v in signal_pvs],
            "n_signal_pvs_event": [len(v) for v in signal_pvs],
        }
    )
    return _add_common_columns(rows, config)


def _add_common_columns(frame, config):
    frame["sample_label"] = config.label
    for name, value in asdict(config).items():
        if name == "label":
            continue
        if isinstance(value, (list, tuple)):
            value = json.dumps(value)
        frame[f"study_{name}"] = np.nan if value is None else value
    return frame


def _preselect_tracks(tracks, pvs, config):
    tracks = dict(tracks)
    set_tracks_pid(tracks, "pi+")
    tracks_pv_association(tracks, pvs, max_dt_chi2=config.max_dt_chi2)
    return apply_cuts(
        tracks,
        [
            cut_min("pt", 200.0),
            cut_min_ip(0.06, dt_chi2=config.max_dt_chi2),
            cut_max("chi2ndof", 10.0),
        ],
    )


def _make_candidates(tracks, pvs, config):
    composite_cuts = [
        cut_max("vertex_chi2", config.max_vertex_chi2),
        cut_max("max_doca", 0.2),
        cut_min("vertex_z", -330.0),
    ]
    if config.max_vertex_time_chi2 is not None:
        composite_cuts.append(
            cut_max("vertex_time_chi2", config.max_vertex_time_chi2)
        )

    return combine(
        [tracks, tracks],
        pvs,
        combination_cuts=[
            cut_max("max_doca", 1.0),
            cut_min(sum_in_tree("pt"), 400.0),
            cut_min("pt", 1000.0),
        ],
        composite_cuts=composite_cuts,
        final_cuts=[
            cut_range("flight_eta", 2.0, 5.0),
            cut_min("mcor", 1000.0),
            all_in_tree(cut_min_ip_chi2(4.0, dt_chi2=config.max_dt_chi2)),
            all_in_tree(cut_min("pt", 200.0)),
            cut_max_ip_chi2(16.0, dt_chi2=config.max_dt_chi2),
        ],
        pv_function=partial(
            composite_pv_association, max_dt_chi2=config.max_dt_chi2
        ),
        compute_pv_compatibility=True,
        require_common_pv_on_time=config.require_common_pv_on_time,
        require_same_best_pv=False,
    )


def _candidate_rows(
    candidates,
    event_info,
    n_tracks_input,
    n_tracks_preselected,
    n_pvs,
    signal_pvs,
    track_sample,
    config,
):
    counts = ak.to_numpy(ak.num(candidates["vertex_x"]))
    event_index = np.repeat(np.arange(len(counts)), counts)
    if len(event_index) == 0:
        return None

    def flat(field):
        return ak.to_numpy(ak.flatten(candidates[field]))

    def daughter(index, field):
        return ak.to_numpy(ak.flatten(get_daughter(candidates, index, field)))

    fdchi2 = flat("fdchi2")
    vertex_chi2 = flat("vertex_chi2")
    d0_pt, d1_pt = daughter(0, "pt"), daughter(1, "pt")
    d0_ipchi2 = daughter(0, "min_ip_chi2")
    d1_ipchi2 = daughter(1, "min_ip_chi2")
    features = np.column_stack(
        [
            np.log(np.maximum(fdchi2, 1e-10)),
            (d0_pt + d1_pt) / 1000.0,
            np.maximum(vertex_chi2, 1e-10),
            np.log(np.maximum(np.minimum(d0_ipchi2, d1_ipchi2), 1e-10)),
        ]
    ).astype(np.float32)
    response = onnx_models(config.model_path).run(features)

    best_pv = flat("best_pv_index").astype(np.int64)
    signal_pvs_for_candidate = [signal_pvs[index] for index in event_index]
    best_is_signal = np.array(
        [
            int(index) in event_signal_pvs
            for index, event_signal_pvs in zip(
                best_pv, signal_pvs_for_candidate
            )
        ]
    )
    d0_charge, d1_charge = daughter(0, "charge"), daughter(1, "charge")
    d0_truth = daughter(0, "mc_truth").astype(bool)
    d1_truth = daughter(1, "mc_truth").astype(bool)

    rows = pd.DataFrame(
        {
            "run_number": event_info["run_number"][event_index],
            "event_number": event_info["event_number"][event_index],
            "track_sample": track_sample,
            "is_candidate": True,
            "n_tracks_input_event": n_tracks_input[event_index],
            "n_tracks_preselected_event": n_tracks_preselected[event_index],
            "n_pvs_event": n_pvs[event_index],
            "signal_pv_indices_json": [
                json.dumps(v) for v in signal_pvs_for_candidate
            ],
            "n_signal_pvs_event": [len(v) for v in signal_pvs_for_candidate],
            "candidate_index": np.concatenate(
                [np.arange(count) for count in counts]
            ),
            "mva_response": response,
            "best_pv_index": best_pv,
            "n_pvs_considered": flat("n_pvs_considered"),
            "candidate_pv_on_time_indices_json": [
                json.dumps(v)
                for v in ak.to_list(
                    ak.flatten(candidates["pv_on_time"], axis=1)
                )
            ],
            "daughters_share_pv_on_time": flat(
                "daughters_have_common_pv_on_time"
            ).astype(bool),
            "daughters_same_best_pv": flat(
                "daughters_have_same_best_pv"
            ).astype(bool),
            "common_pv_indices_json": [
                json.dumps(v)
                for v in ak.to_list(
                    ak.flatten(
                        candidates["daughter_common_pv_on_time"], axis=1
                    )
                )
            ],
            "same_sign": d0_charge * d1_charge > 0,
            "contains_ghost": ~(d0_truth & d1_truth),
            "candidate_from_signal": flat("mc_fromsignal").astype(bool),
            "best_pv_is_signal": best_is_signal,
            "daughter0_track_id": daughter(0, "track_id"),
            "daughter1_track_id": daughter(1, "track_id"),
            "mass": flat("mass"),
            "pt": flat("pt"),
            "vertex_z": flat("vertex_z"),
            "vertex_chi2": vertex_chi2,
            "vertex_time_chi2": flat("vertex_time_chi2"),
            "max_doca": flat("max_doca"),
            "max_doca_chi2": flat("max_doca_chi2"),
            "fdchi2": fdchi2,
            "mcor": flat("mcor"),
            "flight_eta": flat("flight_eta"),
            "composite_ip_chi2": flat("composite_ip_chi2"),
        }
    )
    counters(f"{track_sample} candidates before MVA").add(len(rows))
    return _add_common_columns(rows, config)


def reconstruction(chunk, config=DEFAULT_CONFIG):
    """Return explicit event rows plus all preselected MVA candidate rows."""
    tracks = load_tracks(chunk)
    pvs = load_pvs(chunk)
    event_info = load_event_info(chunk)

    n_pvs = ak.to_numpy(ak.num(pvs["x"]))
    signal_pvs = _signal_pv_indices(tracks, pvs)
    frames = []

    for track_sample in config.track_samples:
        sample_tracks = tracks
        if track_sample == "truth_matched":
            sample_tracks = apply_mask(tracks, tracks["mc_truth"])
        elif track_sample != "all_tracks":
            raise ValueError(f"unknown track sample: {track_sample}")

        n_tracks_input = ak.to_numpy(ak.num(sample_tracks["x"]))
        selected = _preselect_tracks(sample_tracks, pvs, config)
        n_selected = ak.to_numpy(ak.num(selected["x"]))
        frames.append(
            _event_rows(
                event_info,
                n_tracks_input,
                n_selected,
                n_pvs,
                signal_pvs,
                track_sample,
                config,
            )
        )

        candidates = _make_candidates(selected, pvs, config)
        if candidates is not None:
            candidate_rows = _candidate_rows(
                candidates,
                event_info,
                n_tracks_input,
                n_selected,
                n_pvs,
                signal_pvs,
                track_sample,
                config,
            )
            if candidate_rows is not None:
                frames.append(candidate_rows)

    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="ROOT path; wildcards allowed"
    )
    parser.add_argument(
        "--out", required=True, help="output candidate/event Parquet"
    )
    parser.add_argument("--model", default=DEFAULT_CONFIG.model_path)
    parser.add_argument("--label", default=DEFAULT_CONFIG.label)
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-dt-chi2", type=float, default=16.0)
    parser.add_argument("--disable-pv-timing", action="store_true")
    parser.add_argument(
        "--require-common-pv-on-time",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-vertex-chi2", type=float, default=20.0)
    parser.add_argument("--max-vertex-time-chi2", type=float, default=16.0)
    parser.add_argument("--disable-vertex-time-cut", action="store_true")
    parser.add_argument(
        "--track-samples",
        nargs="+",
        choices=("all_tracks", "truth_matched"),
        default=list(DEFAULT_CONFIG.track_samples),
    )
    args = parser.parse_args()

    config = StudyConfig(
        max_dt_chi2=None if args.disable_pv_timing else args.max_dt_chi2,
        require_common_pv_on_time=args.require_common_pv_on_time,
        max_vertex_chi2=args.max_vertex_chi2,
        max_vertex_time_chi2=(
            None if args.disable_vertex_time_cut else args.max_vertex_time_chi2
        ),
        model_path=args.model,
        label=args.label,
        track_samples=tuple(args.track_samples),
    )
    print("TwoTrackMVA study setup:")
    for name, value in asdict(config).items():
        print(f"  {name}: {value}")
    print(
        f"  max_events: {args.max_events or 'all'}\n"
        f"  chunk_size: {args.chunk_size}\n"
        f"  workers: {args.workers}\n"
        f"  output: {args.out}",
        flush=True,
    )
    run_reconstruction(
        partial(reconstruction, config=config),
        input_data=args.input,
        out=Path(args.out),
        max_events=args.max_events or None,
        chunk_size=args.chunk_size,
        workers=args.workers,
        print_throughput=True,
    )
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
