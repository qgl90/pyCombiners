#!/usr/bin/env python3
"""Bs -> mu+mu- reconstruction: run a single combiner mode, save to Parquet."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

from trackcomb import (
    DEFAULT_MAX_DT_CHI2,
    configurable,
    apply_mask,
    apply_cuts,
    all_in_tree,
    any_in_tree,
    candidates_to_dataframe,
    combine,
    composite_pv_association,
    compute_bkgcat,
    count_reco_signal,
    count_true_decays,
    counters,
    rate_counters,
    onnx_models,
    cut_max,
    cut_max_ip,
    cut_max_ip_chi2,
    cut_min,
    cut_min_ip,
    cut_min_ip_chi2,
    cut_range,
    get_daughter,
    load_event_info,
    load_pvs,
    load_tracks,
    pdg_id,
    run_reconstruction,
    set_composite_pid,
    set_tracks_pid,
    sum_in_tree,
    tracks_pv_association,
)


def make_dataframe(candidates, event_info):
    """Convert candidates to DataFrame with event info and truth."""
    df = candidates_to_dataframe(candidates)
    cand_per_evt = ak.to_numpy(ak.num(candidates["vertex_x"]))
    df["run_number"] = np.repeat(event_info["run_number"], cand_per_evt)
    df["event_number"] = np.repeat(event_info["event_number"], cand_per_evt)
    if "bkgcat" not in candidates:
        compute_bkgcat(candidates)
    df["bkgcat"] = ak.to_numpy(ak.flatten(candidates["bkgcat"]))
    df["is_signal"] = df["bkgcat"] <= 10
    return df


def _load(chunk):
    tracks = load_tracks(chunk)
    pvs = load_pvs(chunk)
    info = load_event_info(chunk)
    return tracks, pvs, info


def cheated_reconstruction(chunk):
    tracks, pvs, event_info = _load(chunk)

    n_true = int(np.sum(count_true_decays(tracks, "B(s)0", ["mu+", "mu-"])))

    is_muon = np.abs(tracks["mc_pid"]) == abs(pdg_id("mu+"))
    anc_pids = tracks["mc_ancestor_pids"]
    has_bs = ak.any(np.abs(anc_pids) == abs(pdg_id("B(s)0")), axis=-1)
    tracks = apply_mask(tracks, is_muon & has_bs)

    set_tracks_pid(tracks, "mu+")
    tracks = tracks_pv_association(
        tracks, pvs, max_dt_chi2=DEFAULT_MAX_DT_CHI2
    )
    pos_tracks = apply_mask(tracks, tracks["charge"] > 0)
    neg_tracks = apply_mask(tracks, tracks["charge"] < 0)

    candidates = combine(
        [pos_tracks, neg_tracks],
        pvs,
        pv_function=partial(
            composite_pv_association, max_dt_chi2=DEFAULT_MAX_DT_CHI2
        ),
        require_common_pv_on_time=True,
    )
    if candidates is None:
        rate_counters("cheated efficiency").add(0, n_true)
        return None
    set_composite_pid(candidates, "B(s)0")

    compute_bkgcat(candidates)
    candidates = apply_mask(candidates, candidates["bkgcat"] <= 10)

    n_reco = count_reco_signal(candidates, "B(s)0")
    rate_counters("cheated efficiency").add(n_reco, n_true)

    if n_reco == 0:
        return None

    df = make_dataframe(candidates, event_info)
    counters("candidates").add(len(df))
    return df


def _reconstruct(tracks, pvs, event_info, mode):
    set_tracks_pid(tracks, "mu+")
    use_timing = mode != "full_notime"
    max_dt_chi2 = DEFAULT_MAX_DT_CHI2 if use_timing else None
    tracks = tracks_pv_association(tracks, pvs, max_dt_chi2=max_dt_chi2)
    composite_pv = partial(composite_pv_association, max_dt_chi2=max_dt_chi2)
    pos_tracks = apply_mask(tracks, tracks["charge"] > 0)
    neg_tracks = apply_mask(tracks, tracks["charge"] < 0)

    if mode == "full":
        cuts: dict[str, Any] = {
            "track_cuts": [
                cut_min("pt", 1000),
                cut_min("rich_dll_muon", -5.0),
            ],
            "combination_cuts": [
                cut_max("max_doca", 0.05),
                cut_range("mass", 4700, 6000),
            ],
            "composite_cuts": [
                cut_max("vertex_chi2", 4.0),
                cut_max("pair_time_chi2", 4.0),
                cut_min("pt", 1000),
                any_in_tree(cut_min("pt", 2000)),
            ],
            "final_cuts": [
                cut_min("dira", 0.9995),
                cut_max_ip(0.1, dt_chi2=DEFAULT_MAX_DT_CHI2),
                cut_max_ip_chi2(16, dt_chi2=DEFAULT_MAX_DT_CHI2),
            ],
        }
    elif mode == "full_notime":
        cuts = {
            "track_cuts": [
                cut_min("pt", 1000),
                cut_min("rich_dll_muon", -5.0),
            ],
            "combination_cuts": [
                cut_max("max_doca", 0.05),
                cut_range("mass", 4700, 6000),
            ],
            "composite_cuts": [
                cut_max("vertex_chi2", 4.0),
                cut_min("pt", 1000),
                any_in_tree(cut_min("pt", 2000)),
            ],
            "final_cuts": [
                cut_min("dira", 0.9995),
                cut_max_ip(0.1),
                cut_max_ip_chi2(16),
            ],
        }
    elif mode == "dist":
        cuts = {
            "track_cuts": [cut_min("pt", 800)],
            "combination_cuts": [
                cut_max("max_doca", 0.2),
                cut_range("mass", 4500, 6000),
            ],
            "composite_cuts": [
                cut_max("vertex_chi2", 10.0),
                cut_max("pair_time_chi2", 10.0),
                cut_min("pt", 1000),
                any_in_tree(cut_min("pt", 2000)),
            ],
            "final_cuts": [
                cut_min("dira", 0.9995),
                cut_max_ip(0.1, dt_chi2=DEFAULT_MAX_DT_CHI2),
                cut_max_ip_chi2(16, dt_chi2=DEFAULT_MAX_DT_CHI2),
            ],
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    candidates = combine(
        [pos_tracks, neg_tracks],
        pvs,
        pv_function=composite_pv,
        require_common_pv_on_time=True,
        **cuts,
    )
    if candidates is None:
        return None
    set_composite_pid(candidates, "B(s)0")

    if int(ak.sum(ak.num(candidates["vertex_x"]))) == 0:
        return None

    df = make_dataframe(candidates, event_info)
    counters("candidates").add(len(df))
    return df


@configurable
def reconstruction(chunk, mode="full"):
    tracks, pvs, event_info = _load(chunk)
    return _reconstruct(tracks, pvs, event_info, mode)


def pvtag_reconstruction(chunk, mode, model_path, mva_cut):
    tracks, pvs, event_info = _load(chunk)
    filtered = pvtag_filter(
        tracks, pvs, len(event_info["run_number"]), model_path, mva_cut
    )
    if filtered is None:
        return None
    tracks, pvs = filtered
    return _reconstruct(tracks, pvs, event_info, mode)


def with_pvtag(mode, model_path, mva_cut):
    """Reconstruction with PV tagging prefilter (picklable for workers)."""
    return partial(
        pvtag_reconstruction, mode=mode, model_path=model_path, mva_cut=mva_cut
    )


def pvtag_filter(tracks, pvs, n_events, model_path, mva_cut):
    """Filter tracks/PVs by TwoTrackMVA PV tagging. Returns pair or None."""
    # PV association for all tracks
    tracks = tracks_pv_association(
        tracks, pvs, max_dt_chi2=DEFAULT_MAX_DT_CHI2
    )

    # MVA track preselection
    mva_tracks = apply_cuts(
        tracks,
        [
            cut_min("pt", 200),
            cut_min_ip(0.06, dt_chi2=DEFAULT_MAX_DT_CHI2),
            cut_max("chi2ndof", 10),
        ],
    )
    set_tracks_pid(mva_tracks, "pi+")

    mva_cands = combine(
        [mva_tracks, mva_tracks],
        pvs,
        combination_cuts=[
            cut_max("max_doca", 1.0),
            cut_min(sum_in_tree("pt"), 400),
            cut_min("pt", 1000),
        ],
        composite_cuts=[
            cut_max("vertex_chi2", 20.0),
            cut_max("max_doca", 0.2),
            cut_min("vertex_z", -330),
        ],
        final_cuts=[
            cut_range("flight_eta", 2, 5),
            cut_min("mcor", 1000),
            all_in_tree(cut_min_ip_chi2(4, dt_chi2=DEFAULT_MAX_DT_CHI2)),
            all_in_tree(cut_min("pt", 200)),
            cut_max_ip_chi2(16, dt_chi2=DEFAULT_MAX_DT_CHI2),
        ],
        pv_function=partial(
            composite_pv_association, max_dt_chi2=DEFAULT_MAX_DT_CHI2
        ),
        require_common_pv_on_time=True,
    )

    if mva_cands is None or int(ak.sum(ak.num(mva_cands["vertex_x"]))) == 0:
        return None

    # MVA feature extraction — directly from candidates, no DataFrame
    cand_per_evt = ak.to_numpy(ak.num(mva_cands["vertex_x"]))
    fdchi2 = ak.to_numpy(ak.flatten(mva_cands["fdchi2"]))
    vertex_chi2 = ak.to_numpy(ak.flatten(mva_cands["vertex_chi2"]))
    d0_pt = ak.to_numpy(ak.flatten(get_daughter(mva_cands, 0, "pt")))
    d1_pt = ak.to_numpy(ak.flatten(get_daughter(mva_cands, 1, "pt")))
    min_ipchi2 = np.minimum(
        ak.to_numpy(ak.flatten(get_daughter(mva_cands, 0, "min_ip_chi2"))),
        ak.to_numpy(ak.flatten(get_daughter(mva_cands, 1, "min_ip_chi2"))),
    )

    features = np.column_stack(
        [
            np.log(np.maximum(fdchi2, 1e-10)),
            (d0_pt + d1_pt) / 1000,  # MeV -> GeV (model trained in GeV)
            np.maximum(vertex_chi2, 1e-10),
            np.log(np.maximum(min_ipchi2, 1e-10)),
        ]
    ).astype(np.float32)

    mva_response = onnx_models(model_path).run(features)
    mva_pass = mva_response > mva_cut
    if not mva_pass.any():
        return None

    # Tagged (event_idx, pv_index) pairs — vectorized with key encoding
    event_idx = np.repeat(np.arange(n_events, dtype=np.int64), cand_per_evt)
    best_pv = ak.to_numpy(ak.flatten(mva_cands["best_pv_index"])).astype(
        np.int64
    )

    flat_pv_idx = ak.to_numpy(ak.flatten(pvs["pv_index"])).astype(np.int64)
    stride = int(flat_pv_idx.max()) + 1 if len(flat_pv_idx) > 0 else 1
    tagged_keys = np.unique(event_idx[mva_pass] * stride + best_pv[mva_pass])

    # Filter tracks and PVs by tagged pairs
    flat_bpi = ak.to_numpy(ak.flatten(tracks["best_pv_index"])).astype(
        np.int64
    )
    tracks_per_event = ak.to_numpy(ak.num(tracks["best_pv_index"]))
    track_ei = np.repeat(np.arange(n_events, dtype=np.int64), tracks_per_event)
    track_mask = ak.unflatten(
        np.isin(track_ei * stride + flat_bpi, tagged_keys), tracks_per_event
    )

    pvs_per_event = ak.to_numpy(ak.num(pvs["pv_index"]))
    pv_ei = np.repeat(np.arange(n_events, dtype=np.int64), pvs_per_event)
    pv_mask = ak.unflatten(
        np.isin(pv_ei * stride + flat_pv_idx, tagged_keys), pvs_per_event
    )

    return apply_mask(tracks, track_mask), apply_mask(pvs, pv_mask)


def main():
    parser = argparse.ArgumentParser(description="Bs -> mu+mu- reconstruction")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["cheated", "full", "full_notime", "dist"],
    )
    parser.add_argument(
        "--input", required=True, help="ROOT file path (wildcards allowed)"
    )
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out-dir", default="public/bs_to_mumu/reconstruction"
    )
    parser.add_argument(
        "--out-file", default=None, help="Override output file path"
    )
    parser.add_argument(
        "--pvtag", action="store_true", help="Enable PV tagging prefilter"
    )
    parser.add_argument(
        "--model", default=None, help="ONNX model path (--pvtag)"
    )
    parser.add_argument(
        "--mva-cut", type=float, default=0.9569, help="MVA cut (--pvtag)"
    )
    args = parser.parse_args()
    args.max_events = args.max_events or None

    if args.pvtag and not args.model:
        parser.error("--model is required when using --pvtag")

    if args.mode == "cheated":
        reco_fn = cheated_reconstruction
    else:
        reconstruction.global_bind(mode=args.mode)
        reco_fn = reconstruction

    if args.pvtag:
        reco_fn = with_pvtag(args.mode, args.model, args.mva_cut)

    label = f"{args.mode}_pvtag" if args.pvtag else args.mode
    out_path = (
        Path(args.out_file)
        if args.out_file
        else Path(args.out_dir) / f"{label}.parquet"
    )

    run_reconstruction(
        reco_fn,
        input_data=args.input,
        out=out_path,
        max_events=args.max_events,
        chunk_size=args.chunk_size,
        workers=args.workers,
        print_throughput=True,
    )

    print(f"\nMode: {label}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
