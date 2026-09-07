#!/usr/bin/env python3
"""Bs -> J/psi(-> mu+mu-) phi(-> K+K-) reconstruction."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from trackcomb import (
    apply_cuts,
    apply_mask,
    candidates_to_dataframe,
    combine,
    composite_pv_association,
    compute_bkgcat,
    count_reco_signal,
    count_true_decays,
    counters,
    rate_counters,
    cut_max,
    cut_max_ip_chi2,
    cut_min,
    cut_min_ip_chi2,
    cut_range,
    load_event_info,
    load_pvs,
    load_tracks,
    pdg_id,
    run_reconstruction,
    set_composite_pid,
    set_tracks_pid,
    tracks_pv_association,
)


@dataclass(frozen=True)
class SelectionConfig:
    """PV/timing and fit-quality handles for the HLT2-like selection."""

    max_dt_chi2: float | None = 16.0
    require_common_pv_on_time: bool = True
    min_track_ip_chi2: float = 9.0
    max_jpsi_vertex_chi2: float = 30.0
    max_jpsi_vertex_time_chi2: float | None = 16.0
    min_jpsi_fd_chi2: float = 30.0
    max_phi_vertex_chi2: float = 25.0
    max_phi_vertex_time_chi2: float | None = 16.0
    max_bs_vertex_chi2: float = 9.0
    max_bs_vertex_time_chi2: float | None = 16.0
    max_bs_ip_chi2: float = 25.0
    min_bs_dira: float = 0.9995


DEFAULT_SELECTION = SelectionConfig()


def _pv_function(config):
    return partial(composite_pv_association, max_dt_chi2=config.max_dt_chi2)


def _vertex_cuts(max_chi2, max_time_chi2):
    cuts = [cut_max("vertex_chi2", max_chi2)]
    if max_time_chi2 is not None:
        cuts.append(cut_max("vertex_time_chi2", max_time_chi2))
    return cuts


def _tracks_with_hypothesis(tracks, pvs, particle_id, config):
    """Copy tracks, set the mass-dependent time, then associate timed PVs."""
    selected = dict(tracks)
    set_tracks_pid(selected, particle_id)
    return tracks_pv_association(selected, pvs, max_dt_chi2=config.max_dt_chi2)


def _charge_split(tracks):
    return (
        apply_mask(tracks, tracks["charge"] > 0),
        apply_mask(tracks, tracks["charge"] < 0),
    )


def _combine_jpsi(mu_pos, mu_neg, pvs, apply_hlt2_cuts, config):
    # Every intermediate/final vertex below is a two-body fit with one spatial
    # degree of freedom, so vertex_chi2 is also vertex_chi2/ndof.
    return combine(
        [mu_pos, mu_neg],
        pvs,
        combination_cuts=(
            [
                cut_range("mass", 0.0, 5500.0),
                cut_min("pt", 0.0),
                cut_max("max_doca_chi2", 30.0),
            ]
            if apply_hlt2_cuts
            else None
        ),
        composite_cuts=(
            _vertex_cuts(
                config.max_jpsi_vertex_chi2,
                config.max_jpsi_vertex_time_chi2,
            )
            if apply_hlt2_cuts
            else None
        ),
        final_cuts=(
            [cut_min("fdchi2", config.min_jpsi_fd_chi2)]
            if apply_hlt2_cuts
            else None
        ),
        pv_function=_pv_function(config),
        require_common_pv_on_time=(
            apply_hlt2_cuts and config.require_common_pv_on_time
        ),
    )


def _combine_phi(k_pos, k_neg, pvs, apply_hlt2_cuts, config):
    return combine(
        [k_pos, k_neg],
        pvs,
        combination_cuts=(
            [
                cut_range("mass", 0.0, 1100.0),
                cut_min("pt", 400.0),
                cut_max("max_doca_chi2", 30.0),
            ]
            if apply_hlt2_cuts
            else None
        ),
        composite_cuts=(
            _vertex_cuts(
                config.max_phi_vertex_chi2,
                config.max_phi_vertex_time_chi2,
            )
            if apply_hlt2_cuts
            else None
        ),
        pv_function=_pv_function(config),
        require_common_pv_on_time=(
            apply_hlt2_cuts and config.require_common_pv_on_time
        ),
    )


def _combine_bs(jpsi, phi, pvs, apply_hlt2_cuts, config):
    return combine(
        [jpsi, phi],
        pvs,
        combination_cuts=(
            [cut_range("mass", 4000.0, 6500.0)] if apply_hlt2_cuts else None
        ),
        composite_cuts=(
            _vertex_cuts(
                config.max_bs_vertex_chi2,
                config.max_bs_vertex_time_chi2,
            )
            if apply_hlt2_cuts
            else None
        ),
        final_cuts=(
            [
                cut_max_ip_chi2(
                    config.max_bs_ip_chi2,
                    dt_chi2=config.max_dt_chi2,
                ),
                cut_min("dira", config.min_bs_dira),
            ]
            if apply_hlt2_cuts
            else None
        ),
        pv_function=_pv_function(config),
        require_common_pv_on_time=(
            apply_hlt2_cuts and config.require_common_pv_on_time
        ),
    )


def _load(chunk):
    return load_tracks(chunk), load_pvs(chunk), load_event_info(chunk)


def _select_full_tracks(tracks, pvs, config):
    """Apply the full muon and kaon track selections."""
    mu_tracks = _tracks_with_hypothesis(tracks, pvs, "mu+", config)
    mu_tracks = apply_cuts(
        mu_tracks,
        [
            cut_min("pt", 300.0),
            cut_min("p", 3000.0),
            cut_min_ip_chi2(
                config.min_track_ip_chi2,
                dt_chi2=config.max_dt_chi2,
            ),
            # PID cut intentionally disabled until the DLL inputs are usable:
            # cut_min("rich_dll_muon", 2.0),
        ],
    )
    k_tracks = _tracks_with_hypothesis(tracks, pvs, "K+", config)
    k_tracks = apply_cuts(
        k_tracks,
        [
            cut_min("pt", 250.0),
            cut_min("p", 2000.0),
            cut_min_ip_chi2(
                config.min_track_ip_chi2,
                dt_chi2=config.max_dt_chi2,
            ),
            # PID cut intentionally disabled until the DLL inputs are usable:
            # cut_min("rich_dll_kaon", 2.0),
        ],
    )
    return mu_tracks, k_tracks


def _has_track(tracks, particle_id):
    return ak.to_numpy(
        ak.any(
            tracks["mc_truth"]
            & tracks["mc_fromsignal"]
            & (tracks["mc_pid"] == pdg_id(particle_id)),
            axis=1,
        )
    )


def _has_signal_composite(candidates, particle_id, n_events):
    if candidates is None:
        return np.zeros(n_events, dtype=bool)
    return ak.to_numpy(
        ak.any(
            candidates["mc_truth"]
            & candidates["mc_fromsignal"]
            & (np.abs(candidates["mc_pid"]) == abs(pdg_id(particle_id))),
            axis=1,
        )
    )


def make_dataframe(candidates, event_info, mode, config):
    """Convert candidates to DataFrame with event info and truth."""
    df = candidates_to_dataframe(candidates)
    cand_per_evt = ak.to_numpy(ak.num(candidates["vertex_x"]))
    df["run_number"] = np.repeat(event_info["run_number"], cand_per_evt)
    df["event_number"] = np.repeat(event_info["event_number"], cand_per_evt)
    if "bkgcat" not in candidates:
        compute_bkgcat(candidates)
    df["bkgcat"] = ak.to_numpy(ak.flatten(candidates["bkgcat"]))
    df["is_signal"] = df["bkgcat"] <= 10
    # Preserve every candidate for yield/distribution plots.  Efficiency code
    # uses this flag (or the MC key) to count each generated decay only once.
    df["is_signal_clone"] = df["is_signal"] & df.duplicated(
        ["run_number", "event_number", "mc_key"], keep="first"
    )
    df["reconstruction_mode"] = mode
    df["selection_applied"] = mode in {"cheated_selected", "full"}
    for name, value in asdict(config).items():
        df[f"selection_{name}"] = np.nan if value is None else value
    return df


def _truth_guided_reconstruction(chunk, config, apply_hlt2_cuts):
    """Reconstruct signal candidates from truth-matched fromSignal tracks."""
    tracks, pvs, event_info = _load(chunk)

    anc_pids = tracks["mc_ancestor_pids"]
    has_bs = ak.any(np.abs(anc_pids) == abs(pdg_id("B(s)0")), axis=-1)
    signal_tracks = apply_mask(
        tracks,
        tracks["mc_truth"] & tracks["mc_fromsignal"] & has_bs,
    )
    n_true = int(
        np.sum(
            count_true_decays(
                signal_tracks,
                "B(s)0",
                ["mu+", "mu-", "K+", "K-"],
            )
        )
    )

    if apply_hlt2_cuts:
        mu_tracks, k_tracks = _select_full_tracks(signal_tracks, pvs, config)
    else:
        is_muon = np.abs(signal_tracks["mc_pid"]) == abs(pdg_id("mu+"))
        is_kaon = np.abs(signal_tracks["mc_pid"]) == abs(pdg_id("K+"))
        mu_tracks = _tracks_with_hypothesis(
            apply_mask(signal_tracks, is_muon), pvs, "mu+", config
        )
        k_tracks = _tracks_with_hypothesis(
            apply_mask(signal_tracks, is_kaon), pvs, "K+", config
        )
    mu_pos, mu_neg = _charge_split(mu_tracks)
    k_pos, k_neg = _charge_split(k_tracks)

    label = "cheated selected" if apply_hlt2_cuts else "cheated"
    mode = "cheated_selected" if apply_hlt2_cuts else "cheated"

    jpsi = _combine_jpsi(
        mu_pos,
        mu_neg,
        pvs,
        apply_hlt2_cuts=apply_hlt2_cuts,
        config=config,
    )
    if jpsi is None:
        rate_counters(f"{label} efficiency").add(0, n_true)
        return None
    set_composite_pid(jpsi, "J/psi(1S)")

    phi = _combine_phi(
        k_pos,
        k_neg,
        pvs,
        apply_hlt2_cuts=apply_hlt2_cuts,
        config=config,
    )
    if phi is None:
        rate_counters(f"{label} efficiency").add(0, n_true)
        return None
    set_composite_pid(phi, "phi(1020)")

    bs = _combine_bs(
        jpsi,
        phi,
        pvs,
        apply_hlt2_cuts=apply_hlt2_cuts,
        config=config,
    )
    if bs is None:
        rate_counters(f"{label} efficiency").add(0, n_true)
        return None
    set_composite_pid(bs, "B(s)0")

    compute_bkgcat(bs)
    bs = apply_mask(bs, bs["bkgcat"] <= 10)

    n_reco = count_reco_signal(bs, "B(s)0")
    rate_counters(f"{label} efficiency").add(n_reco, n_true)

    if n_reco == 0:
        return None

    df = make_dataframe(bs, event_info, mode, config)
    counters("candidates").add(len(df))
    return df


def cheated_reconstruction(chunk, config=DEFAULT_SELECTION):
    """Truth-guided signal reconstruction without HLT2 selection cuts."""
    return _truth_guided_reconstruction(
        chunk,
        config=config,
        apply_hlt2_cuts=False,
    )


def cheated_selected_reconstruction(chunk, config=DEFAULT_SELECTION):
    """Truth-guided signal reconstruction with all HLT2-like cuts."""
    return _truth_guided_reconstruction(
        chunk,
        config=config,
        apply_hlt2_cuts=True,
    )


def full_reconstruction(chunk, config=DEFAULT_SELECTION):
    """HLT2-like Bs -> J/psi(mu+mu-) phi(K+K-) reconstruction."""
    tracks, pvs, event_info = _load(chunk)

    mu_tracks, k_tracks = _select_full_tracks(tracks, pvs, config)

    mu_pos, mu_neg = _charge_split(mu_tracks)
    k_pos, k_neg = _charge_split(k_tracks)

    jpsi = _combine_jpsi(
        mu_pos, mu_neg, pvs, apply_hlt2_cuts=True, config=config
    )
    if jpsi is None:
        return None
    set_composite_pid(jpsi, "J/psi(1S)")

    phi = _combine_phi(k_pos, k_neg, pvs, apply_hlt2_cuts=True, config=config)
    if phi is None:
        return None
    set_composite_pid(phi, "phi(1020)")

    bs = _combine_bs(jpsi, phi, pvs, apply_hlt2_cuts=True, config=config)
    if bs is None:
        return None
    set_composite_pid(bs, "B(s)0")

    if int(ak.sum(ak.num(bs["vertex_x"]))) == 0:
        return None

    df = make_dataframe(bs, event_info, "full", config)
    counters("candidates").add(len(df))
    return df


def signal_cutflow_reconstruction(chunk, config=DEFAULT_SELECTION):
    """Return event-level cumulative efficiencies using signal tracks only."""
    tracks, pvs, event_info = _load(chunk)
    n_events = len(event_info["run_number"])

    has_bs = ak.any(
        np.abs(tracks["mc_ancestor_pids"]) == abs(pdg_id("B(s)0")), axis=-1
    )
    signal_tracks = apply_mask(
        tracks,
        tracks["mc_truth"] & tracks["mc_fromsignal"] & has_bs,
    )
    n_true_decays = count_true_decays(
        signal_tracks, "B(s)0", ["mu+", "mu-", "K+", "K-"]
    )
    has_four_signal_long_tracks = n_true_decays > 0

    true_muons = apply_mask(
        signal_tracks,
        np.abs(signal_tracks["mc_pid"]) == abs(pdg_id("mu+")),
    )
    true_kaons = apply_mask(
        signal_tracks,
        np.abs(signal_tracks["mc_pid"]) == abs(pdg_id("K+")),
    )
    cheated_muons = _tracks_with_hypothesis(true_muons, pvs, "mu+", config)
    cheated_kaons = _tracks_with_hypothesis(true_kaons, pvs, "K+", config)
    cheated_mu_pos, cheated_mu_neg = _charge_split(cheated_muons)
    cheated_k_pos, cheated_k_neg = _charge_split(cheated_kaons)
    cheated_jpsi = _combine_jpsi(
        cheated_mu_pos,
        cheated_mu_neg,
        pvs,
        apply_hlt2_cuts=False,
        config=config,
    )
    cheated_phi = _combine_phi(
        cheated_k_pos,
        cheated_k_neg,
        pvs,
        apply_hlt2_cuts=False,
        config=config,
    )
    cheated_bs = None
    if cheated_jpsi is not None and cheated_phi is not None:
        set_composite_pid(cheated_jpsi, "J/psi(1S)")
        set_composite_pid(cheated_phi, "phi(1020)")
        cheated_bs = _combine_bs(
            cheated_jpsi,
            cheated_phi,
            pvs,
            apply_hlt2_cuts=False,
            config=config,
        )
    has_cheated_signal_candidate = _has_signal_composite(
        cheated_bs, "B(s)0", n_events
    )

    mu_tracks, k_tracks = _select_full_tracks(signal_tracks, pvs, config)
    has_selected_muons = _has_track(mu_tracks, "mu+") & _has_track(
        mu_tracks, "mu-"
    )
    has_selected_kaons = _has_track(k_tracks, "K+") & _has_track(
        k_tracks, "K-"
    )
    has_four_selected_signal_tracks = has_selected_muons & has_selected_kaons

    mu_pos, mu_neg = _charge_split(mu_tracks)
    k_pos, k_neg = _charge_split(k_tracks)
    jpsi = _combine_jpsi(
        mu_pos, mu_neg, pvs, apply_hlt2_cuts=True, config=config
    )
    phi = _combine_phi(k_pos, k_neg, pvs, apply_hlt2_cuts=True, config=config)
    has_selected_jpsi = _has_signal_composite(jpsi, "J/psi(1S)", n_events)
    has_selected_phi = _has_signal_composite(phi, "phi(1020)", n_events)

    bs = None
    if jpsi is not None and phi is not None:
        set_composite_pid(jpsi, "J/psi(1S)")
        set_composite_pid(phi, "phi(1020)")
        bs = _combine_bs(jpsi, phi, pvs, apply_hlt2_cuts=True, config=config)
    has_selected_bs = _has_signal_composite(bs, "B(s)0", n_events)

    cheated_stage = has_four_signal_long_tracks & has_cheated_signal_candidate
    muon_stage = cheated_stage & has_selected_muons
    four_track_stage = cheated_stage & has_four_selected_signal_tracks
    jpsi_stage = four_track_stage & has_selected_jpsi
    phi_stage = jpsi_stage & has_selected_phi
    bs_stage = phi_stage & has_selected_bs

    result = pd.DataFrame(
        {
            "run_number": event_info["run_number"],
            "event_number": event_info["event_number"],
            "n_truth_signal_decays": n_true_decays,
            "has_four_signal_long_tracks": has_four_signal_long_tracks,
            "has_cheated_signal_candidate": cheated_stage,
            "passes_signal_muon_selection": muon_stage,
            "passes_four_signal_track_selection": four_track_stage,
            "passes_signal_jpsi_selection": jpsi_stage,
            "passes_signal_phi_selection": phi_stage,
            "passes_signal_bs_selection": bs_stage,
        }
    )
    for name, value in asdict(config).items():
        result[f"selection_{name}"] = np.nan if value is None else value
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Bs -> J/psi phi reconstruction"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["cheated", "cheated_selected", "full", "signal_cutflow"],
    )
    parser.add_argument(
        "--input", required=True, help="ROOT file path (wildcards allowed)"
    )
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-dt-chi2",
        type=float,
        default=DEFAULT_SELECTION.max_dt_chi2,
        help="maximum object-PV dt chi2 before minimum-IP association",
    )
    parser.add_argument(
        "--disable-pv-timing",
        action="store_true",
        help="associate and compute IP quantities against all PVs",
    )
    parser.add_argument(
        "--require-common-pv-on-time",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SELECTION.require_common_pv_on_time,
        help="require daughters to share an eligible PV before combination",
    )
    parser.add_argument(
        "--min-track-ip-chi2",
        type=float,
        default=DEFAULT_SELECTION.min_track_ip_chi2,
    )
    parser.add_argument(
        "--min-jpsi-fd-chi2",
        type=float,
        default=DEFAULT_SELECTION.min_jpsi_fd_chi2,
    )
    parser.add_argument(
        "--max-bs-ip-chi2",
        type=float,
        default=DEFAULT_SELECTION.max_bs_ip_chi2,
    )
    parser.add_argument(
        "--min-bs-dira",
        type=float,
        default=DEFAULT_SELECTION.min_bs_dira,
    )
    parser.add_argument(
        "--max-jpsi-vertex-chi2",
        type=float,
        default=DEFAULT_SELECTION.max_jpsi_vertex_chi2,
    )
    parser.add_argument(
        "--max-phi-vertex-chi2",
        type=float,
        default=DEFAULT_SELECTION.max_phi_vertex_chi2,
    )
    parser.add_argument(
        "--max-bs-vertex-chi2",
        type=float,
        default=DEFAULT_SELECTION.max_bs_vertex_chi2,
    )
    parser.add_argument(
        "--max-jpsi-vertex-time-chi2",
        type=float,
        default=DEFAULT_SELECTION.max_jpsi_vertex_time_chi2,
    )
    parser.add_argument(
        "--max-phi-vertex-time-chi2",
        type=float,
        default=DEFAULT_SELECTION.max_phi_vertex_time_chi2,
    )
    parser.add_argument(
        "--max-bs-vertex-time-chi2",
        type=float,
        default=DEFAULT_SELECTION.max_bs_vertex_time_chi2,
    )
    parser.add_argument(
        "--out-dir", default="public/bs_to_jpsiphi/reconstruction"
    )
    parser.add_argument("--out-file", default=None)
    args = parser.parse_args()
    args.max_events = args.max_events or None

    config = SelectionConfig(
        max_dt_chi2=None if args.disable_pv_timing else args.max_dt_chi2,
        require_common_pv_on_time=args.require_common_pv_on_time,
        min_track_ip_chi2=args.min_track_ip_chi2,
        max_jpsi_vertex_chi2=args.max_jpsi_vertex_chi2,
        max_jpsi_vertex_time_chi2=args.max_jpsi_vertex_time_chi2,
        min_jpsi_fd_chi2=args.min_jpsi_fd_chi2,
        max_phi_vertex_chi2=args.max_phi_vertex_chi2,
        max_phi_vertex_time_chi2=args.max_phi_vertex_time_chi2,
        max_bs_vertex_chi2=args.max_bs_vertex_chi2,
        max_bs_vertex_time_chi2=args.max_bs_vertex_time_chi2,
        max_bs_ip_chi2=args.max_bs_ip_chi2,
        min_bs_dira=args.min_bs_dira,
    )

    if args.mode == "cheated":
        reco_fn = partial(cheated_reconstruction, config=config)
    elif args.mode == "cheated_selected":
        reco_fn = partial(cheated_selected_reconstruction, config=config)
    elif args.mode == "signal_cutflow":
        reco_fn = partial(signal_cutflow_reconstruction, config=config)
    else:
        reco_fn = partial(full_reconstruction, config=config)

    out_path = (
        Path(args.out_file)
        if args.out_file
        else Path(args.out_dir) / f"{args.mode}.parquet"
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

    print(f"\nMode: {args.mode}")
    print(f"Selection: {config}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
