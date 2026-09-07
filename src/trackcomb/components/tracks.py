"""tracks container: BestLong* branches -> track fields, one line each."""

from __future__ import annotations

import awkward as ak

from ..io import read, read_float64, unflatten_2d
from ..models import Container, COV5_LOWER_TRI
from ..physics import compute_default_track_quantities
from ..pid import set_tracks_pid
from ..configurable import configurable

HIT_TYPES = ("TVHits", "UPHits", "FTHits", "MPHits")

_HIT_LEAVES = {
    "TVHits": ("id", "x", "y", "z", "t"),
    "UPHits": ("id", "x", "y", "z"),
    "FTHits": ("id", "x", "z", "dxdy"),
    "MPHits": ("id", "x", "z", "y"),
}

_TRUE_HIT_LEAVES = ("x", "y", "z", "t")


@configurable
def load_tracks(
    chunk: Container,
    hits=("TVHits",),
    rich=True,
    mc=True,
    true_hits=False,
    compute_track_quantities=compute_default_track_quantities,
) -> Container:
    t: Container = {}

    # ---- track state (float64 for fit numerics) --------------------------
    t["x"] = read_float64(chunk, "BestLongState_FirstMeasurement/x")
    t["y"] = read_float64(chunk, "BestLongState_FirstMeasurement/y")
    t["z"] = read_float64(chunk, "BestLongState_FirstMeasurement/z")
    t["tx"] = read_float64(chunk, "BestLongState_FirstMeasurement/tx")
    t["ty"] = read_float64(chunk, "BestLongState_FirstMeasurement/ty")
    t["qop"] = read_float64(chunk, "BestLongState_FirstMeasurement/qop")
    t["chi2ndof"] = read(chunk, "BestLongState_FirstMeasurement/chi2ndof")
    t["ndof"] = read(chunk, "BestLongState_FirstMeasurement/ndof")
    for i, j in COV5_LOWER_TRI:
        t[f"cov_{i}_{j}"] = read_float64(
            chunk, f"BestLongState_FirstMeasurement/cov_{i}_{j}"
        )

    # ---- MC truth ---------------------------------------------------------
    if mc:
        t["mc_truth"] = read(chunk, "BestLongMC/truth") != 0
        t["mc_fromsignal"] = read(chunk, "BestLongMC/fromSignal") != 0
        t["mc_pid"] = read(chunk, "BestLongMC/pid")
        t["mc_key"] = read(chunk, "BestLongMC/key")
        t["mc_pv_key"] = read(chunk, "BestLongMC/pv_key")
        t["mc_charge"] = read(chunk, "BestLongMC/charge")
        t["mc_has_tv"] = read(chunk, "BestLongMC/hasTV") != 0
        t["mc_has_up"] = read(chunk, "BestLongMC/hasUP") != 0
        t["mc_has_mp"] = read(chunk, "BestLongMC/hasMP") != 0
        t["mc_has_ft"] = read(chunk, "BestLongMC/hasFT") != 0
        t["mc_has_t"] = read(chunk, "BestLongMC/hasT") != 0
        t["mc_px"] = read(chunk, "BestLongMC/px")
        t["mc_py"] = read(chunk, "BestLongMC/py")
        t["mc_pz"] = read(chunk, "BestLongMC/pz")
        t["mc_pe"] = read(chunk, "BestLongMC/pe")
        t["mc_ovtx_x"] = read(chunk, "BestLongMC/ovtx_x")
        t["mc_ovtx_y"] = read(chunk, "BestLongMC/ovtx_y")
        t["mc_ovtx_z"] = read(chunk, "BestLongMC/ovtx_z")
        t["mc_ovtx_time"] = read(chunk, "BestLongMC/ovtx_t")
        t["mc_pv_x"] = read(chunk, "BestLongMC/pv_x")
        t["mc_pv_y"] = read(chunk, "BestLongMC/pv_y")
        t["mc_pv_z"] = read(chunk, "BestLongMC/pv_z")
        t["mc_pv_time"] = read(chunk, "BestLongMC/pv_t")
        n_anc = read(chunk, "BestLongMC/n_ancestors")
        t["mc_n_ancestors"] = n_anc
        t["mc_ancestor_pids"] = unflatten_2d(
            read(chunk, "BestLongMC/ancestor_pids"), n_anc
        )
        t["mc_ancestor_keys"] = unflatten_2d(
            read(chunk, "BestLongMC/ancestor_keys"), n_anc
        )

    # ---- RICH PID ----------------------------------------------------------
    if rich:
        t["rich_dll_electron"] = read(chunk, "BestLongRich/DLL_Electron")
        t["rich_dll_muon"] = read(chunk, "BestLongRich/DLL_Muon")
        t["rich_dll_pion"] = read(chunk, "BestLongRich/DLL_Pion")
        t["rich_dll_kaon"] = read(chunk, "BestLongRich/DLL_Kaon")
        t["rich_dll_proton"] = read(chunk, "BestLongRich/DLL_Proton")
        t["rich_dll_deuteron"] = read(chunk, "BestLongRich/DLL_Deuteron")
        t["rich_dll_below_threshold"] = read(
            chunk, "BestLongRich/DLL_BelowThreshold"
        )
        t["rich_has_info"] = read(chunk, "BestLongRich/hasRICHInfo") != 0

    # ---- hits (doubly jagged: per track -> per hit) ------------------------
    for htype in hits:
        assert htype in HIT_TYPES, f"{htype} is an invalid hit type"
        prefix = htype.lower()  # tvhits, uphits, fthits, mphits
        n = read(chunk, f"BestLong{htype}/n")
        t[f"{prefix}_n"] = n
        for leaf in _HIT_LEAVES[htype]:
            t[f"{prefix}_{leaf}"] = unflatten_2d(
                read(chunk, f"BestLong{htype}/{leaf}"), n
            )
        if true_hits:
            for leaf in _TRUE_HIT_LEAVES:
                t[f"true_{prefix}_{leaf}"] = unflatten_2d(
                    read(chunk, f"BestLongTrue{htype}/{leaf}"), n
                )

    t["_type"] = "tracks"
    t["track_id"] = ak.local_index(t["x"], axis=1)
    compute_track_quantities(t)
    # The default track time and track-PV timing residual use the pion mass
    # hypothesis. Reconstruction channels may replace it afterwards.
    set_tracks_pid(t, "pi+")
    return t
