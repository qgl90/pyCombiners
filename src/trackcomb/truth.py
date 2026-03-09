"""MC truth matching and background categorisation."""

from __future__ import annotations

__author__ = [
    "Renato Quagliani <rquaglia@cern.ch>",
    "Jiahui Zhuo <jiahui.zhuo@cern.ch>",
]

from collections import Counter, defaultdict

import awkward as ak
import numpy as np

from .models import n_daughters


def count_true_decays(tracks, mother_pdg, daughter_pdgs=None):
    """Count true MC decays per event by grouping tracks with shared ancestor."""
    from .pid import pdg_id

    if isinstance(mother_pdg, str):
        mother_pdg = pdg_id(mother_pdg)
    mother_pdg = abs(mother_pdg)

    if daughter_pdgs is not None:
        expected = Counter(
            pdg_id(p) if isinstance(p, str) else p for p in daughter_pdgs
        )
    else:
        expected = None

    n_events = len(tracks["mc_pid"])
    counts = np.zeros(n_events, dtype=int)

    # Materialize to Python lists to avoid per-element awkward overhead
    mc_pid_list = tracks["mc_pid"].tolist()
    anc_pids_list = tracks["mc_ancestor_pids"].tolist()
    anc_keys_list = tracks["mc_ancestor_keys"].tolist()

    for evt in range(n_events):
        evt_mc_pids = mc_pid_list[evt]
        evt_anc_pids = anc_pids_list[evt]
        evt_anc_keys = anc_keys_list[evt]
        n_tracks = len(evt_mc_pids)

        # Group track indices by ancestor key where |ancestor_pid| == mother_pdg
        groups: dict[int, list[int]] = defaultdict(list)
        for ti in range(n_tracks):
            for pid, key in zip(evt_anc_pids[ti], evt_anc_keys[ti]):
                if abs(pid) == abs(mother_pdg):
                    groups[key].append(ti)

        if expected is None:
            counts[evt] = len(groups)
        else:
            for key, track_indices in groups.items():
                actual = Counter(int(evt_mc_pids[ti]) for ti in track_indices)
                if all(actual[k] >= v for k, v in expected.items()):
                    counts[evt] += 1

    return counts


def count_reco_signal(candidates, mother_pdg):
    """Count unique reconstructed signal decays per event, then sum."""
    from .pid import pdg_id

    if isinstance(mother_pdg, str):
        mother_pdg = pdg_id(mother_pdg)
    mother_keys = candidates["mc_key"][
        np.abs(candidates["mc_pid"]) == abs(mother_pdg)
    ]
    return int(ak.sum(ak.num(ak.run_lengths(ak.sort(mother_keys)))))


def compute_bkgcat(candidates):
    """Compute LHCb-style background categories into candidates["bkgcat"].

    Codes: 0=Signal, 10=QuasiSignal, 20=PhysBkg, 30=Reflection,
    60=Ghost, 63=Clone, 66=Hierarchy, 100=Pileup,
    110=FromB, 120=FromC, 130=LightParticle (40/50 omitted).
    """
    n_body = n_daughters(candidates)
    if n_body == 0:
        raise ValueError(
            "No daughter{k}_global_index fields found in candidates"
        )

    mother_pdg = _first_nonempty_int(candidates, "pid")

    pools = candidates["_daughter_pools"]
    # Get daughter PIDs from pools
    daughter_pdgs = []
    for k in range(n_body):
        pool = pools[k]
        if "pid" in pool:
            pid_val = pool["pid"]
            if isinstance(pid_val, (int, float, np.integer)):
                daughter_pdgs.append(int(pid_val))
            else:
                gi_arr = candidates[f"daughter{k}_global_index"]
                flat_pid = ak.flatten(pid_val)
                try:
                    first_gi = int(ak.flatten(gi_arr)[0])
                    daughter_pdgs.append(int(np.asarray(flat_pid)[first_gi]))
                except (ValueError, IndexError):
                    daughter_pdgs.append(None)
        else:
            daughter_pdgs.append(None)

    cand_counts = ak.to_numpy(ak.num(candidates["vertex_x"]))
    N_total = int(cand_counts.sum())

    if N_total == 0:
        return ak.unflatten(np.array([], dtype=int), cand_counts)

    # --- Gather per-daughter MC fields via global_index ---
    d_mc_truth, d_mc_pid, d_mc_key = [], [], []
    d_mc_pv_key, d_mc_fromsignal = [], []
    d_anc_pids, d_anc_keys = [], []

    for k in range(n_body):
        pool = pools[k]
        gi = ak.to_numpy(ak.flatten(candidates[f"daughter{k}_global_index"]))

        d_mc_truth.append(np.asarray(ak.flatten(pool["mc_truth"]))[gi])
        d_mc_pid.append(np.asarray(ak.flatten(pool["mc_pid"]))[gi])
        d_mc_key.append(np.asarray(ak.flatten(pool["mc_key"]))[gi])
        d_mc_pv_key.append(np.asarray(ak.flatten(pool["mc_pv_key"]))[gi])
        d_mc_fromsignal.append(
            np.asarray(ak.flatten(pool["mc_fromsignal"]))[gi]
        )
        d_anc_pids.append(ak.flatten(pool["mc_ancestor_pids"], axis=1)[gi])
        d_anc_keys.append(ak.flatten(pool["mc_ancestor_keys"], axis=1)[gi])

    # --- Vectorized category assignment (priority order) ---
    cats = np.full(N_total, 130, dtype=int)
    assigned = np.zeros(N_total, dtype=bool)

    def assign(mask, code):
        nonlocal assigned
        m = mask & ~assigned
        cats[m] = code
        assigned |= m

    # 1. Ghost (60): any daughter has mc_truth == 0
    is_ghost = np.zeros(N_total, dtype=bool)
    for k in range(n_body):
        is_ghost |= d_mc_truth[k] == 0
    assign(is_ghost, 60)

    # 2. Clone (63): two daughters share same mc_key
    is_clone = np.zeros(N_total, dtype=bool)
    for i in range(n_body):
        for j in range(i + 1, n_body):
            is_clone |= d_mc_key[i] == d_mc_key[j]
    assign(is_clone, 63)

    # 3. Hierarchy (66): one daughter's mc_key in another's ancestors
    is_hierarchy = np.zeros(N_total, dtype=bool)
    for i in range(n_body):
        for j in range(n_body):
            if i == j:
                continue
            match = d_anc_keys[j] == d_mc_key[i]
            is_hierarchy |= ak.to_numpy(ak.any(match, axis=-1))
    assign(is_hierarchy, 66)

    # 4. Common ancestor: intersect ancestor key sets
    d0_keys = d_anc_keys[0]
    d0_pids = d_anc_pids[0]
    in_common = ak.ones_like(d0_keys, dtype=bool)
    for k in range(1, n_body):
        match = d0_keys[:, :, np.newaxis] == d_anc_keys[k][:, np.newaxis, :]
        in_common = in_common & ak.any(match, axis=-1)
    has_common = ak.to_numpy(ak.any(in_common, axis=-1))

    # 4a. Reflection (30): common ancestor but wrong daughter PIDs
    correct_pids = np.ones(N_total, dtype=bool)
    for k in range(n_body):
        if daughter_pdgs[k] is not None:
            correct_pids &= np.abs(d_mc_pid[k]) == abs(daughter_pdgs[k])
    assign(has_common & ~correct_pids, 30)

    # 4b. PhysBkg (20): correct daughters but no common ancestor matches mother
    if mother_pdg is not None:
        mother_match = (np.abs(d0_pids) == abs(mother_pdg)) & in_common
        correct_mother = ak.to_numpy(ak.any(mother_match, axis=-1))
    else:
        correct_mother = np.zeros(N_total, dtype=bool)
    assign(has_common & correct_pids & ~correct_mother, 20)

    # 4c. Signal (0) / QuasiSignal (10)
    all_fromsignal = np.ones(N_total, dtype=bool)
    for k in range(n_body):
        all_fromsignal &= d_mc_fromsignal[k] == 1
    assign(has_common & correct_pids & correct_mother & all_fromsignal, 0)
    assign(has_common & correct_pids & correct_mother & ~all_fromsignal, 10)

    # 5. No common ancestor
    # 5a. Pileup (100): daughters from different PVs
    different_pv = np.zeros(N_total, dtype=bool)
    for i in range(n_body):
        for j in range(i + 1, n_body):
            both_valid = (d_mc_pv_key[i] != -1) & (d_mc_pv_key[j] != -1)
            different_pv |= both_valid & (d_mc_pv_key[i] != d_mc_pv_key[j])
    assign(~has_common & different_pv, 100)

    # 5b. FromB (110): any daughter has b-hadron ancestor
    is_from_b = np.zeros(N_total, dtype=bool)
    for k in range(n_body):
        abs_p = np.abs(d_anc_pids[k])
        has_b = (
            ((abs_p // 100) % 10 == 5)
            | ((abs_p // 1000) % 10 == 5)
            | ((abs_p // 10000) % 10 == 5)
        )
        is_from_b |= ak.to_numpy(ak.any(has_b, axis=-1))
    assign(~has_common & is_from_b, 110)

    # 5c. FromC (120): any daughter has c-hadron ancestor
    is_from_c = np.zeros(N_total, dtype=bool)
    for k in range(n_body):
        abs_p = np.abs(d_anc_pids[k])
        has_c = (
            ((abs_p // 100) % 10 == 4)
            | ((abs_p // 1000) % 10 == 4)
            | ((abs_p // 10000) % 10 == 4)
        )
        is_from_c |= ak.to_numpy(ak.any(has_c, axis=-1))
    assign(~has_common & is_from_c, 120)

    # Remaining: 130 (LightParticle) - already default
    candidates["bkgcat"] = ak.unflatten(cats, cand_counts)


def _first_nonempty_int(candidates, field):
    """Extract a scalar PID from a jagged field (first non-empty value)."""
    if field not in candidates:
        return None
    arr = candidates[field]
    # Handle scalar (e.g. candidates["pid"] = 531)
    if isinstance(arr, (int, float)):
        return int(arr)

    if isinstance(arr, np.integer):
        return int(arr)
    for evt in range(len(arr)):
        if len(arr[evt]) > 0:
            return int(arr[evt][0])
    return None
