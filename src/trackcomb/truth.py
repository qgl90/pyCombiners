"""MC truth matching and background categorisation."""

from __future__ import annotations

__author__ = "Renato Quagliani <rquaglia@cern.ch>"

from collections import defaultdict

import awkward as ak
import numpy as np

from .models import infer_n_body


def count_true_decays(tracks, mother_pdg, daughter_pdgs=None):
    """Count true MC decays per event by grouping tracks with shared ancestor.

    Optionally filters by daughter PID pattern.
    """
    from .pid import pdg_id

    if isinstance(mother_pdg, str):
        mother_pdg = pdg_id(mother_pdg)
    mother_pdg = abs(mother_pdg)

    if daughter_pdgs is not None:
        expected = sorted(
            abs(pdg_id(p)) if isinstance(p, str) else abs(p) for p in daughter_pdgs
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
                    break  # one track contributes once per mother type

        if expected is None:
            counts[evt] = len(groups)
        else:
            for key, track_indices in groups.items():
                actual = sorted(abs(evt_mc_pids[ti]) for ti in track_indices)
                if actual == expected:
                    counts[evt] += 1

    return counts


def truth_match_candidates(candidates):
    """Check if all daughters share a common MC ancestor matching the mother.

    Returns jagged bool (events, candidates).
    """
    n_body = infer_n_body(candidates)
    if n_body == 0:
        raise ValueError("No daughter{k}_pool_index fields found in candidates")

    # Read mother and daughter PDGs from the container
    mother_pdg = _extract_scalar_pid(candidates, "pid")
    daughter_pdgs = []
    for k in range(n_body):
        daughter_pdgs.append(_extract_scalar_pid(candidates, f"daughter{k}_pid"))
    expected = (
        sorted(abs(p) for p in daughter_pdgs)
        if all(p is not None for p in daughter_pdgs)
        else None
    )

    # Get daughter pools
    daughter_pools = candidates["_daughter_pools"]

    # Materialize per-pool MC data
    pool_mc_pid = []
    pool_anc_pids = []
    pool_anc_keys = []
    for k in range(n_body):
        pool = daughter_pools[k]
        pool_mc_pid.append(pool["mc_pid"].tolist())
        pool_anc_pids.append(pool["mc_ancestor_pids"].tolist())
        pool_anc_keys.append(pool["mc_ancestor_keys"].tolist())

    # Materialize daughter pool indices
    daughter_pool_idx_lists = [
        candidates[f"daughter{k}_pool_index"].tolist() for k in range(n_body)
    ]
    n_cands_list = ak.num(candidates["vertex_x"]).tolist()
    n_events = len(n_cands_list)

    results = []
    for evt in range(n_events):
        n_cands = n_cands_list[evt]
        matched = np.zeros(n_cands, dtype=bool)

        for ci in range(n_cands):
            # Find ancestor keys matching mother_pdg for each daughter
            ancestor_sets = []
            daughter_mc_pids = []
            for k in range(n_body):
                pool_idx = int(daughter_pool_idx_lists[k][evt][ci])
                anc_pids = pool_anc_pids[k][evt][pool_idx]
                anc_keys = pool_anc_keys[k][evt][pool_idx]
                mc_pid = pool_mc_pid[k][evt][pool_idx]
                daughter_mc_pids.append(mc_pid)

                matching_keys = set()
                for pid, key in zip(anc_pids, anc_keys):
                    if abs(pid) == abs(mother_pdg):
                        matching_keys.add(key)
                ancestor_sets.append(matching_keys)

            # Intersect: require common ancestor across all daughters
            if not ancestor_sets or not ancestor_sets[0]:
                continue
            common = ancestor_sets[0]
            for s in ancestor_sets[1:]:
                common &= s
            if not common:
                continue

            # Optionally check daughter PIDs
            if expected is not None:
                actual = sorted(abs(p) for p in daughter_mc_pids)
                if actual != expected:
                    continue

            matched[ci] = True

        results.append(matched)

    return ak.Array(results)


def bkgcat(candidates):
    """Simplified LHCb-style background categories.

    Returns jagged int (events, candidates) with codes:
      0=Signal, 10=QuasiSignal, 20=PhysBkg, 30=Reflection,
      60=Ghost, 63=Clone, 66=Hierarchy, 100=Pileup,
      110=FromB, 120=FromC, 130=LightParticle.

    Categories 40 (PartReco) and 50 (LowMass) are omitted because
    they need the full MC particle tree.
    """
    n_body = infer_n_body(candidates)
    if n_body == 0:
        raise ValueError("No daughter{k}_pool_index fields found in candidates")

    mother_pdg = _extract_scalar_pid(candidates, "pid")
    daughter_pdgs = []
    for k in range(n_body):
        daughter_pdgs.append(_extract_scalar_pid(candidates, f"daughter{k}_pid"))

    daughter_pools = candidates["_daughter_pools"]

    # Materialize per-pool MC data
    pool_mc_truth = []
    pool_mc_pid = []
    pool_mc_key = []
    pool_mc_pv_key = []
    pool_mc_fromsignal = []
    pool_anc_pids = []
    pool_anc_keys = []
    for k in range(n_body):
        pool = daughter_pools[k]
        pool_mc_truth.append(pool["mc_truth"].tolist())
        pool_mc_pid.append(pool["mc_pid"].tolist())
        pool_mc_key.append(pool["mc_key"].tolist())
        pool_mc_pv_key.append(pool["mc_pv_key"].tolist())
        pool_mc_fromsignal.append(pool["mc_fromsignal"].tolist())
        pool_anc_pids.append(pool["mc_ancestor_pids"].tolist())
        pool_anc_keys.append(pool["mc_ancestor_keys"].tolist())

    daughter_pool_idx_lists = [
        candidates[f"daughter{k}_pool_index"].tolist() for k in range(n_body)
    ]
    n_cands_list = ak.num(candidates["vertex_x"]).tolist()
    n_events = len(n_cands_list)

    results = []
    for evt in range(n_events):
        n_cands = n_cands_list[evt]
        cats = np.full(n_cands, 130, dtype=int)  # default: light particle

        for ci in range(n_cands):
            # Gather per-daughter MC info
            mc_truths = []
            mc_pids = []
            mc_keys = []
            mc_pv_keys = []
            mc_fromsignals = []
            anc_pid_lists = []
            anc_key_lists = []
            for k in range(n_body):
                pool_idx = int(daughter_pool_idx_lists[k][evt][ci])
                mc_truths.append(pool_mc_truth[k][evt][pool_idx])
                mc_pids.append(pool_mc_pid[k][evt][pool_idx])
                mc_keys.append(pool_mc_key[k][evt][pool_idx])
                mc_pv_keys.append(pool_mc_pv_key[k][evt][pool_idx])
                mc_fromsignals.append(pool_mc_fromsignal[k][evt][pool_idx])
                anc_pid_lists.append(pool_anc_pids[k][evt][pool_idx])
                anc_key_lists.append(pool_anc_keys[k][evt][pool_idx])

            # --- Evaluate conditions ---

            # G: any ghost
            G = any(t == 0 for t in mc_truths)
            if G:
                cats[ci] = 60
                continue

            # K: clone — two daughters share same mc_key
            key_set = set()
            K = False
            for mk in mc_keys:
                if mk in key_set:
                    K = True
                    break
                key_set.add(mk)
            if K:
                cats[ci] = 63
                continue

            # L: hierarchy — one daughter's mc_key in another daughter's ancestors
            L = False
            for k1 in range(n_body):
                for k2 in range(n_body):
                    if k1 == k2:
                        continue
                    if mc_keys[k1] in anc_key_lists[k2]:
                        L = True
                        break
                if L:
                    break
            if L:
                cats[ci] = 66
                continue

            # A: all daughters share common MC ancestor key
            ancestor_key_sets = []
            for k in range(n_body):
                ancestor_key_sets.append(set(anc_key_lists[k]))
            common_keys = ancestor_key_sets[0]
            for s in ancestor_key_sets[1:]:
                common_keys = common_keys & s

            A = len(common_keys) > 0

            if A:
                # C: correct PID for all daughters
                C = True
                for k in range(n_body):
                    if daughter_pdgs[k] is not None:
                        if abs(mc_pids[k]) != abs(daughter_pdgs[k]):
                            C = False
                            break

                if not C:
                    cats[ci] = 30  # Reflection
                    continue

                # D: common ancestor PDG matches candidate mother
                D = False
                if mother_pdg is not None:
                    for k in range(n_body):
                        for pid, key in zip(anc_pid_lists[k], anc_key_lists[k]):
                            if key in common_keys and abs(pid) == abs(mother_pdg):
                                D = True
                                break
                        if D:
                            break

                if not D:
                    cats[ci] = 20  # Physics background
                    continue

                # S: all daughters from signal
                S = all(f == 1 for f in mc_fromsignals)
                cats[ci] = 0 if S else 10
            else:
                # !A branch: pileup / bb / cc / light

                # H: pileup — different mc_pv_key
                valid_pv_keys = [k for k in mc_pv_keys if k != -1]
                H = len(set(valid_pv_keys)) > 1 if len(valid_pv_keys) > 1 else False
                if H:
                    cats[ci] = 100
                    continue

                # I: b-hadron ancestor
                I = False
                for k in range(n_body):
                    for pid in anc_pid_lists[k]:
                        if _has_b_quark(pid):
                            I = True
                            break
                    if I:
                        break
                if I:
                    cats[ci] = 110
                    continue

                # J: c-hadron ancestor
                J = False
                for k in range(n_body):
                    for pid in anc_pid_lists[k]:
                        if _has_c_quark(pid):
                            J = True
                            break
                    if J:
                        break
                if J:
                    cats[ci] = 120
                    continue

                cats[ci] = 130  # light particle (default)

        results.append(cats)

    return ak.Array(results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_b_quark(pdg_id):
    """Check if a hadron contains a b quark using PDG numbering."""
    n = abs(pdg_id)
    return (n // 100) % 10 == 5 or (n // 1000) % 10 == 5 or (n // 10000) % 10 == 5


def _has_c_quark(pdg_id):
    """Check if a hadron contains a c quark using PDG numbering."""
    n = abs(pdg_id)
    return (n // 100) % 10 == 4 or (n // 1000) % 10 == 4 or (n // 10000) % 10 == 4


def _extract_scalar_pid(candidates, field):
    """Extract a scalar PID from a jagged field (first non-empty value)."""
    if field not in candidates:
        return None
    arr = candidates[field]
    # Handle scalar (e.g. candidates["pid"] = 531)
    if isinstance(arr, (int, float)):
        return int(arr)
    import numpy as np

    if isinstance(arr, np.integer):
        return int(arr)
    for evt in range(len(arr)):
        if len(arr[evt]) > 0:
            return int(arr[evt][0])
    return None
