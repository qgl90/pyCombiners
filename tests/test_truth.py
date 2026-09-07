"""Unit tests for truth matching functions."""

from __future__ import annotations

__author__ = "Renato Quagliani <rquaglia@cern.ch>"


import unittest

import awkward as ak
import numpy as np

from trackcomb.models import n_daughters
from trackcomb.truth import (
    _first_nonempty_int,
    compute_bkgcat,
    count_true_decays,
)


def _has_b_quark(pdg_id):
    n = abs(pdg_id)
    return (
        (n // 100) % 10 == 5 or (n // 1000) % 10 == 5 or (n // 10000) % 10 == 5
    )


def _has_c_quark(pdg_id):
    n = abs(pdg_id)
    return (
        (n // 100) % 10 == 4 or (n // 1000) % 10 == 4 or (n // 10000) % 10 == 4
    )


def _bkgcat_loop(candidates):
    """Loop-based bkgcat reference implementation for validation."""
    n_body = n_daughters(candidates)
    if n_body == 0:
        raise ValueError(
            "No daughter{k}_global_index fields found in candidates"
        )

    mother_pdg = _first_nonempty_int(candidates, "pid")

    daughter_pools = candidates["_daughter_pools"]
    daughter_pdgs = []
    for k in range(n_body):
        pool = daughter_pools[k]
        if "pid" in pool:
            gi_arr = candidates[f"daughter{k}_global_index"]
            flat_pid = ak.flatten(pool["pid"])
            try:
                first_gi = int(ak.flatten(gi_arr)[0])
                daughter_pdgs.append(int(np.asarray(flat_pid)[first_gi]))
            except (ValueError, IndexError):
                daughter_pdgs.append(None)
        else:
            daughter_pdgs.append(None)

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

    daughter_gi_lists = [
        candidates[f"daughter{k}_global_index"].tolist() for k in range(n_body)
    ]
    pool_offsets_list = []
    for k in range(n_body):
        pool = daughter_pools[k]
        pc = ak.to_numpy(ak.num(pool["x"]))
        off = np.zeros(len(pc) + 1, dtype=np.int64)
        np.cumsum(pc, out=off[1:])
        pool_offsets_list.append(off)

    n_cands_list = ak.num(candidates["vertex_x"]).tolist()
    n_events = len(n_cands_list)

    results = []
    for evt in range(n_events):
        n_cands = n_cands_list[evt]
        cats = np.full(n_cands, 130, dtype=int)

        for ci in range(n_cands):
            mc_truths = []
            mc_pids = []
            mc_keys = []
            mc_pv_keys = []
            mc_fromsignals = []
            anc_pid_lists = []
            anc_key_lists = []
            for k in range(n_body):
                gi = int(daughter_gi_lists[k][evt][ci])
                pool_idx = gi - int(pool_offsets_list[k][evt])
                mc_truths.append(pool_mc_truth[k][evt][pool_idx])
                mc_pids.append(pool_mc_pid[k][evt][pool_idx])
                mc_keys.append(pool_mc_key[k][evt][pool_idx])
                mc_pv_keys.append(pool_mc_pv_key[k][evt][pool_idx])
                mc_fromsignals.append(pool_mc_fromsignal[k][evt][pool_idx])
                anc_pid_lists.append(pool_anc_pids[k][evt][pool_idx])
                anc_key_lists.append(pool_anc_keys[k][evt][pool_idx])

            G = any(t == 0 for t in mc_truths)
            if G:
                cats[ci] = 60
                continue

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

            ancestor_key_sets = [set(anc_key_lists[k]) for k in range(n_body)]
            common_keys = ancestor_key_sets[0]
            for s in ancestor_key_sets[1:]:
                common_keys = common_keys & s

            A = len(common_keys) > 0

            if A:
                C = True
                for k in range(n_body):
                    if daughter_pdgs[k] is not None:
                        if abs(mc_pids[k]) != abs(daughter_pdgs[k]):
                            C = False
                            break

                if not C:
                    cats[ci] = 30
                    continue

                D = False
                if mother_pdg is not None:
                    for k in range(n_body):
                        for pid, key in zip(
                            anc_pid_lists[k], anc_key_lists[k]
                        ):
                            if key in common_keys and abs(pid) == abs(
                                mother_pdg
                            ):
                                D = True
                                break
                        if D:
                            break

                if not D:
                    cats[ci] = 20
                    continue

                S = all(f == 1 for f in mc_fromsignals)
                cats[ci] = 0 if S else 10
            else:
                valid_pv_keys = [k for k in mc_pv_keys if k != -1]
                H = (
                    len(set(valid_pv_keys)) > 1
                    if len(valid_pv_keys) > 1
                    else False
                )
                if H:
                    cats[ci] = 100
                    continue

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

                cats[ci] = 130

        results.append(cats)

    return ak.Array(results)


def _make_tracks_with_mc(n_events=2):
    """Synthetic tracks with MC ancestry: 2 events, Ks and unrelated tracks."""
    mc_pid = ak.Array(
        [
            [211, -211, 211, 13],
            [-211, 211, 321],
        ]
    )
    mc_ancestor_pids = ak.Array(
        [
            [[310], [310], [310], [443]],
            [[310], [310], [421]],
        ]
    )
    mc_ancestor_keys = ak.Array(
        [
            [[100], [100], [200], [500]],
            [[300], [300], [600]],
        ]
    )

    # Minimal track fields needed by combine (for testing truth matching
    # with candidates, we also need spatial + covariance fields)
    rng = np.random.RandomState(42)
    tracks = {
        "mc_pid": mc_pid,
        "mc_ancestor_pids": mc_ancestor_pids,
        "mc_ancestor_keys": mc_ancestor_keys,
        "mc_truth": ak.Array([[1, 1, 1, 1], [1, 1, 1]]),
        "mc_pv_key": ak.Array([[0, 0, 0, 0], [0, 0, 0]]),
    }

    # Add physics fields so combine can run
    for field in ("x", "y", "z", "tx", "ty"):
        tracks[field] = ak.Array(
            [
                rng.randn(4) * 0.1,
                rng.randn(3) * 0.1,
            ]
        )
    tracks["p"] = ak.Array(
        [
            np.abs(rng.randn(4)) * 10 + 5,
            np.abs(rng.randn(3)) * 10 + 5,
        ]
    )
    tracks["charge"] = ak.Array(
        [
            [1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    tracks["time"] = ak.Array(
        [
            rng.randn(4) * 0.01,
            rng.randn(3) * 0.01,
        ]
    )
    tracks["sigma_time"] = ak.Array(
        [
            [0.07] * 4,
            [0.07] * 3,
        ]
    )
    tracks["pt"] = ak.Array(
        [
            np.abs(rng.randn(4)) * 2 + 0.5,
            np.abs(rng.randn(3)) * 2 + 0.5,
        ]
    )
    tracks["eta"] = ak.Array(
        [
            rng.randn(4) * 0.5 + 3.5,
            rng.randn(3) * 0.5 + 3.5,
        ]
    )
    tracks["track_id"] = ak.Array(
        [
            [0, 1, 2, 3],
            [0, 1, 2],
        ]
    )

    # 4x4 covariance (lower triangle)
    for i in range(4):
        for j in range(i + 1):
            if i == j:
                tracks[f"cov_{i}_{j}"] = ak.Array(
                    [
                        [0.001] * 4,
                        [0.001] * 3,
                    ]
                )
            else:
                tracks[f"cov_{i}_{j}"] = ak.Array(
                    [
                        [0.0] * 4,
                        [0.0] * 3,
                    ]
                )

    return tracks


class TestBatchCountTrueDecays(unittest.TestCase):
    """Test count_true_decays."""

    def test_count_without_daughter_filter(self):
        """Without daughter_pdgs, count all groups with matching mother."""
        tracks = _make_tracks_with_mc()
        counts = count_true_decays(tracks, "K(S)0")
        # Event 0: 2 Ks ancestors (key=100 with 2 tracks, key=200 with 1 track)
        # Event 1: 1 Ks ancestor (key=300 with 2 tracks)
        self.assertEqual(counts[0], 2)
        self.assertEqual(counts[1], 1)

    def test_count_with_daughter_filter(self):
        """With daughter_pdgs=["pi+", "pi-"], only complete Ks->pipi counted."""
        tracks = _make_tracks_with_mc()
        counts = count_true_decays(tracks, "K(S)0", ["pi+", "pi-"])
        # Event 0: key=100 has pi+(211) and pi-(-211) -> sorted |pid| = [211, 211] -> match
        #          key=200 has only one track (pi+) -> [211] != [211, 211] -> no match
        # Event 1: key=300 has pi-(−211) and pi+(211) -> sorted |pid| = [211, 211] -> match
        self.assertEqual(counts[0], 1)
        self.assertEqual(counts[1], 1)

    def test_no_matching_mother(self):
        """No tracks with J/psi (443) as mother -> all zeros."""
        tracks = _make_tracks_with_mc()
        # Only trk3 in event 0 has ancestor_pid=443, and there's only 1 such track
        counts = count_true_decays(tracks, "J/psi(1S)", ["mu+", "mu+"])
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[1], 0)

    def test_returns_numpy_array(self):
        tracks = _make_tracks_with_mc()
        counts = count_true_decays(tracks, "K(S)0")
        self.assertIsInstance(counts, np.ndarray)
        self.assertEqual(counts.shape, (2,))


def _truth_match_candidates(candidates):
    """Loop-based truth matching reference implementation for testing."""
    n_body = n_daughters(candidates)
    mother_pdg = _first_nonempty_int(candidates, "pid")

    daughter_pools = candidates["_daughter_pools"]
    daughter_pdgs = []
    for k in range(n_body):
        pool = daughter_pools[k]
        if "pid" in pool:
            gi = candidates[f"daughter{k}_global_index"]
            flat_pid = ak.flatten(pool["pid"])
            try:
                first_gi = int(ak.flatten(gi)[0])
                daughter_pdgs.append(int(np.asarray(flat_pid)[first_gi]))
            except (ValueError, IndexError):
                daughter_pdgs.append(None)
        else:
            daughter_pdgs.append(None)
    expected = (
        sorted(abs(p) for p in daughter_pdgs)
        if all(p is not None for p in daughter_pdgs)
        else None
    )

    pool_mc_pid = []
    pool_anc_pids = []
    pool_anc_keys = []
    for k in range(n_body):
        pool = daughter_pools[k]
        pool_mc_pid.append(pool["mc_pid"].tolist())
        pool_anc_pids.append(pool["mc_ancestor_pids"].tolist())
        pool_anc_keys.append(pool["mc_ancestor_keys"].tolist())

    daughter_gi_lists = [
        candidates[f"daughter{k}_global_index"].tolist() for k in range(n_body)
    ]
    pool_offsets = []
    for k in range(n_body):
        pool = daughter_pools[k]
        pc = ak.to_numpy(ak.num(pool["x"]))
        off = np.zeros(len(pc) + 1, dtype=np.int64)
        np.cumsum(pc, out=off[1:])
        pool_offsets.append(off)

    n_cands_list = ak.num(candidates["vertex_x"]).tolist()
    n_events = len(n_cands_list)

    results = []
    for evt in range(n_events):
        n_cands = n_cands_list[evt]
        matched = np.zeros(n_cands, dtype=bool)
        for ci in range(n_cands):
            ancestor_sets = []
            daughter_mc_pids = []
            for k in range(n_body):
                gi = int(daughter_gi_lists[k][evt][ci])
                pool_idx = gi - int(pool_offsets[k][evt])
                anc_pids = pool_anc_pids[k][evt][pool_idx]
                anc_keys = pool_anc_keys[k][evt][pool_idx]
                mc_pid = pool_mc_pid[k][evt][pool_idx]
                daughter_mc_pids.append(mc_pid)
                matching_keys = set()
                for pid, key in zip(anc_pids, anc_keys):
                    if abs(pid) == abs(mother_pdg):
                        matching_keys.add(key)
                ancestor_sets.append(matching_keys)
            if not ancestor_sets or not ancestor_sets[0]:
                continue
            common = ancestor_sets[0]
            for s in ancestor_sets[1:]:
                common &= s
            if not common:
                continue
            if expected is not None:
                actual = sorted(abs(p) for p in daughter_mc_pids)
                if actual != expected:
                    continue
            matched[ci] = True
        results.append(matched)
    return ak.Array(results)


class TestBatchTruthMatchCandidates(unittest.TestCase):
    """Test _truth_match_candidates with synthetic candidates."""

    def _make_candidates(self, tracks):
        """Fake candidates: evt0 has 1 signal + 2 bkg, evt1 has 1 signal + 1 bkg."""
        # Compute pool offsets for global_index
        # Event 0: 4 tracks (offsets: 0), Event 1: 3 tracks (offsets: 4)
        # global_index = pool_offset[evt] + local_index
        # evt0: local [0,0,0] -> global [0,0,0]; local [1,3,2] -> global [1,3,2]
        # evt1: local [0,0] -> global [4,4]; local [1,2] -> global [5,6]
        return {
            "vertex_x": ak.Array([[0.0, 0.0, 0.0], [0.0, 0.0]]),
            "daughter0_track_id": ak.Array([[0, 0, 0], [0, 0]]),
            "daughter1_track_id": ak.Array([[1, 3, 2], [1, 2]]),
            "daughter0_global_index": ak.Array([[0, 0, 0], [4, 4]]),
            "daughter1_global_index": ak.Array([[1, 3, 2], [5, 6]]),
            # Mother PDG (K_S^0 = 310)
            "pid": ak.Array([[310, 310, 310], [310, 310]]),
            # Reference to the underlying track pools
            "_daughter_pools": [tracks, tracks],
        }

    def test_truth_match_with_daughters(self):
        """Signal candidates are correctly identified."""
        tracks = _make_tracks_with_mc()
        candidates = self._make_candidates(tracks)
        matched = _truth_match_candidates(candidates)
        # Event 0: cand0=True, cand1=False, cand2=False
        self.assertTrue(matched[0][0])
        self.assertFalse(matched[0][1])
        self.assertFalse(matched[0][2])
        # Event 1: cand0=True, cand1=False
        self.assertTrue(matched[1][0])
        self.assertFalse(matched[1][1])

    def test_truth_match_without_pid_field(self):
        """Without 'pid' in pool, only common ancestor matters (no PID check)."""
        tracks = _make_tracks_with_mc()
        # The pool doesn't have 'pid' field (no set_tracks_pid called),
        # so daughter PID check is skipped
        candidates = self._make_candidates(tracks)
        matched = _truth_match_candidates(candidates)
        # Event 0: cand0=True (shared Ks100), cand1=False, cand2=False (diff keys)
        self.assertTrue(matched[0][0])
        self.assertFalse(matched[0][1])
        self.assertFalse(matched[0][2])

    def test_returns_jagged_bool(self):
        tracks = _make_tracks_with_mc()
        candidates = self._make_candidates(tracks)
        matched = _truth_match_candidates(candidates)
        self.assertEqual(len(matched), 2)
        self.assertEqual(len(matched[0]), 3)
        self.assertEqual(len(matched[1]), 2)

    def test_empty_candidates(self):
        """No candidates -> empty boolean arrays."""
        tracks = _make_tracks_with_mc()
        candidates = {
            "vertex_x": ak.Array([[], []]),
            "daughter0_track_id": ak.Array([[], []]),
            "daughter1_track_id": ak.Array([[], []]),
            "daughter0_global_index": ak.Array([[], []]),
            "daughter1_global_index": ak.Array([[], []]),
            "pid": ak.Array([[], []]),
            "_daughter_pools": [tracks, tracks],
        }
        matched = _truth_match_candidates(candidates)
        self.assertEqual(len(matched[0]), 0)
        self.assertEqual(len(matched[1]), 0)


class TestBkgCat(unittest.TestCase):
    """Test bkgcat background category classification."""

    def _make_pool(
        self,
        mc_truth,
        mc_pid,
        mc_key,
        mc_pv_key,
        mc_fromsignal,
        mc_ancestor_pids,
        mc_ancestor_keys,
    ):
        """Build a minimal track pool with MC fields for a single event."""
        n = len(mc_truth)
        rng = np.random.RandomState(0)
        pool = {
            "mc_truth": ak.Array([mc_truth]),
            "mc_pid": ak.Array([mc_pid]),
            "mc_key": ak.Array([mc_key]),
            "mc_pv_key": ak.Array([mc_pv_key]),
            "mc_fromsignal": ak.Array([mc_fromsignal]),
            "mc_ancestor_pids": ak.Array([mc_ancestor_pids]),
            "mc_ancestor_keys": ak.Array([mc_ancestor_keys]),
        }
        for field in ("x", "y", "z", "tx", "ty"):
            pool[field] = ak.Array([rng.randn(n).tolist()])
        pool["p"] = ak.Array([(np.abs(rng.randn(n)) * 10 + 5).tolist()])
        pool["charge"] = ak.Array([[1.0] * n])
        pool["track_id"] = ak.Array([list(range(n))])
        return pool

    def _make_candidates(
        self, pool, daughter0_idx, daughter1_idx, mother_pid, d0_pid, d1_pid
    ):
        """Build a single-event candidates container.

        daughter0_idx/daughter1_idx are local indices within a single-event pool,
        which equal global indices since there's only one event (offset=0).
        d0_pid/d1_pid are set as the pool's 'pid' field via set_tracks_pid-like setup.
        """
        from trackcomb.pid import set_tracks_pid

        n = len(daughter0_idx)
        # Set pid on pool so daughter PIDs can be looked up
        if "pid" not in pool:
            # Assign pid based on d0_pid (all tracks get same pid hypothesis)
            pool["pid"] = ak.Array([[d0_pid] * len(pool["mc_truth"][0])])
            pool["mass"] = ak.Array([[0.0] * len(pool["mc_truth"][0])])
        # For single-event pool, global_index == local_index
        return {
            "vertex_x": ak.Array(
                [daughter0_idx]
            ),  # dummy, just needs right length
            "daughter0_global_index": ak.Array([daughter0_idx]),
            "daughter1_global_index": ak.Array([daughter1_idx]),
            "pid": ak.Array([[mother_pid] * n]),
            "_daughter_pools": [pool, pool],
        }

    def test_signal(self):
        """Cat 0: all from same mother, correct PID, signal mother, fromsignal."""
        pool = self._make_pool(
            mc_truth=[1, 1],
            mc_pid=[211, -211],
            mc_key=[10, 11],
            mc_pv_key=[0, 0],
            mc_fromsignal=[1, 1],
            mc_ancestor_pids=[[310], [310]],
            mc_ancestor_keys=[[100], [100]],
        )
        cands = self._make_candidates(pool, [0], [1], 310, 211, 211)
        compute_bkgcat(cands)
        self.assertEqual(int(cands["bkgcat"][0][0]), 0)

    def test_quasi_signal(self):
        """Cat 10: same as signal but mc_fromsignal=0."""
        pool = self._make_pool(
            mc_truth=[1, 1],
            mc_pid=[211, -211],
            mc_key=[10, 11],
            mc_pv_key=[0, 0],
            mc_fromsignal=[0, 0],
            mc_ancestor_pids=[[310], [310]],
            mc_ancestor_keys=[[100], [100]],
        )
        cands = self._make_candidates(pool, [0], [1], 310, 211, 211)
        compute_bkgcat(cands)
        self.assertEqual(int(cands["bkgcat"][0][0]), 10)

    def test_physics_background(self):
        """Cat 20: correct PID, common ancestor, but wrong mother type."""
        pool = self._make_pool(
            mc_truth=[1, 1],
            mc_pid=[211, -211],
            mc_key=[10, 11],
            mc_pv_key=[0, 0],
            mc_fromsignal=[0, 0],
            mc_ancestor_pids=[[421], [421]],  # D0, not Ks
            mc_ancestor_keys=[[100], [100]],
        )
        # Candidate says mother is Ks (310), but true ancestor is D0 (421)
        cands = self._make_candidates(pool, [0], [1], 310, 211, 211)
        compute_bkgcat(cands)
        self.assertEqual(int(cands["bkgcat"][0][0]), 20)

    def test_reflection(self):
        """Cat 30: common ancestor, right mother, but wrong daughter PID."""
        pool = self._make_pool(
            mc_truth=[1, 1],
            mc_pid=[321, -211],  # true: K+ and pi-, but assigned pi+ pi+
            mc_key=[10, 11],
            mc_pv_key=[0, 0],
            mc_fromsignal=[0, 0],
            mc_ancestor_pids=[[310], [310]],
            mc_ancestor_keys=[[100], [100]],
        )
        cands = self._make_candidates(pool, [0], [1], 310, 211, 211)
        compute_bkgcat(cands)
        self.assertEqual(int(cands["bkgcat"][0][0]), 30)

    def test_ghost(self):
        """Cat 60: one daughter is a ghost (mc_truth=0)."""
        pool = self._make_pool(
            mc_truth=[1, 0],
            mc_pid=[211, 0],
            mc_key=[10, -1],
            mc_pv_key=[0, -1],
            mc_fromsignal=[1, 0],
            mc_ancestor_pids=[[310], []],
            mc_ancestor_keys=[[100], []],
        )
        cands = self._make_candidates(pool, [0], [1], 310, 211, 211)
        compute_bkgcat(cands)
        self.assertEqual(int(cands["bkgcat"][0][0]), 60)

    def test_clone(self):
        """Cat 63: two daughters share same mc_key."""
        pool = self._make_pool(
            mc_truth=[1, 1],
            mc_pid=[211, 211],
            mc_key=[10, 10],  # same mc_key = clone
            mc_pv_key=[0, 0],
            mc_fromsignal=[1, 1],
            mc_ancestor_pids=[[310], [310]],
            mc_ancestor_keys=[[100], [100]],
        )
        cands = self._make_candidates(pool, [0], [1], 310, 211, 211)
        compute_bkgcat(cands)
        self.assertEqual(int(cands["bkgcat"][0][0]), 63)

    def test_hierarchy(self):
        """Cat 66: one daughter's mc_key is in other daughter's ancestor chain."""
        pool = self._make_pool(
            mc_truth=[1, 1],
            mc_pid=[211, -211],
            mc_key=[10, 11],
            mc_pv_key=[0, 0],
            mc_fromsignal=[0, 0],
            mc_ancestor_pids=[[310], [310]],
            mc_ancestor_keys=[
                [100],
                [100, 10],
            ],  # daughter1 has daughter0's mc_key=10 as ancestor
        )
        cands = self._make_candidates(pool, [0], [1], 310, 211, 211)
        compute_bkgcat(cands)
        self.assertEqual(int(cands["bkgcat"][0][0]), 66)

    def test_pileup(self):
        """Cat 100: no common ancestor, different PV keys."""
        pool = self._make_pool(
            mc_truth=[1, 1],
            mc_pid=[211, -211],
            mc_key=[10, 11],
            mc_pv_key=[0, 1],  # different PVs
            mc_fromsignal=[0, 0],
            mc_ancestor_pids=[[310], [421]],  # different ancestors
            mc_ancestor_keys=[[100], [200]],
        )
        cands = self._make_candidates(pool, [0], [1], 310, 211, 211)
        compute_bkgcat(cands)
        self.assertEqual(int(cands["bkgcat"][0][0]), 100)

    def test_from_b_event(self):
        """Cat 110: no common ancestor, same PV, b-hadron in ancestry."""
        pool = self._make_pool(
            mc_truth=[1, 1],
            mc_pid=[211, -211],
            mc_key=[10, 11],
            mc_pv_key=[0, 0],
            mc_fromsignal=[0, 0],
            mc_ancestor_pids=[[511], [421]],  # B0 and D0 ancestors (no common)
            mc_ancestor_keys=[[100], [200]],
        )
        cands = self._make_candidates(pool, [0], [1], 310, 211, 211)
        compute_bkgcat(cands)
        self.assertEqual(int(cands["bkgcat"][0][0]), 110)

    def test_from_c_event(self):
        """Cat 120: no common ancestor, same PV, c-hadron but no b-hadron."""
        pool = self._make_pool(
            mc_truth=[1, 1],
            mc_pid=[211, -211],
            mc_key=[10, 11],
            mc_pv_key=[0, 0],
            mc_fromsignal=[0, 0],
            mc_ancestor_pids=[[421], [111]],  # D0 and pi0 (no common key)
            mc_ancestor_keys=[[100], [200]],
        )
        cands = self._make_candidates(pool, [0], [1], 310, 211, 211)
        compute_bkgcat(cands)
        self.assertEqual(int(cands["bkgcat"][0][0]), 120)

    def test_light_particle(self):
        """Cat 130: no common ancestor, same PV, no b or c hadrons."""
        pool = self._make_pool(
            mc_truth=[1, 1],
            mc_pid=[211, -211],
            mc_key=[10, 11],
            mc_pv_key=[0, 0],
            mc_fromsignal=[0, 0],
            mc_ancestor_pids=[[111], [113]],  # pi0 and rho0 (light)
            mc_ancestor_keys=[[100], [200]],
        )
        cands = self._make_candidates(pool, [0], [1], 310, 211, 211)
        compute_bkgcat(cands)
        self.assertEqual(int(cands["bkgcat"][0][0]), 130)


class TestPropagateMcTruth(unittest.TestCase):
    """Test MC truth propagation inside combine() for hierarchical decays."""

    def _make_tracks_and_pvs(self):
        """Synthetic mu/pi tracks from Bs/Ks with MC ancestry, plus PVs."""
        from trackcomb.pid import set_tracks_pid

        rng = np.random.RandomState(99)
        tracks = {
            "x": ak.Array([rng.randn(4) * 0.1, rng.randn(2) * 0.1]),
            "y": ak.Array([rng.randn(4) * 0.1, rng.randn(2) * 0.1]),
            "z": ak.Array([[100.0] * 4, [100.0] * 2]),
            "tx": ak.Array([rng.randn(4) * 0.01, rng.randn(2) * 0.01]),
            "ty": ak.Array([rng.randn(4) * 0.01, rng.randn(2) * 0.01]),
            "p": ak.Array(
                [
                    (np.abs(rng.randn(4)) * 10 + 20).tolist(),
                    (np.abs(rng.randn(2)) * 10 + 20).tolist(),
                ]
            ),
            "charge": ak.Array([[1.0, -1.0, 1.0, -1.0], [1.0, -1.0]]),
            "time": ak.Array([[0.1] * 4, [0.1] * 2]),
            "sigma_time": ak.Array([[0.01] * 4, [0.01] * 2]),
            "track_id": ak.Array([[0, 1, 2, 3], [0, 1]]),
            # MC truth
            "mc_truth": ak.Array([[1, 1, 1, 1], [1, 1]]),
            "mc_pid": ak.Array([[13, -13, 211, -211], [13, -13]]),
            "mc_key": ak.Array([[10, 11, 12, 13], [20, 21]]),
            "mc_pv_key": ak.Array([[0, 0, 0, 0], [1, 1]]),
            "mc_fromsignal": ak.Array([[1, 1, 0, 0], [1, 1]]),
            # Ancestors ordered most-immediate-first:
            # trk0,1: Bs(531) key=100 → grandparent(999) key=500
            # trk2,3: Ks(310) key=200 → other(888) key=600
            # evt1 trk0,1: Bs(531) key=300 → gp(999) key=700
            "mc_ancestor_pids": ak.Array(
                [
                    [[531, 999], [531, 999], [310, 888], [310, 888]],
                    [[531, 999], [531, 999]],
                ]
            ),
            "mc_ancestor_keys": ak.Array(
                [
                    [[100, 500], [100, 500], [200, 600], [200, 600]],
                    [[300, 700], [300, 700]],
                ]
            ),
        }
        tracks["qop"] = tracks["charge"] / tracks["p"]
        for i in range(5):
            for j in range(i + 1):
                val = 0.001 if i == j else 0.0
                tracks[f"cov_{i}_{j}"] = ak.Array([[val] * 4, [val] * 2])

        set_tracks_pid(tracks, "mu+")

        pvs = {
            "x": ak.Array([[0.0], [0.0]]),
            "y": ak.Array([[0.0], [0.0]]),
            "z": ak.Array([[0.0], [0.0]]),
            "time": ak.Array([[0.0], [0.0]]),
            "sigma_time": ak.Array([[0.05], [0.05]]),
            "cov_0_0": ak.Array([[1e-4], [1e-4]]),
            "cov_1_0": ak.Array([[0.0], [0.0]]),
            "cov_1_1": ak.Array([[1e-4], [1e-4]]),
        }
        return tracks, pvs

    def test_mc_fields_present(self):
        """combine() output should contain all MC truth fields."""
        from trackcomb import combine, apply_mask

        tracks, pvs = self._make_tracks_and_pvs()
        pos = apply_mask(tracks, tracks["charge"] > 0)
        neg = apply_mask(tracks, tracks["charge"] < 0)
        cands = combine([pos, neg], pvs)
        for field in (
            "mc_truth",
            "mc_pid",
            "mc_key",
            "mc_pv_key",
            "mc_fromsignal",
            "mc_ancestor_pids",
            "mc_ancestor_keys",
        ):
            self.assertIn(field, cands, f"Missing MC field: {field}")

    def test_signal_candidate_mc_truth(self):
        """Signal candidates (shared ancestor) get correct mc_pid/key."""
        from trackcomb import combine, apply_mask

        tracks, pvs = self._make_tracks_and_pvs()
        pos = apply_mask(tracks, tracks["charge"] > 0)
        neg = apply_mask(tracks, tracks["charge"] < 0)
        cands = combine([pos, neg], pvs)

        # Find candidates where mc_truth == 1
        flat_truth = ak.to_numpy(ak.flatten(cands["mc_truth"]))
        flat_pid = ak.to_numpy(ak.flatten(cands["mc_pid"]))
        flat_key = ak.to_numpy(ak.flatten(cands["mc_key"]))
        signal_mask = flat_truth == 1
        self.assertTrue(
            np.any(signal_mask), "Should have at least one signal candidate"
        )
        # All signal candidates should have pid=531 (Bs) or pid=310 (Ks)
        for pid in flat_pid[signal_mask]:
            self.assertIn(abs(pid), {531, 310})
        # All signal keys should be positive
        for key in flat_key[signal_mask]:
            self.assertGreater(key, 0)

    def test_nonsignal_candidate_mc_truth(self):
        """Non-signal candidates get mc_truth=0, mc_key=-1, empty ancestors."""
        from trackcomb import combine, apply_mask

        tracks, pvs = self._make_tracks_and_pvs()
        pos = apply_mask(tracks, tracks["charge"] > 0)
        neg = apply_mask(tracks, tracks["charge"] < 0)
        cands = combine([pos, neg], pvs)

        flat_truth = ak.to_numpy(ak.flatten(cands["mc_truth"]))
        flat_key = ak.to_numpy(ak.flatten(cands["mc_key"]))
        flat_pv = ak.to_numpy(ak.flatten(cands["mc_pv_key"]))
        flat_fs = ak.to_numpy(ak.flatten(cands["mc_fromsignal"]))
        nonsig = flat_truth == 0
        if np.any(nonsig):
            np.testing.assert_array_equal(flat_key[nonsig], -1)
            np.testing.assert_array_equal(flat_pv[nonsig], -1)
            np.testing.assert_array_equal(flat_fs[nonsig], 0)

    def test_ancestor_chain_excludes_common_mother(self):
        """Remaining ancestor chain should NOT include the common mother itself."""
        from trackcomb import combine, apply_mask

        tracks, pvs = self._make_tracks_and_pvs()
        pos = apply_mask(tracks, tracks["charge"] > 0)
        neg = apply_mask(tracks, tracks["charge"] < 0)
        cands = combine([pos, neg], pvs)

        flat_truth = ak.to_numpy(ak.flatten(cands["mc_truth"]))
        flat_key = ak.to_numpy(ak.flatten(cands["mc_key"]))
        flat_anc_keys = ak.flatten(cands["mc_ancestor_keys"])

        for i in range(len(flat_truth)):
            if flat_truth[i] == 1:
                mother_key = flat_key[i]
                anc_keys = flat_anc_keys[i].tolist()
                self.assertNotIn(
                    mother_key,
                    anc_keys,
                    f"Common mother key {mother_key} should not be in "
                    f"remaining ancestor chain {anc_keys}",
                )

    def test_fromsignal_requires_all_daughters(self):
        """mc_fromsignal=1 only when ALL daughters have mc_fromsignal=1."""
        from trackcomb import combine, apply_mask

        tracks, pvs = self._make_tracks_and_pvs()
        pos = apply_mask(tracks, tracks["charge"] > 0)
        neg = apply_mask(tracks, tracks["charge"] < 0)
        cands = combine([pos, neg], pvs)

        flat_truth = ak.to_numpy(ak.flatten(cands["mc_truth"]))
        flat_pid = ak.to_numpy(ak.flatten(cands["mc_pid"]))
        flat_fs = ak.to_numpy(ak.flatten(cands["mc_fromsignal"]))

        # Candidates from Bs (pid=531): daughters have fromsignal=1 → mc_fromsignal=1
        bs_mask = (flat_truth == 1) & (np.abs(flat_pid) == 531)
        if np.any(bs_mask):
            np.testing.assert_array_equal(flat_fs[bs_mask], 1)

        # Candidates from Ks (pid=310): daughters have fromsignal=0 → mc_fromsignal=0
        ks_mask = (flat_truth == 1) & (np.abs(flat_pid) == 310)
        if np.any(ks_mask):
            np.testing.assert_array_equal(flat_fs[ks_mask], 0)

    def test_bkgcat_on_propagated_truth(self):
        """bkgcat() should work on candidates with propagated MC truth."""
        from trackcomb import combine, apply_mask, set_composite_pid

        tracks, pvs = self._make_tracks_and_pvs()
        pos = apply_mask(tracks, tracks["charge"] > 0)
        neg = apply_mask(tracks, tracks["charge"] < 0)
        cands = combine([pos, neg], pvs)
        set_composite_pid(cands, "B(s)0")

        compute_bkgcat(cands)
        flat_cats = ak.to_numpy(ak.flatten(cands["bkgcat"]))
        flat_truth = ak.to_numpy(ak.flatten(cands["mc_truth"]))
        flat_pid = ak.to_numpy(ak.flatten(cands["mc_pid"]))

        # Bs signal candidates should get bkgcat 0
        bs_signal = (flat_truth == 1) & (np.abs(flat_pid) == 531)
        if np.any(bs_signal):
            np.testing.assert_array_equal(flat_cats[bs_signal], 0)

    def test_no_mc_fields_skips_propagation(self):
        """combine() should work without MC fields (no propagation)."""
        from trackcomb import combine
        from trackcomb.pid import set_tracks_pid

        tracks = {
            "x": ak.Array([[0.1, 0.2]]),
            "y": ak.Array([[0.01, 0.02]]),
            "z": ak.Array([[100.0, 100.0]]),
            "tx": ak.Array([[0.01, -0.01]]),
            "ty": ak.Array([[0.01, -0.01]]),
            "p": ak.Array([[50.0, 40.0]]),
            "charge": ak.Array([[1.0, -1.0]]),
            "time": ak.Array([[0.1, 0.1]]),
            "sigma_time": ak.Array([[0.01, 0.01]]),
            "track_id": ak.Array([[0, 1]]),
            "qop": ak.Array([[1.0 / 50.0, -1.0 / 40.0]]),
        }
        for i in range(5):
            for j in range(i + 1):
                tracks[f"cov_{i}_{j}"] = ak.Array(
                    [[0.001 if i == j else 0.0] * 2]
                )
        set_tracks_pid(tracks, "mu+")
        pvs = {
            "x": ak.Array([[0.0]]),
            "y": ak.Array([[0.0]]),
            "z": ak.Array([[0.0]]),
            "time": ak.Array([[0.0]]),
            "sigma_time": ak.Array([[0.05]]),
            "cov_0_0": ak.Array([[1e-4]]),
            "cov_1_0": ak.Array([[0.0]]),
            "cov_1_1": ak.Array([[1e-4]]),
        }
        cands = combine([tracks, tracks], pvs)
        # Should not crash and should NOT have mc_truth field
        self.assertNotIn("mc_truth", cands)

    def test_empty_candidates_have_mc_fields(self):
        """Empty candidates (all filtered) should still have MC truth fields."""
        from trackcomb import combine, apply_mask

        tracks, pvs = self._make_tracks_and_pvs()
        pos = apply_mask(tracks, tracks["charge"] > 0)
        neg = apply_mask(tracks, tracks["charge"] < 0)
        # Use impossibly tight cut to get 0 candidates
        cands = combine(
            [pos, neg],
            pvs,
            combination_cuts=[lambda c: c["mass"] > 9999000],
        )
        self.assertIsNone(
            cands, "Should return None when all filtered pre-fit"
        )


class TestBkgcatVectorized(unittest.TestCase):
    """Compare vectorized bkgcat against loop-based _bkgcat_loop."""

    def _make_tracks_and_pvs(self):
        """Reuse the same fixture from TestMCTruthPropagation."""
        from trackcomb.pid import set_tracks_pid

        rng = np.random.RandomState(99)
        tracks = {
            "x": ak.Array([rng.randn(4) * 0.1, rng.randn(2) * 0.1]),
            "y": ak.Array([rng.randn(4) * 0.1, rng.randn(2) * 0.1]),
            "z": ak.Array([[100.0] * 4, [100.0] * 2]),
            "tx": ak.Array([rng.randn(4) * 0.01, rng.randn(2) * 0.01]),
            "ty": ak.Array([rng.randn(4) * 0.01, rng.randn(2) * 0.01]),
            "p": ak.Array(
                [
                    (np.abs(rng.randn(4)) * 10 + 20).tolist(),
                    (np.abs(rng.randn(2)) * 10 + 20).tolist(),
                ]
            ),
            "charge": ak.Array([[1.0, -1.0, 1.0, -1.0], [1.0, -1.0]]),
            "time": ak.Array([[0.1] * 4, [0.1] * 2]),
            "sigma_time": ak.Array([[0.01] * 4, [0.01] * 2]),
            "track_id": ak.Array([[0, 1, 2, 3], [0, 1]]),
            "mc_truth": ak.Array([[1, 1, 1, 1], [1, 1]]),
            "mc_pid": ak.Array([[13, -13, 211, -211], [13, -13]]),
            "mc_key": ak.Array([[10, 11, 12, 13], [20, 21]]),
            "mc_pv_key": ak.Array([[0, 0, 0, 0], [1, 1]]),
            "mc_fromsignal": ak.Array([[1, 1, 0, 0], [1, 1]]),
            "mc_ancestor_pids": ak.Array(
                [
                    [[531, 999], [531, 999], [310, 888], [310, 888]],
                    [[531, 999], [531, 999]],
                ]
            ),
            "mc_ancestor_keys": ak.Array(
                [
                    [[100, 500], [100, 500], [200, 600], [200, 600]],
                    [[300, 700], [300, 700]],
                ]
            ),
        }
        tracks["qop"] = tracks["charge"] / tracks["p"]
        for i in range(5):
            for j in range(i + 1):
                val = 0.001 if i == j else 0.0
                tracks[f"cov_{i}_{j}"] = ak.Array([[val] * 4, [val] * 2])

        set_tracks_pid(tracks, "mu+")

        pvs = {
            "x": ak.Array([[0.0], [0.0]]),
            "y": ak.Array([[0.0], [0.0]]),
            "z": ak.Array([[0.0], [0.0]]),
            "time": ak.Array([[0.0], [0.0]]),
            "sigma_time": ak.Array([[0.05], [0.05]]),
            "cov_0_0": ak.Array([[1e-4], [1e-4]]),
            "cov_1_0": ak.Array([[0.0], [0.0]]),
            "cov_1_1": ak.Array([[1e-4], [1e-4]]),
        }
        return tracks, pvs

    def test_vectorized_matches_loop(self):
        """Vectorized bkgcat must produce identical results to _bkgcat_loop."""
        from trackcomb import apply_mask, combine, set_composite_pid

        tracks, pvs = self._make_tracks_and_pvs()
        pos = apply_mask(tracks, tracks["charge"] > 0)
        neg = apply_mask(tracks, tracks["charge"] < 0)
        cands = combine([pos, neg], pvs)
        set_composite_pid(cands, "B(s)0")

        compute_bkgcat(cands)
        cats_vec = ak.to_numpy(ak.flatten(cands["bkgcat"]))
        cats_loop = ak.to_numpy(ak.flatten(_bkgcat_loop(cands)))

        np.testing.assert_array_equal(cats_vec, cats_loop)

    def test_vectorized_matches_loop_ks(self):
        """Test with K(S)0 PID assignment."""
        from trackcomb import apply_mask, combine, set_composite_pid
        from trackcomb.pid import set_tracks_pid

        tracks, pvs = self._make_tracks_and_pvs()
        set_tracks_pid(tracks, "pi+")
        pos = apply_mask(tracks, tracks["charge"] > 0)
        neg = apply_mask(tracks, tracks["charge"] < 0)
        cands = combine([pos, neg], pvs)
        set_composite_pid(cands, "K(S)0")

        compute_bkgcat(cands)
        cats_vec = ak.to_numpy(ak.flatten(cands["bkgcat"]))
        cats_loop = ak.to_numpy(ak.flatten(_bkgcat_loop(cands)))

        np.testing.assert_array_equal(cats_vec, cats_loop)


if __name__ == "__main__":
    unittest.main()
