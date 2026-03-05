"""Unit tests for truth matching functions."""

from __future__ import annotations

__author__ = "Renato Quagliani <rquaglia@cern.ch>"


import unittest

import awkward as ak
import numpy as np

from trackcomb.truth import bkgcat, count_true_decays, truth_match_candidates


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


class TestBatchTruthMatchCandidates(unittest.TestCase):
    """Test truth_match_candidates with synthetic candidates."""

    def _make_candidates(self, tracks):
        """Fake candidates: evt0 has 1 signal + 2 bkg, evt1 has 1 signal + 1 bkg."""
        return {
            "vertex_x": ak.Array([[0.0, 0.0, 0.0], [0.0, 0.0]]),
            "daughter0_track_id": ak.Array([[0, 0, 0], [0, 0]]),
            "daughter1_track_id": ak.Array([[1, 3, 2], [1, 2]]),
            # pool_index is the local position within the pool (same as
            # track_id here since both daughters draw from the same full pool)
            "daughter0_pool_index": ak.Array([[0, 0, 0], [0, 0]]),
            "daughter1_pool_index": ak.Array([[1, 3, 2], [1, 2]]),
            # Daughter PIDs (pi+ hypothesis for both daughters, PDG 211)
            "daughter0_pid": ak.Array([[211, 211, 211], [211, 211]]),
            "daughter1_pid": ak.Array([[211, 211, 211], [211, 211]]),
            # Mother PDG (K_S^0 = 310)
            "pid": ak.Array([[310, 310, 310], [310, 310]]),
            # Reference to the underlying track pools
            "_daughter_pools": [tracks, tracks],
        }

    def test_truth_match_with_daughters(self):
        """Signal candidates are correctly identified."""
        tracks = _make_tracks_with_mc()
        candidates = self._make_candidates(tracks)
        matched = truth_match_candidates(candidates)
        # Event 0: cand0=True, cand1=False, cand2=False
        self.assertTrue(matched[0][0])
        self.assertFalse(matched[0][1])
        self.assertFalse(matched[0][2])
        # Event 1: cand0=True, cand1=False
        self.assertTrue(matched[1][0])
        self.assertFalse(matched[1][1])

    def test_truth_match_without_daughters(self):
        """Without daughter pid fields, only common ancestor matters."""
        tracks = _make_tracks_with_mc()
        candidates = self._make_candidates(tracks)
        # Remove daughter{k}_pid so daughter check is skipped
        del candidates["daughter0_pid"]
        del candidates["daughter1_pid"]
        matched = truth_match_candidates(candidates)
        # Event 0: cand0=True (shared Ks100), cand1=False, cand2=False (diff keys)
        self.assertTrue(matched[0][0])
        self.assertFalse(matched[0][1])
        self.assertFalse(matched[0][2])

    def test_returns_jagged_bool(self):
        tracks = _make_tracks_with_mc()
        candidates = self._make_candidates(tracks)
        matched = truth_match_candidates(candidates)
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
            "daughter0_pool_index": ak.Array([[], []]),
            "daughter1_pool_index": ak.Array([[], []]),
            "daughter0_pid": ak.Array([[], []]),
            "daughter1_pid": ak.Array([[], []]),
            "pid": ak.Array([[], []]),
            "_daughter_pools": [tracks, tracks],
        }
        matched = truth_match_candidates(candidates)
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
        """Build a single-event candidates container."""
        n = len(daughter0_idx)
        return {
            "vertex_x": ak.Array([daughter0_idx]),  # dummy, just needs right length
            "daughter0_pool_index": ak.Array([daughter0_idx]),
            "daughter1_pool_index": ak.Array([daughter1_idx]),
            "daughter0_pid": ak.Array([[d0_pid] * n]),
            "daughter1_pid": ak.Array([[d1_pid] * n]),
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
        cats = bkgcat(cands)
        self.assertEqual(int(cats[0][0]), 0)

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
        cats = bkgcat(cands)
        self.assertEqual(int(cats[0][0]), 10)

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
        cats = bkgcat(cands)
        self.assertEqual(int(cats[0][0]), 20)

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
        cats = bkgcat(cands)
        self.assertEqual(int(cats[0][0]), 30)

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
        cats = bkgcat(cands)
        self.assertEqual(int(cats[0][0]), 60)

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
        cats = bkgcat(cands)
        self.assertEqual(int(cats[0][0]), 63)

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
        cats = bkgcat(cands)
        self.assertEqual(int(cats[0][0]), 66)

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
        cats = bkgcat(cands)
        self.assertEqual(int(cats[0][0]), 100)

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
        cats = bkgcat(cands)
        self.assertEqual(int(cats[0][0]), 110)

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
        cats = bkgcat(cands)
        self.assertEqual(int(cats[0][0]), 120)

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
        cats = bkgcat(cands)
        self.assertEqual(int(cats[0][0]), 130)


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
        for i in range(4):
            for j in range(i + 1):
                val = 0.001 if i == j else 0.0
                tracks[f"cov_{i}_{j}"] = ak.Array([[val] * 4, [val] * 2])

        tracks = set_tracks_pid(tracks, "mu+")

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
        cands = combine([pos, neg], pvs, use_timing=False)
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
        cands = combine([pos, neg], pvs, use_timing=False)

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
        cands = combine([pos, neg], pvs, use_timing=False)

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
        cands = combine([pos, neg], pvs, use_timing=False)

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
        cands = combine([pos, neg], pvs, use_timing=False)

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
        from trackcomb import combine, apply_mask, pdg_id

        tracks, pvs = self._make_tracks_and_pvs()
        pos = apply_mask(tracks, tracks["charge"] > 0)
        neg = apply_mask(tracks, tracks["charge"] < 0)
        cands = combine([pos, neg], pvs, use_timing=False)
        cands["pid"] = pdg_id("B(s)0")

        cats = bkgcat(cands)
        flat_cats = ak.to_numpy(ak.flatten(cats))
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
        }
        for i in range(4):
            for j in range(i + 1):
                tracks[f"cov_{i}_{j}"] = ak.Array([[0.001 if i == j else 0.0] * 2])
        tracks = set_tracks_pid(tracks, "mu+")
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
        cands = combine([tracks, tracks], pvs, use_timing=False)
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
            [pos, neg], pvs, use_timing=False, vertex_cuts=[lambda c: c["mass"] > 9999]
        )
        for field in (
            "mc_truth",
            "mc_pid",
            "mc_key",
            "mc_pv_key",
            "mc_fromsignal",
            "mc_ancestor_pids",
            "mc_ancestor_keys",
        ):
            self.assertIn(field, cands, f"Missing MC field in empty: {field}")
        self.assertEqual(len(cands["mc_truth"][0]), 0)


if __name__ == "__main__":
    unittest.main()
