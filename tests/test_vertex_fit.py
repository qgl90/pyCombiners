"""Regression test for vertex_fit_3d against per-candidate reference."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import pytest

from trackcomb.physics import vertex_fit_3d

FIXTURE = (
    Path(__file__).resolve().parent / "input" / "test_vertex_fit_reference.npz"
)

COV_KEYS = [
    "cov_0_0",
    "cov_1_0",
    "cov_1_1",
    "cov_2_0",
    "cov_2_1",
    "cov_2_2",
    "cov_3_0",
    "cov_3_1",
    "cov_3_2",
    "cov_3_3",
]


@pytest.fixture
def ref():
    if not FIXTURE.exists():
        pytest.skip(
            "fixture not generated — run scripts/generate_vertex_fit_fixture.py"
        )
    return np.load(FIXTURE)


def _build_comb(ref):
    """Build a minimal comb dict from the fixture arrays."""
    # ref arrays have shape (N_candidates, n_daughters)
    N, n_body = ref["x"].shape
    pools = []
    for k in range(n_body):
        pool = {}
        for field in ["x", "y", "z", "tx", "ty"] + COV_KEYS:
            pool[field] = ak.Array([ref[field][:, k]])
        pools.append(pool)

    comb = {"_daughter_pools": pools}
    for k in range(n_body):
        comb[f"daughter{k}_global_index"] = np.arange(N)
    return comb


class TestVertexFitReference:
    """Verify batch vertex_fit_3d against per-candidate np.linalg reference."""

    def _run(self, ref):
        comb = _build_comb(ref)
        vertex_fit_3d(comb)
        return comb

    def test_vertex_position(self, ref):
        comb = self._run(ref)
        xyz = np.column_stack(
            [comb["vertex_x"], comb["vertex_y"], comb["vertex_z"]]
        )
        np.testing.assert_allclose(xyz, ref["vertex_xyz"], rtol=1e-10)

    def test_spatial_chi2(self, ref):
        comb = self._run(ref)
        np.testing.assert_allclose(
            comb["vertex_chi2"], ref["spatial_chi2"], rtol=1e-10
        )

    def test_vertex_covariance(self, ref):
        comb = self._run(ref)
        cov = np.zeros((len(comb["vertex_x"]), 3, 3))
        for i in range(3):
            for j in range(i + 1):
                cov[:, i, j] = comb[f"vertex_cov_{i}_{j}"]
                cov[:, j, i] = comb[f"vertex_cov_{i}_{j}"]
        np.testing.assert_allclose(cov, ref["vertex_cov"], rtol=1e-10)
