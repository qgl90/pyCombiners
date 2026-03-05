"""Regression test for vertex_fit_xyz against per-candidate reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trackcomb.physics import vertex_fit_xyz

FIXTURE = Path(__file__).resolve().parent / "input" / "test_vertex_fit_reference.npz"

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


class TestVertexFitReference:
    """Verify batch vertex_fit_xyz against per-candidate np.linalg reference."""

    def _run(self, ref):
        cov_fields = {k: ref[k] for k in COV_KEYS}
        return vertex_fit_xyz(
            ref["x"], ref["y"], ref["z"], ref["tx"], ref["ty"], cov_fields
        )

    def test_vertex_position(self, ref):
        xyz, _, _ = self._run(ref)
        np.testing.assert_allclose(xyz, ref["vertex_xyz"], rtol=1e-10)

    def test_spatial_chi2(self, ref):
        _, chi2, _ = self._run(ref)
        np.testing.assert_allclose(chi2, ref["spatial_chi2"], rtol=1e-10)

    def test_vertex_covariance(self, ref):
        _, _, cov = self._run(ref)
        np.testing.assert_allclose(cov, ref["vertex_cov"], rtol=1e-10)
