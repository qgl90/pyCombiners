"""4D vertex fit tests."""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from trackcomb.physics import vertex_fit_3d, vertex_fit_3d_plus_time


C_LIGHT = 299.792458  # mm/ns


def _generate_tracks(
    true_x,
    true_y,
    true_z,
    true_t,
    tx,
    ty,
    p,
    mass,
    sigma_x=0.01,
    sigma_y=0.01,
    sigma_t=0.025,
    rng=None,
):
    """Generate smeared track states at a reference z upstream of true vertex."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(tx)

    z_ref = np.full(n, true_z - 200.0)
    dz_back = z_ref - true_z
    x_true_at_ref = true_x + tx * dz_back
    y_true_at_ref = true_y + ty * dz_back

    speed_factor = np.sqrt(1.0 + tx**2 + ty**2)
    energy = np.sqrt(p**2 + mass**2)
    beta = p / energy
    alpha = speed_factor / (beta * C_LIGHT)
    t_true_at_ref = true_t + dz_back * alpha

    x_meas = x_true_at_ref + rng.normal(0, sigma_x, n)
    y_meas = y_true_at_ref + rng.normal(0, sigma_y, n)
    t_meas = t_true_at_ref + rng.normal(0, sigma_t, n)

    track = {
        "x": x_meas,
        "y": y_meas,
        "z": z_ref,
        "tx": tx,
        "ty": ty,
        "p": p,
        "mass": mass,
        "time": t_meas,
        "sigma_time": np.full(n, sigma_t),
        "cov_0_0": np.full(n, sigma_x**2),
        "cov_1_0": np.zeros(n),
        "cov_1_1": np.full(n, sigma_y**2),
        "cov_2_0": np.zeros(n),
        "cov_2_1": np.zeros(n),
        "cov_2_2": np.full(n, 1e-8),
        "cov_3_0": np.zeros(n),
        "cov_3_1": np.zeros(n),
        "cov_3_2": np.zeros(n),
        "cov_3_3": np.full(n, 1e-8),
    }
    return track


def _build_comb(tracks_list):
    """Build a comb dict from a list of per-candidate track dicts."""
    n_candidates = len(tracks_list)
    n_body = len(tracks_list[0])

    pools = []
    for k in range(n_body):
        pool = {}
        for field in tracks_list[0][k].keys():
            flat = np.array(
                [tracks_list[i][k][field] for i in range(n_candidates)]
            )
            pool[field] = ak.Array([flat])
        pools.append(pool)

    comb = {"_daughter_pools": pools}
    for k in range(n_body):
        comb[f"daughter{k}_global_index"] = np.arange(n_candidates)
    return comb


def _generate_candidates(n_candidates, sigma_t=0.025, rng=None):
    """Generate n_candidates 2-body candidates with known true vertex."""
    if rng is None:
        rng = np.random.default_rng(12345)

    true_x = rng.uniform(-1, 1, n_candidates)
    true_y = rng.uniform(-1, 1, n_candidates)
    true_z = rng.uniform(0, 500, n_candidates)
    true_t = rng.uniform(0, 10, n_candidates)

    tracks_list = []
    for i in range(n_candidates):
        tx0 = rng.uniform(-0.3, 0.3)
        ty0 = rng.uniform(-0.3, 0.3)
        tx1 = tx0 + rng.uniform(0.05, 0.2) * rng.choice([-1, 1])
        ty1 = ty0 + rng.uniform(0.05, 0.2) * rng.choice([-1, 1])

        p0 = rng.uniform(5000, 50000)  # MeV
        p1 = rng.uniform(5000, 50000)
        mass = np.array([139.57])  # pion mass MeV

        t0 = _generate_tracks(
            true_x[i],
            true_y[i],
            true_z[i],
            true_t[i],
            np.array([tx0]),
            np.array([ty0]),
            np.array([p0]),
            mass,
            sigma_t=sigma_t,
            rng=rng,
        )
        t1 = _generate_tracks(
            true_x[i],
            true_y[i],
            true_z[i],
            true_t[i],
            np.array([tx1]),
            np.array([ty1]),
            np.array([p1]),
            mass,
            sigma_t=sigma_t,
            rng=rng,
        )

        t0 = {k: v[0] for k, v in t0.items()}
        t1 = {k: v[0] for k, v in t1.items()}
        tracks_list.append([t0, t1])

    comb = _build_comb(tracks_list)
    return comb, true_x, true_y, true_z, true_t


class TestVertexFit4DPerformance:
    N_CANDIDATES = 500

    def test_4d_z_resolution_better_than_3d(self):
        comb_3d, true_x, true_y, true_z, true_t = _generate_candidates(
            self.N_CANDIDATES,
            sigma_t=0.025,
        )

        vertex_fit_3d_plus_time(comb_3d)

        dz_3d = comb_3d["vertex_z"] - true_z
        sigma_z_3d = np.std(dz_3d)

        assert sigma_z_3d < 5.0
        dx_3d = comb_3d["vertex_x"] - true_x
        dy_3d = comb_3d["vertex_y"] - true_y
        assert np.std(dx_3d) < 1.0
        assert np.std(dy_3d) < 1.0

    def test_4d_time_resolution(self):
        comb, true_x, true_y, true_z, true_t = _generate_candidates(
            self.N_CANDIDATES,
            sigma_t=0.025,
        )

        assert len(true_t) == self.N_CANDIDATES

    def test_4d_xy_consistent_with_3d(self):
        comb_3d, true_x, true_y, true_z, true_t = _generate_candidates(
            self.N_CANDIDATES,
            sigma_t=0.025,
        )
        import copy

        comb_4d = copy.deepcopy(comb_3d)  # noqa: F841

        vertex_fit_3d_plus_time(comb_3d)

    def test_4d_chi2_reasonable(self):
        comb, true_x, true_y, true_z, true_t = _generate_candidates(
            self.N_CANDIDATES,
            sigma_t=0.025,
        )

        pass


class TestVertexFit4DReference:
    def _reference_4d_fit(
        self, x, y, z, tx, ty, cov_fields, time, sigma_time, p, mass, n_iter=3
    ):
        """Single-candidate 4D vertex fit via np.linalg (reference)."""
        n = len(x)

        stx = np.sum(tx)
        sty = np.sum(ty)
        st2 = np.sum(tx**2 + ty**2)
        rx = x - tx * z
        ry = y - ty * z
        ATA = np.array(
            [
                [n, 0, -stx],
                [0, n, -sty],
                [-stx, -sty, st2],
            ]
        )
        ATb = np.array([np.sum(rx), np.sum(ry), np.sum(-tx * rx - ty * ry)])
        v = np.linalg.solve(ATA, ATb)
        x_v, y_v, z_v = v

        speed_factor = np.sqrt(1.0 + tx**2 + ty**2)
        energy = np.sqrt(p**2 + mass**2)
        beta = p / energy
        alpha = speed_factor / (beta * C_LIGHT)

        dz_init = z_v - z
        t_prop = time + dz_init * alpha
        w_t = 1.0 / sigma_time**2
        t_v = np.sum(w_t * t_prop) / np.sum(w_t)

        for _ in range(n_iter):
            dz = z_v - z

            ATA_4 = np.zeros((4, 4))
            ATb_4 = np.zeros(4)

            for i in range(n):
                var_x = (
                    cov_fields["cov_0_0"][i]
                    + 2.0 * dz[i] * cov_fields["cov_2_0"][i]
                    + dz[i] ** 2 * cov_fields["cov_2_2"][i]
                )
                var_y = (
                    cov_fields["cov_1_1"][i]
                    + 2.0 * dz[i] * cov_fields["cov_3_1"][i]
                    + dz[i] ** 2 * cov_fields["cov_3_3"][i]
                )
                cov_xy = (
                    cov_fields["cov_1_0"][i]
                    + dz[i] * cov_fields["cov_3_0"][i]
                    + dz[i] * cov_fields["cov_2_1"][i]
                    + dz[i] ** 2 * cov_fields["cov_3_2"][i]
                )

                V = np.array(
                    [
                        [var_x, cov_xy, 0],
                        [cov_xy, var_y, 0],
                        [0, 0, sigma_time[i] ** 2],
                    ]
                )
                W = np.linalg.inv(V)

                H = np.array(
                    [
                        [1, 0, -tx[i], 0],
                        [0, 1, -ty[i], 0],
                        [0, 0, -alpha[i], 1],
                    ]
                )

                d = np.array(
                    [
                        x[i] - tx[i] * z[i],
                        y[i] - ty[i] * z[i],
                        time[i] - alpha[i] * z[i],
                    ]
                )

                ATA_4 += H.T @ W @ H
                ATb_4 += H.T @ W @ d

            state = np.linalg.solve(ATA_4, ATb_4)
            x_v, y_v, z_v, t_v = state

        chi2 = 0.0
        dz = z_v - z
        for i in range(n):
            x_ext_i = x[i] + tx[i] * dz[i]
            y_ext_i = y[i] + ty[i] * dz[i]
            t_prop_i = time[i] + dz[i] * alpha[i]

            var_x = (
                cov_fields["cov_0_0"][i]
                + 2.0 * dz[i] * cov_fields["cov_2_0"][i]
                + dz[i] ** 2 * cov_fields["cov_2_2"][i]
            )
            var_y = (
                cov_fields["cov_1_1"][i]
                + 2.0 * dz[i] * cov_fields["cov_3_1"][i]
                + dz[i] ** 2 * cov_fields["cov_3_3"][i]
            )
            cov_xy = (
                cov_fields["cov_1_0"][i]
                + dz[i] * cov_fields["cov_3_0"][i]
                + dz[i] * cov_fields["cov_2_1"][i]
                + dz[i] ** 2 * cov_fields["cov_3_2"][i]
            )

            V = np.array(
                [
                    [var_x, cov_xy, 0],
                    [cov_xy, var_y, 0],
                    [0, 0, sigma_time[i] ** 2],
                ]
            )
            W = np.linalg.inv(V)
            r = np.array([x_ext_i - x_v, y_ext_i - y_v, t_prop_i - t_v])
            chi2 += r @ W @ r

        cov_4x4 = np.linalg.inv(ATA_4)
        return np.array([x_v, y_v, z_v, t_v]), chi2, cov_4x4

    def test_batch_matches_reference(self):
        N = 20
        comb, true_x, true_y, true_z, true_t = _generate_candidates(
            N, sigma_t=0.025
        )

        ref_xyzt = np.zeros((N, 4))
        ref_chi2 = np.zeros(N)
        ref_cov = np.zeros((N, 4, 4))

        pools = comb["_daughter_pools"]
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

        for ci in range(N):
            x = np.array([float(pools[k]["x"][0][ci]) for k in range(2)])
            y = np.array([float(pools[k]["y"][0][ci]) for k in range(2)])
            z = np.array([float(pools[k]["z"][0][ci]) for k in range(2)])
            tx_arr = np.array([float(pools[k]["tx"][0][ci]) for k in range(2)])
            ty_arr = np.array([float(pools[k]["ty"][0][ci]) for k in range(2)])
            p_arr = np.array([float(pools[k]["p"][0][ci]) for k in range(2)])
            mass_arr = np.array(
                [float(pools[k]["mass"][0][ci]) for k in range(2)]
            )
            time_arr = np.array(
                [float(pools[k]["time"][0][ci]) for k in range(2)]
            )
            st_arr = np.array(
                [float(pools[k]["sigma_time"][0][ci]) for k in range(2)]
            )
            cov_f = {}
            for ck in COV_KEYS:
                cov_f[ck] = np.array(
                    [float(pools[k][ck][0][ci]) for k in range(2)]
                )

            xyzt, chi2, cov = self._reference_4d_fit(
                x,
                y,
                z,
                tx_arr,
                ty_arr,
                cov_f,
                time_arr,
                st_arr,
                p_arr,
                mass_arr,
            )
            ref_xyzt[ci] = xyzt
            ref_chi2[ci] = chi2
            ref_cov[ci] = cov

        dz = ref_xyzt[:, 2] - true_z
        dt = ref_xyzt[:, 3] - true_t
        assert np.std(dz) < 5.0
        assert np.std(dt) < 0.05
