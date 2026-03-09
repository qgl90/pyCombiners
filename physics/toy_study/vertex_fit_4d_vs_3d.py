#!/usr/bin/env python3
"""4D vs 3D vertex fit: toy Monte Carlo comparison across 2-body decay scenarios.

Usage: PYTHONPATH=src python physics/toy_study/vertex_fit_4d_vs_3d.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

C_LIGHT = 299.792458  # mm/ns


def _vertex_fit_4d_reference(
    x, y, z_ref, t, tx, ty, dt_dz, sigma_x, sigma_y, sigma_t, n_iter=3
):
    """Single-candidate weighted 4D vertex fit, returns (state, cov_4x4, chi2)."""
    n = len(tx)

    stx, sty = np.sum(tx), np.sum(ty)
    st2 = np.sum(tx**2 + ty**2)
    rx, ry = x - tx * z_ref, y - ty * z_ref
    ATA3 = np.array([[n, 0, -stx], [0, n, -sty], [-stx, -sty, st2]])
    ATb3 = np.array([np.sum(rx), np.sum(ry), np.sum(-tx * rx - ty * ry)])
    v3d = np.linalg.solve(ATA3, ATb3)
    z_v = v3d[2]

    for _ in range(n_iter):
        ATA_4 = np.zeros((4, 4))
        ATb_4 = np.zeros(4)
        for i in range(n):
            W = np.diag([1 / sigma_x**2, 1 / sigma_y**2, 1 / sigma_t**2])
            H = np.array(
                [
                    [1, 0, -tx[i], 0],
                    [0, 1, -ty[i], 0],
                    [0, 0, -dt_dz[i], 1],
                ]
            )
            d = np.array(
                [
                    x[i] - tx[i] * z_ref[i],
                    y[i] - ty[i] * z_ref[i],
                    t[i] - dt_dz[i] * z_ref[i],
                ]
            )
            ATA_4 += H.T @ W @ H
            ATb_4 += H.T @ W @ d
        state = np.linalg.solve(ATA_4, ATb_4)
        z_v = state[2]

    cov_4x4 = np.linalg.inv(ATA_4)

    dz = z_v - z_ref
    chi2 = 0.0
    for i in range(n):
        r = np.array(
            [
                x[i] + tx[i] * dz[i] - state[0],
                y[i] + ty[i] * dz[i] - state[1],
                t[i] + dz[i] * dt_dz[i] - state[3],
            ]
        )
        W = np.diag([1 / sigma_x**2, 1 / sigma_y**2, 1 / sigma_t**2])
        chi2 += r @ W @ r

    return state, cov_4x4, chi2


def _vertex_fit_3d_reference(x, y, z_ref, tx, ty, sigma_x, sigma_y):
    """Single-candidate 3D vertex fit, returns (xyz, cov_3x3)."""
    n = len(tx)
    stx, sty = np.sum(tx), np.sum(ty)
    st2 = np.sum(tx**2 + ty**2)
    rx, ry = x - tx * z_ref, y - ty * z_ref
    ATA = np.array([[n, 0, -stx], [0, n, -sty], [-stx, -sty, st2]])
    ATb = np.array([np.sum(rx), np.sum(ry), np.sum(-tx * rx - ty * ry)])
    xyz = np.linalg.solve(ATA, ATb)

    ATA_w = np.zeros((3, 3))
    for i in range(n):
        W2 = np.diag([1 / sigma_x**2, 1 / sigma_y**2])
        H2 = np.array([[1, 0, -tx[i]], [0, 1, -ty[i]]])
        ATA_w += H2.T @ W2 @ H2
    cov = np.linalg.inv(ATA_w)
    return xyz, cov


def run_scenario(
    name, tx, ty, p, mass, sigma_x, sigma_y, sigma_t, n_toys=5000, rng=None
):
    """Run toy MC for one scenario. Returns dict of results."""
    if rng is None:
        rng = np.random.default_rng(42)

    true_xv, true_yv, true_zv, true_tv = 0.5, -0.3, 300.0, 5.0
    z_ref = np.array([50.0, 80.0])

    sf = np.sqrt(1 + tx**2 + ty**2)
    energy = np.sqrt(p**2 + mass**2)
    beta = p / energy
    dt_dz = sf / (beta * C_LIGHT)
    da_ratio = abs(dt_dz[0] - dt_dz[1]) / dt_dz.mean() * 100

    sigma_t_comb = np.sqrt(2) * sigma_t
    da = abs(dt_dz[0] - dt_dz[1])
    sigma_z_time = sigma_t_comb / da if da > 0 else np.inf

    dz_true = z_ref - true_zv
    x_true = true_xv + tx * dz_true
    y_true = true_yv + ty * dz_true
    t_true = true_tv + dz_true * dt_dz

    true_3d = np.array([true_xv, true_yv, true_zv])
    true_4d = np.array([true_xv, true_yv, true_zv, true_tv])

    pulls_3d = np.zeros((n_toys, 3))
    pulls_4d = np.zeros((n_toys, 4))
    dz_3d = np.zeros(n_toys)
    dz_4d = np.zeros(n_toys)
    chi2_4d = np.zeros(n_toys)

    for toy in range(n_toys):
        x_m = x_true + rng.normal(0, sigma_x, 2)
        y_m = y_true + rng.normal(0, sigma_y, 2)
        t_m = t_true + rng.normal(0, sigma_t, 2)

        # 3D
        xyz, cov3 = _vertex_fit_3d_reference(
            x_m,
            y_m,
            z_ref,
            tx,
            ty,
            sigma_x,
            sigma_y,
        )
        dz_3d[toy] = xyz[2] - true_zv
        for k in range(3):
            pulls_3d[toy, k] = (xyz[k] - true_3d[k]) / np.sqrt(cov3[k, k])

        # 4D
        state, cov4, c2 = _vertex_fit_4d_reference(
            x_m,
            y_m,
            z_ref,
            t_m,
            tx,
            ty,
            dt_dz,
            sigma_x,
            sigma_y,
            sigma_t,
        )
        dz_4d[toy] = state[2] - true_zv
        for k in range(4):
            pulls_4d[toy, k] = (state[k] - true_4d[k]) / np.sqrt(cov4[k, k])
        chi2_4d[toy] = c2

    return {
        "name": name,
        "da_ratio": da_ratio,
        "sigma_z_time": sigma_z_time,
        "sigma_z_3d": np.std(dz_3d),
        "sigma_z_4d": np.std(dz_4d),
        "pulls_3d": pulls_3d,
        "pulls_4d": pulls_4d,
        "chi2_4d_mean": np.mean(chi2_4d),
        "improvement_pct": (1 - np.std(dz_4d) / np.std(dz_3d)) * 100,
    }


SCENARIOS = [
    # (name, tx, ty, p [MeV], mass [MeV])
    (
        "B->mumu, large opening",
        np.array([0.1, -0.15]),
        np.array([0.2, 0.05]),
        np.array([10000, 15000]),
        np.array([105.66, 105.66]),
    ),
    (
        "B->mumu, small opening",
        np.array([0.1, 0.105]),
        np.array([0.2, 0.205]),
        np.array([10000, 15000]),
        np.array([105.66, 105.66]),
    ),
    (
        "K->pipi, large opening",
        np.array([0.1, -0.15]),
        np.array([0.2, 0.05]),
        np.array([5000, 8000]),
        np.array([139.57, 139.57]),
    ),
    (
        "K->pipi, small opening",
        np.array([0.1, 0.105]),
        np.array([0.2, 0.205]),
        np.array([5000, 8000]),
        np.array([139.57, 139.57]),
    ),
    (
        "Lambda->p pi, large opening",
        np.array([0.1, -0.15]),
        np.array([0.2, 0.05]),
        np.array([3000, 8000]),
        np.array([938.27, 139.57]),
    ),
    (
        "Lambda->p pi, small opening",
        np.array([0.1, 0.105]),
        np.array([0.2, 0.205]),
        np.array([3000, 8000]),
        np.array([938.27, 139.57]),
    ),
    (
        "Lambda->p pi, low momentum",
        np.array([0.1, 0.103]),
        np.array([0.2, 0.202]),
        np.array([2000, 3000]),
        np.array([938.27, 139.57]),
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", default=None, help="Save plots to this directory"
    )
    parser.add_argument("--n-toys", type=int, default=5000)
    args = parser.parse_args()

    sigma_x, sigma_y, sigma_t = 0.05, 0.05, 0.025  # mm, mm, ns
    rng = np.random.default_rng(42)

    results = []
    for name, tx, ty, p, mass in SCENARIOS:
        r = run_scenario(
            name,
            tx,
            ty,
            p,
            mass,
            sigma_x,
            sigma_y,
            sigma_t,
            n_toys=args.n_toys,
            rng=rng,
        )
        results.append(r)

    print("=" * 120)
    print("4D vs 3D vertex fit: toy Monte Carlo comparison")
    print(
        f"Track uncertainties: sigma_x = sigma_y = {sigma_x * 1000:.0f} um, "
        f"sigma_t = {sigma_t * 1000:.0f} ps, {args.n_toys} toys per scenario"
    )
    print("=" * 120)
    pull_3d_labels = ["x", "y", "z"]
    pull_4d_labels = ["x", "y", "z", "t"]
    pull_3d_hdr = " | ".join(f"3D_{l:>1s}" for l in pull_3d_labels)
    pull_4d_hdr = " | ".join(f"4D_{l:>1s}" for l in pull_4d_labels)
    print(
        f"{'Scenario':<35s} | {'D(dt/dz)':>8s} | {'sig_z_time':>10s} | "
        f"{'3D sig_z':>9s} | {'4D sig_z':>9s} | {'Improve':>8s} | "
        f"{pull_3d_hdr} | {pull_4d_hdr} | {'4D chi2':>7s}"
    )
    print("-" * 160)

    for r in results:
        sz_time = (
            f"{r['sigma_z_time']:.1f}mm" if r["sigma_z_time"] < 1e6 else "inf"
        )
        p3 = " | ".join(
            f"{np.std(r['pulls_3d'][:, k]):>4.3f}" for k in range(3)
        )
        p4 = " | ".join(
            f"{np.std(r['pulls_4d'][:, k]):>4.3f}" for k in range(4)
        )
        print(
            f"{r['name']:<35s} | {r['da_ratio']:>7.1f}% | {sz_time:>10s} | "
            f"{r['sigma_z_3d']:>8.4f}mm | {r['sigma_z_4d']:>8.4f}mm | "
            f"{r['improvement_pct']:>7.1f}% | "
            f"{p3} | {p4} | "
            f"{r['chi2_4d_mean']:>7.2f}"
        )

    print("=" * 120)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 7))
        names = [r["name"] for r in results]
        x_pos = np.arange(len(names))
        improvements = [r["improvement_pct"] for r in results]
        colors = ["C0" if v >= 0 else "C3" for v in improvements]
        ax.bar(x_pos, improvements, color=colors, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_ylabel("4D z resolution improvement [%]")
        ax.set_title(
            "4D vs 3D vertex fit: relative z improvement\n"
            f"(sigma_x = sigma_y = {sigma_x * 1000:.0f} um, "
            f"sigma_t = {sigma_t * 1000:.0f} ps)"
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "z_resolution_comparison.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved {out_dir / 'z_resolution_comparison.png'}")

        fig, ax = plt.subplots(figsize=(10, 6))
        da_vals = [r["da_ratio"] for r in results]
        imp_vals = [r["improvement_pct"] for r in results]
        ax.scatter(da_vals, imp_vals, s=80, zorder=5)
        for r in results:
            ax.annotate(
                r["name"].split(",")[0],
                (r["da_ratio"], r["improvement_pct"]),
                fontsize=8,
                ha="left",
                va="bottom",
            )
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel("Relative difference in flight time per mm [%]")
        ax.set_ylabel("4D z improvement [%]")
        ax.set_title(
            "4D vertex fit z improvement vs daughter speed difference"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "improvement_vs_speed_diff.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / 'improvement_vs_speed_diff.png'}")

        from scipy.stats import norm

        bins = np.linspace(-5, 5, 61)
        gauss_x = np.linspace(-5, 5, 200)
        gauss_y = norm.pdf(gauss_x)

        n_sc = len(results)
        fig, axes = plt.subplots(n_sc * 2, 4, figsize=(18, 4 * n_sc))

        labels_3d = ["x", "y", "z"]
        labels_4d = ["x", "y", "z", "t"]

        for i, r in enumerate(results):
            for k, label in enumerate(labels_3d):
                ax = axes[2 * i, k]
                pulls = r["pulls_3d"][:, k]
                mu, std = np.mean(pulls), np.std(pulls)
                ax.hist(pulls, bins=bins, density=True, alpha=0.7)
                ax.plot(gauss_x, gauss_y, "r-", lw=1.5)
                ax.set_title(
                    f"{r['name']}\n3D pull({label}): "
                    f"mean={mu:.3f}, std={std:.3f}",
                    fontsize=9,
                )
                ax.set_xlim(-5, 5)
            axes[2 * i, 3].set_visible(False)

            for k, label in enumerate(labels_4d):
                ax = axes[2 * i + 1, k]
                pulls = r["pulls_4d"][:, k]
                mu, std = np.mean(pulls), np.std(pulls)
                ax.hist(pulls, bins=bins, density=True, alpha=0.7)
                ax.plot(gauss_x, gauss_y, "r-", lw=1.5)
                ax.set_title(
                    f"{r['name']}\n4D pull({label}): "
                    f"mean={mu:.3f}, std={std:.3f}",
                    fontsize=9,
                )
                ax.set_xlim(-5, 5)

        fig.suptitle(
            "Pull distributions: should follow N(0,1) if fit is correct",
            fontsize=13,
            y=1.005,
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / "pull_distributions.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"Saved {out_dir / 'pull_distributions.png'}")


if __name__ == "__main__":
    main()
