#!/usr/bin/env python3
"""Signal vs background distribution comparison for reconstructed candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CHANNELS = {
    "ks_to_pipi": {
        "decay_latex": r"K_S^0 \to \pi^+\pi^-",
        "prefix": "ks",
        "observables": [
            ("mass", r"$m(\pi\pi)$ [MeV]"),
            ("vtx_chi2", r"Vertex $\chi^2$"),
            ("time_chi2", r"Pair time $\chi^2$"),
            ("doca", r"DOCA [mm]"),
            ("dira", r"DIRA"),
            ("mother_pt", r"$p_T(K_S^0)$ [MeV]"),
            ("mother_ip", r"$K_S^0$ IP [mm]"),
            ("mother_ip_chi2", r"$K_S^0$ IP $\chi^2$"),
            ("min_track_ip", r"min track IP [mm]"),
            ("max_track_ip", r"max track IP [mm]"),
            ("min_track_pt", r"min track $p_T$ [MeV]"),
            ("max_track_pt", r"max track $p_T$ [MeV]"),
        ],
    },
    "bs_to_mumu": {
        "decay_latex": r"B_s^0 \to \mu^+\mu^-",
        "prefix": "bs",
        "observables": [
            ("mass", r"$m(\mu\mu)$ [MeV]"),
            ("vtx_chi2", r"Vertex $\chi^2$"),
            ("time_chi2", r"Pair time $\chi^2$"),
            ("doca", r"DOCA [mm]"),
            ("dira", r"DIRA"),
            ("mother_pt", r"$p_T(B_s^0)$ [MeV]"),
            ("mother_ip", r"$B_s^0$ IP [mm]"),
            ("mother_ip_chi2", r"$B_s^0$ IP $\chi^2$"),
            ("min_track_ip", r"min track IP [mm]"),
            ("max_track_ip", r"max track IP [mm]"),
            ("min_track_pt", r"min track $p_T$ [MeV]"),
            ("max_track_pt", r"max track $p_T$ [MeV]"),
        ],
    },
    "bs_to_jpsiphi": {
        "decay_latex": r"B_s^0 \to J/\psi(\mu\mu)\,\phi(KK)",
        "prefix": "bs",
        "observables": [
            ("mass", r"$m(B_s^0)$ [MeV]"),
            ("jpsi_mass", r"$m(J/\psi)$ [MeV]"),
            ("phi_mass", r"$m(\phi)$ [MeV]"),
            ("vtx_chi2", r"$B_s^0$ vertex $\chi^2$"),
            ("jpsi_vtx_chi2", r"$J/\psi$ vertex $\chi^2$"),
            ("phi_vtx_chi2", r"$\phi$ vertex $\chi^2$"),
            ("time_chi2", r"$B_s^0$ pair time $\chi^2$"),
            ("doca", r"$B_s^0$ DOCA [mm]"),
            ("jpsi_doca", r"$J/\psi$ DOCA [mm]"),
            ("phi_doca", r"$\phi$ DOCA [mm]"),
            ("dira", r"DIRA"),
            ("mother_pt", r"$p_T(B_s^0)$ [MeV]"),
            ("mother_ip", r"$B_s^0$ IP [mm]"),
            ("mother_ip_chi2", r"$B_s^0$ IP $\chi^2$"),
            ("fdchi2", r"$B_s^0$ FD $\chi^2$"),
            ("min_track_pt", r"min track $p_T$ [MeV]"),
            ("max_track_pt", r"max track $p_T$ [MeV]"),
            ("min_track_ip", r"min track IP [mm]"),
            ("max_track_ip", r"max track IP [mm]"),
            ("min_track_ipchi2", r"min track IP $\chi^2$"),
        ],
    },
}


def _auto_range(all_values, lo_pct=1, hi_pct=99):
    """Compute plot range from percentiles of combined data."""
    lo = np.percentile(all_values, lo_pct)
    hi = np.percentile(all_values, hi_pct)
    margin = (hi - lo) * 0.05
    return lo - margin, hi + margin


def _leaf_daughter_cols(subset, suffix):
    """Find leaf-level daughter columns ending with suffix (skipping intermediates)."""
    cols = []
    for c in subset.columns:
        if not c.startswith("daughter") or not c.endswith(suffix):
            continue
        parts = c.split("_")
        n_daughter = sum(1 for p in parts if p.startswith("daughter"))
        if n_daughter >= 2:
            cols.append(c)
    if not cols:
        cols = sorted(
            c
            for c in subset.columns
            if c.startswith("daughter") and c.endswith(suffix)
        )
    return sorted(cols)


def _build_obs_arrays(subset):
    """Build dict of observable name -> numpy array for a DataFrame subset."""
    obs = {}
    obs["mass"] = subset["mass"].values
    obs["vtx_chi2"] = subset["vertex_chi2"].values
    if "pair_time_chi2" in subset.columns:
        obs["time_chi2"] = subset["pair_time_chi2"].values
    if "max_doca" in subset.columns:
        obs["doca"] = subset["max_doca"].values
    obs["dira"] = subset["dira"].fillna(0.0).values
    obs["mother_pt"] = subset["pt"].values
    obs["mother_ip"] = subset["composite_ip"].fillna(0.0).values
    obs["mother_ip_chi2"] = subset["composite_ip_chi2"].fillna(0.0).values
    if "fdchi2" in subset.columns:
        obs["fdchi2"] = subset["fdchi2"].fillna(0.0).values

    for prefix, label in [("daughter0", "jpsi"), ("daughter1", "phi")]:
        mass_col = f"{prefix}_mass"
        if mass_col in subset.columns:
            obs[f"{label}_mass"] = subset[mass_col].values
        chi2_col = f"{prefix}_vertex_chi2"
        if chi2_col in subset.columns:
            obs[f"{label}_vtx_chi2"] = subset[chi2_col].values
        doca_col = f"{prefix}_max_doca"
        if doca_col in subset.columns:
            obs[f"{label}_doca"] = subset[doca_col].values

    pt_cols = _leaf_daughter_cols(subset, "_pt")
    if pt_cols:
        daughter_pts = subset[pt_cols].values
        obs["min_track_pt"] = np.nanmin(daughter_pts, axis=1)
        obs["max_track_pt"] = np.nanmax(daughter_pts, axis=1)

    ip_cols = _leaf_daughter_cols(subset, "_min_ip")
    if ip_cols:
        daughter_ips = subset[ip_cols].values
        obs["min_track_ip"] = np.nanmin(daughter_ips, axis=1)
        obs["max_track_ip"] = np.nanmax(daughter_ips, axis=1)

    ipchi2_cols = _leaf_daughter_cols(subset, "_min_ip_chi2")
    if ipchi2_cols:
        daughter_ipchi2s = subset[ipchi2_cols].values
        obs["min_track_ipchi2"] = np.nanmin(daughter_ipchi2s, axis=1)

    return obs


def main():
    parser = argparse.ArgumentParser(
        description="Signal vs background distribution comparison",
    )
    parser.add_argument("--dist", required=True, help="Path to dist.parquet")
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for plots"
    )
    parser.add_argument(
        "--channel", required=True, choices=CHANNELS, help="Decay channel"
    )
    parser.add_argument(
        "--lumi", default="", help="Luminosity label for plot titles"
    )
    args = parser.parse_args()

    ch = CHANNELS[args.channel]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.read_parquet(args.dist)
    sig = df[df["is_signal"]].copy()
    bkg = df[~df["is_signal"]].copy()
    n_sig = len(sig)
    n_bkg = len(bkg)

    print(f"\n{'=' * 60}")
    print(f"Channel: {args.channel}")
    print(f"Signal:     {n_sig}")
    print(f"Background: {n_bkg}")
    print(
        f"Purity:     {n_sig}/{n_sig + n_bkg} = {n_sig / max(n_sig + n_bkg, 1) * 100:.1f}%"
    )
    print(f"{'=' * 60}")

    sig_obs = _build_obs_arrays(sig)
    bkg_obs = _build_obs_arrays(bkg)

    header = (
        f"{'observable':<20}  {'1%':>8}  {'5%':>8}  {'median':>8}  "
        f"{'95%':>8}  {'99%':>8}"
    )
    sep = "-" * 68

    for label, obs_dict, count in [
        ("Signal", sig_obs, n_sig),
        ("Background", bkg_obs, n_bkg),
    ]:
        if count == 0:
            continue
        print(f"\n{label} percentiles ({count}):")
        print(header)
        print(sep)
        for name in obs_dict:
            vals = obs_dict[name]
            p1, p5, p50, p95, p99 = np.percentile(vals, [1, 5, 50, 95, 99])
            print(
                f"{name:<20}  {p1:>8.4f}  {p5:>8.4f}  {p50:>8.4f}  "
                f"{p95:>8.4f}  {p99:>8.4f}"
            )

    import matplotlib.pyplot as plt
    from trackcomb.plot import make_figure

    # Filter to observables that exist in the data
    available = [
        (n, xl) for n, xl in ch["observables"] if n in sig_obs or n in bkg_obs
    ]
    n_vars = len(available)
    ncols = min(n_vars, 3)
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = make_figure(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    axes = list(axes.flatten())

    for ax, (name, xlabel) in zip(axes, available):
        sig_v = sig_obs.get(name, np.array([]))
        bkg_v = bkg_obs.get(name, np.array([]))
        xrange = (
            _auto_range(sig_v)
            if len(sig_v) > 0
            else (_auto_range(bkg_v) if len(bkg_v) > 0 else (0, 1))
        )
        bins = np.linspace(xrange[0], xrange[1], 51)
        if len(sig_v) > 0:
            ax.hist(
                sig_v,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.5,
                color="steelblue",
                label=f"Signal ({len(sig_v)})",
            )
        if len(bkg_v) > 0:
            ax.hist(
                bkg_v,
                bins=bins,
                density=True,
                histtype="step",
                color="red",
                linewidth=1.5,
                label=f"Bkg ({len(bkg_v)})",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalised")
        ax.set_xlim(xrange)
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    for ax in axes[n_vars:]:
        ax.set_visible(False)

    lumi_tag = f" [{args.lumi}]" if args.lumi else ""
    fig.suptitle(
        f"${ch['decay_latex']}$ signal vs background (normalised){lumi_tag}"
    )
    fig.tight_layout()
    prefix = ch["prefix"]
    fig.savefig(out_dir / f"{prefix}_observables.png", dpi=150)
    print(f"\nSaved {out_dir / f'{prefix}_observables.png'}")


if __name__ == "__main__":
    main()
