"""Plotting utilities for candidate distributions and signal/background comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .models import CombinationResult


def plot_mass(
    signal: Sequence[CombinationResult],
    background: Sequence[CombinationResult],
    *,
    pdg_mass: float | None = None,
    xlabel: str = r"$m$ [GeV]",
    title: str = "Invariant mass",
    mass_range: tuple[float, float] | None = None,
    bins: int = 50,
    out_path: str | Path | None = None,
):
    """Plot signal vs background mass distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig_m = [c.candidate_p4.mass for c in signal]
    bkg_m = [c.candidate_p4.mass for c in background]

    fig, ax = plt.subplots(figsize=(8, 5))
    r = mass_range
    if bkg_m:
        ax.hist(bkg_m, bins=bins, range=r, histtype="stepfilled",
                alpha=0.4, color="grey", label=f"Background ({len(bkg_m)})")
    if sig_m:
        ax.hist(sig_m, bins=bins, range=r, histtype="stepfilled",
                alpha=0.8, color="red", label=f"Signal ({len(sig_m)})")
    if pdg_mass is not None:
        ax.axvline(pdg_mass, color="black", linestyle="--", linewidth=1,
                   label=f"PDG ({pdg_mass*1e3:.1f} MeV)")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Candidates", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig, ax


def plot_distributions(
    signal: Sequence[CombinationResult],
    background: Sequence[CombinationResult],
    observables: Sequence[tuple[str, Callable, str, tuple[float, float], int]],
    *,
    title: str = "Signal vs Background (normalised)",
    out_path: str | Path | None = None,
):
    """Multi-panel signal vs background comparison.

    Parameters
    ----------
    observables : sequence of (name, extractor, xlabel, range, bins)
        Each entry defines one panel.  ``extractor`` is a callable
        ``CombinationResult -> float``.

    Example
    -------
    >>> obs = [
    ...     ("mass", lambda c: c.candidate_p4.mass, r"$m$ [GeV]", (0.4, 0.6), 50),
    ...     ("vtx_chi2", lambda c: c.vertex_chi2, r"Vertex $\\chi^2$", (0, 25), 50),
    ... ]
    >>> plot_distributions(sig, bkg, obs, out_path="dist.png")
    """
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_obs = len(observables)
    n_cols = min(3, n_obs)
    n_rows = math.ceil(n_obs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_obs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (name, extractor, xlabel, xrange, nbins) in enumerate(observables):
        ax = axes[i]
        sig_vals = [extractor(c) for c in signal]
        bkg_vals = [extractor(c) for c in background]
        if bkg_vals:
            ax.hist(bkg_vals, bins=nbins, range=xrange, histtype="step",
                    linewidth=1.5, color="grey", label="Bkg", density=True)
        if sig_vals:
            ax.hist(sig_vals, bins=nbins, range=xrange, histtype="step",
                    linewidth=2, color="red", label="Sig", density=True)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Norm.", fontsize=11)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
        ax.grid(True, alpha=0.3)

    for j in range(n_obs, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_efficiency_vs_cut(
    signal: Sequence[CombinationResult],
    background: Sequence[CombinationResult],
    n_true: int,
    extractor: Callable[[CombinationResult], float],
    cut_values: Sequence[float],
    *,
    xlabel: str = "Cut value",
    title: str = "Efficiency vs Purity scan",
    out_path: str | Path | None = None,
):
    """Plot efficiency and purity vs a sliding cut value.

    Candidates with ``extractor(c) <= cut`` are accepted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    effs, purs = [], []
    for cut in cut_values:
        ns = sum(1 for c in signal if extractor(c) <= cut)
        nb = sum(1 for c in background if extractor(c) <= cut)
        eff = ns / max(n_true, 1) * 100
        pur = ns / max(ns + nb, 1) * 100
        effs.append(eff)
        purs.append(pur)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(cut_values, effs, "o-", color="blue", label="Efficiency")
    ax1.set_xlabel(xlabel, fontsize=13)
    ax1.set_ylabel("Efficiency [%]", color="blue", fontsize=13)
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(cut_values, purs, "s--", color="red", label="Purity")
    ax2.set_ylabel("Purity [%]", color="red", fontsize=13)
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_title(title, fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig, (ax1, ax2)
