"""Plotting helpers with LHCb2 style and package watermark."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

import trackcomb

LABEL = f"pyCombiners v{trackcomb.__version__}"


def make_figure(*args, **kwargs):
    """Create a figure with LHCb2 style and package watermark."""
    hep.style.use("LHCb2")
    kwargs.setdefault("figsize", (16, 12))
    fig, ax = plt.subplots(*args, **kwargs)
    fig.text(
        0.02,
        0.98,
        LABEL,
        fontsize=14,
        color="gray",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    return fig, ax
