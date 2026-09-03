"""Shared helpers for PID performance analysis."""

from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except ImportError:
    plt = None
    LogNorm = None

import mplhep as hep

hep.style.use("LHCb2")

PDG_FALLBACK = {
    "e-": 11,
    "e+": -11,
    "mu-": 13,
    "mu+": -13,
    "pi+": 211,
    "pi-": -211,
    "K+": 321,
    "K-": -321,
    "p": 2212,
    "p+": 2212,
    "pbar": -2212,
    "anti-p": -2212,
    "d+": 1000010020,
    "deuteron": 1000010020,
}

DEFAULT_BINS: dict[str, np.ndarray] = {
    "p": np.linspace(0, 100000, 50),
    "pt": np.linspace(0, 10000, 50),
    "eta": np.linspace(1.5, 6.5, 50),
}

DEFAULT_LABELS = {
    "p": r"$p$ [MeV/$c$]",
    "pt": r"$p_\mathrm{T}$ [MeV/$c$]",
    "eta": r"$\eta$",
}

DEFAULT_COLORS = {
    0.01: "C4",
    0.03: "C3",
    0.05: "C0",
    0.10: "C1",
    0.15: "C2",
}

SAMPLE_COLORS = [
    "#222222",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
]
SAMPLE_MARKERS = ["o", "s", "D", "^", "v", "P"]


def resolve_pdg_id(name_or_id, pdg_id_fn: Callable | None = None) -> int:
    """Resolve a particle name or integer to a signed PDG ID."""
    if isinstance(name_or_id, (int, np.integer)):
        return int(name_or_id)
    if pdg_id_fn is not None:
        try:
            return int(pdg_id_fn(name_or_id))
        except Exception:
            pass
    if name_or_id in PDG_FALLBACK:
        return PDG_FALLBACK[name_or_id]
    raise ValueError(
        f"Cannot resolve PDG id for {name_or_id!r}. "
        "Pass an integer or a pdg_id() callable."
    )


def binom_err(p: float, n: int) -> float:
    if n <= 0 or not np.isfinite(p):
        return np.nan
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n))


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def species_label(name_or_id) -> str:
    return str(name_or_id).replace("+", "").replace("-", "").replace("~", "")


def require_matplotlib() -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting")
