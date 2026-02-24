"""Particle-hypothesis helpers used in mass-assignment workflows.

This module exposes named hypothesis builders that can be used directly in the
combiner API instead of raw numeric masses.
"""

from __future__ import annotations

from .models import ParticleHypothesis

_PION = ParticleHypothesis(name="pi", mass=0.13957039, pdg_id=211)
_KAON = ParticleHypothesis(name="K", mass=0.493677, pdg_id=321)
_PROTON = ParticleHypothesis(name="p", mass=0.93827208816, pdg_id=2212)
_MUON = ParticleHypothesis(name="mu", mass=0.1056583755, pdg_id=13)
_ELECTRON = ParticleHypothesis(name="e", mass=0.00051099895, pdg_id=11)

_NAME_TO_HYPOTHESIS: dict[str, ParticleHypothesis] = {
    "pi": _PION,
    "pion": _PION,
    "k": _KAON,
    "kaon": _KAON,
    "p": _PROTON,
    "proton": _PROTON,
    "mu": _MUON,
    "muon": _MUON,
    "e": _ELECTRON,
    "electron": _ELECTRON,
}


def make_pion() -> ParticleHypothesis:
    """Return the standard charged-pion mass hypothesis."""
    return _PION


def make_kaon() -> ParticleHypothesis:
    """Return the standard charged-kaon mass hypothesis."""
    return _KAON


def make_proton() -> ParticleHypothesis:
    """Return the proton mass hypothesis."""
    return _PROTON


def make_muon() -> ParticleHypothesis:
    """Return the muon mass hypothesis."""
    return _MUON


def make_electron() -> ParticleHypothesis:
    """Return the electron mass hypothesis."""
    return _ELECTRON


def particle_hypothesis_from_name(name: str) -> ParticleHypothesis:
    """Resolve a short particle name (e.g. `pi`, `kaon`) into a hypothesis."""
    key = name.strip().lower()
    try:
        return _NAME_TO_HYPOTHESIS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_NAME_TO_HYPOTHESIS))
        raise ValueError(
            f"Unknown particle hypothesis name '{name}'. Supported names: {supported}"
        ) from exc
