"""PDG lookup and mass hypothesis assignment."""

from __future__ import annotations

import awkward as ak
from particle import Particle
from .configurable import configurable


def _lookup_particle(particle_id):
    """Resolve particle name or int to a Particle object."""
    if isinstance(particle_id, int):
        return Particle.from_pdgid(particle_id)
    matches = Particle.findall(particle_id)
    if len(matches) != 1:
        raise ValueError(
            f"Particle name {particle_id!r} matched {len(matches)} particles "
            f"(expected exactly 1). Matches: {matches}"
        )
    return matches[0]


def pdg_id(particle_id) -> int:
    """Look up signed PDG ID from a particle name or pass through int."""
    if isinstance(particle_id, int):
        return particle_id
    return int(_lookup_particle(particle_id).pdgid)


def pdg_mass(particle_id) -> float:
    """Look up particle mass in MeV from a name or PDG ID."""
    return _lookup_particle(particle_id).mass


@configurable
def set_tracks_pid(tracks, particle_id, fit_track_time=True):
    """Add mass, pid fields and fit t0 for a given particle hypothesis."""
    from .physics import fit_track_t0

    p = _lookup_particle(particle_id)
    shape_like = ak.ones_like(tracks["x"])
    tracks["mass"] = shape_like * p.mass
    tracks["pid"] = shape_like * int(p.pdgid)

    if fit_track_time and ("tvhits_z" in tracks) and ("tvhits_t" in tracks):
        fit_track_t0(tracks)


def set_composite_pid(candidates, particle_id):
    """Set the mother particle PDG ID on candidates."""
    candidates["pid"] = pdg_id(particle_id)
