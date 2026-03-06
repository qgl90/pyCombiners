"""PDG lookup and mass hypothesis assignment."""

from __future__ import annotations


def pdg_id(particle_id) -> int:
    """Look up signed PDG ID from a particle name or pass through int."""
    from particle import Particle

    if isinstance(particle_id, int):
        return particle_id
    matches = Particle.findall(particle_id)
    if len(matches) != 1:
        raise ValueError(
            f"Particle name {particle_id!r} matched {len(matches)} particles "
            f"(expected exactly 1). Matches: {matches}"
        )
    return int(matches[0].pdgid)


def set_tracks_pid(tracks, particle_id):
    """Add mass and pid fields to tracks for a given particle hypothesis."""
    import awkward as ak
    from particle import Particle

    if isinstance(particle_id, int):
        p = Particle.from_pdgid(particle_id)
    else:
        matches = Particle.findall(particle_id)
        if len(matches) != 1:
            raise ValueError(
                f"Particle name {particle_id!r} matched {len(matches)} particles "
                f"(expected exactly 1). Matches: {matches}"
            )
        p = matches[0]

    mass_mev = p.mass  # MeV (native LHCb unit)
    pdg_id = int(p.pdgid)

    shape_like = ak.ones_like(tracks["x"])
    out = {**tracks}
    out["mass"] = shape_like * mass_mev
    out["pid"] = shape_like * pdg_id
    return out
