"""Stepwise decay-chain examples using composite candidates as track-like inputs.

Demonstrated chains:
- B -> J/psi(mu mu) K
- B -> J/psi(mu mu) Phi(K K)

The key abstraction is `combination_to_track_state`, which converts a
combination output into a new `TrackState` carrying:
- fitted `(x, y, z, time)`,
- inferred slopes `(tx, ty)` from candidate momentum,
- approximate `cov4`,
- propagated source-track provenance.
"""

from __future__ import annotations
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


from pathlib import Path

from trackcomb import (
    CombinationCuts,
    ParticleCombiner,
    ParticleHypothesis,
    combination_to_track_state,
    make_kaon,
    make_muon,
)
from trackcomb.io import (
    load_primary_vertices_json,
    load_tracks_json,
    write_results_table,
)

M_JPSI = 3.0969
M_PHI = 1.019461


def _track_map(tracks):
    """Map track_id to TrackState."""
    return {t.track_id: t for t in tracks}


def _as_composite_tracks(candidates, base_tracks, prefix):
    """Convert accepted combinations into track-like composite objects."""
    tmap = _track_map(base_tracks)
    out = []
    for i, cand in enumerate(candidates):
        constituents = [tmap[tid] for tid in cand.track_ids]
        out.append(combination_to_track_state(cand, constituents, f"{prefix}_{i}"))
    return out


def main() -> int:
    """Run stepwise examples and write B-candidate tables."""
    tracks = load_tracks_json("examples/tracks.json")
    pvs = load_primary_vertices_json("examples/primary_vertex.json")
    combiner = ParticleCombiner()

    # 1) Build J/psi -> mu+ mu-
    jpsi_results = combiner.combine(
        tracks=tracks,
        primary_vertices=pvs,
        n_body=2,
        mass_hypotheses=[[make_muon(), make_muon()]],
        cuts=CombinationCuts(
            allowed_charge_patterns=("+-", "-+"),
            min_mass=2.8,
            max_mass=3.4,
            max_doca=5.0,
        ),
    )
    jpsi_tracks = _as_composite_tracks(jpsi_results, tracks, "JPSI")

    # Kaon-like tracks (very simple placeholder PID rule).
    kaon_tracks = [t for t in tracks if t.rich_dll_k > t.rich_dll_pi]

    # 2) B -> J/psi K (stage-2 uses composite J/psi as one track-like input)
    b_to_jpsi_k = []
    for jpsi in jpsi_tracks:
        for kaon in kaon_tracks:
            stage = combiner.combine(
                tracks=[jpsi, kaon],
                primary_vertices=pvs,
                n_body=2,
                mass_hypotheses=[[ParticleHypothesis(name="J/psi", mass=M_JPSI), make_kaon()]],
                cuts=CombinationCuts(min_mass=4.8, max_mass=5.8, max_doca=5.0),
            )
            b_to_jpsi_k.extend(stage)

    # 3) Phi -> K K
    phi_results = combiner.combine(
        tracks=kaon_tracks,
        primary_vertices=pvs,
        n_body=2,
        mass_hypotheses=[[make_kaon(), make_kaon()]],
        cuts=CombinationCuts(
            allowed_charge_patterns=("+-", "-+"),
            min_mass=0.98,
            max_mass=1.06,
            max_doca=5.0,
        ),
    )
    phi_tracks = _as_composite_tracks(phi_results, kaon_tracks, "PHI")

    # 4) B -> J/psi Phi (composite + composite stage)
    b_to_jpsi_phi = []
    for jpsi in jpsi_tracks:
        for phi in phi_tracks:
            stage = combiner.combine(
                tracks=[jpsi, phi],
                primary_vertices=pvs,
                n_body=2,
                mass_hypotheses=[
                    [
                        ParticleHypothesis(name="J/psi", mass=M_JPSI),
                        ParticleHypothesis(name="phi", mass=M_PHI),
                    ]
                ],
                cuts=CombinationCuts(min_mass=4.8, max_mass=5.8, max_doca=5.0),
            )
            b_to_jpsi_phi.extend(stage)

    out_dir = Path("examples")
    write_results_table(out_dir / "b_to_jpsi_k.parquet", b_to_jpsi_k)
    write_results_table(out_dir / "b_to_jpsi_phi.parquet", b_to_jpsi_phi)
    print(f"Wrote {len(b_to_jpsi_k)} B->J/psi K candidates to {out_dir / 'b_to_jpsi_k.parquet'}")
    print(f"Wrote {len(b_to_jpsi_phi)} B->J/psi Phi candidates to {out_dir / 'b_to_jpsi_phi.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
