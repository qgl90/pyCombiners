"""Public package exports for the particle-combination framework."""
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


from .combiner import ParticleCombiner, TrackCombiner
from .composite import combination_to_track_state
from .decay import Decay, combine, make_decay
from .io import candidates_to_dataframe, iter_events_root, load_events_root, load_pvs_root, load_tracks_root
from .models import (
    CombinationCuts,
    CombinationResult,
    EventInput,
    LorentzVector,
    ParticleHypothesis,
    PrimaryVertex,
    TrackPreselection,
    TrackState,
)
from .overlap import has_shared_tracks, remove_overlaps
from .pid import (
    make_electron,
    make_kaon,
    make_muon,
    make_pion,
    make_proton,
    particle_hypothesis_from_name,
)
from .truth import (
    count_true_decays,
    filter_tracks_by_ancestor,
    get_true_decay_groups,
    truth_match,
    truth_match_candidates,
)
from .utils import best_candidate, filter_candidates

__all__ = [
    # Core
    "ParticleCombiner",
    "TrackCombiner",
    "combination_to_track_state",
    # Decay + combine
    "Decay",
    "make_decay",
    "combine",
    # Models
    "TrackState",
    "PrimaryVertex",
    "EventInput",
    "LorentzVector",
    "CombinationResult",
    "ParticleHypothesis",
    "TrackPreselection",
    "CombinationCuts",
    # PID
    "make_pion",
    "make_kaon",
    "make_proton",
    "make_muon",
    "make_electron",
    "particle_hypothesis_from_name",
    # I/O
    "candidates_to_dataframe",
    "load_tracks_root",
    "load_pvs_root",
    "iter_events_root",
    "load_events_root",
    # Truth
    "truth_match",
    "truth_match_candidates",
    "count_true_decays",
    "filter_tracks_by_ancestor",
    "get_true_decay_groups",
    # Overlap
    "has_shared_tracks",
    "remove_overlaps",
    # Utils
    "filter_candidates",
    "best_candidate",
]
