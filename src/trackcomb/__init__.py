"""Package exports."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("track-combination-framework")
except PackageNotFoundError:
    __version__ = "dev"

from .models import (
    Container,
    CutFunction,
    n_daughters,
    get_daughter,
    gather_daughters_stack,
    apply_mask,
    apply_cuts,
    cut_min,
    cut_max,
    cut_range,
    any_in_tree,
    all_in_tree,
    sum_in_tree,
    min_in_tree,
    max_in_tree,
    pick_inner,
    gather_jagged,
)
from .io import (
    event_stream,
    read,
    load_events,
    candidates_to_dataframe,
)
from .components import (
    load_tracks,
    load_pvs,
    load_event_info,
    load_reconstructible_tracks,
    load_calo_clusters,
    load_reconstructible_calo_clusters,
)
from .physics import (
    compute_track_pv_pairs,
    tracks_pv_association,
    compute_default_track_quantities,
    fit_track_t0,
    vertex_fit_3d,
    vertex_fit_3d_plus_time,
    composite_pv_association,
)
from .combiner import combine
from .truth import compute_bkgcat, count_reco_signal, count_true_decays

from .pid import (
    pdg_id,
    pdg_mass,
    set_composite_pid,
    set_tracks_pid,
)

from .plot import make_figure
from .configurable import configurable
from .counters import counters, rate_counters, print_counters
from .onnx import onnx_models
from .runner import run_reconstruction

__all__ = [
    # Container utilities
    "Container",
    "CutFunction",
    "n_daughters",
    "apply_mask",
    "apply_cuts",
    "cut_min",
    "cut_max",
    "cut_range",
    "get_daughter",
    "gather_daughters_stack",
    "any_in_tree",
    "all_in_tree",
    "sum_in_tree",
    "min_in_tree",
    "max_in_tree",
    "pick_inner",
    "gather_jagged",
    # IO
    "event_stream",
    "read",
    "load_events",
    "load_tracks",
    "load_pvs",
    "load_event_info",
    "load_reconstructible_tracks",
    "load_calo_clusters",
    "load_reconstructible_calo_clusters",
    "candidates_to_dataframe",
    # Physics (user-facing only)
    "compute_track_pv_pairs",
    "tracks_pv_association",
    "compute_default_track_quantities",
    "fit_track_t0",
    "vertex_fit_3d",
    "vertex_fit_3d_plus_time",
    "composite_pv_association",
    # Pipeline
    "combine",
    # Truth matching
    "compute_bkgcat",
    "count_reco_signal",
    "count_true_decays",
    # PID
    "pdg_id",
    "pdg_mass",
    "set_composite_pid",
    "set_tracks_pid",
    # Plotting
    "make_figure",
    # Runner
    "run_reconstruction",
    # Utilities
    "configurable",
    "counters",
    "rate_counters",
    "print_counters",
    "onnx_models",
]
