"""Package exports."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("track-combination-framework")
except PackageNotFoundError:
    __version__ = "dev"

from .models import (
    Container,
    CutFunction,
    infer_n_body,
    apply_mask,
    apply_cuts,
    cut_min,
    cut_max,
    cut_range,
    pick_along_inner,
)
from .io import (
    load_tracks_root,
    load_pvs_root,
    load_events_root,
    iter_events_root,
    candidates_to_parquet,
    candidates_to_dataframe,
    extract_daughter_fields,
)
from .physics import (
    ip_to_pvs,
    flight_corrected_dt,
    tracks_pv_association,
)
from .combiner import combine
from .truth import bkgcat, count_true_decays, truth_match_candidates

from .pid import (
    pdg_id,
    set_tracks_pid,
)

from .plot import make_figure

__all__ = [
    # Container utilities
    "Container",
    "CutFunction",
    "infer_n_body",
    "apply_mask",
    "apply_cuts",
    "cut_min",
    "cut_max",
    "cut_range",
    "pick_along_inner",
    # IO
    "load_tracks_root",
    "load_pvs_root",
    "load_events_root",
    "iter_events_root",
    "candidates_to_parquet",
    "candidates_to_dataframe",
    "extract_daughter_fields",
    # Physics (user-facing only)
    "ip_to_pvs",
    "flight_corrected_dt",
    "tracks_pv_association",
    # Pipeline
    "combine",
    # Truth matching
    "bkgcat",
    "count_true_decays",
    "truth_match_candidates",
    # PID
    "pdg_id",
    "set_tracks_pid",
    # Plotting
    "make_figure",
]
