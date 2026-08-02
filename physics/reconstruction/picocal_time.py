"""PicoCal per-(area x section) time alignment constants.

Measured on the 100k Bs_phigamma sample (gamma_time_resolution.py);
production-version specific. Delete once aligned upstream.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

# alignment constants [ns], indexed by cellID area (1-7; index 0 unused)
_BIAS_FRONT = np.array(
    [np.nan, 1.264, 1.261, 1.258, 1.255, 1.281, 1.276, 1.341]
)
_BIAS_BACK = np.array(
    [np.nan, 0.951, 0.946, 0.946, 0.948, 1.092, 1.094, 1.259]
)

# single-section aligned resolutions [ns] -> weighted-mean weights
SIGMA_FRONT = 0.050
SIGMA_BACK = 0.036
# measured estimator resolutions [ns] (DSCB, gamma_time_aligned.py)
SIGMA_WMEAN = 0.039
SIGMA_FRONT_ONLY = 0.051
SIGMA_BACK_ONLY = 0.042


def add_seed_area(clusters):
    """Add seed cellID, area and per-section time-alignment bias to a
    calo cluster container (requires load_calo_clusters(entries=True)).

    The seed cell is the highest-energy entry; area = cellID bits 24-26.
    Clusters without entries get area 0 -> NaN bias -> NaN photon time.
    """
    iseed = ak.argmax(clusters["entry_e"], axis=2, keepdims=True)
    cid = ak.values_astype(
        ak.fill_none(ak.firsts(clusters["entry_cellid"][iseed], axis=2), 0),
        np.int64,
    )
    clusters["seed_cellid"] = cid
    area = (cid >> 24) & 0x7
    clusters["area"] = area
    flat_area = ak.to_numpy(ak.flatten(area))
    counts = ak.num(area)
    clusters["time_bias_front"] = ak.unflatten(_BIAS_FRONT[flat_area], counts)
    clusters["time_bias_back"] = ak.unflatten(_BIAS_BACK[flat_area], counts)
    return clusters
