"""pvs container: PVState/PVMC branches -> PV fields."""

from __future__ import annotations

import awkward as ak

from ..io import read, read_float64
from ..models import Container
from ..configurable import configurable

_COV4_LOWER_TRI = [(i, j) for i in range(4) for j in range(i + 1)]


@configurable
def load_pvs(chunk: Container, mc=True) -> Container:
    p: Container = {}

    p["x"] = read_float64(chunk, "PVState/x")
    p["y"] = read_float64(chunk, "PVState/y")
    p["z"] = read_float64(chunk, "PVState/z")
    p["time"] = read_float64(chunk, "PVState/t")
    p["ndof"] = read(chunk, "PVState/ndof")
    p["chi2ndof"] = read(chunk, "PVState/chi2ndof")
    for i, j in _COV4_LOWER_TRI:
        p[f"cov_{i}_{j}"] = read_float64(chunk, f"PVState/cov_{i}_{j}")

    if mc:
        p["mc_key"] = read(chunk, "PVMC/key")
        p["mc_x"] = read(chunk, "PVMC/x")
        p["mc_y"] = read(chunk, "PVMC/y")
        p["mc_z"] = read(chunk, "PVMC/z")
        p["mc_time"] = read(chunk, "PVMC/t")

    p["sigma_time"] = ak.where(
        p["cov_3_3"] > 0.0, abs(p["cov_3_3"]) ** 0.5, 0.0
    )

    p["_type"] = "pvs"
    p["pv_index"] = ak.local_index(p["x"], axis=1)
    return p
