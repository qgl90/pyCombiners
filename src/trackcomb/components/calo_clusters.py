"""calo cluster containers: PicoCalClusters(+MC) and reconstructible clusters."""

from __future__ import annotations

import awkward as ak

from ..io import read, unflatten_2d
from ..models import Container
from ..configurable import configurable


@configurable
def load_calo_clusters(chunk: Container, mc=True, entries=False) -> Container:
    c: Container = {}

    c["x"] = read(chunk, "PicoCalClusters/cluster_x")
    c["y"] = read(chunk, "PicoCalClusters/cluster_y")
    c["z"] = read(chunk, "PicoCalClusters/cluster_z")
    c["e"] = read(chunk, "PicoCalClusters/cluster_e")
    c["time"] = read(chunk, "PicoCalClusters/cluster_t")
    c["front_x"] = read(chunk, "PicoCalClusters/cluster_front_x")
    c["front_y"] = read(chunk, "PicoCalClusters/cluster_front_y")
    c["front_z"] = read(chunk, "PicoCalClusters/cluster_front_z")
    c["front_e"] = read(chunk, "PicoCalClusters/cluster_front_e")
    c["front_time"] = read(chunk, "PicoCalClusters/cluster_front_t")
    c["back_x"] = read(chunk, "PicoCalClusters/cluster_back_x")
    c["back_y"] = read(chunk, "PicoCalClusters/cluster_back_y")
    c["back_z"] = read(chunk, "PicoCalClusters/cluster_back_z")
    c["back_e"] = read(chunk, "PicoCalClusters/cluster_back_e")
    c["back_time"] = read(chunk, "PicoCalClusters/cluster_back_t")
    c["type"] = read(chunk, "PicoCalClusters/cluster_type")
    c["n_entries"] = read(chunk, "PicoCalClusters/cluster_n_entries")

    # per-digit info inside each cluster (large; off by default)
    if entries:
        n_entries = c["n_entries"]
        for leaf in ("cellID", "status", "e", "fraction", "t", "x", "y", "z"):
            c[f"entry_{leaf.lower()}"] = unflatten_2d(
                read(chunk, f"PicoCalClusters/entry_{leaf}"), n_entries
            )

    if mc:
        n_matches = read(chunk, "PicoCalClustersMC/cluster_n_matches")
        c["mc_n_matches"] = n_matches
        c["mc_leaf_weight_sum"] = read(
            chunk, "PicoCalClustersMC/cluster_leaf_weight_sum"
        )
        c["mc_true_min_time"] = read(chunk, "PicoCalClustersMC/true_min_t")
        c["mc_true_max_time"] = read(chunk, "PicoCalClustersMC/true_max_t")
        c["mc_true_seed_time"] = read(chunk, "PicoCalClustersMC/true_seed_t")
        # per-match info (doubly jagged: cluster -> matched MC particle)
        for leaf, name in [
            ("match_weight", "mc_match_weight"),
            ("match_pid", "mc_match_pid"),
            ("match_key", "mc_match_key"),
            ("match_pv_key", "mc_match_pv_key"),
            ("match_charge", "mc_match_charge"),
            ("match_px", "mc_match_px"),
            ("match_py", "mc_match_py"),
            ("match_pz", "mc_match_pz"),
            ("match_pe", "mc_match_pe"),
            ("match_ovtx_x", "mc_match_ovtx_x"),
            ("match_ovtx_y", "mc_match_ovtx_y"),
            ("match_ovtx_z", "mc_match_ovtx_z"),
            ("match_ovtx_t", "mc_match_ovtx_time"),
            ("match_pv_x", "mc_match_pv_x"),
            ("match_pv_y", "mc_match_pv_y"),
            ("match_pv_z", "mc_match_pv_z"),
            ("match_pv_t", "mc_match_pv_time"),
        ]:
            c[name] = unflatten_2d(
                read(chunk, f"PicoCalClustersMC/{leaf}"), n_matches
            )
        c["mc_match_fromsignal"] = (
            unflatten_2d(
                read(chunk, "PicoCalClustersMC/match_fromSignal"), n_matches
            )
            != 0
        )
        # ancestors are triply jagged: cluster -> match -> ancestor.
        # Stored flat per event; expose per-match flat lists + sizes.
        match_n_anc = unflatten_2d(
            read(chunk, "PicoCalClustersMC/match_n_ancestors"), n_matches
        )
        c["mc_match_n_ancestors"] = match_n_anc
        flat_n_anc = ak.flatten(
            read(chunk, "PicoCalClustersMC/match_n_ancestors")
        )
        for leaf, name in [
            ("ancestor_pids", "mc_match_ancestor_pids"),
            ("ancestor_keys", "mc_match_ancestor_keys"),
        ]:
            flat = ak.flatten(read(chunk, f"PicoCalClustersMC/{leaf}"))
            per_match = ak.unflatten(flat, flat_n_anc)
            per_cluster = ak.unflatten(per_match, ak.flatten(n_matches))
            c[name] = ak.unflatten(per_cluster, ak.num(n_matches))

    c["_type"] = "calo_clusters"
    c["cluster_id"] = ak.local_index(c["x"], axis=1)
    return c


def load_reconstructible_calo_clusters(chunk: Container) -> Container:
    r: Container = {}

    for leaf in (
        "key",
        "pid",
        "pv_key",
        "charge",
        "px",
        "py",
        "pz",
        "pe",
        "deposit",
        "deposit_front",
        "deposit_back",
        "n_digits",
    ):
        r[leaf] = read(chunk, f"ReconstructiblePicoCalClusters/{leaf}")
    r["from_signal"] = (
        read(chunk, "ReconstructiblePicoCalClusters/fromSignal") != 0
    )
    for leaf, name in [
        ("ovtx_x", "ovtx_x"),
        ("ovtx_y", "ovtx_y"),
        ("ovtx_z", "ovtx_z"),
        ("ovtx_t", "ovtx_time"),
        ("pv_x", "pv_x"),
        ("pv_y", "pv_y"),
        ("pv_z", "pv_z"),
        ("pv_t", "pv_time"),
    ]:
        r[name] = read(chunk, f"ReconstructiblePicoCalClusters/{leaf}")

    n_anc = read(chunk, "ReconstructiblePicoCalClusters/n_ancestors")
    r["n_ancestors"] = n_anc
    r["ancestor_pids"] = unflatten_2d(
        read(chunk, "ReconstructiblePicoCalClusters/ancestor_pids"), n_anc
    )
    r["ancestor_keys"] = unflatten_2d(
        read(chunk, "ReconstructiblePicoCalClusters/ancestor_keys"), n_anc
    )

    r["_type"] = "reconstructible_calo_clusters"
    r["reconstructible_id"] = ak.local_index(r["pid"], axis=1)
    return r
