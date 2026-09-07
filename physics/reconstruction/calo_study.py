#!/usr/bin/env python3
"""Dump direct-Bs photons and their PicoCal clusters for the calo study.

One row per (photon, matched cluster), ordered by match weight; the
photon set comes from the cluster MC-match lists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from trackcomb import (
    counters,
    load_calo_clusters,
    load_event_info,
    run_reconstruction,
)

# cluster-level fields copied per matched cluster
_CLUSTER_FIELDS = [
    ("x", "cluster_x"),
    ("y", "cluster_y"),
    ("z", "cluster_z"),
    ("time", "cluster_time"),
    ("front_e", "cluster_front_e"),
    ("front_x", "cluster_front_x"),
    ("front_y", "cluster_front_y"),
    ("front_z", "cluster_front_z"),
    ("front_time", "cluster_front_time"),
    ("back_e", "cluster_back_e"),
    ("back_x", "cluster_back_x"),
    ("back_y", "cluster_back_y"),
    ("back_z", "cluster_back_z"),
    ("back_time", "cluster_back_time"),
    ("seed_cellid", "cluster_seed_cellid"),
]


def reconstruction(chunk):
    c = load_calo_clusters(chunk, entries=True)
    info = load_event_info(chunk)
    n_events = len(info["run_number"])

    # seed cell = highest-energy entry; its cellID carries the packed
    # geometry bits (FB / cellIdx / row / col / modType / area / calo)
    iseed = ak.argmax(c["entry_e"], axis=2, keepdims=True)
    c["seed_cellid"] = ak.values_astype(
        ak.fill_none(ak.firsts(c["entry_cellid"][iseed], axis=2), 0),
        np.int64,
    )

    w = c["mc_match_weight"]
    ib = ak.argmax(w, axis=2, keepdims=True)

    def best(field):
        return ak.firsts(c[field][ib], axis=2)

    best_key = ak.fill_none(best("mc_match_key"), -999999)
    best_w = ak.fill_none(best("mc_match_weight"), np.nan)
    best_pid = ak.fill_none(best("mc_match_pid"), 0)
    best_pe = ak.fill_none(best("mc_match_pe"), np.nan)
    first_anc = ak.fill_none(
        ak.firsts(ak.firsts(c["mc_match_ancestor_pids"][ib], axis=2), axis=-1),
        0,
    )
    is_dbs = (best_pid == 22) & (np.abs(first_anc) == 531)

    leaf = c["mc_leaf_weight_sum"]
    # the best match's origin vertex = the photon's own production vertex
    ovtx = {
        n: ak.fill_none(best(f"mc_match_ovtx_{s}"), np.nan)
        for s, n in [
            ("x", "gamma_ovtx_x"),
            ("y", "gamma_ovtx_y"),
            ("z", "gamma_ovtx_z"),
            ("time", "gamma_ovtx_time"),
        ]
    }
    parts = []
    for iev in range(n_events):
        sel = ak.to_numpy(is_dbs[iev])
        if not sel.any():
            continue
        bk = ak.to_numpy(best_key[iev])
        bw = ak.to_numpy(best_w[iev])
        bpe = ak.to_numpy(best_pe[iev])
        ce = ak.to_numpy(c["e"][iev])
        lf = ak.to_numpy(leaf[iev])
        cl = {n: ak.to_numpy(c[s][iev]) for s, n in _CLUSTER_FIELDS}
        ov = {n: ak.to_numpy(v[iev]) for n, v in ovtx.items()}
        for key in np.unique(bk[sel]):
            mine = bk == key
            order = np.argsort(-bw[mine])
            idx = np.where(mine)[0][order]
            n_cl = len(idx)
            parts.append(
                pd.DataFrame(
                    {
                        "run_number": np.repeat(info["run_number"][iev], n_cl),
                        "event_number": np.repeat(
                            info["event_number"][iev], n_cl
                        ),
                        "gamma_key": np.repeat(key, n_cl),
                        "gamma_e_true": np.repeat(bpe[idx[0]], n_cl),
                        "gamma_is_direct_bs": np.ones(n_cl, dtype=bool),
                        **{
                            n: np.repeat(v[idx[0]], n_cl)
                            for n, v in ov.items()
                        },
                        "cluster_e": ce[idx],
                        **{n: v[idx] for n, v in cl.items()},
                        "cluster_match_weight": bw[idx],
                        "cluster_purity": bw[idx]
                        / np.where(lf[idx] > 0, lf[idx], np.nan),
                        "cluster_rank": np.arange(n_cl, dtype=np.int64),
                        "n_matched_clusters": np.repeat(n_cl, n_cl),
                        "sum_matched_e": np.repeat(ce[idx].sum(), n_cl),
                    }
                )
            )
            counters("direct-Bs photons").add(1)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Dump direct-Bs photons from cluster MC-match lists"
    )
    parser.add_argument(
        "--input", required=True, help="ROOT file path (wildcards allowed)"
    )
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out-file",
        default="public/calo_study/reconstruction/bgamma_matches.parquet",
    )
    args = parser.parse_args()

    out_path = Path(args.out_file)
    run_reconstruction(
        reconstruction,
        input_data=args.input,
        out=out_path,
        max_events=args.max_events or None,
        chunk_size=args.chunk_size,
        workers=args.workers,
        print_throughput=True,
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
