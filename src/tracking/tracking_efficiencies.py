#!/usr/bin/env python3
"""Produce and plot tracking efficiency and ghost/fake-rate data."""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from trackcomb import (
    counters,
    load_event_info,
    load_reconstructible_tracks,
    load_tracks,
    run_reconstruction,
)


COMMON_TAGS = (
    "from_signal",
    "positive_charge",
    "negative_charge",
    "from_beauty",
    "from_charm",
)

TRACK_TYPES = {
    "long": ("has_velo", "has_t"),
    "down": ("has_ut", "has_t"),
    "longft": ("has_velo", "has_ft"),
    "longmp": ("has_velo", "has_mp"),
}


def _flat(values):
    return np.asarray(ak.flatten(values))


def _compute_track_kinematics(tracks):
    """Compute only the reconstructed kinematics needed by this study."""
    qop = tracks["qop"]
    tracks["p"] = 1.0 / abs(qop)
    tracks["charge"] = ak.where(qop > 0, 1, -1)
    tx, ty = tracks["tx"], tracks["ty"]
    norm = np.sqrt(1.0 + tx**2 + ty**2)
    dz = 1.0 / norm
    tracks["pt"] = tracks["p"] * np.sqrt(tx**2 + ty**2) / norm
    tracks["eta"] = 0.5 * np.log((1.0 + dz) / np.maximum(1.0 - dz, 1e-30))
    return tracks


def _from_flavour(ancestor_pids, quark):
    """Whether an ancestor PDG ID contains the requested heavy quark."""
    apid = abs(ancestor_pids)
    is_hadron = apid >= 100
    contains = (
        ((apid // 10) % 10 == quark)
        | ((apid // 100) % 10 == quark)
        | ((apid // 1000) % 10 == quark)
    )
    return ak.any(is_hadron & contains, axis=2)


def _repeat_event(values, counts):
    return np.repeat(np.asarray(values), np.asarray(counts))


def _reconstructible_frame(reconstructible, event_info):
    counts = ak.to_numpy(ak.num(reconstructible["p"]))
    charge = _flat(reconstructible["charge"])
    n_rows = int(counts.sum())
    frame = pd.DataFrame(
        {
            "row_type": np.repeat("reconstructible", n_rows),
            "run_number": _repeat_event(event_info["run_number"], counts),
            "event_number": _repeat_event(event_info["event_number"], counts),
            "object_index": _flat(reconstructible["reconstructible_id"]),
            "mc_key": np.full(n_rows, -1, dtype=np.int64),
            "truth_matched": np.zeros(n_rows, dtype=bool),
            "is_unique_truth_match": np.zeros(n_rows, dtype=bool),
            "truth_pid": _flat(reconstructible["pid"]),
            "truth_p": _flat(reconstructible["p"]),
            "truth_pt": _flat(reconstructible["pt"]),
            "truth_eta": _flat(reconstructible["eta"]),
            "truth_phi": _flat(reconstructible["phi"]),
            "reco_p": np.full(n_rows, np.nan),
            "reco_pt": np.full(n_rows, np.nan),
            "reco_eta": np.full(n_rows, np.nan),
            "reco_chi2ndof": np.full(n_rows, np.nan),
            "from_signal": _flat(reconstructible["from_signal"]),
            "has_velo": _flat(reconstructible["has_velo"]),
            "has_ut": _flat(reconstructible["has_ut"]),
            "has_mp": _flat(reconstructible["has_mp"]),
            "has_ft": _flat(reconstructible["has_ft"]),
            "has_t": _flat(reconstructible["has_t"]),
            "positive_charge": charge > 0,
            "negative_charge": charge < 0,
            "from_beauty": _flat(reconstructible["from_beauty"]),
            "from_charm": _flat(reconstructible["from_charm"]),
            "from_strange_reconstructible": _flat(
                reconstructible["from_strange"]
            ),
        }
    )
    _set_track_type_tags(frame)
    return frame


def _long_track_frame(tracks, event_info):
    counts = ak.to_numpy(ak.num(tracks["p"]))
    n_rows = int(counts.sum())
    truth_matched = _flat(tracks["mc_truth"]).astype(bool)
    mc_px = _flat(tracks["mc_px"])
    mc_py = _flat(tracks["mc_py"])
    mc_pz = _flat(tracks["mc_pz"])
    truth_p = np.sqrt(mc_px**2 + mc_py**2 + mc_pz**2)
    truth_pt = np.sqrt(mc_px**2 + mc_py**2)
    truth_eta = np.arcsinh(mc_pz / np.maximum(truth_pt, 1e-30))
    truth_phi = np.arctan2(mc_py, mc_px)
    mc_charge = _flat(tracks["mc_charge"])

    frame = pd.DataFrame(
        {
            "row_type": np.repeat("long", n_rows),
            "run_number": _repeat_event(event_info["run_number"], counts),
            "event_number": _repeat_event(event_info["event_number"], counts),
            "object_index": _flat(tracks["track_id"]),
            "mc_key": _flat(tracks["mc_key"]),
            "truth_matched": truth_matched,
            "is_unique_truth_match": np.zeros(n_rows, dtype=bool),
            "truth_pid": _flat(tracks["mc_pid"]),
            "truth_p": np.where(truth_matched, truth_p, np.nan),
            "truth_pt": np.where(truth_matched, truth_pt, np.nan),
            "truth_eta": np.where(truth_matched, truth_eta, np.nan),
            "truth_phi": np.where(truth_matched, truth_phi, np.nan),
            "reco_p": _flat(tracks["p"]),
            "reco_pt": _flat(tracks["pt"]),
            "reco_eta": _flat(tracks["eta"]),
            "reco_chi2ndof": _flat(tracks["chi2ndof"]),
            "from_signal": truth_matched & _flat(tracks["mc_fromsignal"]),
            "has_velo": truth_matched & _flat(tracks["mc_has_tv"]),
            "has_ut": truth_matched & _flat(tracks["mc_has_up"]),
            "has_mp": truth_matched & _flat(tracks["mc_has_mp"]),
            "has_ft": truth_matched & _flat(tracks["mc_has_ft"]),
            "has_t": truth_matched & _flat(tracks["mc_has_t"]),
            "positive_charge": truth_matched & (mc_charge > 0),
            "negative_charge": truth_matched & (mc_charge < 0),
            "from_beauty": truth_matched
            & _flat(_from_flavour(tracks["mc_ancestor_pids"], 5)),
            "from_charm": truth_matched
            & _flat(_from_flavour(tracks["mc_ancestor_pids"], 4)),
            # There is no consistently equivalent Long-track tag.
            "from_strange_reconstructible": pd.array(
                [pd.NA] * n_rows, dtype="boolean"
            ),
        }
    )
    _set_track_type_tags(frame)

    matched = frame["truth_matched"] & (frame["mc_key"] >= 0)
    duplicated = frame.loc[matched].duplicated(
        ["run_number", "event_number", "mc_key"], keep="first"
    )
    frame.loc[matched, "is_unique_truth_match"] = ~duplicated.to_numpy()
    return frame


def reconstruction(chunk):
    """Return reconstructible-MC and reconstructed-Long rows for one chunk."""
    tracks = load_tracks(
        chunk,
        hits=(),
        rich=False,
        compute_track_quantities=_compute_track_kinematics,
    )
    reconstructible = load_reconstructible_tracks(chunk)
    event_info = load_event_info(chunk)

    reco_frame = _long_track_frame(tracks, event_info)
    truth_frame = _reconstructible_frame(reconstructible, event_info)
    counters("Long tracks").add(len(reco_frame))
    counters("truth-matched Long tracks").add(
        int(reco_frame["truth_matched"].sum())
    )
    counters("MC reconstructible tracks").add(len(truth_frame))
    return pd.concat([truth_frame, reco_frame], ignore_index=True)


def _set_track_type_tags(frame):
    """Add the standard reconstructibility categories from primitive flags."""
    for name, (first, second) in TRACK_TYPES.items():
        frame[name] = frame[first].astype(bool) & frame[second].astype(bool)
    return frame


def _selection(frame, track_type, tags):
    selected = frame[track_type].astype(bool).to_numpy().copy()
    for tag in tags:
        selected &= frame[tag].fillna(False).astype(bool).to_numpy()
    return selected


def _ratio(values_num, values_den, bins):
    numerator, _ = np.histogram(values_num, bins=bins)
    denominator, _ = np.histogram(values_den, bins=bins)
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan, dtype=float),
        where=denominator > 0,
    )
    uncertainty = np.sqrt(
        np.clip(ratio * (1.0 - ratio), 0.0, None) / np.maximum(denominator, 1)
    )
    return numerator, denominator, ratio, uncertainty


def _performance_tables(frame, track_type, tags):
    _set_track_type_tags(frame)
    reconstructible = frame[frame["row_type"] == "reconstructible"]
    long_tracks = frame[frame["row_type"] == "long"]

    denominator = reconstructible[
        _selection(reconstructible, track_type, tags)
    ]
    numerator = long_tracks[
        long_tracks["truth_matched"]
        & long_tracks["is_unique_truth_match"]
        & _selection(long_tracks, track_type, tags)
    ]
    fake_numerator = long_tracks[~long_tracks["truth_matched"]]

    definitions = {
        "pt": (
            "truth_pt",
            "reco_pt",
            1000.0 * np.linspace(0, 10, 100),
            1e-3,
        ),
        "eta": ("truth_eta", "reco_eta", np.linspace(1.5, 5.5, 100), 1.0),
        "p": (
            "truth_p",
            "reco_p",
            1000.0 * np.linspace(0, 100, 100),
            1e-3,
        ),
    }
    rows = []
    for variable, (
        truth_field,
        reco_field,
        bins,
        scale,
    ) in definitions.items():
        n_eff, d_eff, efficiency, efficiency_err = _ratio(
            numerator[truth_field].dropna(),
            denominator[truth_field].dropna(),
            bins,
        )
        n_fake, d_fake, fake_rate, fake_rate_err = _ratio(
            fake_numerator[reco_field].dropna(),
            long_tracks[reco_field].dropna(),
            bins,
        )
        for index in range(len(bins) - 1):
            rows.append(
                {
                    "track_type": track_type,
                    "selection_tags": ",".join(tags),
                    "variable": variable,
                    "bin_low": bins[index] * scale,
                    "bin_high": bins[index + 1] * scale,
                    "efficiency_numerator": n_eff[index],
                    "efficiency_denominator": d_eff[index],
                    "efficiency": efficiency[index],
                    "efficiency_uncertainty": efficiency_err[index],
                    "fake_numerator": n_fake[index],
                    "fake_denominator": d_fake[index],
                    "fake_rate": fake_rate[index],
                    "fake_rate_uncertainty": fake_rate_err[index],
                }
            )
    return pd.DataFrame(rows)


def _plot_metric(table, value, uncertainty, ylabel, output, selection_label):
    import matplotlib.pyplot as plt

    from trackcomb.plot import make_figure

    fig, axes = make_figure(1, 3, figsize=(21 * 1.5, 6 * 1.5))
    labels = {"pt": r"$p_T$ [GeV]", "eta": r"$\eta$", "p": r"$p$ [GeV]"}
    for axis, variable in zip(axes, ("pt", "eta", "p")):
        points = table[table["variable"] == variable]
        centers = 0.5 * (points["bin_low"] + points["bin_high"])
        widths = 0.5 * (points["bin_high"] - points["bin_low"])
        axis.errorbar(
            centers,
            100.0 * points[value],
            xerr=widths,
            yerr=100.0 * points[uncertainty],
            fmt="o",
            capsize=2,
        )
        axis.set_xlabel(labels[variable])
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 105.0)
        axis.grid(True, alpha=0.3)
    fig.suptitle(f"{ylabel.removesuffix(' [%]')} ({selection_label})")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _infer_label(dataframe_path):
    stem = Path(dataframe_path).stem
    for prefix in ("tracking_particles_", "tracking_particles"):
        if stem.startswith(prefix):
            return stem.removeprefix(prefix).lstrip("_")
    return stem


def make_plots(dataframe_path, plot_dir, track_type, tags, label=None):
    """Read the produced Parquet and write efficiency/fake-rate products."""
    frame = pd.read_parquet(dataframe_path)
    table = _performance_tables(frame, track_type, tags)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    label = label or _infer_label(dataframe_path)
    suffix = f"_{label}" if label else ""
    table["sample_label"] = label
    table_path = plot_dir / f"tracking_performance_binned{suffix}.parquet"
    efficiency_path = plot_dir / f"tracking_efficiency{suffix}.png"
    ghost_path = plot_dir / f"tracking_ghost_rate{suffix}.png"
    table.to_parquet(table_path, index=False)
    _plot_metric(
        table,
        "efficiency",
        "efficiency_uncertainty",
        "Tracking efficiency [%]",
        efficiency_path,
        f"{label}: " + " & ".join((track_type, *tags)),
    )
    _plot_metric(
        table,
        "fake_rate",
        "fake_rate_uncertainty",
        "Ghost rate [%]",
        ghost_path,
        f"{label}: all reconstructed Long tracks",
    )
    print(f"Saved {table_path}")
    print(f"Saved {efficiency_path}")
    print(f"Saved {ghost_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Produce Long-track efficiency/fake-rate Parquet and plots"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", help="ROOT path; wildcards allowed")
    source.add_argument(
        "--dataframe", help="existing tracking Parquet (plot only)"
    )
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out",
        default="public/tracking_efficiency/tracking_particles.parquet",
        help="output dataframe when reading ROOT input",
    )
    parser.add_argument(
        "--plot-dir",
        default="public/tracking_efficiency",
        help="plot output directory",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="sample label; inferred from the output Parquet name by default",
    )
    parser.add_argument(
        "--track-type",
        choices=TRACK_TYPES,
        default="long",
        help="MC reconstructibility category used for efficiency",
    )
    parser.add_argument(
        "--all-track-types",
        action="store_true",
        help="plot every track type in a separate subdirectory",
    )
    parser.add_argument(
        "--selection-tags",
        nargs="*",
        choices=COMMON_TAGS,
        default=["from_signal"],
        help="additional ANDed truth tags; pass with no values for inclusive",
    )
    parser.add_argument(
        "--list-tags", action="store_true", help="print common selectable tags"
    )
    args = parser.parse_args()

    if args.list_tags:
        print("Track reconstructibility categories:")
        for name, fields in TRACK_TYPES.items():
            print(f"  {name}: {fields[0]} & {fields[1]}")
        print("Common reconstructible/matched-Long tags:")
        print("  " + "\n  ".join(COMMON_TAGS))
        print("Reconstructible-only (stored, not selectable): from_strange")

    if not args.input and not args.dataframe:
        if args.list_tags:
            return
        parser.error("one of --input or --dataframe is required")

    if args.dataframe:
        dataframe_path = Path(args.dataframe)
    else:
        dataframe_path = Path(args.out)
        run_reconstruction(
            reconstruction,
            input_data=args.input,
            out=dataframe_path,
            max_events=args.max_events or None,
            chunk_size=args.chunk_size,
            workers=args.workers,
            print_throughput=True,
        )
        print(f"Saved {dataframe_path}")

    track_types = TRACK_TYPES if args.all_track_types else (args.track_type,)
    for track_type in track_types:
        plot_dir = (
            Path(args.plot_dir) / track_type
            if args.all_track_types
            else args.plot_dir
        )
        make_plots(
            dataframe_path,
            plot_dir,
            track_type,
            args.selection_tags,
            args.label,
        )


if __name__ == "__main__":
    main()
