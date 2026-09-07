# TwoTrackMVA timed-PV study

This study measures the TwoTrackMVA candidate rate, the number of distinct PVs
selected by those candidates, the signal-PV selection efficiency, and the effect
of reconstructed ghost tracks. Its Parquet files are the primary output; the
provided plotting scripts are only convenient default views.

To run the complete `1p0e34` timed/common-on-time versus `0p2e34`
untimed/all-pairs study, including the explicit MVA threshold scan and all
comparisons, run:

```bash
physics/analysis/two_track_mva_pv_study/run_timing_luminosity_study.sh
```

The script intentionally contains the complete raw commands, with no loops or
helper functions. Edit the visible arguments directly to change the 40,000-event
limit, chunk size, worker count, inputs, or output paths.

## Selection and association sequence

For every event the producer does the following independently for `all_tracks`
and `truth_matched` tracks:

1. Assign the pion mass hypothesis to every input track.
2. Associate each track to all PVs passing `dt_chi2 < max_dt_chi2` (default 16).
3. Calculate transverse IP/IP chi2 to that PV subset and select tracks using the
   resulting timed `minIP`.
4. Form both same-sign and opposite-sign two-track combinations. By default the
   two daughters must share at least one index in their on-time-PV lists. The
   old, stricter same-best-PV condition is not imposed.
5. Apply the spatial vertex chi2 cut and require the two track times at the fitted
   vertex to agree with `vertex_time_chi2 < 16`.
6. Treat the fitted two-body object as a new object: time-gate the PVs again using
   its improved fitted time/error, then choose its best PV by minimum transverse
   IP and calculate all PV-dependent quantities from that association.
7. Evaluate, but do not cut on, the TwoTrackMVA response. This permits arbitrary
   downstream threshold scans without rerunning reconstruction.

`--no-require-common-pv-on-time` retains all pairs while still writing the common
PV and same-best-PV flags. It is intended for explicitly studying the effect of
the daughter-PV compatibility requirement. `--disable-pv-timing` makes all PVs
eligible; it does not mean “no association.”

## Produce a reusable candidate Parquet

Run from the repository root. The example below uses the same environment and
EOS layout as the other reconstruction studies:

```bash
./myenv/run PYTHONPATH=src python3 physics/reconstruction/two_track_mva_pv_study.py \
  --input '/eos/lhcb/wg/rta/WP6/tdr_u2_august2026/middle-scenario-july2026_0p2e34/Bs_Jpsimm_Phi/moore/Bs_Jpsimm_Phi_13144011_0p2e34_slot_*event*.root' \
  --max-events 40000 --chunk-size 1000 --workers 20 \
  --max-dt-chi2 16 --max-vertex-time-chi2 16 \
  --label 0p2e34 \
  --out two_track_mva/0p2e34_candidates.parquet
```

Repeat with the `1p0e34` and `1p3e34` input patterns and corresponding labels.
To retain pairs failing the common-on-time-PV condition for a policy comparison,
add `--no-require-common-pv-on-time`.

Every input event has one `is_candidate=False` row per track sample. These rows
are essential: events producing zero candidates remain in all rate denominators.
Candidate rows contain the MVA response, charge category, best PV index, common
daughter PV indices, same-best-PV flag, signal-PV flag, ghost content, vertex
space/time chi2, and the observables used by the model.

## Scan the MVA requirement

```bash
./myenv/run PYTHONPATH=src python3 physics/analysis/two_track_mva_pv_study/analyze.py \
  --input two_track_mva/0p2e34_candidates.parquet \
  --out two_track_mva/0p2e34_scan.parquet \
  --plot-dir two_track_mva/plots_0p2e34 \
  --thresholds 0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.95 0.96 0.97 0.98 0.99
```

The summary Parquet stores all three charge categories, all daughter-PV
policies, and both track samples. `all_pairs`, `common_on_time`, and
`same_best_pv` make the compatibility requirement directly comparable. The
unrestricted curve requires a producer run with
`--no-require-common-pv-on-time`; otherwise `all_pairs` and `common_on_time`
necessarily contain the same already-filtered candidates. Definitions are:

- `candidates_per_event`: selected candidates divided by all input events;
- `pv_selected_event_fraction`: fraction of all input events in which the
  selected candidates identify at least one best PV;
- `mean_unique_pvs_per_selected_event`: mean number of distinct best-PV indices,
  evaluated only among events having at least one selected PV;
- `mean_selected_pvs_per_event`: unconditional auxiliary quantity, including
  events with zero selected PVs;
- `signal_pv_retention_efficiency`: fraction of all input events with an
  identifiable reconstructed signal PV for which the selected unique-PV set
  contains that signal PV. Its denominator is fixed across the MVA scan, so it
  can only stay constant or decrease as the cut is tightened;
- `signal_pv_correctness_given_selected_event`: fraction of selected events for
  which the selected-PV set contains the signal PV. Its denominator changes
  with the MVA cut, so this conditional correctness can increase as incorrectly
  associated events are rejected. `signal_pv_efficiency_given_selected_event`
  remains as a backward-compatible alias;
- `signal_pv_efficiency`: signal reco-PV indices selected at least once divided
  by all reconstructable signal reco-PV indices in the input events. This older,
  unconditional quantity remains in the Parquet but is not used in the default
  plots;
- `all_tracks` versus `truth_matched`: exact rerun with and without ghost tracks
  in the input pool. Their difference quantifies the ghost-track contribution.

At startup the analyzer prints the producer configuration recovered from the
candidate Parquet, the hard-coded reconstruction sequence, the thresholds and
categories being scanned, and the exact event/PV efficiency definitions. Older
candidate Parquets remain supported; producer fields that predate persistence
of the model path and requested track samples are reported as not stored.

`raw_selected_pv_multiplicity.png` shows raw selected-event counts versus MVA
working point and unique selected-PV multiplicity. At the nominal working point
it also shows the two-dimensional distribution of all reconstructed PVs versus
selected PVs, and the total-PV spectrum of selected events. The corresponding
event-level values are saved in `<scan-name>_pv_multiplicity.parquet`. Use
`--plot-pv-policy` to choose `all_pairs`, `common_on_time`, or `same_best_pv`.

`signal_pv_retention_efficiency.png` shows the event-level probability that the
signal PV belongs to the selected unique-PV set. Only events where a
truth-matched reconstructed signal PV can be identified enter the fixed
denominator. The same retention metric appears in the third panel of
`mva_threshold_scan.png`.

`unique_selected_pvs_per_processed_event.png` is the simpler one-dimensional
view at one working point. Its x axis is the number of distinct selected best
PVs and its y axis is the raw number of processed events. Zero is included and
means that no candidate selected a valid PV in that event. The default working
point is `0.9569`; change it with `--multiplicity-cut`. Its event-level source is
saved as `<scan-name>_processed_event_pvs.parquet`.

## Compare luminosities or labels

```bash
./myenv/run PYTHONPATH=src python3 physics/analysis/two_track_mva_pv_study/compare.py \
  --input 0p2e34 two_track_mva/0p2e34_scan.parquet \
  --input 1p0e34 two_track_mva/1p0e34_scan.parquet \
  --input 1p3e34 two_track_mva/1p3e34_scan.parquet \
  --track-sample all_tracks --pv-policy common_on_time --charge all \
  --out two_track_mva/comparison.png
```

The comparison also writes `comparison.parquet`, containing the concatenated
labelled scan tables for custom plotting.

To compare samples produced with different timing/PV policies, use `--series`
with `LABEL PARQUET PV_POLICY` for each curve. For example, compare a timed
high-luminosity sample against an untimed low-luminosity sample, while also
showing the stricter same-best-PV interpretation:

```bash
./myenv/run PYTHONPATH=src python3 physics/analysis/two_track_mva_pv_study/compare.py \
  --series 1p0e34_timed_common two_track_mva_1p0e34/bs_scan.parquet common_on_time \
  --series 1p0e34_timed_same_best two_track_mva_1p0e34/bs_scan.parquet same_best_pv \
  --series 0p2e34_no_timing two_track_mva_no_timing/0p2e34/bs_scan.parquet all_pairs \
  --track-sample all_tracks --charge all \
  --out two_track_mva/timing_luminosity_comparison.png
```

Repeat with `--track-sample truth_matched` to compare after removing ghost
tracks. This deliberately compares both luminosity and timing configuration;
the series labels and output Parquet preserve the policy used for every curve.
