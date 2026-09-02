# Long-track efficiency and fake-rate study

`src/tracking/tracking_efficiencies.py` produces a reusable particle-level Parquet and,
in the same invocation, plots Long-track efficiency and ghost rate versus `pt`, `eta`, `p`, and
`phi`, plus momentum resolution and bias versus true `p`, `eta`, and `phi`.

## Run on ROOT input

From the repository root:

```bash
./myenv/run PYTHONPATH=src python3 src/tracking/tracking_efficiencies.py \
  --input '/eos/lhcb/wg/rta/WP6/tdr_u2_august2026/middle-scenario-july2026_0p2e34/Bs_Jpsimm_Phi/moore/Bs_Jpsimm_Phi_13144011_0p2e34_slot_*_100_event_container.root' \
  --max-events 1000 \
  --chunk-size 100 \
  --workers 20 \
  --out bs_jpsi_phi_tracking/tracking_particles_0p2e34.parquet \
  --plot-dir bs_jpsi_phi_tracking
```

The default efficiency selection is track type `long` plus `from_signal`. The available track-type
definitions are:

```text
long   = has_velo & has_t
down   = has_ut   & has_t
longft = has_velo & has_ft
longmp = has_velo & has_mp
```

The selected track type is required consistently in both numerator and denominator.

Pass `--all-track-types` to make a separate plot directory for `long`, `down`,
`longft`, and `longmp` in one invocation. This is the mode used by the CI
performance-studies pipeline.

The sample label is inferred from the Parquet name, so using the same plot directory for several
luminosities does not overwrite earlier results. For the command above the output directory
contains:

- `tracking_efficiency_0p2e34.png`: efficiency versus truth `pt`, `eta`, and `p`;
- `tracking_ghost_rate_0p2e34.png`: ghost fraction versus reconstructed `pt`, `eta`, `p`, and
  `phi`;
- `tracking_performance_binned_0p2e34.parquet`: bin edges, raw numerators/denominators, ratios,
  and binomial uncertainties.
- `momentum_resolution/deltap_over_p_vs_{p,eta,phi}_0p2e34.png`: Gaussian-core momentum
  resolution and bias;
- `momentum_resolution/momentum_resolution_binned_0p2e34.parquet`: fitted means, widths,
  uncertainties, fit ranges, status, and populations for every truth-kinematic bin;
- `momentum_resolution/gaussian_fit_checks_vs_{p,eta,phi}_0p2e34.pdf`: multipage
  per-bin residual histograms with the MAD seed window, final fit window, and Gaussian overlay.

The main Parquet contains both `row_type == "reconstructible"` denominator rows and
`row_type == "long"` reconstructed-track rows, so alternative analyses can be performed without
rerunning reconstruction.

Every efficiency and ghost-rate panel also shows its denominator spectrum on a secondary y-axis.
For efficiency this is the selected MC-reconstructible population in truth kinematics; for ghost
rate it is the full reconstructed BestLong population in reconstructed kinematics. Comparison
plots show one dashed denominator spectrum per sample in the same colour as its rate curve.

## Definitions

For track-type tag `T` and an optional common truth-tag selection `S`:

```text
efficiency = unique truth-matched Long tracks satisfying T & S
             -------------------------------------------------
                    MCReconstructible particles satisfying T & S

ghost rate = reconstructed Long tracks without a truth match
             ------------------------------------------------
                     all reconstructed Long tracks
```

The efficiency uses truth kinematics. The ghost-rate numerator and denominator both use
reconstructed kinematics because an unmatched track has no valid truth particle. Repeated Long
tracks matched to the same `(run,
event, mc_key)` count once in the efficiency numerator, while all reconstructed tracks remain in
the ghost-rate denominator.

## Momentum resolution and bias

For every truth-matched reconstructed Long track, the signed residual is

```text
delta_p_over_p = (p_reco - p_true) / p_true
```

Tracks are sliced in bins of true `p`, true `eta`, or true `phi`. The suggested default fit range
is obtained in two stages: seed the core with `median +/- 3 * 1.4826 * MAD`, then iteratively
refit and retain `mean +/- 3 * sigma`. The final `mean +/- 3 * sigma` interval is stored as
`fit_low_percent` and `fit_high_percent`; change both three-sigma selections with
`--resolution-fit-sigma`. A bin is not fitted when fewer than `--min-resolution-entries`
(default 20) remain in its robust core.

The fitted Gaussian width is the momentum resolution and its mean is the momentum bias; both
are reported in percent. The signed residual is required to measure bias, while the positive
Gaussian width measures the magnitude of the resolution corresponding to
`|p_reco-p_true|/p_true`. The diagnostic PDFs show every bin, including skipped bins, and annotate
the total and fitted populations, fit status, mean, and width. This makes the chosen fit window
and any non-Gaussian tails directly inspectable.

## Alternative truth tags

List the supported common tags:

```bash
PYTHONPATH=src python3 src/tracking/tracking_efficiencies.py --list-tags
```

Use one or more tags as an AND selection:

```bash
# Inclusive efficiency for the `long` reconstructibility category
PYTHONPATH=src python3 src/tracking/tracking_efficiencies.py \
  --dataframe bs_jpsi_phi/tracking_efficiency/0p2_lumi.parquet \
  --track-type long \
  --selection-tags \
  --plot-dir bs_jpsi_phi/tracking_efficiency/inclusive

# Beauty-origin tracks in the `longmp` category
PYTHONPATH=src python3 src/tracking/tracking_efficiencies.py \
  --dataframe bs_jpsi_phi/tracking_efficiency/0p2_lumi.parquet \
  --track-type longmp \
  --selection-tags from_beauty \
  --plot-dir bs_jpsi_phi/tracking_efficiency/from_beauty_longmp
```

The additional tags available consistently on MCReconstructible particles and matched Long-track
truth are `from_signal`, `positive_charge`, `negative_charge`, `from_beauty`, and `from_charm`.
The primitive acceptance flags and all four derived track-type columns are retained in the main
Parquet. The input also provides `from_strange` on MCReconstructible particles, but no directly
equivalent matched-Long field exists. It is stored as `from_strange_reconstructible` only on
reconstructible rows and is deliberately not offered as a common efficiency selection.

## Compare luminosity samples

Once the three particle Parquets have been produced, compare every track type with:

```bash
./myenv/run PYTHONPATH=src python3 src/tracking/compare_tracking_efficiencies.py \
  --input 0p2e34 bs_jpsi_phi_tracking/tracking_particles_0p2e34.parquet \
  --input 1p0e34 bs_jpsi_phi_tracking/tracking_particles_1p0e34.parquet \
  --input 1p3e34 bs_jpsi_phi_tracking/tracking_particles_1p3e34.parquet \
  --selection-tags from_signal \
  --out-dir bs_jpsi_phi_tracking/comparison
```

This writes one efficiency comparison PNG for each of `long`, `down`, `longft`, and `longmp`, plus
`tracking_ghost_rate_comparison.png`. It also writes `tracking_comparison_summary.parquet` with the
integrated raw counts/rates and `tracking_comparison_binned.parquet` with every plotted bin. To
compare only selected categories, add for example `--track-types long longft`.

## Pipeline

The repository pipeline produces the tracking particle Parquet, plots all four
track types, and runs the PID study on the same input sample:

```bash
snakemake --snakefile workflow/performance_studies.smk -c4
```
