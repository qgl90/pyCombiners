# Primary-vertex resolution study

`src/pv/pv_resolution.py` reads the aligned `PVState/*` and `PVMC/*` arrays, writes one
Parquet row per reconstructed PV with a valid MC match, and measures PV bias and resolution
versus reconstructed-PV `ndof`.

The loader asserts that `PVState` and `PVMC` contain the same number of entries in every event.
Their common per-event array index is retained as `pv_index`. Entries with `PVMC/key == -1` are
then removed; no independent matching or reordering is performed.

## Run from ROOT input

From the repository root:

```bash
./myenv/run PYTHONPATH=src python3 src/pv/pv_resolution.py \
  --input '/eos/lhcb/path/to/*event*.root' \
  --max-events 40000 \
  --chunk-size 1000 \
  --workers 20 \
  --label 0p2e34 \
  --out pv_resolution/pv_residuals_0p2e34.parquet \
  --plot-dir pv_resolution/0p2e34
```

Plot again without reading ROOT:

```bash
./myenv/run PYTHONPATH=src python3 src/pv/pv_resolution.py \
  --dataframe pv_resolution/pv_residuals_0p2e34.parquet \
  --label 0p2e34 \
  --plot-dir pv_resolution/0p2e34
```

The dataframe retains reconstructed and truth `x`, `y`, `z`, and time, `ndof`, `chi2ndof`, the
full lower triangle of the reconstructed 4x4 covariance, and the four residuals
`PVState - PVMC`.

## Resolution definition

For each `ndof` bin and each of `x`, `y`, `z`, and time, the script fits the core of

```text
delta = PVState coordinate - matched PVMC coordinate
```

using the same robust Gaussian procedure as the tracking-resolution study. The Gaussian mean is
the bias and its width is the resolution. The seed interval is
`median +/- 3 * 1.4826 * MAD`; the final suggested range is `mean +/- 3 * sigma`. Override the
range multiplier with `--fit-sigma` and the minimum core population with
`--min-fit-entries`.

The default fixed `ndof` edges are suitable for comparing independently produced samples. They
can be replaced with, for example:

```bash
--ndof-edges 0 10 20 30 40 60 80 100 150 200 300 500 1000
```

## Outputs

- `pv_residuals_<label>.parquet`: reusable matched-PV dataframe;
- `pv_resolution_fits_<label>.parquet`: per-bin fit populations, status, ranges, bias,
  resolution, and uncertainties;
- `pv_residuals_vs_ndof_<label>.png`: residual-density maps versus `ndof`;
- `pv_bias_resolution_vs_ndof_<label>.png`: fitted Gaussian means and widths;
- `pv_gaussian_fit_checks_{x,y,z,time}_<label>.png`: all per-bin residual distributions,
  robust ranges, and Gaussian overlays.

The input coordinates are assumed to use the event-tuple units: millimetres for `x`, `y`, `z`
and nanoseconds for time.
