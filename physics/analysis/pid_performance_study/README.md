# Long-track PID performance study

The PID study writes a reusable Long-track Parquet and plots kaon efficiency
against pion misidentification using the RICH kaon DLL. The Parquet contains
reconstructed kinematics, RICH responses, MC truth, and run/event identifiers,
so alternative working points can be evaluated without rereading ROOT files.

From the repository root:

```bash
PYTHONPATH=src python3 physics/reconstruction/pid_performance.py \
  --input 'input/*.root' \
  --max-events 1000 --chunk-size 100 --workers 4 \
  --out-file public/pid/reconstruction/tracks.parquet \
  --out-dir public/pid/analysis \
  --out-tag sample
```

This produces global and kinematically binned ROC curves, efficiency versus
`p`, `pt`, and `eta` at fixed pion misidentification probabilities, a 2D
`eta`-`pt` map, and CSV tables containing the plotted values.

Existing Parquets can be plotted or compared independently:

```bash
PYTHONPATH=src python3 -m pid \
  --input low_lumi low_lumi.parquet \
  --input high_lumi high_lumi.parquet \
  --signal K+ --background pi+ --dll-field rich_dll_kaon \
  --target 0.01 --target 0.05 \
  --out-dir public/pid/comparison --out-tag luminosity
```

## Pipeline

The combined PID/tracking pipeline runs both studies and makes all plots:

```bash
snakemake --snakefile workflow/performance_studies.smk -c4
```

Use `--configfile ci/performance_studies.yaml` for the 1k-event CI sample.
