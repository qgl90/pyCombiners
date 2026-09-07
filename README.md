# pyCombiners  [![pipeline status](https://gitlab.cern.ch/rquaglia/pyCombiners/badges/main/pipeline.svg)](https://gitlab.cern.ch/rquaglia/pyCombiners/-/commits/main)


This is more like a MooreAnalysis framework for Upgrade II trigger studies.

The idea is to allow users to create complex decay chains on top of U2 reconstruction, using
Moore-like syntax in Python scripts, and also create final tuples to perform physics analysis.

The final tuple can be stored in any format you want: ROOT, parquet, json, ...etc.

The core library is located in `src/`, while the decay chain reconstruction scripts are stored in
`physics/reconstruction/` and the scripts that analyze the output of reconstruction to make plots
and conclusions are stored in `physics/analysis/`.

A central pipeline is configured with `snakemake`, it is used to connect the reconstruction output
to the input of analysis. But you can always run stuff separately with all the flexibility of
configuring your own pipeline for your study.

## Project Structure

```
pyCombiners/
├── src/trackcomb/              # Core Python package (combiner, physics, IO, truth, ...)
├── physics/
│   ├── reconstruction/         # Reconstruction scripts (one per decay channel)
│   └── analysis/               # Analysis/plotting scripts (organized by study type)
├── workflow/                   # One snakefile per pipeline (--snakefile workflow/<name>.smk)
├── tests/                      # pytest test suite
├── models/                     # ONNX MVA model files
├── scripts/                    # Utility scripts (ONNX conversion, fixture generation, ...)
├── config/                     # One yaml per pipeline (inputs, event counts)
└── environment.yaml            # Conda environment definition
```

## Input Format

The design philosophy of this project is "KISS", keep it simple and stupid. Our reconstruction
data is dumped in one `EventTuple` TTree per file, where all information is stored in a SoA
format without introducing new C++ structs. Each entry corresponds to an event, each property is
a `std::vector<T>` branch, and branches are organized in two-level `Group/leaf` names:

```
BestLongState_FirstMeasurement/{x, y, z, tx, ty, qop, cov_i_j, ...}
BestLongMC/{pid, key, pv_key, fromSignal, ancestor_pids, n_ancestors, ...}
BestLongRich/{DLL_Muon, DLL_Kaon, ..., hasRICHInfo}
BestLongTVHits/{n, id, x, y, z, t}          # per-hit lists, sized by n
PVState/..., PVMC/..., EventInfo/...
ReconstructibleTracks/..., PicoCalClusters*/...
```

Multi-level information (hits per track, ancestors per track) uses a flat vector + size vector,
recovered with `ak.unflatten`. The component loaders handle all of this.

**Important**: leaf basenames repeat across groups (`x` appears in many groups), so branches
must be read by their **full path** (`tree["BestLongTVHits/x"]`). Never use uproot
`filter_name`/wildcard reads on this format — same-named leaves get silently merged.

## Units

All quantities follow native LHCb conventions: momentum, mass and energy in **MeV**, spatial
coordinates in **mm**, time in **ns**.

## Event Model

Since our data is loaded as a big Dict, it is natural to design our event model to the philosophy:
**EVERYTHING IS Dict**.

The container of tracks is a Dict, the container of secondary vertices is a Dict, the container
of PVs is a Dict, everything is a Dict.

The advantage of this design is we can easily add new information to the Dict. One example would
be to associate best PV to each track:

```python
tracks = load_tracks_from_event(...)
pvs = load_pvs(...)

tracks_pv_association(tracks, pvs, max_dt_chi2=3.5)
```

In this example, `tracks_pv_association` will create new keys in the tracks dict, so variables
like `min_ip`, `min_ip_chi2`, `pv_on_time`, and `best_pv_x/y/z` become
available. The timing requirement first defines the eligible PV subset; IP and
best PV are then evaluated on that subset. Passing neither `max_dt` nor
`max_dt_chi2` keeps all PVs and therefore gives the spatial-only association.

Then the selection of tracks becomes something very simple:

```python
good_tracks = apply_cuts(tracks, [cut_min_ip(threshold, dt_chi2=3.5)])
```

This Dict design also allows us to define different cuts easily without relying on any Functor
framework. Just use the built-in cut helpers:

```python
from trackcomb import cut_min, cut_max, cut_max_ip_chi2, cut_range

my_cuts = [
    cut_min("pt", 500),  # pt >= 500 MeV
    cut_max_ip_chi2(16, dt_chi2=3.5),  # timed-PV IP chi2 <= 16
    cut_range("mass", 470, 520),  # mass in [470, 520] MeV
]
```

Or just use a lambda for anything more complex:

```python
my_cuts = [
    cut_min("pt", 500),
    lambda c: c["daughter0_pt"] + c["daughter1_pt"] > 1000,  # sum pt cut
    lambda c: c["mass"] - 498 < 20,  # asymmetric mass window
]
```

## Decay Chain Reconstruction

The core of this framework is the `combine()` function. It takes track pools and PVs, builds all
combinations, performs vertex fit, timing fit, computes kinematics, associates best PV, and
propagates MC truth. All in one call, fully vectorized over events and candidates.

See [docs/physics.md](docs/physics.md) for the detailed formulas (IP, vertex fit, DOCA, time fit,
DIRA, fdchi2, mcor, etc).

A typical reconstruction script looks like this:

```python
from functools import partial

from trackcomb import (
    event_stream,
    load_tracks,
    load_pvs,
    load_event_info,
    set_tracks_pid,
    apply_mask,
    tracks_pv_association,
    combine,
    composite_pv_association,
    cut_min,
    cut_max,
    cut_range,
    candidates_to_dataframe,
    set_composite_pid,
)

for chunk in event_stream("input/1p0E34_Bs_mumu/*.root", chunk_size=100):
    # Load only what you need (hits/rich/mc are switchable kwargs)
    tracks = load_tracks(chunk)  # hits=("TVHits",), rich=True, mc=True
    pvs = load_pvs(chunk)
    info = load_event_info(chunk)

    # Track-PV association, mass hypothesis
    tracks_pv_association(tracks, pvs, max_dt_chi2=3.5)
    set_tracks_pid(tracks, "mu+")
    pos = apply_mask(tracks, tracks["charge"] > 0)
    neg = apply_mask(tracks, tracks["charge"] < 0)

    # Combine
    candidates = combine(
        [pos, neg],
        pvs,
        track_cuts=[cut_min("pt", 1000), cut_min("rich_dll_muon", -5)],
        combination_cuts=[
            cut_max("max_doca", 0.05),
            cut_range("mass", 4700, 6000),
        ],
        composite_cuts=[cut_max("vertex_chi2", 4)],
        final_cuts=[cut_min("dira", 0.9995)],
        pv_function=partial(composite_pv_association, max_dt_chi2=3.5),
        require_common_pv_on_time=True,
    )
    if candidates is None:
        continue
    set_composite_pid(candidates, "B(s)0")
    df = candidates_to_dataframe(candidates)
    ...
```

For production use, wrap the per-chunk logic in a function and let
`run_reconstruction(fn, input_path, out="bs.parquet")` drive the loop — it streams results
into a single Parquet file with constant memory, handles wildcards and skips corrupt files.

The output candidates are also Dicts, with track-compatible fields (`x`, `y`, `z`, `tx`, `ty`,
`p`, `charge`, `time`, `track_id`, covariance). This means candidates can be fed back into
another `combine()` call to build hierarchical decays.

## Truth Matching

MC truth is propagated automatically by `combine()`, each candidate gets `mc_truth`, `mc_pid`,
`mc_key`, `mc_fromsignal`, and full ancestor chains. The framework also provides an LHCb-style
`bkgcat()` function that classifies candidates into Signal, QuasiSignal, Ghost, Clone, Reflection,
Pileup, FromB, FromC, etc.

See [docs/truth_matching.md](docs/truth_matching.md) for the full documentation and category
decision tree.

## Installation

```bash
# Create the conda environment
conda env create -n pyCombiner -f environment.yaml
conda activate pyCombiner

# Install the package in development mode
pip install -e .
```

## Input Data

Input ntuples are not tracked in git. The current samples live at
`/shared/jzhuo/run5/renato_test_middle-scenario-June2026-RichV3/<sample>/*.root`
(EventTuple format, 100 events per file). Small local subsets can be merged into
`input/` with hadd:

```bash
hadd input/eventtuple_bsmumu_1p0e34_1000evts.root \
     $(ls /shared/.../1p0E34_Bs_mumu/*.root | head -10)
```

## Running the Pipelines

Each pipeline is a standalone snakefile under `workflow/` with its own config
under `config/` — flat keys, no coupling between pipelines:

```yaml
# config/bs_to_mumu.yaml
output_dir: public/bs_to_mumu
input: input/eventtuple_bsmumu_1p0e34_1000evts.root
max_events_full: 1000
```

Each pipeline owns its output directory: parquets go to
`<output_dir>/reconstruction/` and plots to `<output_dir>/analysis/`
(one sub-directory per study when a pipeline has several). Overriding
`output_dir` relocates the whole tree.

Input paths may contain wildcards — corrupt files are skipped with a warning.
The committed configs are the default mode of each pipeline; anything else
(quick tests, other samples/luminosities) is a command-line override or a
private yaml:

The five pipelines are `bs_to_phigamma`, `calo_study`, `bs_to_mumu`,
`ks_to_pipi` and `bs_to_jpsiphi`:

```bash
snakemake --snakefile workflow/bs_to_phigamma.smk -c16
snakemake --snakefile workflow/calo_study.smk -c16
snakemake --snakefile workflow/bs_to_mumu.smk -c8

# Dry-run to check the DAG
snakemake --snakefile workflow/bs_to_phigamma.smk -n

# Quick test: cap events, redirect output
snakemake --snakefile workflow/bs_to_phigamma.smk -c8 \
    --config max_events=1000 output_dir=public/test

# Big/custom job: bring your own config (e.g. another luminosity)
snakemake --snakefile workflow/bs_to_mumu.smk -c16 --configfile my_1p5e34.yaml
```

Reconstruction rules take all cores of the invocation (`--workers` = snakemake
`threads`); analysis rules are single-core so snakemake runs them in parallel.

You can also run reconstruction scripts directly (wildcards allowed in --input):

```bash
PYTHONPATH=src python3 physics/reconstruction/bs_to_mumu.py \
    --mode full \
    --input input/eventtuple_bsmumu_1p0e34_1000evts.root \
    --max-events 1000 \
    --out-dir output/
```

## Continuous Integration

Besides formatting and unit tests (shared runners), every merge request runs
all five pipelines as 1k-event smoke jobs on a dedicated shell runner
(tag `pyCombiner-cpu`). The jobs use `--configfile ci/<pipeline>.yaml` to
point at local copies of the inputs under `/data/ci/input/` on the runner VM;
`ci/ensure_env.sh` rebuilds the micromamba env whenever `environment.yaml`
changes. The `analysis/` plots and all logs are kept as job artifacts —
7 days for merge requests, forever on `main`.

## Plotting

The framework provides a `make_figure()` helper that sets up LHCb2 style and adds a version
watermark. It's a drop-in replacement for `plt.subplots()`:

```python
from trackcomb import make_figure

fig, ax = make_figure()
ax.hist(df["mass"], bins=100)
fig.savefig("mass.png")
```

The version label (e.g. "pyCombiners v0.1.0") is read automatically from the installed package
metadata via `trackcomb.__version__`.

## Running Tests

```bash
pip install -e . --no-deps
pytest tests/ -v
```
