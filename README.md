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
├── workflow/
│   ├── reconstruction.smk      # Snakemake rules for reconstruction
│   └── analysis/               # Snakemake rules for each analysis study
├── tests/                      # pytest test suite
├── models/                     # ONNX MVA model files
├── scripts/                    # Utility scripts (ONNX conversion, fixture generation, ...)
├── Snakefile                   # Top-level Snakemake entry point
├── config.yaml                 # Pipeline configuration (luminosities, channels, modes)
└── environment.yaml            # Conda environment definition
```

## Input Format

The design philosophy of this project is "KISS", keep it simple and stupid. Our reconstruction
data is dumped in a big TTree/RNTuple where all information is stored in a SoA format without
introducing new C++ struct or anything like that. Each entry of the TTree corresponds to an
event, and each property is stored in `std::vector<T>`, so instead of

```cpp
struct Track { float x, y, z, ...; };
std::vector<Track> tracks;
```

We have

```cpp
std::vector<float> tracks_x, tracks_y, tracks_z, ...
```

For multi-level information like

```cpp
struct Hit { float x, y, z; };
struct Track { std::vector<Hit> hits; };
std::vector<Track> tracks;
```

we use a flatten vector + size vector to store it

```cpp
std::vector<int>   tracks_hits_n; // size vector, number of hits in each track
std::vector<float> tracks_hits_x, tracks_hits_y, tracks_hits_z, ...
```

The advantage of this design is it's very easy to load such data with uproot, awkward array fits
very well with this design. Data will be loaded as a big dict:

```python
data["tracks_x"]      # array of floats
data["tracks_y"]      # array of floats
...
data["tracks_hits_x"] # flat array of floats
data["tracks_hits_n"] # size array
```

The multi-level information can be recovered easily:

```python
data["tracks_hits_x"] = ak.unflatten(data["tracks_hits_x"], data["tracks_hits_n"])
```

In the end we have:

```python
{
    "tracks_x": [trackA.x, trackB.x, ...],
    ...
    "tracks_hits_x": [[trackA.hits[0].x, trackA.hits[1].x, ...], [trackB.hits[0].x, ...], ...],
}
```

## Event Model

Since our data is loaded as a big Dict, it is natural to design our event model to the philosophy:
**EVERYTHING IS Dict**.

The container of tracks is a Dict, the container of secondary vertices is a Dict, the container
of PVs is a Dict, everything is a Dict.

The advantage of this design is we can easily add new information to the Dict. One example would
be to associate best PV to each track:

```python
tracks = load_tracks_from_event(...)
pvs    = load_pvs(...)

selected_pvs = select_pvs(pvs)
tracks_pv_association(tracks, selected_pvs)
```

In this example, `tracks_pv_association` will create new keys in the tracks dict, so variables
like `min_ip` and `min_ip_chi2` and `best_pv_x/y/z` become available.

Then the selection of tracks becomes something very simple:

```python
good_tracks = apply_mask(tracks, tracks["min_ip"] > threshold)
```

This Dict design also allows us to define different cuts easily without relying on any Functor
framework. Just use the built-in cut helpers:

```python
from trackcomb import cut_min, cut_max, cut_range

my_cuts = [
    cut_min("pt", 0.5),           # pt >= 0.5 GeV
    cut_max("min_ip_chi2", 16),   # IP chi2 <= 16
    cut_range("mass", 0.47, 0.52) # mass in [0.47, 0.52] GeV
]
```

Or just use a lambda for anything more complex:

```python
my_cuts = [
    cut_min("pt", 0.5),
    lambda c: c["daughter0_pt"] + c["daughter1_pt"] > 1.0,  # sum pt cut
    lambda c: c["mass"] - 0.498 < 0.02,                     # asymmetric mass window
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
from trackcomb import (
    load_events_root, set_tracks_pid, apply_mask,
    tracks_pv_association, combine,
    cut_min, cut_max, cut_range,
    candidates_to_dataframe, extract_daughter_fields,
    pdg_id,
)

# Load data
tracks, pvs, info = load_events_root("input/ntuple_bsmumu_1p5e34_1000evts.root")

# Track-PV association
tracks_pv_association(tracks, pvs)

# Split tracks by charge and assign mass hypothesis
pos = apply_mask(tracks, tracks["charge"] > 0)
neg = apply_mask(tracks, tracks["charge"] < 0)
set_tracks_pid(pos, "mu+")
set_tracks_pid(neg, "mu+")

# Combine
candidates = combine(
    [pos, neg], pvs,
    track_cuts=[cut_min("pt", 0.5), cut_min("min_ip", 0.05)],
    combination_cuts=[cut_max("max_doca", 0.2), cut_min("dira", 0.9995)],
    vertex_cuts=[cut_range("mass", 5.0, 5.8), cut_min("pt", 1.0)],
)
candidates["pid"] = pdg_id("B(s)0")

# Export to parquet
df = candidates_to_dataframe(candidates)
for k, v in extract_daughter_fields(candidates).items():
    df[k] = v
df.to_parquet("bs_candidates.parquet")
```

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

Input ntuples are not tracked in git. You need to set up the `input/` directory yourself.

The easiest way is to symlink to the EOS storage:

```bash
ln -s /eos/lhcb/user/j/jzhuo/pyCombiners/input ./input
```

Alternatively, you can customize `config.yaml` to point input paths to wherever your data lives.
The config file defines the full run matrix: luminosity points (`1p5e34`, `1p3e34`, `1p0e34`,
`run3`), channels (`ks_to_pipi`, `bs_to_mumu`, `track_pv_association`, `two_track_mva`,
`bs_to_mumu_pvtag`), and modes (`cheated`, `full`, `dist`, ...). Each channel entry has an
`input` field pointing to the ROOT ntuple:

```yaml
luminosities:
  1p5e34:
    channels:
      bs_to_mumu:
        input: input/ntuple_bsmumu_1p5e34_1000evts.root
        modes:
          full:
            max_events: 1000
```

## Running the Pipeline

```bash
# Run the entire pipeline
snakemake -j4

# Run only reconstruction
snakemake all_reconstruction -j4

# Run a specific target
snakemake public/dev/1p5e34/reconstruction/bs_to_mumu/full.parquet -j1

# Dry-run to check the DAG
snakemake -n
```

You can also run reconstruction scripts directly:

```bash
PYTHONPATH=src python3 physics/reconstruction/bs_to_mumu.py \
    --mode full \
    --input input/ntuple_bsmumu_1p5e34_1000evts.root \
    --tree BestLongTracks/TrackTuple \
    --max-events 1000 \
    --out-dir output/
```

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
