# Physics studies

Each sub-directory targets a different aspect of the analysis:

- **truth_study/**: cheated reconstruction (truth-filtered tracks only,
  no combinatorial background).  Fast; useful for seeing signal shapes
  and deciding where to put cuts.
- **distribution_study/**: full reconstruction with loose cuts, used to check
  signal vs background distributions.
- **performance_study/**: evaluate the final cuts, plotting distributions and
  efficiencies.

## Environment setup

Use `micromamba` or `conda` or any conda-like environment tool

```bash
micromamba create -f environment.yaml run5
micromamba activate run5
```

Import the core library

```bash
export PYTHONPATH=src
```

Or, install the project

```bash
pip install -e .
```

## Running a study

```bash
# quick truth-level check
python physics/truth_study/ks_to_pipi.py \
    --input /eos/lhcb/user/j/jzhuo/run5/ntuples/minbias_1p5e32_1000evts.root --max-events 200

# full distribution study
python physics/distribution_study/ks_to_pipi.py \
    --input /eos/lhcb/user/j/jzhuo/run5/ntuples/minbias_1p5e32_1000evts.root --max-events 200

# performance study
python physics/performance_study/ks_to_pipi.py \
    --input /eos/lhcb/user/j/jzhuo/run5/ntuples/minbias_1p5e32_1000evts.root --max-events 1000
```

All scripts accept `--out-dir` to control where plots go (defaults to
a `plots*/` sub-directory next to the script).  Use `--help` for the
full list of options.
