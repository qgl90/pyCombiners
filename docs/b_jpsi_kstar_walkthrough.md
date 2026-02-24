# B -> J/psi K* Walkthrough (Synthetic 1000-Event Sample)

This walkthrough builds a realistic synthetic example for:

- `B -> J/psi(mu+ mu-) K*(K+ pi-)`
- 1000 events total
- 20% signal events with one truth B decay
- 80% background-only events
- B-study mass window: `5000-6000 MeV`

The workflow is split in two scripts:

1. Generate fake dataset and run staged combiners into one pandas output table.
2. Study/plot the resulting composite masses from that table.

## 1) Generate + Combine

From repository root:

```bash
PYTHONPATH=src python3 examples/b_jpsi_kstar_fake_and_combine.py \
  --n-events 1000 \
  --signal-fraction 0.20 \
  --seed 12345 \
  --out-events examples/output_bjpsikstar_events.json \
  --out-truth examples/output_bjpsikstar_truth.json \
  --out-candidates examples/output_bjpsikstar_candidates.parquet
```

Outputs:

- `examples/output_bjpsikstar_events.json`:
  - event list with tracks + PVs
- `examples/output_bjpsikstar_truth.json`:
  - event-level truth track ids (`J/psi`, `K*`, `B`)
- `examples/output_bjpsikstar_candidates.parquet`:
  - staged candidate table (`JPSI`, `KSTAR`, `B`)
- `examples/output_bjpsikstar_candidates.summary.json`:
  - compact count/purity summary

## 2) Study/Plot Composite Masses

```bash
python3 examples/b_jpsi_kstar_study.py \
  --input examples/output_bjpsikstar_candidates.parquet \
  --out-dir examples/output_bjpsikstar_study \
  --b-min-mev 5000 \
  --b-max-mev 6000
```

Outputs:

- `examples/output_bjpsikstar_study/summary.json`
- `examples/output_bjpsikstar_study/jpsi_mass_mev.png`
- `examples/output_bjpsikstar_study/kstar_mass_mev.png`
- `examples/output_bjpsikstar_study/b_mass_mev_window.png`
- `examples/output_bjpsikstar_study/b_mass_vs_vertex_chi2.png`

## 3) Inspect with pandas

```python
import pandas as pd

df = pd.read_parquet("examples/output_bjpsikstar_candidates.parquet")

print(df.groupby("stage").size())
print(df[df["is_truth"]].groupby("stage").size())

b = df.query("stage == 'B' and 5000 <= candidate_mass_mev <= 6000")
print("B rows in window:", len(b))
print("B truth rows in window:", int(b["is_truth"].sum()))
print("B purity in window:", float(b["is_truth"].mean()) if len(b) else 0.0)
```

## Notes on the generation model

- Signal events use physically consistent two-body decays:
  - `B -> J/psi + K*`
  - `J/psi -> mu+ mu-`
  - `K* -> K+ pi-`
- Background tracks are random and include PID-like and timing information.
- The staged combiner flow is:
  1. `mu mu` candidates for `J/psi`
  2. `K pi` candidates for `K*`
  3. combine `J/psi` and `K*` composites into `B` candidates
