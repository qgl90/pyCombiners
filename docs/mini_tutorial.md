# Mini Tutorial

## 1. Run directly from source (no install)

```bash
PYTHONPATH=src python -m trackcomb \
  --tracks examples/tracks.json \
  --primary-vertices examples/primary_vertex.json \
  --masses examples/masses_2body.json \
  --n-body 2 \
  --out output_2body.parquet
```

## 2. Add preselection on tracks

```bash
track-combiner \
  --tracks examples/tracks.json \
  --primary-vertices examples/primary_vertex.json \
  --masses examples/masses_2body.json \
  --n-body 2 \
  --min-track-pt 0.5 \
  --max-track-eta 5.0 \
  --min-track-ip-to-any-pv 0.05 \
  --out output_2body_prefiltered.parquet
```

## 3. Add candidate cuts (DOCA, vertex, mass, kinematics)

```bash
track-combiner \
  --tracks examples/tracks.json \
  --primary-vertices examples/primary_vertex.json \
  --masses examples/masses_3body.json \
  --n-body 3 \
  --max-doca 1.0 \
  --max-vertex-chi2 200.0 \
  --max-vertex-time-chi2 20.0 \
  --max-pair-time-chi2 10.0 \
  --min-mass 0.2 \
  --max-mass 10.0 \
  --min-pair-pt 0.02 \
  --min-pair-eta -10 \
  --max-pair-eta 10 \
  --allowed-charge-patterns "++,--,+-,-+" \
  --out output_3body_filtered.parquet
```

## 4. Event usage: tracks list + PV list

```python
from trackcomb import CombinationCuts, ParticleCombiner, TrackPreselection

combiner = ParticleCombiner()
results = combiner.combine(
    tracks=tracks_for_event,
    primary_vertices=pvs_for_event,
    n_body=3,
    mass_hypotheses=[[0.13957, 0.13957, 0.13957]],
    preselection=TrackPreselection(min_pt=0.5, min_ip_to_any_pv=0.05),
    cuts=CombinationCuts(max_doca=1.0, min_mass=0.2, max_mass=10.0),
)
```

## 5. Inspect / plot the output table

```bash
python examples/inspect_table.py \
  --input output_3body_filtered.parquet \
  --plot
```

## 6. Estimate signal/background around known peaks

`K_S -> pi pi` sideband study:

```bash
python examples/peak_study.py \
  --input output_3body_filtered.parquet \
  --channel ks_pipi \
  --plot
```

`D+ -> K pi pi` sideband study:

```bash
python examples/peak_study.py \
  --input output_3body_filtered.parquet \
  --channel dplus_kpipi \
  --plot
```

## 7. Run staged combinations (composite candidates as tracks)

```bash
python examples/stepwise_decay_examples.py
```

This runs:
- `J/psi(mu mu)` as a 2-body stage.
- `Phi(KK)` as a 2-body stage.
- `B -> J/psi K` and `B -> J/psi Phi` as second-stage combinations where the
  intermediate candidates are converted to track-like objects with
  `(x, y, tx, ty, z, cov4, time, sigma_time, source_track_ids)`.
