# Track Combination Framework

Event-level particle-combination framework for track containers and PV containers.

## Core Concepts

- `TrackState`: `(x, y, tx, ty, time)` at `z`, with `cov4(x,y,tx,ty)`, `sigma_time`, momentum `p`
- Track charge support: `charge` in `{+1, -1}` (or `0` if needed)
- Optional per-track PID-like info:
  - `hasRICH1`, `hasRICH2`
  - `richDLL_pi`, `richDLL_k`, `richDLL_p`, `richDLL_e`
  - `hasCALO`, `caloDLL_e`
- `PrimaryVertex`: `pv_id, x, y, z, cov3, time, sigma_time`
- `ParticleCombiner`: builds 2/3/4-body combinations from tracks
- Named particle hypotheses:
  - `make_pion`, `make_kaon`, `make_proton`, `make_muon`, `make_electron`
  - pass these directly in `mass_hypotheses`
- Hierarchical/staged combinations via `combination_to_track_state(...)`:
  - treat accepted combination outputs as new track-like objects
  - preserve source-track provenance (`source_track_ids`)
  - reuse them in higher-level decay chains
- 4D vertexing: fit `(x, y, z, time)` per candidate
- Time propagation uses mass-dependent `beta = p/sqrt(p^2 + m^2)` when evaluating timing chi2
- Pair/candidate metrics: DOCA, vertex chi2, mass, pair `pT`, pair `eta`, timing chi2
- Track preselection: `pT`, `eta`, minimum IP wrt all PVs in event

## Install (Optional)

```bash
pip install -e .
```

## Run Without Install

```bash
PYTHONPATH=src python -m trackcomb \
  --tracks examples/tracks.json \
  --primary-vertices examples/primary_vertex.json \
  --masses examples/masses_2body.json \
  --n-body 2 \
  --out output_2body.parquet
```

Plain local script (no installation):

```bash
PYTHONPATH=src python examples/multi_event_api.py
```

Documentation webpage (local):

```bash
python3 -m http.server 8080 --directory docs/web
# open http://localhost:8080
```

## CLI Example With Cuts

```bash
track-combiner \
  --tracks examples/tracks.json \
  --primary-vertices examples/primary_vertex.json \
  --masses examples/masses_3body.json \
  --n-body 3 \
  --min-track-pt 0.5 \
  --min-track-ip-to-any-pv 0.05 \
  --max-doca 1.0 \
  --max-vertex-chi2 200.0 \
  --max-pair-time-chi2 10.0 \
  --min-mass 0.2 \
  --max-mass 10.0 \
  --min-pair-pt 0.02 \
  --allowed-charge-patterns "+-,-+" \
  --out output_3body.parquet
```

## Event Mode (Tracks + PV List)

One event typically has:

- `tracks[event]`: list of track states
- `pvs[event]`: list of primary vertices

Python API:

```python
from trackcomb import CombinationCuts, ParticleCombiner, PrimaryVertex, TrackPreselection, TrackState

tracks: list[TrackState] = ...
pvs: list[PrimaryVertex] = ...

combiner = ParticleCombiner()
results = combiner.combine(
    tracks=tracks,
    primary_vertices=pvs,
    n_body=3,
    mass_hypotheses=[[0.13957, 0.13957, 0.13957]],
    preselection=TrackPreselection(min_pt=0.5, min_ip_to_any_pv=0.05),
    cuts=CombinationCuts(max_doca=1.0, min_mass=0.2, max_mass=10.0),
)
```

Multi-event API (`event -> tracks + PVs`):

```python
from trackcomb import EventInput, ParticleCombiner, make_kaon, make_pion

events: list[EventInput] = ...
results = ParticleCombiner().combine_events(
    events=events,
    n_body=2,
    mass_hypotheses=[[make_kaon(), make_pion()]],
)
```

Multi-event CLI:

```bash
PYTHONPATH=src python -m trackcomb \
  --events examples/events.json \
  --masses examples/masses_pid_2body.json \
  --n-body 2 \
  --out output_events_2body.parquet
```

## Stepwise Decay Chains (Composite -> Track Abstraction)

Example script:
- `/Users/renato/Documents/New project/pyCombiners/examples/stepwise_decay_examples.py`

This demonstrates:
1. `J/psi -> mu mu` (2-body)
2. Convert accepted `J/psi` candidates to track-like composites
3. Build `B -> J/psi K` by combining composite `J/psi` with kaon tracks
4. Build `Phi -> K K`, convert to composite tracks
5. Build `B -> J/psi Phi` from two composite candidates

## Write Your Own Channel Script (Template)

Template script:
- `/Users/renato/Documents/New project/pyCombiners/examples/new_decay_channel_template.py`

Run:

```bash
PYTHONPATH=src python3 examples/new_decay_channel_template.py \
  --input-events examples/events.json \
  --output examples/custom_channel_output.parquet \
  --channel dplus_kpipi
```

## Output Fields (Per Combination)

- `vertex_xyz`, `vertex_time`
- `vertex_cov_xyz`, `vertex_sigma_time`
- `vertex_chi2`, `vertex_time_chi2`, `pair_time_chi2`
- `doca_pairs` (`doca12`, `doca13`, ...)
- `candidate_p4` (`px,py,pz,e,mass`)
- `pair_pt`, `pair_eta`
- `track_min_ip`, `track_min_ip_chi2`
- `track_pid_info` (RICH/CALO fields propagated from input tracks)
- `charge_pattern`, `total_charge`
- `source_track_ids` (provenance of original input tracks)
- `event_id`
- `particle_hypotheses`
- `best_pv_id`

Output is written as a tabular file based on extension:
- `.parquet` (recommended)
- `.csv`
- `.pkl`

## Examples and Tutorial

- Mini tutorial: `/Users/renato/Documents/New project/pyCombiners/docs/mini_tutorial.md`
- Docs webpage: `/Users/renato/Documents/New project/pyCombiners/docs/web/index.html`
- Table inspection helper: `/Users/renato/Documents/New project/pyCombiners/examples/inspect_table.py`
- Peak/SB study helper: `/Users/renato/Documents/New project/pyCombiners/examples/peak_study.py`
- Multi-event API example: `/Users/renato/Documents/New project/pyCombiners/examples/multi_event_api.py`
- New channel template: `/Users/renato/Documents/New project/pyCombiners/examples/new_decay_channel_template.py`
- Physics review notes: `/Users/renato/Documents/New project/pyCombiners/docs/physics_review.md`
- Custom scripts:
  - `/Users/renato/Documents/New project/pyCombiners/examples/custom_analysis.py`
  - `/Users/renato/Documents/New project/pyCombiners/examples/custom_scripts/top_candidates.py`
  - `/Users/renato/Documents/New project/pyCombiners/examples/custom_scripts/filter_and_dump.py`

## CI (GitLab)

- Pipeline: `/Users/renato/Documents/New project/pyCombiners/.gitlab-ci.yml`
