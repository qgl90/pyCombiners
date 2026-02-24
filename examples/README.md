# examples Folder

This folder provides runnable scripts and sample input data for common workflows.

## Files

- `tracks.json`
  - Sample event track container.
  - Includes state/covariance/time, charge, and optional PID-like fields.

- `events.json`
  - Sample multi-event input where each event contains its own track list and PV list.

- `primary_vertex.json`
  - Sample PV container with one or more primary vertices and covariance/time fields.

- `masses_2body.json`, `masses_3body.json`, `masses_4body.json`, `masses_pid_2body.json`
  - Mass-hypothesis sets for 2/3/4-body combinations.
  - `masses_pid_2body.json` shows named particle entries (`pi`, `kaon`, ...).

- `inspect_table.py`
  - Loads output tables (`.parquet/.csv/.pkl`) for quick inspection and optional scatter plotting.

- `peak_study.py`
  - Offline signal/background study helper using signal-window + sideband estimation.
  - Includes presets for `ks_pipi` and `dplus_kpipi`.

- `stepwise_decay_examples.py`
  - Demonstrates hierarchical combinations:
    - `B -> J/psi(mu mu) K`
    - `B -> J/psi(mu mu) Phi(KK)`
  - Shows how combination outputs become track-like inputs via `combination_to_track_state`.

- `multi_event_api.py`
  - Demonstrates `ParticleCombiner.combine_events(...)` with `make_pion/make_kaon` hypotheses.
  - Writes a parquet table tagged with `event_id`.

- `new_decay_channel_template.py`
  - Step-by-step template to:
    - adapt custom input format into `EventInput`
    - run a direct 3-body channel (`D+ -> K pi pi`)
    - run a staged channel (`B -> J/psi(mu mu) K`).

- `custom_analysis.py`
  - Minimal example custom callback to post-process results.

- `custom_scripts/`
  - Additional custom callbacks for ranking and filtered dumping.

## How It Interacts With Other Folders

- Imports API from `src/trackcomb/`.
- Data schemas match the loaders in `src/trackcomb/io.py`.
- Demonstrated workflows are documented in `docs/`.
