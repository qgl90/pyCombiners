# src/trackcomb Folder

Core package implementing event-level track/PV combination workflows.

## Files

- `__init__.py`
  - Public API exports (`ParticleCombiner`, models, staged helper).

- `__main__.py`
  - Enables `python -m trackcomb`.

- `models.py`
  - Immutable data models:
    - `TrackState`
    - `PrimaryVertex`
    - `LorentzVector`
    - `CombinationResult`
    - preselection/cut configuration classes.

- `physics.py`
  - Numeric/physics helpers:
    - track->Lorentz conversion
    - vertex fit `(x,y,z,time)`
    - DOCA, IP/IPchi2, pair kinematics
    - matrix utilities.

- `combiner.py`
  - High-level combination engine (`ParticleCombiner`):
    - track preselection
    - n-body generation
    - candidate observable calculation
    - cut application
    - output object construction.

- `composite.py`
  - Staged-combination helper converting a `CombinationResult` into track-like
    `TrackState` for hierarchical decay chains.

- `io.py`
  - Input/output:
    - load tracks/PVs/mass hypotheses from JSON
    - write flattened output tables (`parquet/csv/pkl`).

- `cli.py`
  - Command-line interface wiring parser -> load -> combine -> write -> optional custom script.

## Internal Interaction Flow

1. `cli.py` parses arguments and calls loaders in `io.py`.
2. `combiner.py` consumes `TrackState`/`PrimaryVertex` from `models.py`.
3. `combiner.py` calls math routines in `physics.py`.
4. `io.py` serializes `CombinationResult` into analysis table rows.
5. Optional staged workflows use `composite.py` to feed one combination stage into the next.
