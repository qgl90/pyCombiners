# tests Folder

This folder contains unit tests for core combiner behavior and cut logic.

## Files

- `test_combiner_operations.py`
  - Validates:
    - combination counts
    - kinematic outputs
    - vertex fit behavior
    - DOCA/IP output presence
    - preselection behavior
    - charge-pattern filtering
    - event-batch combination tagging
    - named particle-hypothesis helper support.

- `test_chi2_filters.py`
  - Validates candidate filtering by:
    - spatial vertex chi2
    - pairwise time chi2
    - combined cut behavior
    - mass/beta-corrected time propagation.

- `test_io_loaders.py`
  - Validates JSON input loaders for:
    - multi-event payloads (`events`)
    - named/custom particle mass hypotheses.

## How It Interacts With Other Folders

- Imports and tests code from `src/trackcomb/`.
- Acts as guardrails for changes in physics logic and selection semantics.
