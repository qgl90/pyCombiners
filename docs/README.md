# docs Folder

This folder contains human-oriented documentation for framework usage and physics reasoning.

## Files

- `mini_tutorial.md`
  - Practical, command-first guide to run the combiner, apply cuts, inspect output tables, and run peak studies.
  - Intended for day-to-day usage and quick onboarding.

- `physics_review.md`
  - Technical review of current vertexing/combination logic, caveats, and recommended cut strategies.
  - Focused on resonance studies such as `K_S -> pi pi` and `D -> K pi pi`.

- `web/`
  - Static docs webpage for interactive, sectioned reading.
  - Includes a step-by-step workflow for adapting new input schemas and writing
    custom decay-channel scripts.
  - Run locally with:
    - `python3 -m http.server 8080 --directory docs/web`
    - open `http://localhost:8080`

## How It Interacts With Other Folders

- References scripts in `examples/` for runnable workflows.
- Describes behavior implemented in `src/trackcomb/`.
- Reflects coverage and assumptions validated by tests in `tests/`.
