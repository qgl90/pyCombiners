# Physics/Logic Review Notes

This note reviews the current track vertexing and n-body combination logic for offline peak studies such as:

- `K_S -> pi+ pi-`
- `D+ -> K- pi+ pi+` and charge-conjugate

## What is implemented well

- 3D line-based common-vertex fit + weighted time average (`fit_vertex_xyz_t`).
- Pairwise DOCA calculation for geometric consistency.
- Flexible candidate cuts (`mass`, `pt`, `eta`, DOCA, vertex chi2, timing chi2).
- Event-level PV list and per-track min-IP vs PVs.
- Charge-pattern filtering for 2-body and 3-body combinations.
- Analysis-ready tabular output (parquet/csv/pkl) with flattened per-track features.
- Staged-combination abstraction: intermediate resonances can be converted into
  track-like composites and reused in higher-level decays.

## Important caveats (math/logic)

1. Spatial fit weights do not fully include slope covariance in the normal equations.
   - Current xyz fit solves unweighted line equations first, then evaluates chi2 with propagated xy covariance.
   - For precision analyses, use a fully weighted fit (or Kalman-style vertex fit) where covariance enters the solve itself.

2. Timing model now propagates each track time to the fitted vertex with
   mass-dependent beta.
   - For each mass hypothesis, per-track `time(z_ref)` is transported to
     `time(z_vertex)` using `beta = p/sqrt(p^2 + m^2)`.
   - Vertex-time and pairwise-time chi2 values are therefore hypothesis-dependent.
   - Spatial and temporal solves are still separated (pragmatic approximation).

3. No explicit secondary-vertex displacement observables yet.
   - For `K_S` and charm, add:
     - flight distance/significance from chosen PV to candidate vertex
     - pointing angle / `cos(theta)` between PV->SV vector and candidate momentum
   - These usually give strong background suppression.

4. Candidate-PV association is voting-based from track min-IP.
   - Good pragmatic default, but for high pile-up events, consider selecting PV by best candidate-level pointing/FD significance.

5. Composite-track covariance is approximate in staged workflows.
   - Composite `cov4` uses fitted vertex covariance for x/y and heuristic slope
     covariance mixing from constituents.
   - For precision measurements, propagate full covariance through the full
     decay-chain fit (or refit the full chain jointly).

## Suggested cut strategy (starting point)

These are only initial values; tune on data/MC and detector resolution.

### `K_S -> pi+ pi-`

- `n_body=2`
- charge patterns: `+-,-+`
- mass hypotheses: `(pi, pi)`
- candidate mass window around `m(K_S)=0.497611 GeV`
- tight pair DOCA
- displaced topology:
  - require finite flight distance from PV
  - pointing close to PV->SV line
- optional track PID:
  - favor pion-like RICH response
  - veto electron-like CALO where needed

### `D+ -> K- pi+ pi+` (+ charge conjugate)

- `n_body=3`
- charge patterns: `-++,+--`
- mass hypotheses:
  - apply kaon mass to one track and pion masses to two tracks
  - include permutations if kaon track is unknown
- mass window around `m(D+)=1.86966 GeV`
- require moderate candidate `pt`
- tighter vertex chi2 and pairwise DOCA cuts than loose preselection
- displaced topology with PV association:
  - larger SV displacement significance than prompt background
  - strong pointing requirement
- PID:
  - identify kaon candidate with RICH DLL separation (`K` vs `pi`)

## Offline signal/background estimation

Use sideband subtraction around the mass peak:

1. Select channel-like candidates (charge pattern, topology, PID, etc.).
2. Define:
   - signal window: `[m0 - w_sig, m0 + w_sig]`
   - sidebands: `[m0 - w_out, m0 - w_in]` and `[m0 + w_in, m0 + w_out]`
3. Estimate local background density from sidebands and scale to signal width.
4. Compute:
   - `S_est = N_signal_window - B_est`
   - `S/B`
   - `S/sqrt(S+B)`

The helper script `examples/peak_study.py` automates this workflow.
