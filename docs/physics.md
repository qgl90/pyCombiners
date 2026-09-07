# Physics Computations

This document describes the physics calculations implemented in the framework.

## Track state convention

The original track state in ROOT is a 5-parameter vector `(x, y, tx, ty, q/p)` at a reference
z-plane, where `tx = dx/dz`, `ty = dy/dz` are the slopes and `q/p` is charge over momentum
(in 1/MeV). The covariance matrix in ROOT is 5x5 over these parameters.

During loading, the IO layer converts `q/p` into scalar fields `p` (in GeV) and `charge` (+1/-1),
and keeps only the upper-left 4x4 block of the covariance over `(x, y, tx, ty)`. The 5th row/col
(q/p) is dropped because the current physics computations (IP, vertex fit, DOCA) only use spatial
information.

Loaded tracks receive a pion mass hypothesis by default. This makes the mass-dependent track-time
fit available before track-PV association; a reconstruction channel may replace the hypothesis
later with `set_tracks_pid`.

Extrapolation to a new z is a straight line:

```
x(z') = x + tx * (z' - z)
y(z') = y + ty * (z' - z)
```

The track covariance internally is a 4x4 matrix over `(x, y, tx, ty)`, stored as lower-triangular
fields `cov_i_j` in the container.

## Impact parameter (IP)

The IP of a track to a PV is the transverse distance at the PV z-plane:

```
dx = x + tx * (z_pv - z) - x_pv
dy = y + ty * (z_pv - z) - y_pv
IP = sqrt(dx^2 + dy^2)
```

The IP chi2 accounts for the track and PV covariance. The track covariance is propagated to the
PV z:

```
var_x  = c00 + 2*dz*c20 + dz^2*c22
var_y  = c11 + 2*dz*c31 + dz^2*c33
cov_xy = c10 + dz*c30 + dz*c21 + dz^2*c32
```

Then the total covariance is `V_track(z_pv) + V_pv`, and the chi2 is the 2D Mahalanobis distance:

```
IP_chi2 = (dx, dy) @ V_total^{-1} @ (dx, dy)^T
```

## Flight-corrected time residual

For track-PV timing association, the raw time difference `t_track - t_pv` is corrected for the
time-of-flight from PV to the track reference z:

```
speed_factor = sqrt(1 + tx^2 + ty^2)
t_flight = (z_track - z_pv) * speed_factor / c
dt = t_track - t_flight - t_pv
```

where `c = 299.792458 mm/ns`.

## Best PV selection

For tracks and composites, IP, IP chi2, `dt`, and `dt_chi2` are first computed relative to every
PV. A requested `dt` or `dt_chi2` limit then creates an eligible-PV mask, and the best PV is the
eligible one with the smallest transverse IP. The reconstruction working point is
`dt_chi2 <= 3.5`. If neither timing limit is supplied, all PVs are eligible and association is
based only on spatial IP/IP chi2. If a requested timing mask is empty, `min_ip` and
`min_ip_chi2` are NaN and `best_pv_index` is `-1`; it does not fall back to all PVs.

The stored `pv_ip`, `pv_ip_chi2`, `pv_dt`, and `pv_dt_chi2` arrays allow the timing window to be
chosen at selection time:

```python
cut_min_ip(0.06, dt_chi2=3.5)
cut_min_ip_chi2(4.0, dt_chi2=3.5)
# no timing restriction:
cut_min_ip(0.06)
```

`compute_track_pv_pairs` also provides `dt_chi2` for every track-PV pair:

```
dt_chi2 = dt_corrected^2 / (sigma_track_time^2 + sigma_pv_time^2)
```

`pvs_on_time_for_tracks` returns the event-local PV indices passing a `dt` and/or `dt_chi2` cut
for each track. `tracks_on_time_for_pvs` provides the reverse mapping, including an optional track
mask when only a subset such as signal tracks is wanted.

`reduce_track_pv_pairs` applies the timing mask to the all-PV pair table and returns aligned
event/track/selected-PV lists containing the original PV index, IP, IP chi2, dt, and dt chi2.

`set_track_pv_ip_statistics` writes the compatible-PV multiplicity, smallest IP, second-smallest
IP, and their PV indices into the track container. Its optional `pv_mask` selects the PV indices to
consider separately for each track; without it, all PVs are considered. A supplied empty mask does
not fall back: it gives `n_pvs = 0`, NaN IP values, and indices of `-1`.

For composites, the same logic applies but the IP is computed by extrapolating the composite
flight direction (as a straight line using `tx = dx/dz`, `ty = dy/dz`) back to each PV z.

## Vertex fit

The spatial vertex fit is a least-squares minimization. Each track contributes two constraint
equations:

```
x + tx * (z_v - z) = x_v
y + ty * (z_v - z) = y_v
```

This gives the design matrix rows `H_i = [[1, 0, -tx_i], [0, 1, -ty_i]]` and the normal
equations `(A^T A) v = A^T b` where `v = (x_v, y_v, z_v)`.

The normal equation matrix is:

```
A^T A = [[n,   0,    -sum(tx)        ],
         [0,   n,    -sum(ty)        ],
         [-sum(tx), -sum(ty), sum(tx^2+ty^2)]]
```

When track covariances are available, the fit is weighted: each track's 2x2 covariance
`V_i(z_v)` is propagated to the vertex z, inverted to get weight `W_i = V_i^{-1}`, and the
weighted normal equations `(H^T W H) v = H^T W b` are solved.

The vertex covariance is `cov_v = (H^T W H)^{-1}`, stored as a 3x3 symmetric matrix with fields
`vertex_cov_i_j`.

The spatial chi2 is `sum_i (dx_i, dy_i) @ W_i @ (dx_i, dy_i)^T` where `(dx_i, dy_i)` is the
residual of track i at vertex z.

## DOCA (Distance of Closest Approach)

For two straight-line tracks with direction vectors `u_0`, `u_1` and positions `P_0`, `P_1`,
the DOCA is the minimum distance between the two lines. This reduces to solving:

```
[[a, -b], [-b, c]] @ [s, t] = [-d, e]
```

where `a = u_0 . u_0`, `b = u_0 . u_1`, `c = u_1 . u_1`, `d = u_0 . w_0`, `e = u_1 . w_0`,
and `w_0 = P_0 - P_1`. The DOCA is then `|w_0 + s*u_0 - t*u_1|`.

For parallel tracks (`det ~ 0`), a simplified formula is used.

For n-body combinations, DOCA is computed for all unique pairs, stored as `doca12`, `doca13`, etc.
The maximum over all pairs is stored as `max_doca`.

The corresponding `doca12_chi2`, etc. use the transverse separation at the two closest-approach
points. Each track's 5x5 covariance, ordered as `(x, y, tx, ty, q/p)`, is propagated from its own
reference z to its closest-approach z with the straight-line transport Jacobian. The two propagated
position covariance blocks are summed and used for a 2D Mahalanobis chi2. Summary fields are
`min_doca_chi2` and `max_doca_chi2`.

## Vertex time fit

Track times are first propagated to the vertex z, correcting for mass-dependent speed:

```
path = (z_v - z_track) * sqrt(1 + tx^2 + ty^2)
E = sqrt(p^2 + m^2)
beta = p / E
t_propagated = t_track + path / (beta * c)
```

The vertex time is a weighted average:

```
w_i = 1 / sigma_t_i^2
t_v = sum(w_i * t_i) / sum(w_i)
sigma_t_v = 1 / sqrt(sum(w_i))
```

The vertex time chi2 is `sum_i (t_i - t_v)^2 / sigma_t_i^2`.

### Pairwise time chi2

An alternative metric, useful for timing quality cuts. For each unique pair of daughters (i, j):

```
chi2_ij = (t_i - t_j)^2 / (sigma_i^2 + sigma_j^2)
```

The mean over all pairs is returned.

## Composite-PV association

Before fitting, `combine` can optionally require daughters to share the same timing-qualified
best PV (`require_same_best_pv=True`) or at least one entry in their `pv_on_time` lists
(`require_common_pv_on_time=True`). These requirements only select daughter combinations.

After the vertex and timing fits, the new composite is independently associated to its best PV.
Its own time and reduced time uncertainty are used to rebuild the `dt_chi2` mask; the daughter
intersection is not reused as the composite mask.
The following quantities are computed for each (candidate, PV) pair:

### Composite IP and IP chi2

Same as track IP but using the composite flight direction and propagated vertex covariance.

### DIRA (Direction Angle)

Cosine of the angle between the composite momentum and flight direction (SV - PV):

```
DIRA = (p . f) / (|p| * |f|)
```

where `f = (x_sv - x_pv, y_sv - y_pv, z_sv - z_pv)`.

### FD chi2 (Flight Distance Significance)

Full 3D Mahalanobis distance:

```
delta = (x_sv - x_pv, y_sv - y_pv, z_sv - z_pv)
V_total = V_sv + V_pv       (3x3 covariance matrices)
fdchi2 = delta^T @ V_total^{-1} @ delta
```

### Flight eta

Pseudorapidity of the flight direction vector (PV to SV):

```
flight_eta = arctanh(fz / |f|)
```

### Corrected mass (mcor)

Accounts for missing transverse momentum (e.g. neutrinos):

```
p_perp^2 = |p x f|^2 / |f|^2
mcor = sqrt(m_vis^2 + p_perp^2) + sqrt(p_perp^2)
```

### Time residual and time chi2

The composite time is propagated back to the PV:

```
flight_time = (f . p_hat) / (beta * c)
t_at_pv = t_vertex - flight_time
time_residual = t_at_pv - t_pv
```

The time chi2 includes the vertex time uncertainty, PV time uncertainty, and flight distance
uncertainty propagated through `beta * c`:

```
sigma_flight^2 = (p_hat^T @ V_sv @ p_hat + p_hat^T @ V_pv @ p_hat) / (beta*c)^2
sigma_total^2 = sigma_t_vertex^2 + sigma_t_pv^2 + sigma_flight^2
time_chi2 = time_residual^2 / sigma_total^2
```
