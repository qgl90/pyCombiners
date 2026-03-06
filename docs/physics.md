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

For tracks, the best PV is the one with the smallest IP, optionally pre-filtered by
`|dt_corrected| < threshold`. If no PV passes the time cut, the filter is dropped and all PVs
are considered.

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

After the vertex and timing fits, each composite candidate is associated to its best PV.
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
