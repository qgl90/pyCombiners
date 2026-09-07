# Signal-track to PV association study

This study starts from reconstructed long tracks, keeps tracks marked `mc_fromsignal`, and compares
their IP ranking after different track-PV timing preselections. It produces one Parquet row per
signal track and timing threshold.

For every scan point the ordering is:

```text
all_pairs = IP(track, all event PVs)
mask = abs(dt) <= threshold            # or dt_chi2 <= threshold
considered_pvs = reduce(all_pairs, mask)
min_ip, second_min_ip = rank(considered_pvs.ip)
```

The IP to every PV is calculated once because an individual track-PV IP does not depend on the
other PVs. The timing mask then reduces the PV indices separately for each track, and only this
reduced object is used for IP ranking.

In the reusable API the reduction is optional and represented by the boolean PV-index mask:

```python
# No reduction: rank IP against every event PV.
set_track_pv_ip_statistics(tracks, all_pairs, pv_mask=None)

# Reduction: rank IP only against PV indices selected for each track.
set_track_pv_ip_statistics(tracks, all_pairs, pv_mask=mask)
```

## Input

Run from the repository root with the `pyCombiner` environment activated and the package available
through `PYTHONPATH=src`.

The following is a real Upgrade-II `Bs_Jpsimm_Phi` input pattern already used by this repository:

```bash
INPUT='/eos/lhcb/wg/rta/WP6/tdr_u2_august2026/middle-scenario-july2026_1p0e34/Bs_Jpsimm_Phi/moore/*_slot_*event_container.root'
```

Check that files are visible before launching a large job:

```bash
ls $INPUT | head
```

For a quick test, replace `--max-events 20000` below with `--max-events 100` and use
`--workers 1`.

## Produce the scan dataframe

```bash
PYTHONPATH=src python3 physics/reconstruction/signal_track_pv_study.py \
  --input "$INPUT" \
  --max-events 20000 \
  --chunk-size 100 \
  --workers 10 \
  --dt-thresholds 0.01 0.02 0.03 0.05 0.07 0.10 0.20 0.50 \
  --dt-chi2-thresholds 1 4 9 16 25 50 100 \
  --out public/pv_association/signal_track_pv_scan.parquet
```

The `dt` thresholds are absolute flight-corrected time residuals in ns. The `dt_chi2` thresholds
are dimensionless. Passing both lists creates independent scan rows: it does not apply both cuts at
the same time. The producer also writes one explicit no-timing reference row with
`timing_metric == "none"` and `timing_threshold == inf`. For that row the PV mask is unrestricted,
so IP quantities are ranked against every PV in the event.

The most relevant output columns are:

- `timing_metric`, `timing_threshold`: timing preselection represented by the row.
- `n_pvs_considered`: number of PVs passing that preselection for the signal track.
- `considered_pv_indices`: event-local indices of the PV subset passing the timing cut.
- `considered_pv_ip`, `considered_pv_ip_chi2`: per-PV values for that same subset.
- `considered_pv_dt`, `considered_pv_dt_chi2`: timing values that defined the subset.
- `min_ip`, `second_min_ip`: smallest and second-smallest IP inside the subset.
- `min_ip_pv_index`, `second_min_ip_pv_index`: corresponding event-local PV indices.
- `min_ip_all_pvs`, `second_min_ip_all_pvs`: reference values before timing selection.
- `true_pv_on_time`, `best_is_true_pv`: MC validation columns.
- `n_tracks_on_time_true_pv`, `n_tracks_on_time_best_pv`: reverse PV-to-track multiplicities using
  the full reconstructed track collection.

The five `considered_pv_*` list columns are aligned element by element. For example, element `i` of
`considered_pv_ip` is the IP to PV index `considered_pv_indices[i]` and passed the stored timing
criterion. This makes the timing reduction and subsequent IP ranking inspectable directly from the
Parquet.

There is deliberately no fallback to all PVs. If no PV passes, `n_pvs_considered == 0`, `min_ip`
and `second_min_ip` are `NaN`, all considered-PV lists are empty, and both summary PV indices are
`-1`. If exactly one PV passes, `min_ip` is defined but `second_min_ip` is `NaN`. Normal pandas
comparisons reject these `NaN` rows, so such a track does not pass an IP selection.

## Plot the scan points

The analyzer reads the thresholds directly from the scan Parquet; the scan grid does not need to be
repeated on the command line:

```bash
PYTHONPATH=src python3 \
  physics/analysis/pv_association_study/track_pv_timing_scan.py \
  --input bs_jpsi_phi/signal_track_pv_study/0p2_lumi.parquet \
  --out-dir bs_jpsi_phi/signal_track_pv_study/plots \
  --lumi 0p2e34
```

This writes `track_pv_timing_scan.png` with separate `dt` and `dt_chi2` rows and
`track_pv_timing_scan_summary.parquet` containing one row per actual scan point. The plotted
quantities are separated into three columns: signal-track efficiency, the track-weighted mean
number of selected PVs per signal track, and the PV-weighted mean number of on-time tracks per PV.
Each panel includes the corresponding no-time-cut reference derived from the complete event track
and PV collections; it is not approximated using the loosest timing scan point. In the plot this
reference is drawn as a star at a labelled `No cut` endpoint, one bounded logarithmic step after the
last finite threshold. The stored infinite threshold is therefore never used as an axis coordinate
and cannot make the finite scan unreadable.

## Evaluate an IP selection

For example, require `min_ip <= 0.10 mm` and `second_min_ip >= 0.20 mm`:

```bash
PYTHONPATH=src python3 \
  physics/analysis/pv_association_study/signal_track_ip_efficiency.py \
  --input public/pv_association/signal_track_pv_scan.parquet \
  --min-ip-max 0.10 \
  --second-ip-min 0.20 \
  --out public/pv_association/ip_efficiency.parquet
```

The reported denominator is all `mc_fromsignal` tracks for each timing threshold. The table also
reports `n_zero_pvs`, the fraction retaining at least one PV, and the fraction retaining a second
PV. Therefore losses caused purely by the timing preselection remain visible in the efficiency.

Available cut directions are `--min-ip-min`, `--min-ip-max`, `--second-ip-min`, and
`--second-ip-max`; any combination can be supplied.

The helper is optional: all quantities needed for alternative selections and plots are stored in
`signal_track_pv_scan.parquet`. No candidate is removed when the dataframe is produced, including
tracks with zero compatible PVs. Consequently users can define different IP cuts without rerunning
reconstruction.

## Analyse the Parquet directly

The following standalone example reads only the scan Parquet, applies an IP selection, calculates
the signal-track efficiency for every timing threshold, and makes a plot:

```bash
python3 - <<'PY'
import matplotlib.pyplot as plt
import pandas as pd

path = "public/pv_association/signal_track_pv_scan.parquet"
df = pd.read_parquet(path)

# Example selection. NaN comparisons are False, so tracks with zero PVs or
# without a second PV fail naturally.
df["selected"] = (
    (df["min_ip"] <= 0.10)
    & (df["second_min_ip"] >= 0.20)
)

summary = (
    df.groupby(["timing_metric", "timing_threshold"], as_index=False)
    .agg(
        n_signal_tracks=("track_index", "size"),
        n_selected=("selected", "sum"),
        n_zero_pvs=("n_pvs_considered", lambda values: (values == 0).sum()),
        fraction_with_pv=(
            "n_pvs_considered", lambda values: (values >= 1).mean()
        ),
        fraction_with_second_pv=(
            "n_pvs_considered", lambda values: (values >= 2).mean()
        ),
    )
)
summary["efficiency"] = summary["n_selected"] / summary["n_signal_tracks"]
print(summary.to_string(index=False))

for metric, group in summary.groupby("timing_metric"):
    plt.plot(
        group["timing_threshold"],
        group["efficiency"],
        marker="o",
        label=metric,
    )

plt.xlabel("Timing threshold")
plt.ylabel("Signal-track efficiency")
plt.legend()
plt.tight_layout()
plt.savefig("public/pv_association/ip_efficiency.png", dpi=150)
PY
```

To inspect the available columns interactively:

```bash
python3 -c "import pandas as pd; d=pd.read_parquet('public/pv_association/signal_track_pv_scan.parquet'); print(d.columns.tolist()); print(d.head())"
```
