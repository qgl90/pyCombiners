# Truth Matching

The framework provides MC truth matching and LHCb-style background category out of the box.
Truth information is propagated automatically during `combine()`, so you don't need to do anything
special to use it.

## How it works

When `combine()` builds candidates, it looks at the MC ancestor chains of all daughters. If all
daughters share a common MC ancestor whose PDG ID matches the candidate's mother PID, the candidate
is truth-matched. The ancestor PDG IDs, keys, and per-daughter MC PIDs are all stored in the
candidate container automatically.

The fields propagated by `combine()` are:

| Field | Description |
|---|---|
| `mc_truth` | Jagged bool , whether each candidate is truth-matched |
| `mc_pid` | MC PDG ID of the common ancestor |
| `mc_key` | MC key of the common ancestor |
| `mc_fromsignal` | Whether all daughters come from the signal decay |
| `mc_ancestor_pids` | Doubly-jagged , ancestor PDG IDs per candidate |
| `mc_ancestor_keys` | Doubly-jagged , ancestor MC keys per candidate |

## truth_match_candidates

If you want to run truth matching manually (e.g. re-check after extra cuts), use:

```python
from trackcomb import truth_match_candidates

matched = truth_match_candidates(
    candidates
)  # jagged bool (events, candidates)
```

This checks if all daughters share a common MC ancestor matching the candidate's `pid` field,
and optionally verifies daughter PIDs match the expected `daughter{k}_pid`.

## count_true_decays

Count how many true MC decays exist per event, useful for computing efficiency denominators:

```python
from trackcomb import count_true_decays

# Count all true Ks decays (any daughter combination)
n_true_ks = count_true_decays(tracks, "K(S)0")

# Count only Ks -> pi+ pi- (require specific daughters)
n_true_ks = count_true_decays(tracks, "K(S)0", daughter_pdgs=["pi+", "pi-"])
```

## bkgcat , Background Categorization

The `bkgcat()` function assigns an LHCb-style background category to each candidate. This is
useful for understanding the composition of your selection:

```python
from trackcomb.truth import bkgcat

cats = bkgcat(candidates)  # jagged int (events, candidates)
```

### Category codes

The decision tree goes like this: first check for pathological cases (ghost, clone, hierarchy),
then check if daughters share a common ancestor, and finally classify by ancestor type.

| Code | Name | Condition |
|---|---|---|
| 0 | Signal | Common ancestor with correct mother PDG, correct daughter PIDs, all from signal |
| 10 | QuasiSignal | Same as Signal but not all daughters flagged as `from_signal` |
| 20 | PhysBkg | Common ancestor exists but wrong mother PDG |
| 30 | Reflection | Common ancestor with correct mother but wrong daughter PIDs |
| 60 | Ghost | At least one daughter is a ghost track (`mc_truth == 0`) |
| 63 | Clone | Two daughters share the same `mc_key` |
| 66 | Hierarchy | One daughter's `mc_key` appears in another's ancestor chain |
| 100 | Pileup | Daughters come from different PVs (`mc_pv_key`) |
| 110 | FromB | No common ancestor, but at least one daughter has a b-hadron ancestor |
| 120 | FromC | No common ancestor, at least one daughter has a c-hadron ancestor |
| 130 | LightParticle | None of the above (default) |

Categories 40 (PartReco) and 50 (LowMass) from the full LHCb scheme are omitted because they
require the complete MC particle tree, which is not available in our SoA tuple format.

### Decision flow

```
Ghost? ──yes──> 60
  │no
Clone? ──yes──> 63
  │no
Hierarchy? ──yes──> 66
  │no
Common ancestor? ──yes──> Correct daughter PIDs?
  │no                       │no ──> 30 (Reflection)
  │                         │yes
  │                       Correct mother PDG?
  │                         │no ──> 20 (PhysBkg)
  │                         │yes
  │                       All from signal?
  │                         │yes ──> 0 (Signal)
  │                         │no  ──> 10 (QuasiSignal)
  │
Different PVs? ──yes──> 100 (Pileup)
  │no
b-hadron ancestor? ──yes──> 110 (FromB)
  │no
c-hadron ancestor? ──yes──> 120 (FromC)
  │no
  └──> 130 (LightParticle)
```

### Example usage

```python
import awkward as ak
from trackcomb.truth import bkgcat

cats = bkgcat(candidates)

# Count signal candidates across all events
n_signal = ak.sum(cats == 0)

# Get signal-only candidates
from trackcomb import apply_mask

signal_mask = cats == 0
signal_candidates = apply_mask(candidates, signal_mask)
```

### Note on `from_signal`

Many MC samples do not have the `from_signal` flag properly set, in which case `bkgcat` will
return 10 (QuasiSignal) instead of 0 (Signal) for true signal candidates. If you are working with
such samples, use `cats <= 10` instead of `cats == 0` to select signal.
