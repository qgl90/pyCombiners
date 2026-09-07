#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

mkdir -p bs_jpsiphi_1p0e34_timed/reconstruction
mkdir -p bs_jpsiphi_1p0e34_timed/analysis

# Truth-assisted reconstructible signal denominator.
./myenv/run PYTHONPATH=src python3 \
  physics/reconstruction/bs_to_jpsiphi.py \
  --mode cheated \
  --input '/home/rquaglia/work/samples/tdr_u2_august2026/middle-scenario-july2026_1p0e34/Bs_Jpsimm_Phi/moore-rquaglia-2September2026/*_event_container.root' \
  --max-events 40000 \
  --chunk-size 1000 \
  --workers 20 \
  --max-dt-chi2 16 \
  --require-common-pv-on-time \
  --min-track-ip-chi2 9 \
  --min-jpsi-fd-chi2 30 \
  --max-jpsi-vertex-chi2 30 \
  --max-jpsi-vertex-time-chi2 16 \
  --max-phi-vertex-chi2 25 \
  --max-phi-vertex-time-chi2 16 \
  --max-bs-vertex-chi2 9 \
  --max-bs-vertex-time-chi2 16 \
  --max-bs-ip-chi2 25 \
  --min-bs-dira 0.9995 \
  --out-file bs_jpsiphi_1p0e34_timed/reconstruction/cheated.parquet

# The same truth-guided signal tracks, now with every HLT2-like track,
# combination, vertex, timing, and PV-dependent selection enabled.
./myenv/run PYTHONPATH=src python3 \
  physics/reconstruction/bs_to_jpsiphi.py \
  --mode cheated_selected \
  --input '/home/rquaglia/work/samples/tdr_u2_august2026/middle-scenario-july2026_1p0e34/Bs_Jpsimm_Phi/moore-rquaglia-2September2026/*_event_container.root' \
  --max-events 40000 \
  --chunk-size 1000 \
  --workers 20 \
  --max-dt-chi2 16 \
  --require-common-pv-on-time \
  --min-track-ip-chi2 9 \
  --min-jpsi-fd-chi2 30 \
  --max-jpsi-vertex-chi2 30 \
  --max-jpsi-vertex-time-chi2 16 \
  --max-phi-vertex-chi2 25 \
  --max-phi-vertex-time-chi2 16 \
  --max-bs-vertex-chi2 9 \
  --max-bs-vertex-time-chi2 16 \
  --max-bs-ip-chi2 25 \
  --min-bs-dira 0.9995 \
  --out-file bs_jpsiphi_1p0e34_timed/reconstruction/cheated_selected.parquet

# Full HLT2-like selection. Muon and kaon DLL cuts are currently disabled in
# bs_to_jpsiphi.py; all other track, vertex, timing, and PV cuts are active.
./myenv/run PYTHONPATH=src python3 \
  physics/reconstruction/bs_to_jpsiphi.py \
  --mode full \
  --input '/home/rquaglia/work/samples/tdr_u2_august2026/middle-scenario-july2026_1p0e34/Bs_Jpsimm_Phi/moore-rquaglia-2September2026/*_event_container.root' \
  --max-events 40000 \
  --chunk-size 1000 \
  --workers 20 \
  --max-dt-chi2 16 \
  --require-common-pv-on-time \
  --min-track-ip-chi2 9 \
  --min-jpsi-fd-chi2 30 \
  --max-jpsi-vertex-chi2 30 \
  --max-jpsi-vertex-time-chi2 16 \
  --max-phi-vertex-chi2 25 \
  --max-phi-vertex-time-chi2 16 \
  --max-bs-vertex-chi2 9 \
  --max-bs-vertex-time-chi2 16 \
  --max-bs-ip-chi2 25 \
  --min-bs-dira 0.9995 \
  --out-file bs_jpsiphi_1p0e34_timed/reconstruction/full.parquet

# Event-level cut flow using only truth-matched fromSignal tracks. This pass
# records cumulative survival through the four-track, J/psi, phi, and Bs stages.
./myenv/run PYTHONPATH=src python3 \
  physics/reconstruction/bs_to_jpsiphi.py \
  --mode signal_cutflow \
  --input '/home/rquaglia/work/samples/tdr_u2_august2026/middle-scenario-july2026_1p0e34/Bs_Jpsimm_Phi/moore-rquaglia-2September2026/*_event_container.root' \
  --max-events 40000 \
  --chunk-size 1000 \
  --workers 20 \
  --max-dt-chi2 16 \
  --require-common-pv-on-time \
  --min-track-ip-chi2 9 \
  --min-jpsi-fd-chi2 30 \
  --max-jpsi-vertex-chi2 30 \
  --max-jpsi-vertex-time-chi2 16 \
  --max-phi-vertex-chi2 25 \
  --max-phi-vertex-time-chi2 16 \
  --max-bs-vertex-chi2 9 \
  --max-bs-vertex-time-chi2 16 \
  --max-bs-ip-chi2 25 \
  --min-bs-dira 0.9995 \
  --out-file bs_jpsiphi_1p0e34_timed/reconstruction/signal_cutflow.parquet

# Signal efficiency, efficiency versus kinematics, and Bs/Jpsi/phi mass plots.
./myenv/run PYTHONPATH=src python3 \
  physics/analysis/decay_performance_study/decay_performance.py \
  --cheated bs_jpsiphi_1p0e34_timed/reconstruction/cheated.parquet \
  --cheated-selected bs_jpsiphi_1p0e34_timed/reconstruction/cheated_selected.parquet \
  --full bs_jpsiphi_1p0e34_timed/reconstruction/full.parquet \
  --cutflow bs_jpsiphi_1p0e34_timed/reconstruction/signal_cutflow.parquet \
  --channel bs_to_jpsiphi \
  --lumi 1.0e34 \
  --tag 'dt_chi2 < 16; vertex-time chi2 < 16; DLL cuts disabled' \
  --out-dir bs_jpsiphi_1p0e34_timed/analysis

echo
echo 'Signal-selection efficiency table:'
column -s, -t < bs_jpsiphi_1p0e34_timed/analysis/bs_selection_efficiency.csv

echo
echo 'Signal-only cumulative selection cut flow:'
column -s, -t < bs_jpsiphi_1p0e34_timed/analysis/bs_selection_cutflow.csv

echo
echo 'Full-reconstruction background-category candidate yields:'
column -s, -t < bs_jpsiphi_1p0e34_timed/analysis/bs_bkgcat_yields.csv

echo
echo 'Plots and tables saved under bs_jpsiphi_1p0e34_timed/analysis'
