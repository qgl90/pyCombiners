#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

mkdir -p two_track_mva_comparison/1p0e34_timed/bs/plots
mkdir -p two_track_mva_comparison/1p0e34_timed/minbias/plots
mkdir -p two_track_mva_comparison/0p2e34_no_timing/bs/plots
mkdir -p two_track_mva_comparison/0p2e34_no_timing/minbias/plots
mkdir -p two_track_mva_comparison/comparisons

# 1p0e34: timing enabled, dt_chi2 < 16, vertex-time chi2 < 16,
# and at least one common on-time PV required for the daughter tracks.
./myenv/run PYTHONPATH=src python3 physics/reconstruction/two_track_mva_pv_study.py \
  --input '/home/rquaglia/work/samples/tdr_u2_august2026/middle-scenario-july2026_1p0e34/Bs_Jpsimm_Phi/moore-rquaglia-2September2026/*_event_container.root' \
  --out two_track_mva_comparison/1p0e34_timed/bs/candidates.parquet \
  --label Bs_1p0e34_timed_common \
  --max-events 40000 \
  --chunk-size 1000 \
  --workers 20 \
  --track-samples all_tracks truth_matched \
  --max-dt-chi2 16 \
  --max-vertex-time-chi2 16 \
  --require-common-pv-on-time

./myenv/run PYTHONPATH=src python3 physics/reconstruction/two_track_mva_pv_study.py \
  --input '/home/rquaglia/work/samples/tdr_u2_august2026/middle-scenario-july2026_1p0e34/MINBIAS/moore-rquaglia-2September2026/*_event_container.root' \
  --out two_track_mva_comparison/1p0e34_timed/minbias/candidates.parquet \
  --label MINBIAS_1p0e34_timed_common \
  --max-events 40000 \
  --chunk-size 1000 \
  --workers 20 \
  --track-samples all_tracks truth_matched \
  --max-dt-chi2 16 \
  --max-vertex-time-chi2 16 \
  --require-common-pv-on-time

# 0p2e34: all timing selections disabled and all daughter pairs retained.
./myenv/run PYTHONPATH=src python3 physics/reconstruction/two_track_mva_pv_study.py \
  --input '/home/rquaglia/work/samples/tdr_u2_august2026/middle-scenario-july2026_0p2e34/Bs_Jpsimm_Phi/moore-rquaglia-2September2026/*_event_container.root' \
  --out two_track_mva_comparison/0p2e34_no_timing/bs/candidates.parquet \
  --label Bs_0p2e34_no_timing_all_pairs \
  --max-events 40000 \
  --chunk-size 1000 \
  --workers 20 \
  --track-samples all_tracks truth_matched \
  --disable-pv-timing \
  --disable-vertex-time-cut \
  --no-require-common-pv-on-time

./myenv/run PYTHONPATH=src python3 physics/reconstruction/two_track_mva_pv_study.py \
  --input '/home/rquaglia/work/samples/tdr_u2_august2026/middle-scenario-july2026_0p2e34/MINBIAS/moore-rquaglia-2September2026/*_event_container.root' \
  --out two_track_mva_comparison/0p2e34_no_timing/minbias/candidates.parquet \
  --label MINBIAS_0p2e34_no_timing_all_pairs \
  --max-events 40000 \
  --chunk-size 1000 \
  --workers 20 \
  --track-samples all_tracks truth_matched \
  --disable-pv-timing \
  --disable-vertex-time-cut \
  --no-require-common-pv-on-time

# Explicit threshold scans.
./myenv/run PYTHONPATH=src python3 physics/analysis/two_track_mva_pv_study/analyze.py \
  --input two_track_mva_comparison/1p0e34_timed/bs/candidates.parquet \
  --out two_track_mva_comparison/1p0e34_timed/bs/scan.parquet \
  --plot-dir two_track_mva_comparison/1p0e34_timed/bs/plots \
  --plot-pv-policy common_on_time \
  --multiplicity-cut 0.9569 \
  --charge all \
  --thresholds 0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.95 0.9569 0.96 0.97 0.98 0.99

./myenv/run PYTHONPATH=src python3 physics/analysis/two_track_mva_pv_study/analyze.py \
  --input two_track_mva_comparison/1p0e34_timed/minbias/candidates.parquet \
  --out two_track_mva_comparison/1p0e34_timed/minbias/scan.parquet \
  --plot-dir two_track_mva_comparison/1p0e34_timed/minbias/plots \
  --plot-pv-policy common_on_time \
  --multiplicity-cut 0.9569 \
  --charge all \
  --thresholds 0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.95 0.9569 0.96 0.97 0.98 0.99

./myenv/run PYTHONPATH=src python3 physics/analysis/two_track_mva_pv_study/analyze.py \
  --input two_track_mva_comparison/0p2e34_no_timing/bs/candidates.parquet \
  --out two_track_mva_comparison/0p2e34_no_timing/bs/scan.parquet \
  --plot-dir two_track_mva_comparison/0p2e34_no_timing/bs/plots \
  --plot-pv-policy all_pairs \
  --multiplicity-cut 0.9569 \
  --charge all \
  --thresholds 0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.95 0.9569 0.96 0.97 0.98 0.99

./myenv/run PYTHONPATH=src python3 physics/analysis/two_track_mva_pv_study/analyze.py \
  --input two_track_mva_comparison/0p2e34_no_timing/minbias/candidates.parquet \
  --out two_track_mva_comparison/0p2e34_no_timing/minbias/scan.parquet \
  --plot-dir two_track_mva_comparison/0p2e34_no_timing/minbias/plots \
  --plot-pv-policy all_pairs \
  --multiplicity-cut 0.9569 \
  --charge all \
  --thresholds 0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.95 0.9569 0.96 0.97 0.98 0.99

# Compare the timed 1p0e34 scans with the untimed 0p2e34 scans.
./myenv/run PYTHONPATH=src python3 physics/analysis/two_track_mva_pv_study/compare.py \
  --series 1p0e34_timed_common two_track_mva_comparison/1p0e34_timed/bs/scan.parquet common_on_time \
  --series 0p2e34_no_timing_all_pairs two_track_mva_comparison/0p2e34_no_timing/bs/scan.parquet all_pairs \
  --track-sample all_tracks \
  --charge all \
  --out two_track_mva_comparison/comparisons/bs_all_tracks.png

./myenv/run PYTHONPATH=src python3 physics/analysis/two_track_mva_pv_study/compare.py \
  --series 1p0e34_timed_common two_track_mva_comparison/1p0e34_timed/bs/scan.parquet common_on_time \
  --series 0p2e34_no_timing_all_pairs two_track_mva_comparison/0p2e34_no_timing/bs/scan.parquet all_pairs \
  --track-sample truth_matched \
  --charge all \
  --out two_track_mva_comparison/comparisons/bs_truth_matched.png

./myenv/run PYTHONPATH=src python3 physics/analysis/two_track_mva_pv_study/compare.py \
  --series 1p0e34_timed_common two_track_mva_comparison/1p0e34_timed/minbias/scan.parquet common_on_time \
  --series 0p2e34_no_timing_all_pairs two_track_mva_comparison/0p2e34_no_timing/minbias/scan.parquet all_pairs \
  --track-sample all_tracks \
  --charge all \
  --out two_track_mva_comparison/comparisons/minbias_all_tracks.png

./myenv/run PYTHONPATH=src python3 physics/analysis/two_track_mva_pv_study/compare.py \
  --series 1p0e34_timed_common two_track_mva_comparison/1p0e34_timed/minbias/scan.parquet common_on_time \
  --series 0p2e34_no_timing_all_pairs two_track_mva_comparison/0p2e34_no_timing/minbias/scan.parquet all_pairs \
  --track-sample truth_matched \
  --charge all \
  --out two_track_mva_comparison/comparisons/minbias_truth_matched.png
