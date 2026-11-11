#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
causal_python="${CAUSAL_PYTHON:-python}"
package=li_junhao_trajectronpp
scripts="$package/scripts"
root="$package/outputs/causal_v2"
runs="$root/runs"
results="$root/results"
mkdir -p "$results"

"$causal_python" "$scripts/audit_independent_formal.py"
"$causal_python" "$scripts/aggregate_independent_results.py" --runs-root "$runs" --output "$results/results.csv"
"$causal_python" "$scripts/aggregate_spatial.py" --runs-root "$runs" --output "$results/spatial.csv"
"$causal_python" "$scripts/analyze_statistics.py" --input "$results/results.csv" --output-dir "$results"
"$causal_python" "$scripts/analyze_sample_budget.py" --runs-root "$runs" --output "$results/sample_budget.csv"
"$causal_python" "$scripts/analyze_independent_diagnostics.py" --runs-root "$runs" --data-root data/causal_v2 --output-dir "$results"
"$causal_python" "$scripts/generate_failure_analysis.py" --runs-root "$runs" --data-root data/causal_v2 --output-dir "$results"
"$causal_python" "$scripts/benchmark_independent_runtime.py" --runs-root "$runs" --output "$results/runtime.csv"
"$causal_python" "$scripts/prepare_causal_report.py"
bash "$package/report/build_latex.sh"
