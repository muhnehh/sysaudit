#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY="/home/muham/tf-env/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Missing Python at $PY"
  exit 1
fi

echo "[1/4] Running Gemma stress baseline fp16..."
"$PY" src/run_experiment.py --config configs/gemma_stress_baseline.yaml

echo "[2/4] Running Gemma stress int4..."
"$PY" src/run_experiment.py --config configs/gemma_stress_int4.yaml

echo "[3/4] Regenerating summaries..."
"$PY" src/analysis/generate_extended_report.py

echo "[4/4] Done. See results/final_plan_summary.md"