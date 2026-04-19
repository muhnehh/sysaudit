#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY="/home/muham/tf-env/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Missing Python at $PY"
  exit 1
fi

echo "[1/6] Building jailbreak-stress dataset..."
"$PY" src/data_loader.py \
  --build-jailbreak-stress \
  --base-dataset data/prompts/final_dataset.jsonl \
  --output data/prompts/final_dataset_jailbreak_stress.jsonl

echo "[2/6] Running stress baseline fp16..."
"$PY" src/run_experiment.py --config configs/qwen_stress_baseline.yaml

echo "[3/6] Running stress int4..."
"$PY" src/run_experiment.py --config configs/qwen_stress_int4.yaml

if [[ "${RUN_STRESS_INT8:-0}" == "1" ]]; then
  echo "[4/6] Running stress int8 (optional)..."
  "$PY" src/run_experiment.py --config configs/qwen_stress_int8.yaml
else
  echo "[4/6] Skipping stress int8 (set RUN_STRESS_INT8=1 to enable)."
fi

echo "[5/6] Regenerating summaries..."
"$PY" src/analysis/generate_extended_report.py

echo "[6/6] Done. See results/final_plan_extended.md"
