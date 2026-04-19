#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY="/home/muham/tf-env/bin/python"
mkdir -p results/logs
LOG="results/logs/overnight_stress_completion_$(date +%Y%m%d_%H%M%S).log"

active_pid="$(ps -ef | awk '/bash \.\/run_stress_sequence.sh/ && !/awk/ {print $2; exit}')"

{
  echo "[start] $(date -Is)"
  echo "[info] log=$LOG"
  if [[ -n "${active_pid:-}" ]]; then
    echo "[wait] waiting for active stress sequence pid=$active_pid"
    tail --pid="$active_pid" -f /dev/null
    echo "[wait] active stress sequence finished"
  else
    echo "[wait] no active stress sequence found"
  fi

  echo "[run] starting qwen_stress_int8"
  "$PY" src/run_experiment.py --config configs/qwen_stress_int8.yaml

  echo "[run] regenerating final reports"
  "$PY" src/analysis/generate_extended_report.py --project-root . --runs-dir results/runs

  echo "[done] $(date -Is)"
} >> "$LOG" 2>&1

echo "$LOG"