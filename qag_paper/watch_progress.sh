#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="/home/muham/tf-env/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing Python at $PY"
  exit 1
fi

while true; do
  clear
  "$PY" - <<'PY'
import glob
import json
import os
import time
from pathlib import Path

root = Path('/home/muham/sysaudit/qag_paper/results/runs')
runs = sorted(glob.glob(str(root / '*_*_*')), key=os.path.getmtime, reverse=True)
if not runs:
    print('No run directories found yet.')
    raise SystemExit(0)

latest = Path(runs[0])
progress = latest / 'progress.json'
summary = latest / 'summary.json'

print('QAG Live Progress')
print('=' * 60)
print(f'Run dir: {latest}')

if progress.exists():
    data = json.loads(progress.read_text(encoding='utf-8'))
    total = max(int(data.get('total', 0)), 1)
    completed = int(data.get('completed', 0))
    pct = (completed / total) * 100
    eta = data.get('eta_seconds', None)
    elapsed = float(data.get('elapsed_seconds', 0.0))
    avg_ms = float(data.get('avg_latency_ms', 0.0))
    stage = data.get('stage', 'unknown')
    status = data.get('status', 'unknown')

    bar_len = 40
    filled = int(bar_len * completed / total)
    bar = '#' * filled + '-' * (bar_len - filled)

    print(f"Status: {status}")
    print(f"Stage:  {stage}")
    print(f"[{bar}] {completed}/{total} ({pct:.1f}%)")
    print(f"Elapsed: {elapsed/60:.1f} min")
    if eta is None:
        print('ETA: unknown (loading stage)')
    else:
        print(f"ETA: {float(eta)/60:.1f} min")
    print(f"Avg latency: {avg_ms:.0f} ms/prompt")
else:
    print('Progress file not created yet.')

if summary.exists():
    s = json.loads(summary.read_text(encoding='utf-8'))
    print('\nLast summary available:')
    print(f"  Experiment: {s.get('experiment_name')}")
    print(f"  Refusal rate harmful: {s.get('refusal_rate_harmful')}")
    print(f"  Runtime min: {float(s.get('total_runtime_seconds', 0))/60:.1f}")

print('\n(Refreshing every 3 seconds; Ctrl+C to exit)')
PY
  sleep 3
done
