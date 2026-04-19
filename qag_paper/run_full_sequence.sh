#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f venv/bin/activate ]]; then
  echo "ERROR: venv not found at ./venv. Create it first."
  exit 1
fi

source venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('Environment OK')"

python src/data_loader.py --build-dataset || true
wc -l data/prompts/final_dataset.jsonl

python src/run_experiment.py --config configs/baseline.yaml
python gate1_pilot_check.py

python src/run_experiment.py --config configs/int8.yaml
python src/run_experiment.py --config configs/int4.yaml
python gate2_effect_check.py

python src/run_experiment.py --config configs/qwen_baseline.yaml
python src/run_experiment.py --config configs/qwen_int4.yaml

echo "Run sequence complete. Continue with notebooks for RDPS/FPS and figures."
