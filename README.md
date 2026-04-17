# Systematic Audit of Prefill Jailbreak Detection Methods

This repository benchmarks four published jailbreak detection methods under one evaluation pipeline:

- Refusal Direction (RepE baseline)
- TrajGuard
- Jailbreaking Leaves a Trace (JLT)
- JBShield

The implementation is optimized for constrained hardware (target: 6GB VRAM) by using 4-bit quantized inference and one-prompt-at-a-time processing.

## Project Structure

```text
sysaudit/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── run_benchmark.py
│   ├── utils.py
│   └── methods/
│       ├── base_detector.py
│       ├── refusal_direction.py
│       ├── trajguard.py
│       ├── jailbreaking_leaves_trace.py
│       └── jbshield.py
├── results/
│   ├── metrics.csv
│   ├── figures/
│   └── logs/
├── requirements.txt
└── setup.sh
```

## Environment Setup

### 1) Create environment and install dependencies

Linux/macOS:

```bash
bash setup.sh
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Model access

Default model in code:

- `meta-llama/Llama-3.1-8B-Instruct`

If your environment cannot access gated weights, either:

- authenticate with Hugging Face (`huggingface-cli login`) and optionally pass `--hf-token`, or
- pass another compatible causal LM at runtime with `--model-name`.

Example open model for smoke runs:

- `sshleifer/tiny-gpt2`

## Data Preparation

The data loader downloads and preprocesses:

- AdvBench harmful prompts
- JailbreakBench artifact prompts via the official `jailbreakbench` package
- Alpaca benign instructions

Then it creates:

- `benign_prompts.jsonl`
- `jailbreak_prompts.jsonl`
- `train_prompts.jsonl`
- `val_prompts.jsonl`
- `test_prompts.jsonl`

Run:

```bash
python -m src.data_loader \
  --raw-dir data/raw \
  --processed-dir data/processed \
  --benign-count 500 \
  --advbench-count 500 \
  --jailbreakbench-count 500 \
  --jbb-artifact-methods PAIR,GCG,JBC,DSN,prompt_with_random_search \
  --jbb-artifact-models vicuna-13b-v1.5,llama-2-7b-chat-hf \
  --seed 42
```

If artifact loading fails due network issues, the loader automatically falls back to CSV/Hugging Face sources.

## Run Full Benchmark

```bash
python -m src.run_benchmark \
  --prepare-data \
  --raw-dir data/raw \
  --processed-dir data/processed \
  --results-dir results \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --hf-token <your_hf_token_if_needed> \
  --max-context-tokens 2048 \
  --max-memory-gb 6 \
  --seed 42
```

## Run Smoke Benchmark (Fast Validation)

Use this before long full runs to validate end-to-end execution:

```bash
python -m src.run_benchmark \
  --processed-dir data/processed \
  --results-dir results/smoke \
  --model-name sshleifer/tiny-gpt2 \
  --methods refusal_direction,trajguard,jlt,jbshield \
  --train-limit 8 \
  --val-limit 6 \
  --test-limit 6 \
  --max-context-tokens 256 \
  --seed 42
```

## Phase 2: Pilot Go/No-Go Gate

Run a constrained pilot (default total budget: 50 prompts split across train/val/test),
then auto-generate per-method and overall gate decisions:

```bash
python -m src.run_pilot_gate \
  --processed-dir data/processed \
  --results-dir results/pilot_gate \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --hf-token <your_hf_token_if_needed> \
  --pilot-size 50 \
  --min-iid-auroc 0.70 \
  --max-auroc-drop 0.25 \
  --max-latency-ms 2500 \
  --max-memory-overhead-mb 2048 \
  --fail-on-gate-fail
```

Outputs:

- `results/pilot_gate/metrics.csv`
- `results/pilot_gate/logs/pilot_gate_report.json`
- `results/pilot_gate/logs/pilot_gate_summary.csv`

Checkpoint interpretation note (April 2026):

- The committed pilot report currently reflects a CPU-safe fallback run (`sshleifer/tiny-gpt2`) triggered when CUDA was unavailable for a likely large model.
- Treat this as pipeline/smoke evidence only, not as target-model validation for `meta-llama/Llama-3.1-8B-Instruct`.

## Phase 3: Arabic Backup Track

Run a backup benchmark track using Arabic-script prompts discovered in processed splits.
If insufficient Arabic prompts exist, the runner emits a skipped report (or fails in strict mode).

```bash
python -m src.run_arabic_backup \
  --processed-dir data/processed \
  --results-dir results/arabic_backup \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --hf-token <your_hf_token_if_needed> \
  --min-arabic-char-ratio 0.15 \
  --min-per-split 4
```

Outputs:

- `results/arabic_backup/metrics.csv` (if completed)
- `results/arabic_backup/logs/arabic_track_report.json`

New execution controls in `src.run_benchmark`:

- `--methods`: comma-separated detector subset
- `--train-limit`, `--val-limit`, `--test-limit`: optional capped record counts for fast runs
- `--hf-token`: optional token for gated model repos

New data controls in `src.data_loader`:

- `--jbb-artifact-methods`: comma-separated artifact attack methods
- `--jbb-artifact-models`: comma-separated target models for artifact retrieval
- `--disable-jbb-artifacts`: force fallback data paths

## Metrics Produced

For each method and protocol split:

- AUROC
- F1 at tuned threshold
- FPR at 95% TPR
- Average detection latency (ms)
- Average memory overhead (MB)

Cross-dataset protocol is included:

- Tune on AdvBench-oriented split
- Evaluate OOD on JailbreakBench-oriented test set

## Output Artifacts

Minimal retained audit artifacts in this repository:

- `results/metrics.csv`
- `results/logs/benchmark_run.json`
- `results/pilot_gate/metrics.csv`
- `results/pilot_gate/logs/benchmark_run.json`
- `results/pilot_gate/logs/pilot_gate_report.json`
- `results/pilot_gate/logs/pilot_gate_summary.csv`
- `results/arabic_backup/logs/arabic_track_report.json`

Generated run artifacts (intentionally git-ignored by default):

- `results/**/figures/*.png`
- `results/**/logs/artifacts/*.npy`

## 6GB VRAM Notes

- Uses 4-bit quantization when CUDA is available.
- Runs prompts one at a time (no batching).
- Clears CUDA cache after hidden-state extraction.
- Keeps context length capped by `--max-context-tokens`.

## Reproducibility

- Fixed random seed via `--seed` (default 42).
- Deterministic data sampling and split generation.
- Detector thresholds tuned on validation split and persisted in run logs.

## Troubleshooting

- OOM errors:
  - Reduce `--max-context-tokens`.
  - Use a smaller model.
  - Ensure CUDA and bitsandbytes are correctly installed.
- Gated model errors (401 / access denied):
  - Run `huggingface-cli login`.
  - Pass `--hf-token`.
  - Or switch to an open model via `--model-name`.
- Dataset download interruptions:
  - Re-run `python -m src.data_loader` (the loader now retries URLs and falls back to Hugging Face datasets).
- Slow runtime:
  - Reduce sample counts during smoke testing.
  - Lower TrajGuard `max_new_tokens` in code for quick checks.
