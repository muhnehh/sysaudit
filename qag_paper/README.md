# Quantization Alignment Gap (QAG): Stress-Testing Refusal Safety Under Quantization

This subproject implements the hardware-constrained QAG pipeline for:

- fp16 vs INT8 vs INT4 refusal behavior shift (QAG)
- Refusal Direction Preservation Score (RDPS)
- Flip Predictability Score (FPS)
- Stress-tested refusal behavior under jailbreak prompt transforms
- Safety-helpfulness-latency tradeoff reporting

Designed for RTX 2060 Max-Q (6GB VRAM), batch size 1, sequential hidden-state extraction.

## Quick Start

1. Create and activate Python 3.11 venv.
2. Install pinned packages (see requirements and install order in master context).
3. Build dataset once:

```bash
python src/data_loader.py --build-dataset
```

4. Run baseline:

```bash
python src/run_experiment.py --config configs/baseline.yaml
python gate1_pilot_check.py
```

Progress monitoring during long runs:

```bash
./watch_progress.sh
```

This shows stage, completed/total prompts, elapsed time, ETA, and average latency.
Each run also writes `results/runs/<run_dir>/progress.json` with the same fields.

5. If GATE-1 passes, run quantization conditions:

```bash
python src/run_experiment.py --config configs/int8.yaml
python src/run_experiment.py --config configs/int4.yaml
python gate2_effect_check.py
```

6. Replication model:

```bash
python src/run_experiment.py --config configs/qwen_baseline.yaml
python src/run_experiment.py --config configs/qwen_int4.yaml
```

7. Recompute corrected summary and extended tables:

```bash
python src/analysis/generate_extended_report.py
```

8. Build and run jailbreak stress benchmark:

```bash
python src/data_loader.py \
	--build-jailbreak-stress \
	--base-dataset data/prompts/final_dataset.jsonl \
	--output data/prompts/final_dataset_jailbreak_stress.jsonl

python src/run_experiment.py --config configs/qwen_stress_baseline.yaml
python src/run_experiment.py --config configs/qwen_stress_int4.yaml
```

Or run the helper sequence:

```bash
./run_stress_sequence.sh
```

9. Profile INT8 vs INT4 CUDA behavior:

```bash
python src/diagnostics/profile_quant_kernels.py --model-id Qwen/Qwen2.5-1.5B-Instruct
```

10. Run scheme comparisons (bnb/GPTQ/AWQ configs):

```bash
python src/run_scheme_comparison.py --configs \
	configs/qwen_stress_int4.yaml \
	configs/qwen_gptq_template.yaml \
	configs/qwen_awq_template.yaml
```

Note: GPTQ/AWQ dependencies may require a separate Python environment from this pinned stack.

11. Analysis notebooks:

- notebooks/02_representation_analysis.ipynb
- notebooks/03_fps_and_figures.ipynb

## Hard Constraints

- Never load two models at once.
- Never alter dataset after final_dataset.sha256 is recorded.
- Refusal classifier updates must be versioned and reported (run `generate_extended_report.py` after changes).
- Log every run into results/runs with metadata (hardware + git hash).

## Files

- src/data_loader.py: dataset builder + locked refusal classifier
- src/analysis/generate_extended_report.py: corrected metrics + pairwise stats + mismatch audit
- src/models.py: safe model loading/unloading with VRAM checks
- src/decoding/autoregressive.py: deterministic generation + optional hidden capture
- src/metrics.py: QAG + bootstrap CI + McNemar test
- src/probes/hidden_state_capture.py: RD extraction, RDPS, FPS
- src/evaluation/refusal_classifier.py: GPT-judge agreement validation
- src/run_experiment.py: main orchestrator
- gate1_pilot_check.py, gate2_effect_check.py: gate checkpoints
- run_stress_sequence.sh: end-to-end stress benchmark helper
