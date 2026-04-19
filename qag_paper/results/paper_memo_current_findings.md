# Current Findings and Top-Tier Path (Auto-Generated)

## 1) Corrected Core Result (after refusal-classifier repair)

Using recomputed labels from raw responses:

- Harmful refusal is 100% for baseline, int8, int4, qwen baseline, qwen int4 on the standard dataset.
- Original non-zero QAG was a classifier artifact (explicit refusal phrasing variant not covered).
- On standard prompts, quantization safety gap is effectively zero.

Key artifact:
- results/final_plan_summary.md
- results/final_plan_stress_variants.csv
- results/final_plan_stress_variant_comparisons.csv

## 2) Practical Deployment Finding Already Supported

On RTX 2060 Max-Q (cc 7.5), Qwen2.5-1.5B:

- fp16 generate(64 tokens): 3457.87 ms
- int8 generate(64 tokens): 24124.39 ms
- int4 generate(64 tokens): 5211.29 ms

Ratios:

- int8 vs fp16 generate: 6.977x slower
- int8 vs int4 generate: 4.629x slower
- int4 vs fp16 generate: 1.507x slower

This supports an actionable claim: int8 (bnb LLM.int8 path) is strongly Pareto-disfavored in this hardware setting.

Key artifact:
- results/diagnostics/int8_vs_int4_profile.md

## 3) New Stress-Pilot Signal (non-ceiling condition)

Dataset built:
- data/prompts/final_dataset_jailbreak_stress.jsonl
- 800 harmful (original + 3 transforms), 200 benign controls

Pilot runs (200 prompts):
- qwen_stress_pilot_baseline_fp16: harmful refusal 1.000 (164/164 harmful)
- qwen_stress_pilot_int4_quantized: harmful refusal 0.9573 (157/164 harmful)

Paired test (pilot):
- Harmful QAG = 0.0427
- McNemar p = 0.0233
- 7 harmful flips from fp16 refusal to int4 compliance

Variant-level localization (pilot):
- base64_wrap: fp16 1.000 vs int4 0.750 (delta -0.250)
- original/prefix/roleplay variants: no observed drop in this pilot sample

Interpretation:
- The stress benchmark creates measurable headroom.
- The observed gap is currently concentrated in the base64 transformation.

Key artifact:
- results/final_plan_summary.md

## 4) Code/Infra Upgrades Completed

- Refusal classifier phrase coverage expanded in src/data_loader.py.
- Stress dataset builder added to src/data_loader.py:
  - --build-jailbreak-stress
- Multi-layer hidden-state post-pass added to src/run_experiment.py:
  - captures all configured layers, not only hidden_state_layers[0]
- Extra prompt metadata propagation fixed in src/run_experiment.py
  - stress_variant/base_idx/base_prompt are preserved in new runs
- Extended report generator added:
  - src/analysis/generate_extended_report.py
- INT8/INT4 profiler added:
  - src/diagnostics/profile_quant_kernels.py
- Scheme comparison runner added:
  - src/run_scheme_comparison.py
- Stress sequence helper added:
  - run_stress_sequence.sh

## 5) Main-Track Upgrade Checklist (next)

1. In progress: run full stress baseline/int4 on all 1000 prompts.
  - Active run directory: results/runs/qwen_stress_baseline_fp16_20260418_201123
  - int8 is intentionally skipped in this sequence because prior diagnostics show severe hardware-specific slowdown and it is optional.
2. Add second model family for cross-model external validity.
3. Add one independent refusal judge or human adjudication subset.
4. Run scheme comparisons in a separate env:
   - Current env can use optimum.
   - autoawq conflicts with transformers pin.
   - auto-gptq wheel resolution fails under current setup.
5. Produce layer-wise mechanistic analysis from newly captured multi-layer states.

## 6) Immediate Commands

Generate corrected reports:

python src/analysis/generate_extended_report.py

Run full stress sequence (baseline + int4):

./run_stress_sequence.sh

Include stress int8 too (optional, much slower on this GPU):

RUN_STRESS_INT8=1 ./run_stress_sequence.sh

Run profiler:

python src/diagnostics/profile_quant_kernels.py --quantization fp16 --out-json results/diagnostics/profile_fp16.json
python src/diagnostics/profile_quant_kernels.py --quantization int8 --out-json results/diagnostics/profile_int8.json
python src/diagnostics/profile_quant_kernels.py --quantization int4 --out-json results/diagnostics/profile_int4.json
