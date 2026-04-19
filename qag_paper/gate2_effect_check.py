import json
from pathlib import Path

from src.metrics import compute_qag
from src.utils import load_jsonl


baseline = load_jsonl(
    str(sorted(Path("results/runs").glob("baseline_fp16_*"))[-1] / "results.jsonl")
)
int4 = load_jsonl(
    str(sorted(Path("results/runs").glob("int4_quantized_*"))[-1] / "results.jsonl")
)

qag = compute_qag(baseline, int4, "harmful")
print(json.dumps(qag, indent=2))

if not qag["exceeds_threshold"]:
    print(f"\n!!! GATE-2 FAIL: QAG = {qag['qag']:.4f} < 0.05 threshold.")
    print("!!! The effect is too small for the intended mechanistic claim.")
else:
    print(f"\nGATE-2 PASS: QAG = {qag['qag']:.4f} >= 0.05 threshold.")
    print("Proceed to mechanistic analysis.")
