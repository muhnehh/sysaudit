"""Profile fp16/int8/int4 latency and top CUDA ops for one prompt."""

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bitsandbytes as bnb
import torch
from torch.profiler import ProfilerActivity, profile

from src.models import load_model_and_tokenizer, unload_model


DEFAULT_PROMPT = "Explain in two sentences why reducing food waste matters."


def quant_to_value(name: str):
    if name == "fp16":
        return None
    return name


def time_forward_and_generate(model, tokenizer, prompt: str) -> tuple[float, float]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        _ = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        start.record()
        _ = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        end.record()
        torch.cuda.synchronize()
        forward_ms = float(start.elapsed_time(end))

    with torch.no_grad():
        start.record()
        _ = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        end.record()
        torch.cuda.synchronize()
        generate_ms = float(start.elapsed_time(end))

    return forward_ms, generate_ms


def profile_top_cuda_ops(model, tokenizer, prompt: str, row_limit: int = 20) -> list[dict]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _ = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

    events = []
    for evt in prof.key_averages():
        cuda_ms = float(evt.cuda_time_total / 1000.0)
        if cuda_ms <= 0:
            continue
        events.append(
            {
                "op": evt.key,
                "cuda_time_ms": round(cuda_ms, 3),
                "cpu_time_ms": round(float(evt.cpu_time_total / 1000.0), 3),
                "calls": int(evt.count),
            }
        )

    events.sort(key=lambda x: x["cuda_time_ms"], reverse=True)
    return events[:row_limit]


def run_profile(model_id: str, prompt: str, out_json: Path, quantizations: list[str]) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this profiling script.")

    device_name = torch.cuda.get_device_name(0)
    cap_major, cap_minor = torch.cuda.get_device_capability(0)

    report = {
        "timestamp_unix": time.time(),
        "gpu_name": device_name,
        "gpu_compute_capability": f"{cap_major}.{cap_minor}",
        "torch_version": torch.__version__,
        "bitsandbytes_version": getattr(bnb, "__version__", "unknown"),
        "model_id": model_id,
        "prompt": prompt,
        "profiles": {},
    }

    for quant_name in quantizations:
        quant_value = quant_to_value(quant_name)
        print(f"Profiling {model_id} [{quant_name}]...")
        model, tokenizer = load_model_and_tokenizer(
            model_id=model_id,
            quantization=quant_value,
            device="cuda",
        )

        forward_ms, generate_ms = time_forward_and_generate(model, tokenizer, prompt)
        top_ops = profile_top_cuda_ops(model, tokenizer, prompt)

        report["profiles"][quant_name] = {
            "forward_ms": round(forward_ms, 2),
            "generate_64tok_ms": round(generate_ms, 2),
            "top_cuda_ops": top_ops,
        }

        unload_model(model)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== Profiling Summary ===")
    print(f"GPU: {report['gpu_name']} (cc {report['gpu_compute_capability']})")
    for quant_name in quantizations:
        stats = report["profiles"][quant_name]
        print(
            f"{quant_name}: forward={stats['forward_ms']}ms, "
            f"generate_64tok={stats['generate_64tok_ms']}ms"
        )
    print(f"\nWrote {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--out-json",
        default="results/diagnostics/int8_vs_int4_profile.json",
    )
    parser.add_argument(
        "--quantization",
        choices=["all", "fp16", "int8", "int4"],
        default="all",
        help="Profile all quantization modes or exactly one mode.",
    )
    args = parser.parse_args()

    quantizations = ["fp16", "int8", "int4"]
    if args.quantization != "all":
        quantizations = [args.quantization]

    run_profile(
        model_id=args.model_id,
        prompt=args.prompt,
        out_json=Path(args.out_json),
        quantizations=quantizations,
    )


if __name__ == "__main__":
    main()
