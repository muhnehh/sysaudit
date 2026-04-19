"""
Main orchestration script.
Usage: python src/run_experiment.py --config configs/baseline.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import classify_refusal
from src.decoding.autoregressive import (
    extract_last_token_hidden_states,
    generate_response,
)
from src.metrics import bootstrap_ci, compute_refusal_rate
from src.models import load_model_and_tokenizer, unload_model
from src.utils import (
    load_jsonl,
    load_yaml,
    make_run_dir,
    save_jsonl,
    save_run_metadata,
    set_all_seeds,
    sha256_file,
)


def run_experiment(config_path: str) -> dict:
    config = load_yaml(config_path)
    set_all_seeds(config["seed"])

    run_dir = make_run_dir(config["results_dir"], config["experiment_name"])
    progress_path = run_dir / "progress.json"

    def update_progress(
        *,
        status: str,
        stage: str,
        completed: int,
        total: int,
        elapsed_seconds: float,
        eta_seconds: float | None,
        avg_latency_ms: float,
    ) -> None:
        payload = {
            "experiment_name": config["experiment_name"],
            "status": status,
            "stage": stage,
            "completed": completed,
            "total": total,
            "elapsed_seconds": round(elapsed_seconds, 1),
            "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
            "avg_latency_ms": round(avg_latency_ms, 2),
        }
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    update_progress(
        status="running",
        stage="initializing",
        completed=0,
        total=0,
        elapsed_seconds=0.0,
        eta_seconds=None,
        avg_latency_ms=0.0,
    )

    print("\n" + "=" * 60)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Run dir: {run_dir}")
    print("=" * 60 + "\n")

    dataset_path = config["benchmark_path"]
    hash_path = dataset_path.replace(".jsonl", ".sha256")
    if Path(hash_path).exists():
        with open(hash_path, encoding="utf-8") as f:
            expected_hash = f.read().strip()
        actual_hash = sha256_file(dataset_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Dataset hash mismatch! Expected {expected_hash}, got {actual_hash}. "
                "The dataset has been modified. Experiments are invalid."
            )
        print(f"Dataset integrity verified: {actual_hash[:16]}...")
    else:
        print("WARNING: No hash file found. Proceeding without integrity check.")

    save_run_metadata(run_dir, config)

    prompts = load_jsonl(dataset_path)
    if config.get("max_prompts") is not None:
        prompts = prompts[: int(config["max_prompts"])]
    print(f"Loaded {len(prompts)} prompts")

    update_progress(
        status="running",
        stage="loading_model",
        completed=0,
        total=len(prompts),
        elapsed_seconds=0.0,
        eta_seconds=None,
        avg_latency_ms=0.0,
    )

    print("[1/3] Preparing run context...")

    print("[2/3] Loading model and tokenizer...")
    load_start = time.time()
    model, tokenizer = load_model_and_tokenizer(
        config["model_id"],
        config.get("quantization"),
        config.get("device", "cuda"),
    )
    print(f"[2/3] Model ready in {time.time() - load_start:.1f}s")

    results = []
    harmful_count = sum(1 for p in prompts if p.get("label") == "harmful")
    benign_count = sum(1 for p in prompts if p.get("label") == "benign")
    print("[3/3] Starting generation loop...")
    print(
        f"\nGenerating responses for {len(prompts)} prompts "
        f"({harmful_count} harmful, {benign_count} benign)..."
    )
    print("Estimated time: ~40-90 minutes on RTX 2060 Max-Q. Do not interrupt.\n")

    start_time = time.time()
    latency_total = 0.0

    update_progress(
        status="running",
        stage="generating",
        completed=0,
        total=len(prompts),
        elapsed_seconds=0.0,
        eta_seconds=None,
        avg_latency_ms=0.0,
    )

    pbar = tqdm(
        total=len(prompts),
        desc=f"{config['experiment_name']}",
        unit="prompt",
        dynamic_ncols=True,
    )

    for i, item in enumerate(prompts):
        result = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=item["prompt"],
            model_id=config["model_id"],
            max_new_tokens=config.get("max_new_tokens", 128),
            temperature=config.get("temperature", 0.0),
            seed=config["seed"],
            capture_layer=None,
        )

        for meta_key, meta_val in item.items():
            if meta_key == "prompt":
                continue
            if meta_key not in result:
                result[meta_key] = meta_val

        result["idx"] = item.get("idx", i)
        result["label"] = item.get("label", "unknown")
        result["category"] = item.get("category", "unknown")
        result["refusal"] = classify_refusal(result["response"])
        result["experiment"] = config["experiment_name"]
        result["model_id"] = config["model_id"]
        result["quantization"] = config.get("quantization")
        results.append(result)
        latency_total += result["latency_ms"]

        processed = i + 1
        elapsed = time.time() - start_time
        rate = processed / max(elapsed, 1e-8)
        remaining = (len(prompts) - processed) / max(rate, 1e-8)
        avg_latency = latency_total / processed

        pbar.update(1)
        pbar.set_postfix(
            elapsed_min=f"{elapsed/60:.1f}",
            eta_min=f"{remaining/60:.1f}",
            avg_ms=f"{avg_latency:.0f}",
            refresh=False,
        )

        update_progress(
            status="running",
            stage="generating",
            completed=processed,
            total=len(prompts),
            elapsed_seconds=elapsed,
            eta_seconds=remaining,
            avg_latency_ms=avg_latency,
        )

    pbar.close()

    capture_layers = config.get("hidden_state_layers", [])
    if config.get("capture_hidden_states", False) and capture_layers:
        capture_labels = set(config.get("hidden_state_capture_labels", ["harmful"]))
        capture_refusal = config.get("hidden_state_capture_refusal", None)
        capture_limit = int(config.get("max_hidden_state_prompts", 100))

        selected_indices = []
        for result_idx, row in enumerate(results):
            if row.get("label") not in capture_labels:
                continue
            if capture_refusal is not None and int(row.get("refusal", 0)) != int(capture_refusal):
                continue

            selected_indices.append(result_idx)
            if len(selected_indices) >= capture_limit:
                break

        if selected_indices:
            print(
                "\nCapturing hidden states in post-pass for "
                f"{len(selected_indices)} prompts at layers {capture_layers}..."
            )

            update_progress(
                status="running",
                stage="capturing_hidden_states",
                completed=len(results),
                total=len(results),
                elapsed_seconds=time.time() - start_time,
                eta_seconds=None,
                avg_latency_ms=latency_total / max(len(results), 1),
            )

            for hs_rank, result_idx in enumerate(selected_indices, start=1):
                row = results[result_idx]
                by_layer = extract_last_token_hidden_states(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=row["prompt"],
                    model_id=config["model_id"],
                    capture_layers=capture_layers,
                )

                row["hidden_states_by_layer"] = {
                    str(layer): by_layer[layer].tolist() for layer in capture_layers
                }

                first_layer = capture_layers[0]
                first_vec = by_layer[first_layer]
                row["hidden_state"] = first_vec.tolist()
                row["hidden_state_layer"] = first_layer
                row["hidden_state_layers"] = capture_layers
                row["hidden_state_dim"] = int(first_vec.shape[-1])

                if hs_rank % 10 == 0 or hs_rank == len(selected_indices):
                    elapsed = time.time() - start_time
                    update_progress(
                        status="running",
                        stage="capturing_hidden_states",
                        completed=len(results),
                        total=len(results),
                        elapsed_seconds=elapsed,
                        eta_seconds=None,
                        avg_latency_ms=latency_total / max(len(results), 1),
                    )
                    print(f"  Hidden-state capture progress: {hs_rank}/{len(selected_indices)}")

    total_time = time.time() - start_time
    save_jsonl(results, str(run_dir / "results.jsonl"))

    harmful_results = [r for r in results if r["label"] == "harmful"]
    refusal_rate = compute_refusal_rate(results, "harmful")
    _, lo, hi = bootstrap_ci(results, "harmful")

    summary = {
        "experiment_name": config["experiment_name"],
        "model_id": config["model_id"],
        "quantization": config.get("quantization"),
        "n_total": len(results),
        "n_harmful": len(harmful_results),
        "refusal_rate_harmful": round(refusal_rate, 4),
        "refusal_rate_ci_95": [round(lo, 4), round(hi, 4)],
        "n_refused": sum(r["refusal"] for r in harmful_results),
        "n_complied": sum(1 - r["refusal"] for r in harmful_results),
        "total_runtime_seconds": round(total_time, 1),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / max(len(results), 1), 2),
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    update_progress(
        status="completed",
        stage="completed",
        completed=len(results),
        total=len(results),
        elapsed_seconds=total_time,
        eta_seconds=0.0,
        avg_latency_ms=summary["avg_latency_ms"],
    )

    print("\n" + "=" * 60)
    print(f"RESULTS: {config['experiment_name']}")
    print(f"  Refusal rate (harmful): {refusal_rate:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"  Runtime: {total_time/60:.1f} minutes")
    print(f"  Results saved to: {run_dir}")
    print("=" * 60 + "\n")

    unload_model(model)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    summary = run_experiment(args.config)
    print(json.dumps(summary, indent=2))
