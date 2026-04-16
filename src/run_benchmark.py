from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from src.data_loader import prepare_datasets
from src.evaluation import (
    evaluate_and_summarize,
    plot_generalization_heatmap,
    plot_latency_bars,
    plot_roc_curves,
    save_metrics_csv,
)
from src.methods.base_detector import BaseDetector
from src.methods.jailbreaking_leaves_trace import JailbreakingLeavesTraceDetector
from src.methods.jbshield import JBShieldDetector
from src.methods.refusal_direction import RefusalDirectionDetector
from src.methods.trajguard import TrajGuardDetector
from src.utils import RuntimeConfig, HiddenStateRuntime, read_jsonl, set_seed, ensure_dir



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run systematic jailbreak detector benchmark.")

    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--max-memory-gb", type=int, default=6)
    parser.add_argument("--device-map", type=str, default="auto")

    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-data", action="store_true")

    parser.add_argument("--benign-count", type=int, default=500)
    parser.add_argument("--advbench-count", type=int, default=500)
    parser.add_argument("--jailbreakbench-count", type=int, default=500)

    return parser.parse_args()



def _load_split(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required split not found: {path}")
    return read_jsonl(path)



def _split_for_ood_protocol(
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    adv_train = [
        dict(r)
        for r in train_records
        if int(r["label"]) == 0 or str(r.get("source", "")).lower() == "advbench"
    ]
    adv_val = [
        dict(r)
        for r in val_records
        if int(r["label"]) == 0 or str(r.get("source", "")).lower() == "advbench"
    ]
    ood_test = [
        dict(r)
        for r in test_records
        if int(r["label"]) == 0 or str(r.get("source", "")).lower() == "jailbreakbench"
    ]

    if not adv_train:
        adv_train = [dict(r) for r in train_records]
    if not adv_val:
        adv_val = [dict(r) for r in val_records]
    if not ood_test:
        ood_test = [dict(r) for r in test_records]

    return adv_train, adv_val, ood_test



def _build_detector_factories(
    runtime: HiddenStateRuntime,
    artifacts_dir: Path,
) -> Dict[str, Callable[[], BaseDetector]]:
    ensure_dir(artifacts_dir)

    return {
        "refusal_direction": lambda: RefusalDirectionDetector(
            runtime=runtime,
            layer=20,
            threshold=0.0,
            vector_path=artifacts_dir / "refusal_direction_vector.npy",
        ),
        "trajguard": lambda: TrajGuardDetector(
            runtime=runtime,
            layer=20,
            window_size=5,
            consecutive_windows=2,
            threshold=0.20,
            max_new_tokens=24,
        ),
        "jlt": lambda: JailbreakingLeavesTraceDetector(
            runtime=runtime,
            layers=(15, 20, 25),
            threshold=0.0,
        ),
        "jbshield": lambda: JBShieldDetector(
            runtime=runtime,
            layer=20,
            threshold=0.0,
            jailbreak_vector_path=artifacts_dir / "jbshield_jailbreak_vector.npy",
            toxicity_vector_path=artifacts_dir / "jbshield_toxicity_vector.npy",
            use_logistic_calibration=True,
        ),
    }



def _maybe_prepare_data(args: argparse.Namespace) -> None:
    train_path = args.processed_dir / "train_prompts.jsonl"
    val_path = args.processed_dir / "val_prompts.jsonl"
    test_path = args.processed_dir / "test_prompts.jsonl"

    if args.prepare_data or not (train_path.exists() and val_path.exists() and test_path.exists()):
        stats = prepare_datasets(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            benign_count=args.benign_count,
            advbench_count=args.advbench_count,
            jailbreakbench_count=args.jailbreakbench_count,
            seed=args.seed,
            force_download=False,
        )
        print("Prepared datasets:")
        for key, value in stats.items():
            print(f"  {key}: {value}")



def _persist_run_log(log_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(log_rows), f, indent=2)



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ensure_dir(args.results_dir)
    ensure_dir(args.results_dir / "figures")
    ensure_dir(args.results_dir / "logs")

    _maybe_prepare_data(args)

    train_records = _load_split(args.processed_dir / "train_prompts.jsonl")
    val_records = _load_split(args.processed_dir / "val_prompts.jsonl")
    test_records = _load_split(args.processed_dir / "test_prompts.jsonl")

    adv_train, adv_val, ood_test = _split_for_ood_protocol(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
    )

    runtime = HiddenStateRuntime(
        RuntimeConfig(
            model_name=args.model_name,
            max_context_tokens=args.max_context_tokens,
            max_memory_gb=args.max_memory_gb,
            device_map=args.device_map,
        )
    )

    detector_factories = _build_detector_factories(
        runtime=runtime,
        artifacts_dir=args.results_dir / "logs" / "artifacts",
    )

    metric_rows: List[Dict[str, Any]] = []
    run_logs: List[Dict[str, Any]] = []

    iid_runs = []
    ood_runs = []

    for method_name, make_detector in detector_factories.items():
        print(f"Running method: {method_name}")

        detector_iid = make_detector()
        detector_iid.fit(train_records)
        detector_iid.tune_threshold(val_records)

        iid_run, iid_summary = evaluate_and_summarize(
            detector=detector_iid,
            records=test_records,
            dataset_name="iid_test",
        )
        iid_summary["protocol"] = "standard"
        metric_rows.append(iid_summary)
        iid_runs.append(iid_run)

        detector_ood = make_detector()
        detector_ood.fit(adv_train)
        detector_ood.tune_threshold(adv_val)

        ood_run, ood_summary = evaluate_and_summarize(
            detector=detector_ood,
            records=ood_test,
            dataset_name="ood_jailbreakbench",
        )
        ood_summary["protocol"] = "train_on_advbench_only"
        metric_rows.append(ood_summary)
        ood_runs.append(ood_run)

        run_logs.append(
            {
                "method": method_name,
                "iid_threshold": float(detector_iid.threshold),
                "ood_threshold": float(detector_ood.threshold),
                "train_size": len(train_records),
                "val_size": len(val_records),
                "test_size": len(test_records),
                "adv_train_size": len(adv_train),
                "adv_val_size": len(adv_val),
                "ood_test_size": len(ood_test),
            }
        )

    metrics_df = save_metrics_csv(metric_rows, args.results_dir / "metrics.csv")
    _persist_run_log(run_logs, args.results_dir / "logs" / "benchmark_run.json")

    plot_roc_curves(
        iid_runs,
        output_path=args.results_dir / "figures" / "roc_iid.png",
        title="ROC Curves (IID Test)",
    )
    plot_roc_curves(
        ood_runs,
        output_path=args.results_dir / "figures" / "roc_ood.png",
        title="ROC Curves (OOD JailbreakBench)",
    )
    plot_latency_bars(
        metrics_df,
        output_path=args.results_dir / "figures" / "latency_iid.png",
        dataset_name="iid_test",
    )
    plot_generalization_heatmap(
        metrics_df,
        output_path=args.results_dir / "figures" / "generalization_heatmap.png",
        iid_dataset="iid_test",
        ood_dataset="ood_jailbreakbench",
    )

    print("Benchmark complete.")
    print(f"Metrics: {args.results_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
