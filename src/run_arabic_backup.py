from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from src.utils import ensure_dir, read_jsonl, write_jsonl

ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Arabic backup benchmark track from Arabic-script prompts."
    )

    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--max-memory-gb", type=int, default=6)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit quantization when CUDA is available.",
    )

    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/arabic_backup"))

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        type=str,
        default="refusal_direction,trajguard,jlt,jbshield",
        help="Comma-separated detector methods to evaluate.",
    )

    parser.add_argument(
        "--min-arabic-char-ratio",
        type=float,
        default=0.15,
        help="Minimum Arabic-script character ratio for a prompt to enter Arabic track.",
    )
    parser.add_argument(
        "--min-per-split",
        type=int,
        default=4,
        help="Minimum prompts required per split after filtering or fallback re-split.",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=0,
        help="Optional max prompts per split after filtering (0 means no cap).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if Arabic prompts are insufficient instead of writing a skipped report.",
    )

    return parser.parse_args()


def _parse_methods(methods_arg: str) -> List[str]:
    methods = [x.strip() for x in methods_arg.split(",") if x.strip()]
    if not methods:
        raise ValueError("At least one method must be provided via --methods.")
    return methods


def _prompt_text(record: Mapping[str, Any]) -> str:
    raw_prompt = str(record.get("raw_prompt", "") or "").strip()
    if raw_prompt:
        return raw_prompt
    return str(record.get("prompt", "") or "").strip()


def _arabic_char_ratio(text: str) -> float:
    if not text:
        return 0.0

    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0

    arabic_chars = sum(1 for ch in chars if ARABIC_CHAR_RE.match(ch))
    return float(arabic_chars / len(chars))


def _is_arabic_record(record: Mapping[str, Any], min_ratio: float) -> bool:
    text = _prompt_text(record)
    if not text:
        return False
    return _arabic_char_ratio(text) >= min_ratio


def _dedupe_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen_keys = set()
    deduped: List[Dict[str, Any]] = []
    for record in records:
        rec = dict(record)
        rec_key = (str(rec.get("id", "")).strip(), _prompt_text(rec))
        if rec_key in seen_keys:
            continue
        seen_keys.add(rec_key)
        deduped.append(rec)
    return deduped


def _cap_split(records: Sequence[Mapping[str, Any]], max_per_split: int, seed: int) -> List[Dict[str, Any]]:
    output = [dict(r) for r in records]
    if max_per_split <= 0 or len(output) <= max_per_split:
        return output

    rng = random.Random(seed)
    rng.shuffle(output)
    return output[:max_per_split]


def _fallback_resplit(
    pooled_records: Sequence[Mapping[str, Any]],
    min_per_split: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    records = [dict(r) for r in pooled_records]
    rng = random.Random(seed)
    rng.shuffle(records)

    n_total = len(records)
    if n_total < (3 * min_per_split):
        return [], [], []

    train_n = max(min_per_split, int(round(n_total * 0.60)))
    val_n = max(min_per_split, int(round(n_total * 0.20)))
    test_n = n_total - train_n - val_n

    if test_n < min_per_split:
        shortage = min_per_split - test_n

        reducible_train = max(0, train_n - min_per_split)
        reduce_train = min(shortage, reducible_train)
        train_n -= reduce_train
        shortage -= reduce_train

        reducible_val = max(0, val_n - min_per_split)
        reduce_val = min(shortage, reducible_val)
        val_n -= reduce_val
        shortage -= reduce_val

        test_n = n_total - train_n - val_n
        if shortage > 0 or test_n < min_per_split:
            return [], [], []

    train_records = records[:train_n]
    val_records = records[train_n : train_n + val_n]
    test_records = records[train_n + val_n :]
    return train_records, val_records, test_records


def _build_benchmark_command(args: argparse.Namespace, processed_dir: Path) -> List[str]:
    command: List[str] = [
        sys.executable,
        "-m",
        "src.run_benchmark",
        "--processed-dir",
        str(processed_dir),
        "--results-dir",
        str(args.results_dir),
        "--model-name",
        args.model_name,
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--max-memory-gb",
        str(args.max_memory_gb),
        "--device-map",
        args.device_map,
        "--seed",
        str(args.seed),
        "--methods",
        args.methods,
    ]

    if args.hf_token:
        command.extend(["--hf-token", args.hf_token])
    if args.disable_4bit:
        command.append("--disable-4bit")

    return command


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2)


def main() -> None:
    args = parse_args()
    methods = _parse_methods(args.methods)

    train_path = args.processed_dir / "train_prompts.jsonl"
    val_path = args.processed_dir / "val_prompts.jsonl"
    test_path = args.processed_dir / "test_prompts.jsonl"

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Required split not found: {path}")

    train_records = read_jsonl(train_path)
    val_records = read_jsonl(val_path)
    test_records = read_jsonl(test_path)

    train_ar = [r for r in train_records if _is_arabic_record(r, args.min_arabic_char_ratio)]
    val_ar = [r for r in val_records if _is_arabic_record(r, args.min_arabic_char_ratio)]
    test_ar = [r for r in test_records if _is_arabic_record(r, args.min_arabic_char_ratio)]

    train_ar = _dedupe_records(train_ar)
    val_ar = _dedupe_records(val_ar)
    test_ar = _dedupe_records(test_ar)

    if len(train_ar) < args.min_per_split or len(val_ar) < args.min_per_split or len(test_ar) < args.min_per_split:
        pooled = _dedupe_records(train_ar + val_ar + test_ar)
        train_ar, val_ar, test_ar = _fallback_resplit(
            pooled_records=pooled,
            min_per_split=args.min_per_split,
            seed=args.seed,
        )

    train_ar = _cap_split(train_ar, args.max_per_split, seed=args.seed + 1)
    val_ar = _cap_split(val_ar, args.max_per_split, seed=args.seed + 2)
    test_ar = _cap_split(test_ar, args.max_per_split, seed=args.seed + 3)

    report_path = args.results_dir / "logs" / "arabic_track_report.json"
    ensure_dir(args.results_dir)
    ensure_dir(args.results_dir / "logs")

    if len(train_ar) < args.min_per_split or len(val_ar) < args.min_per_split or len(test_ar) < args.min_per_split:
        reason = (
            "Insufficient Arabic-script prompts for backup track "
            f"(train={len(train_ar)}, val={len(val_ar)}, test={len(test_ar)}, "
            f"min_per_split={args.min_per_split})."
        )
        payload = {
            "status": "skipped",
            "reason": reason,
            "criteria": {
                "min_arabic_char_ratio": args.min_arabic_char_ratio,
                "min_per_split": args.min_per_split,
                "max_per_split": args.max_per_split,
            },
            "counts": {
                "train": len(train_ar),
                "val": len(val_ar),
                "test": len(test_ar),
            },
            "methods": methods,
        }
        _write_report(report_path, payload)
        print(reason)
        print(f"Arabic track report: {report_path}")
        if args.strict:
            raise RuntimeError(reason)
        return

    arabic_processed_dir = args.results_dir / "processed_arabic"
    ensure_dir(arabic_processed_dir)
    write_jsonl(train_ar, arabic_processed_dir / "train_prompts.jsonl")
    write_jsonl(val_ar, arabic_processed_dir / "val_prompts.jsonl")
    write_jsonl(test_ar, arabic_processed_dir / "test_prompts.jsonl")

    benchmark_command = _build_benchmark_command(args=args, processed_dir=arabic_processed_dir)

    print("Running Arabic backup benchmark with splits:")
    print(f"  train={len(train_ar)}, val={len(val_ar)}, test={len(test_ar)}")
    print(f"  methods={methods}")

    subprocess.run(benchmark_command, check=True)

    payload = {
        "status": "completed",
        "criteria": {
            "min_arabic_char_ratio": args.min_arabic_char_ratio,
            "min_per_split": args.min_per_split,
            "max_per_split": args.max_per_split,
        },
        "counts": {
            "train": len(train_ar),
            "val": len(val_ar),
            "test": len(test_ar),
        },
        "methods": methods,
        "benchmark_command": benchmark_command,
        "metrics_csv": str(args.results_dir / "metrics.csv"),
    }
    _write_report(report_path, payload)

    print("Arabic backup benchmark complete.")
    print(f"Arabic track report: {report_path}")


if __name__ == "__main__":
    main()
