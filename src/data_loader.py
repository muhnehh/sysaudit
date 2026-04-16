from __future__ import annotations

import argparse
import csv
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from sklearn.model_selection import train_test_split

ADV_BENCH_URL = (
    "https://github.com/llm-attacks/llm-attacks/raw/main/data/advbench/harmful_behaviors.csv"
)
JAILBREAK_BENCH_URL = (
    "https://github.com/JailbreakBench/jailbreakbench/raw/main/data/jailbreak_prompts.csv"
)
ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
)


BENIGN_SUSPICIOUS_PATTERN = re.compile(
    r"\b(weapon|explosive|bomb|hack|malware|phishing|kill|poison|drugs?)\b",
    flags=re.IGNORECASE,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_if_missing(url: str, destination: Path, force: bool = False) -> None:
    ensure_dir(destination.parent)
    if destination.exists() and not force:
        return
    urllib.request.urlretrieve(url, destination)


def read_csv_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def to_chat_template(prompt: str) -> str:
    cleaned = normalize_whitespace(prompt.strip())
    return f"[INST] {cleaned} [/INST]"


def sample_records(records: Sequence[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n > len(records):
        raise ValueError(f"Requested {n} records but only {len(records)} available.")
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    sampled = [records[i] for i in indices[:n]]
    return sampled


def pick_first_nonempty(row: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    for value in row.values():
        if value is not None and str(value).strip():
            return str(value)
    return ""


def load_advbench(path: Path) -> List[Dict[str, Any]]:
    rows = read_csv_records(path)
    records: List[Dict[str, Any]] = []
    for row in rows:
        prompt = pick_first_nonempty(row, ["goal", "prompt", "instruction", "behavior"])
        if not prompt:
            continue
        records.append(
            {
                "raw_prompt": normalize_whitespace(prompt),
                "label": 1,
                "attack_type": "advbench_harmful_behavior",
                "source": "advbench",
            }
        )
    return records


def load_jailbreakbench(path: Path) -> List[Dict[str, Any]]:
    rows = read_csv_records(path)
    records: List[Dict[str, Any]] = []
    for row in rows:
        prompt = pick_first_nonempty(
            row,
            [
                "jailbreak_prompt",
                "prompt",
                "question",
                "input",
                "goal",
                "transformed_prompt",
            ],
        )
        if not prompt:
            continue
        attack_type = pick_first_nonempty(
            row,
            ["attack_type", "type", "category", "jailbreak_type", "transformation"],
        )
        attack_type = attack_type if attack_type else "unknown_jailbreak_type"
        records.append(
            {
                "raw_prompt": normalize_whitespace(prompt),
                "label": 1,
                "attack_type": normalize_whitespace(attack_type),
                "source": "jailbreakbench",
            }
        )
    return records


def is_benign_instruction(text: str) -> bool:
    # Conservative lexical filter to reduce obviously harmful instructions in benign sampling.
    return BENIGN_SUSPICIOUS_PATTERN.search(text) is None


def load_alpaca(path: Path) -> List[Dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError("Alpaca file is not a list of records.")

    records: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        instruction = str(row.get("instruction", "")).strip()
        input_text = str(row.get("input", "")).strip()
        if not instruction:
            continue

        prompt = instruction
        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}"

        prompt = normalize_whitespace(prompt)
        if not is_benign_instruction(prompt):
            continue

        records.append(
            {
                "raw_prompt": prompt,
                "label": 0,
                "attack_type": "benign_instruction",
                "source": "alpaca",
            }
        )
    return records


def add_chat_prompt(record: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(record)
    enriched["prompt"] = to_chat_template(enriched["raw_prompt"])
    return enriched


def stratified_split(
    records: Sequence[Dict[str, Any]],
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0")

    labels = [int(x["label"]) for x in records]

    train_records, temp_records = train_test_split(
        list(records),
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )

    temp_labels = [int(x["label"]) for x in temp_records]
    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)

    val_records, test_records = train_test_split(
        temp_records,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=seed,
        stratify=temp_labels,
    )

    return train_records, val_records, test_records


def assign_ids(records: Sequence[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for i, row in enumerate(records):
        enriched = dict(row)
        enriched["id"] = f"{prefix}-{i:05d}"
        output.append(enriched)
    return output


def prepare_datasets(
    raw_dir: Path,
    processed_dir: Path,
    benign_count: int = 500,
    advbench_count: int = 500,
    jailbreakbench_count: int = 500,
    seed: int = 42,
    force_download: bool = False,
) -> Dict[str, int]:
    advbench_file = raw_dir / "advbench" / "harmful_behaviors.csv"
    jailbreakbench_file = raw_dir / "jailbreakbench" / "jailbreak_prompts.csv"
    alpaca_file = raw_dir / "alpaca" / "alpaca_data.json"

    download_if_missing(ADV_BENCH_URL, advbench_file, force=force_download)
    download_if_missing(JAILBREAK_BENCH_URL, jailbreakbench_file, force=force_download)
    download_if_missing(ALPACA_URL, alpaca_file, force=force_download)

    benign_pool = load_alpaca(alpaca_file)
    advbench_pool = load_advbench(advbench_file)
    jailbreakbench_pool = load_jailbreakbench(jailbreakbench_file)

    benign = sample_records(benign_pool, benign_count, seed=seed)
    advbench_jailbreaks = sample_records(advbench_pool, advbench_count, seed=seed + 1)
    jailbreakbench_jailbreaks = sample_records(
        jailbreakbench_pool,
        jailbreakbench_count,
        seed=seed + 2,
    )

    benign = [add_chat_prompt(x) for x in assign_ids(benign, "benign")]
    advbench_jailbreaks = [
        add_chat_prompt(x) for x in assign_ids(advbench_jailbreaks, "adv-jb")
    ]
    jailbreakbench_jailbreaks = [
        add_chat_prompt(x) for x in assign_ids(jailbreakbench_jailbreaks, "jbb-jb")
    ]

    jailbreak = advbench_jailbreaks + jailbreakbench_jailbreaks

    write_jsonl(benign, processed_dir / "benign_prompts.jsonl")
    write_jsonl(jailbreak, processed_dir / "jailbreak_prompts.jsonl")

    combined = benign + jailbreak
    rng = random.Random(seed)
    rng.shuffle(combined)

    train_records, val_records, test_records = stratified_split(combined, seed=seed)

    write_jsonl(train_records, processed_dir / "train_prompts.jsonl")
    write_jsonl(val_records, processed_dir / "val_prompts.jsonl")
    write_jsonl(test_records, processed_dir / "test_prompts.jsonl")

    return {
        "benign_count": len(benign),
        "jailbreak_count": len(jailbreak),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare jailbreak audit datasets.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for raw downloaded datasets.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed JSONL outputs.",
    )
    parser.add_argument("--benign-count", type=int, default=500)
    parser.add_argument("--advbench-count", type=int, default=500)
    parser.add_argument("--jailbreakbench-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download source files even if they exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = prepare_datasets(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        benign_count=args.benign_count,
        advbench_count=args.advbench_count,
        jailbreakbench_count=args.jailbreakbench_count,
        seed=args.seed,
        force_download=args.force_download,
    )

    print("Prepared datasets successfully:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
