"""Run multiple quantization-scheme configs sequentially and aggregate summaries."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="List of config YAML files to execute in order.",
    )
    parser.add_argument(
        "--out-json",
        default="results/scheme_comparison_summary.json",
        help="Where to save the aggregate summary.",
    )
    args = parser.parse_args()

    summaries = []
    for cfg in args.configs:
        print(f"\n=== Running {cfg} ===")
        summary = run_experiment(cfg)
        summary["config_path"] = cfg
        summaries.append(summary)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
