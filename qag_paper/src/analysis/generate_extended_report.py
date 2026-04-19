"""Generate corrected summary and extended analysis tables from completed run artifacts."""

import argparse
from collections import defaultdict
import csv
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scipy import stats

from src.data_loader import classify_refusal
from src.metrics import bootstrap_ci, compute_qag
from src.utils import load_jsonl


DEFAULT_RUN_PATTERNS = {
    "baseline_fp16": "baseline_fp16_*",
    "int8_quantized": "int8_quantized_*",
    "int4_quantized": "int4_quantized_*",
    "qwen_baseline_fp16": "qwen_baseline_fp16_*",
    "qwen_int4_quantized": "qwen_int4_quantized_*",
    "gemma_stress_baseline_fp16": "gemma_stress_baseline_fp16_*",
    "gemma_stress_int4_quantized": "gemma_stress_int4_quantized_*",
    "qwen_stress_baseline_fp16": "qwen_stress_baseline_fp16_*",
    "qwen_stress_int8_quantized": "qwen_stress_int8_quantized_*",
    "qwen_stress_int4_quantized": "qwen_stress_int4_quantized_*",
    "qwen_stress_pilot_baseline_fp16": "qwen_stress_pilot_baseline_fp16_*",
    "qwen_stress_pilot_int4_quantized": "qwen_stress_pilot_int4_quantized_*",
}


def latest_run_dirs(runs_dir: Path, patterns: dict[str, str]) -> dict[str, Path]:
    latest = {}
    for key, pattern in patterns.items():
        candidates = sorted(glob.glob(str(runs_dir / pattern)))
        if candidates:
            latest[key] = Path(candidates[-1]).resolve()
    return latest


def relabel_results(rows: list[dict]) -> tuple[list[dict], int, list[dict]]:
    relabeled = []
    mismatch_count = 0
    mismatch_examples = []

    for row in rows:
        updated = dict(row)
        new_refusal = classify_refusal(row.get("response", ""))
        old_refusal = row.get("refusal")

        if old_refusal is not None and int(old_refusal) != int(new_refusal):
            mismatch_count += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append(
                    {
                        "idx": row.get("idx"),
                        "label": row.get("label"),
                        "old_refusal": int(old_refusal),
                        "new_refusal": int(new_refusal),
                        "prompt": row.get("prompt", "")[:180],
                        "response": row.get("response", "")[:220],
                    }
                )

        updated["refusal"] = int(new_refusal)
        relabeled.append(updated)

    return relabeled, mismatch_count, mismatch_examples


def summarize_rows(run_key: str, run_dir: Path, rows: list[dict]) -> dict:
    harmful = [r for r in rows if r.get("label") == "harmful"]
    benign = [r for r in rows if r.get("label") == "benign"]

    harmful_rate, harmful_lo, harmful_hi = bootstrap_ci(rows, "harmful")
    benign_rate, benign_lo, benign_hi = bootstrap_ci(rows, "benign")

    harmful_refused = sum(int(r.get("refusal", 0)) for r in harmful)
    benign_refused = sum(int(r.get("refusal", 0)) for r in benign)

    summary_path = run_dir / "summary.json"
    summary_json = {}
    if summary_path.exists():
        summary_json = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "run_key": run_key,
        "run_dir": str(run_dir),
        "experiment_name": summary_json.get("experiment_name", run_key),
        "model_id": summary_json.get("model_id", rows[0].get("model_id", "")),
        "quantization": summary_json.get("quantization", rows[0].get("quantization")),
        "n_total": len(rows),
        "n_harmful": len(harmful),
        "n_benign": len(benign),
        "harmful_refusal_rate": round(harmful_rate, 4),
        "harmful_ci_low": round(harmful_lo, 4),
        "harmful_ci_high": round(harmful_hi, 4),
        "benign_refusal_rate": round(benign_rate, 4),
        "benign_ci_low": round(benign_lo, 4),
        "benign_ci_high": round(benign_hi, 4),
        "n_harmful_refused": harmful_refused,
        "n_harmful_complied": len(harmful) - harmful_refused,
        "n_benign_refused": benign_refused,
        "n_benign_complied": len(benign) - benign_refused,
        "avg_latency_ms": round(
            sum(float(r.get("latency_ms", 0.0)) for r in rows) / max(len(rows), 1),
            2,
        ),
        "mean_output_tokens": round(
            sum(float(r.get("output_token_count", 0)) for r in rows) / max(len(rows), 1),
            2,
        ),
        "total_runtime_seconds": summary_json.get("total_runtime_seconds"),
    }


def benchmark_idx_to_variant_map(run_dir: Path, project_root: Path) -> dict[int, str]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return {}

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    benchmark_path = metadata.get("config", {}).get("benchmark_path")
    if not benchmark_path:
        return {}

    benchmark_file = (project_root / benchmark_path).resolve()
    if not benchmark_file.exists():
        return {}

    idx_to_variant = {}
    for item in load_jsonl(str(benchmark_file)):
        idx = item.get("idx")
        variant = item.get("stress_variant")
        if idx is None or not variant:
            continue

        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue

        idx_to_variant[idx_int] = str(variant)

    return idx_to_variant


def enrich_rows_with_stress_variant(rows: list[dict], idx_to_variant: dict[int, str]) -> list[dict]:
    if not idx_to_variant:
        return rows

    enriched = []
    for row in rows:
        out = dict(row)
        if not out.get("stress_variant"):
            idx = out.get("idx")
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                idx_int = None

            if idx_int is not None and idx_int in idx_to_variant:
                out["stress_variant"] = idx_to_variant[idx_int]

        enriched.append(out)

    return enriched


def summarize_stress_variants(run_key: str, rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("label") != "harmful":
            continue
        variant = row.get("stress_variant")
        if not variant:
            continue
        grouped[str(variant)].append(row)

    out = []
    for variant in sorted(grouped):
        subset = grouped[variant]
        refused = sum(int(r.get("refusal", 0)) for r in subset)
        out.append(
            {
                "run_key": run_key,
                "stress_variant": variant,
                "n_harmful": len(subset),
                "n_harmful_refused": refused,
                "n_harmful_complied": len(subset) - refused,
                "harmful_refusal_rate": round(refused / max(len(subset), 1), 4),
            }
        )

    return out


def paired_mcnemar(
    base_rows: list[dict],
    other_rows: list[dict],
    label: str,
    stress_variant: str | None = None,
) -> tuple[int, int, float]:
    base_by_idx = {
        r["idx"]: r
        for r in base_rows
        if r.get("label") == label
        and (stress_variant is None or r.get("stress_variant") == stress_variant)
    }
    other_by_idx = {
        r["idx"]: r
        for r in other_rows
        if r.get("label") == label
        and (stress_variant is None or r.get("stress_variant") == stress_variant)
    }

    b = 0  # base refused, other complied
    c = 0  # base complied, other refused

    for idx in sorted(set(base_by_idx).intersection(other_by_idx)):
        base_ref = int(base_by_idx[idx].get("refusal", 0))
        other_ref = int(other_by_idx[idx].get("refusal", 0))
        if base_ref == 1 and other_ref == 0:
            b += 1
        elif base_ref == 0 and other_ref == 1:
            c += 1

    if b + c == 0:
        return b, c, 1.0

    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - stats.chi2.cdf(stat, df=1))
    return b, c, p_value


def flip_counts(
    base_rows: list[dict],
    other_rows: list[dict],
    label: str,
    stress_variant: str | None = None,
) -> tuple[int, int]:
    base_by_idx = {
        r["idx"]: r
        for r in base_rows
        if r.get("label") == label
        and (stress_variant is None or r.get("stress_variant") == stress_variant)
    }
    other_by_idx = {
        r["idx"]: r
        for r in other_rows
        if r.get("label") == label
        and (stress_variant is None or r.get("stress_variant") == stress_variant)
    }

    comply_to_refuse = 0
    refuse_to_comply = 0
    for idx in sorted(set(base_by_idx).intersection(other_by_idx)):
        base_ref = int(base_by_idx[idx].get("refusal", 0))
        other_ref = int(other_by_idx[idx].get("refusal", 0))
        if base_ref == 0 and other_ref == 1:
            comply_to_refuse += 1
        elif base_ref == 1 and other_ref == 0:
            refuse_to_comply += 1

    return comply_to_refuse, refuse_to_comply


def build_reports(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir).resolve()
    root = Path(args.project_root).resolve()

    latest = latest_run_dirs(runs_dir, DEFAULT_RUN_PATTERNS)
    if not latest:
        raise SystemExit("No runs found for default patterns.")

    relabeled = {}
    mismatch_summary = {}
    metrics_rows = []
    stress_variant_rows = []

    for run_key, run_dir in latest.items():
        results_path = run_dir / "results.jsonl"
        if not results_path.exists():
            continue

        rows = load_jsonl(str(results_path))
        idx_to_variant = benchmark_idx_to_variant_map(run_dir, root)
        rows = enrich_rows_with_stress_variant(rows, idx_to_variant)
        relabeled_rows, mismatch_count, mismatch_examples = relabel_results(rows)
        relabeled[run_key] = relabeled_rows
        mismatch_summary[run_key] = {
            "count": mismatch_count,
            "examples": mismatch_examples,
        }

        metrics_rows.append(summarize_rows(run_key, run_dir, relabeled_rows))
        stress_variant_rows.extend(summarize_stress_variants(run_key, relabeled_rows))

    metrics_rows.sort(key=lambda r: r["experiment_name"])

    summary_fieldnames = [
        "experiment_name",
        "run_dir",
        "model_id",
        "quantization",
        "n_total",
        "n_harmful",
        "harmful_refusal_rate",
        "harmful_ci_low",
        "harmful_ci_high",
        "n_harmful_refused",
        "n_harmful_complied",
        "total_runtime_seconds",
        "avg_latency_ms",
    ]

    extended_fieldnames = [
        "experiment_name",
        "run_key",
        "run_dir",
        "model_id",
        "quantization",
        "n_total",
        "n_harmful",
        "n_benign",
        "harmful_refusal_rate",
        "harmful_ci_low",
        "harmful_ci_high",
        "benign_refusal_rate",
        "benign_ci_low",
        "benign_ci_high",
        "n_harmful_refused",
        "n_harmful_complied",
        "n_benign_refused",
        "n_benign_complied",
        "avg_latency_ms",
        "mean_output_tokens",
        "total_runtime_seconds",
        "stored_refusal_mismatches",
    ]

    summary_csv_path = Path(args.summary_csv)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({k: row.get(k) for k in summary_fieldnames})

    extended_csv_path = Path(args.extended_csv)
    extended_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with extended_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=extended_fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            out = dict(row)
            out["stored_refusal_mismatches"] = mismatch_summary[row["run_key"]]["count"]
            writer.writerow({k: out.get(k) for k in extended_fieldnames})

    comparisons = []
    stress_variant_comparisons = []
    pair_defs = [
        ("baseline_fp16", "int8_quantized", "baseline_vs_int8"),
        ("baseline_fp16", "int4_quantized", "baseline_vs_int4"),
        ("qwen_baseline_fp16", "qwen_int4_quantized", "qwen_baseline_vs_qwen_int4"),
        (
            "gemma_stress_baseline_fp16",
            "gemma_stress_int4_quantized",
            "gemma_stress_baseline_vs_int4",
        ),
        (
            "qwen_stress_baseline_fp16",
            "qwen_stress_int4_quantized",
            "qwen_stress_baseline_vs_int4",
        ),
        (
            "qwen_stress_baseline_fp16",
            "qwen_stress_int8_quantized",
            "qwen_stress_baseline_vs_int8",
        ),
        (
            "qwen_stress_pilot_baseline_fp16",
            "qwen_stress_pilot_int4_quantized",
            "qwen_stress_pilot_baseline_vs_int4",
        ),
    ]

    for base_key, other_key, name in pair_defs:
        if base_key not in relabeled or other_key not in relabeled:
            continue

        harmful_qag = compute_qag(relabeled[base_key], relabeled[other_key], "harmful")
        benign_qag = compute_qag(relabeled[base_key], relabeled[other_key], "benign")

        h_ctf, h_rtc = flip_counts(relabeled[base_key], relabeled[other_key], "harmful")
        b_ctf, b_rtc = flip_counts(relabeled[base_key], relabeled[other_key], "benign")

        h_b, h_c, h_p = paired_mcnemar(relabeled[base_key], relabeled[other_key], "harmful")
        b_b, b_c, b_p = paired_mcnemar(relabeled[base_key], relabeled[other_key], "benign")

        comparisons.append(
            {
                "comparison": name,
                "base_key": base_key,
                "other_key": other_key,
                "harmful_qag": harmful_qag["qag"],
                "harmful_mcnemar_p": round(h_p, 4),
                "benign_qag": benign_qag["qag"],
                "benign_mcnemar_p": round(b_p, 4),
                "harmful_comply_to_refuse": h_ctf,
                "harmful_refuse_to_comply": h_rtc,
                "benign_comply_to_refuse": b_ctf,
                "benign_refuse_to_comply": b_rtc,
                "harmful_mcnemar_b": h_b,
                "harmful_mcnemar_c": h_c,
                "benign_mcnemar_b": b_b,
                "benign_mcnemar_c": b_c,
            }
        )

        base_variants = {
            r.get("stress_variant")
            for r in relabeled[base_key]
            if r.get("label") == "harmful" and r.get("stress_variant")
        }
        other_variants = {
            r.get("stress_variant")
            for r in relabeled[other_key]
            if r.get("label") == "harmful" and r.get("stress_variant")
        }

        for variant in sorted(base_variants.intersection(other_variants)):
            base_subset = [r for r in relabeled[base_key] if r.get("stress_variant") == variant]
            other_subset = [r for r in relabeled[other_key] if r.get("stress_variant") == variant]

            harmful_qag_variant = compute_qag(base_subset, other_subset, "harmful")
            h_ctf_v, h_rtc_v = flip_counts(
                relabeled[base_key],
                relabeled[other_key],
                "harmful",
                stress_variant=variant,
            )
            h_b_v, h_c_v, h_p_v = paired_mcnemar(
                relabeled[base_key],
                relabeled[other_key],
                "harmful",
                stress_variant=variant,
            )

            n_variant = sum(
                1
                for r in relabeled[base_key]
                if r.get("label") == "harmful" and r.get("stress_variant") == variant
            )

            stress_variant_comparisons.append(
                {
                    "comparison": name,
                    "stress_variant": variant,
                    "n_harmful": n_variant,
                    "harmful_qag": harmful_qag_variant["qag"],
                    "harmful_mcnemar_p": round(h_p_v, 4),
                    "harmful_comply_to_refuse": h_ctf_v,
                    "harmful_refuse_to_comply": h_rtc_v,
                    "harmful_mcnemar_b": h_b_v,
                    "harmful_mcnemar_c": h_c_v,
                }
            )

    stress_variant_fieldnames = [
        "run_key",
        "experiment_name",
        "quantization",
        "stress_variant",
        "n_harmful",
        "n_harmful_refused",
        "n_harmful_complied",
        "harmful_refusal_rate",
    ]

    metrics_by_key = {row["run_key"]: row for row in metrics_rows}
    for row in stress_variant_rows:
        run_meta = metrics_by_key.get(row["run_key"], {})
        row["experiment_name"] = run_meta.get("experiment_name", row["run_key"])
        row["quantization"] = run_meta.get("quantization")

    stress_variant_rows.sort(key=lambda r: (r["run_key"], r["stress_variant"]))

    stress_variant_csv_path = Path(args.stress_variant_csv)
    stress_variant_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with stress_variant_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stress_variant_fieldnames)
        writer.writeheader()
        for row in stress_variant_rows:
            writer.writerow({k: row.get(k) for k in stress_variant_fieldnames})

    stress_variant_cmp_fieldnames = [
        "comparison",
        "stress_variant",
        "n_harmful",
        "harmful_qag",
        "harmful_mcnemar_p",
        "harmful_comply_to_refuse",
        "harmful_refuse_to_comply",
        "harmful_mcnemar_b",
        "harmful_mcnemar_c",
    ]

    stress_variant_cmp_csv_path = Path(args.stress_variant_comparison_csv)
    stress_variant_cmp_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with stress_variant_cmp_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stress_variant_cmp_fieldnames)
        writer.writeheader()
        for row in stress_variant_comparisons:
            writer.writerow({k: row.get(k) for k in stress_variant_cmp_fieldnames})

    summary_md_lines = []
    summary_md_lines.append("# Final Plan Summary (Recomputed)")
    summary_md_lines.append("")
    summary_md_lines.append("Classifier corrections were applied by re-labeling refusal from raw responses.")
    summary_md_lines.append("")
    summary_md_lines.append(
        "| Experiment | Quantization | Harmful refusal | Benign over-refusal | Runtime (min) | Avg latency (ms) |"
    )
    summary_md_lines.append("|---|---:|---:|---:|---:|---:|")
    for row in metrics_rows:
        runtime_min = None
        if row["total_runtime_seconds"] is not None:
            runtime_min = float(row["total_runtime_seconds"]) / 60.0
        runtime_str = f"{runtime_min:.1f}" if runtime_min is not None else "NA"
        summary_md_lines.append(
            "| "
            f"{row['experiment_name']} | {row['quantization']} | "
            f"{row['harmful_refusal_rate']:.3f} "
            f"[{row['harmful_ci_low']:.3f}, {row['harmful_ci_high']:.3f}] | "
            f"{row['benign_refusal_rate']:.3f} "
            f"[{row['benign_ci_low']:.3f}, {row['benign_ci_high']:.3f}] | "
            f"{runtime_str} | "
            f"{row['avg_latency_ms']:.1f} |"
        )

    summary_md_lines.append("")
    summary_md_lines.append("## Pairwise Comparisons")
    summary_md_lines.append("")
    summary_md_lines.append(
        "| Comparison | Harmful QAG | Harmful p | Benign QAG | Benign p | Harmful flips (C->R / R->C) | Benign flips (C->R / R->C) |"
    )
    summary_md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for cmp_row in comparisons:
        summary_md_lines.append(
            "| "
            f"{cmp_row['comparison']} | "
            f"{cmp_row['harmful_qag']:.4f} | "
            f"{cmp_row['harmful_mcnemar_p']:.4f} | "
            f"{cmp_row['benign_qag']:.4f} | "
            f"{cmp_row['benign_mcnemar_p']:.4f} | "
            f"{cmp_row['harmful_comply_to_refuse']} / {cmp_row['harmful_refuse_to_comply']} | "
            f"{cmp_row['benign_comply_to_refuse']} / {cmp_row['benign_refuse_to_comply']} |"
        )

    if stress_variant_rows:
        summary_md_lines.append("")
        summary_md_lines.append("## Stress Variant Breakdown (Harmful Only)")
        summary_md_lines.append("")
        summary_md_lines.append("| Run | Variant | N | Refusal rate |")
        summary_md_lines.append("|---|---|---:|---:|")
        for row in stress_variant_rows:
            summary_md_lines.append(
                "| "
                f"{row['run_key']} | "
                f"{row['stress_variant']} | "
                f"{row['n_harmful']} | "
                f"{row['harmful_refusal_rate']:.4f} |"
            )

    if stress_variant_comparisons:
        summary_md_lines.append("")
        summary_md_lines.append("## Stress Variant Pairwise (Harmful Only)")
        summary_md_lines.append("")
        summary_md_lines.append(
            "| Comparison | Variant | N | Harmful QAG | Harmful p | Flips (C->R / R->C) |"
        )
        summary_md_lines.append("|---|---|---:|---:|---:|---:|")
        for row in stress_variant_comparisons:
            summary_md_lines.append(
                "| "
                f"{row['comparison']} | "
                f"{row['stress_variant']} | "
                f"{row['n_harmful']} | "
                f"{row['harmful_qag']:.4f} | "
                f"{row['harmful_mcnemar_p']:.4f} | "
                f"{row['harmful_comply_to_refuse']} / {row['harmful_refuse_to_comply']} |"
            )

    summary_md_lines.append("")
    summary_md_lines.append("## Classifier Mismatch Audit")
    summary_md_lines.append("")
    for row in metrics_rows:
        run_key = row["run_key"]
        mismatch_count = mismatch_summary[run_key]["count"]
        summary_md_lines.append(
            f"- {run_key}: {mismatch_count} stored-label mismatches after recomputation"
        )

    summary_md_path = Path(args.summary_md)
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.write_text("\n".join(summary_md_lines) + "\n", encoding="utf-8")

    extended_md_lines = []
    extended_md_lines.append("# Extended Quantization Analysis")
    extended_md_lines.append("")
    extended_md_lines.append("## Run Artifacts Used")
    extended_md_lines.append("")
    for key, run_dir in latest.items():
        try:
            run_dir_display = run_dir.relative_to(root)
        except ValueError:
            run_dir_display = run_dir
        extended_md_lines.append(f"- {key}: {run_dir_display}")

    extended_md_lines.append("")
    extended_md_lines.append("## Machine-readable Comparison JSON")
    extended_md_lines.append("")
    extended_md_lines.append("```json")
    extended_md_lines.append(json.dumps(comparisons, indent=2))
    extended_md_lines.append("```")

    if stress_variant_comparisons:
        extended_md_lines.append("")
        extended_md_lines.append("## Stress Variant Comparison JSON (Harmful)")
        extended_md_lines.append("")
        extended_md_lines.append("```json")
        extended_md_lines.append(json.dumps(stress_variant_comparisons, indent=2))
        extended_md_lines.append("```")

    extended_md_path = Path(args.extended_md)
    extended_md_path.parent.mkdir(parents=True, exist_ok=True)
    extended_md_path.write_text("\n".join(extended_md_lines) + "\n", encoding="utf-8")

    details_json_path = Path(args.details_json)
    details_json_path.parent.mkdir(parents=True, exist_ok=True)
    details_json_path.write_text(
        json.dumps(
            {
                "latest_runs": {k: str(v) for k, v in latest.items()},
                "metrics_rows": metrics_rows,
                "pairwise": comparisons,
                "stress_variant_metrics": stress_variant_rows,
                "stress_variant_pairwise": stress_variant_comparisons,
                "mismatch_summary": mismatch_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_md_path}")
    print(f"Wrote {extended_csv_path}")
    print(f"Wrote {extended_md_path}")
    print(f"Wrote {stress_variant_csv_path}")
    print(f"Wrote {stress_variant_cmp_csv_path}")
    print(f"Wrote {details_json_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--summary-csv", default="results/final_plan_summary.csv")
    parser.add_argument("--summary-md", default="results/final_plan_summary.md")
    parser.add_argument("--extended-csv", default="results/final_plan_extended.csv")
    parser.add_argument("--extended-md", default="results/final_plan_extended.md")
    parser.add_argument("--stress-variant-csv", default="results/final_plan_stress_variants.csv")
    parser.add_argument(
        "--stress-variant-comparison-csv",
        default="results/final_plan_stress_variant_comparisons.csv",
    )
    parser.add_argument("--details-json", default="results/final_plan_details.json")
    args = parser.parse_args()

    build_reports(args)


if __name__ == "__main__":
    main()
