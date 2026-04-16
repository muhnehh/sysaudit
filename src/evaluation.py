from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.methods.base_detector import BaseDetector
from src.utils import HiddenStateRuntime, ensure_dir


@dataclass
class DetectorRunOutput:
    method_name: str
    dataset_name: str
    labels: np.ndarray
    scores: np.ndarray
    latencies_ms: np.ndarray
    memory_overhead_mb: np.ndarray
    threshold: float



def fpr_at_target_tpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = int(np.argmin(np.abs(tpr - target_tpr)))
    return float(fpr[idx])



def run_detector(
    detector: BaseDetector,
    records: Sequence[Mapping[str, Any]],
    dataset_name: str,
) -> DetectorRunOutput:
    labels = []
    scores = []
    latencies = []
    memory_deltas = []

    runtime: HiddenStateRuntime = detector.runtime

    for row in records:
        prompt = str(row["prompt"])
        label = int(row["label"])

        runtime.reset_peak_memory_stats()
        before_mb = runtime.cuda_memory_allocated_mb()

        start = time.perf_counter()
        score = float(detector.detect(prompt))
        latency_ms = (time.perf_counter() - start) * 1000.0

        peak_mb = runtime.cuda_peak_memory_mb()
        delta_mb = max(0.0, peak_mb - before_mb)

        labels.append(label)
        scores.append(score)
        latencies.append(latency_ms)
        memory_deltas.append(delta_mb)

    return DetectorRunOutput(
        method_name=detector.name,
        dataset_name=dataset_name,
        labels=np.asarray(labels, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float32),
        latencies_ms=np.asarray(latencies, dtype=np.float32),
        memory_overhead_mb=np.asarray(memory_deltas, dtype=np.float32),
        threshold=float(detector.threshold),
    )



def summarize_run(output: DetectorRunOutput) -> Dict[str, Any]:
    y_true = output.labels
    y_score = output.scores
    threshold = output.threshold

    if len(np.unique(y_true)) >= 2:
        auroc = float(roc_auc_score(y_true, y_score))
    else:
        auroc = float("nan")

    y_pred = (y_score >= threshold).astype(np.int64)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr95 = float(fpr_at_target_tpr(y_true, y_score, target_tpr=0.95))

    return {
        "method": output.method_name,
        "dataset": output.dataset_name,
        "auroc": auroc,
        "f1": f1,
        "fpr_at_95_tpr": fpr95,
        "threshold": threshold,
        "latency_ms_avg": float(np.mean(output.latencies_ms)) if len(output.latencies_ms) else 0.0,
        "latency_ms_std": float(np.std(output.latencies_ms)) if len(output.latencies_ms) else 0.0,
        "memory_overhead_mb_avg": float(np.mean(output.memory_overhead_mb))
        if len(output.memory_overhead_mb)
        else 0.0,
        "num_samples": int(len(y_true)),
    }



def save_metrics_csv(rows: Sequence[Mapping[str, Any]], output_csv_path: Path) -> pd.DataFrame:
    ensure_dir(output_csv_path.parent)
    df = pd.DataFrame(list(rows))
    df.to_csv(output_csv_path, index=False)
    return df



def plot_roc_curves(
    runs: Sequence[DetectorRunOutput],
    output_path: Path,
    title: str,
) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 6))

    for run in runs:
        if len(np.unique(run.labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(run.labels, run.scores)
        curve_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{run.method_name} (AUC={curve_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def plot_latency_bars(metrics_df: pd.DataFrame, output_path: Path, dataset_name: str) -> None:
    ensure_dir(output_path.parent)
    subset = metrics_df[metrics_df["dataset"] == dataset_name].copy()

    if subset.empty:
        return

    plt.figure(figsize=(8, 5))
    sns.barplot(data=subset, x="method", y="latency_ms_avg", palette="crest")
    plt.ylabel("Average Detection Latency (ms)")
    plt.xlabel("Method")
    plt.title(f"Latency Comparison ({dataset_name})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def plot_generalization_heatmap(
    metrics_df: pd.DataFrame,
    output_path: Path,
    iid_dataset: str = "iid_test",
    ood_dataset: str = "ood_jailbreakbench",
) -> None:
    ensure_dir(output_path.parent)

    methods = sorted(metrics_df["method"].unique().tolist())
    rows = []
    for method in methods:
        iid_row = metrics_df[
            (metrics_df["method"] == method) & (metrics_df["dataset"] == iid_dataset)
        ]
        ood_row = metrics_df[
            (metrics_df["method"] == method) & (metrics_df["dataset"] == ood_dataset)
        ]

        if iid_row.empty or ood_row.empty:
            continue

        iid_auroc = float(iid_row.iloc[0]["auroc"])
        ood_auroc = float(ood_row.iloc[0]["auroc"])
        rows.append(
            {
                "method": method,
                "iid_auroc": iid_auroc,
                "ood_auroc": ood_auroc,
                "auroc_drop": iid_auroc - ood_auroc,
            }
        )

    if not rows:
        return

    matrix = pd.DataFrame(rows).set_index("method")[["iid_auroc", "ood_auroc", "auroc_drop"]]

    plt.figure(figsize=(7, max(3, len(matrix) * 0.7)))
    sns.heatmap(matrix, annot=True, cmap="mako", fmt=".3f", cbar=True)
    plt.title("Cross-Dataset Generalization")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def evaluate_and_summarize(
    detector: BaseDetector,
    records: Sequence[Mapping[str, Any]],
    dataset_name: str,
) -> Tuple[DetectorRunOutput, Dict[str, Any]]:
    run_output = run_detector(detector=detector, records=records, dataset_name=dataset_name)
    summary = summarize_run(run_output)
    return run_output, summary
