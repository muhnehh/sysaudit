from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score

from src.utils import HiddenStateRuntime


@dataclass
class DetectionOutput:
    score: float
    flagged: bool


class BaseDetector(ABC):
    """Abstract detector interface used by all benchmarked methods."""

    def __init__(
        self,
        runtime: HiddenStateRuntime,
        name: str,
        threshold: float = 0.5,
    ) -> None:
        self.runtime = runtime
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def detect(self, prompt: str) -> float:
        """Return a jailbreak score where larger values indicate higher jailbreak likelihood."""

    def fit(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Optional calibration hook for methods requiring training statistics."""

    def detect_with_intervention(self, prompt: str) -> Dict[str, Any]:
        score = float(self.detect(prompt))
        return {
            "score": score,
            "flagged": bool(score >= self.threshold),
            "intervention_applied": False,
        }

    def predict(self, prompt: str) -> DetectionOutput:
        score = float(self.detect(prompt))
        return DetectionOutput(score=score, flagged=bool(score >= self.threshold))

    def score_records(self, records: Sequence[Mapping[str, Any]]) -> List[float]:
        return [float(self.detect(str(r["prompt"]))) for r in records]

    def tune_threshold(self, validation_records: Sequence[Mapping[str, Any]]) -> float:
        if not validation_records:
            raise ValueError("Validation records are required for threshold tuning.")

        y_true = np.asarray([int(r["label"]) for r in validation_records], dtype=np.int64)
        y_scores = np.asarray(self.score_records(validation_records), dtype=np.float32)

        score_min = float(np.min(y_scores))
        score_max = float(np.max(y_scores))

        if np.isclose(score_min, score_max):
            self.threshold = score_min
            return self.threshold

        best_threshold = self.threshold
        best_f1 = -1.0

        for threshold in np.linspace(score_min, score_max, num=200):
            y_pred = (y_scores >= threshold).astype(np.int64)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)

        self.threshold = best_threshold
        return self.threshold
