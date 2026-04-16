from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.methods.base_detector import BaseDetector
from src.utils import HiddenStateRuntime, cosine_similarity, ensure_dir


class RefusalDirectionDetector(BaseDetector):
    """RepE-style refusal direction detector using last-token prefill activations."""

    def __init__(
        self,
        runtime: HiddenStateRuntime,
        layer: int = 20,
        threshold: float = 0.0,
        vector_path: Optional[Path] = None,
        max_fit_samples_per_class: int = 256,
        seed: int = 42,
    ) -> None:
        super().__init__(runtime=runtime, name="refusal_direction", threshold=threshold)
        self.layer = layer
        self.vector_path = vector_path
        self.max_fit_samples_per_class = max_fit_samples_per_class
        self.seed = seed
        self.refusal_vector: Optional[np.ndarray] = None

        if self.vector_path is not None and self.vector_path.exists():
            self.refusal_vector = np.load(self.vector_path).astype(np.float32)
            self._normalize_vector()

    def _normalize_vector(self) -> None:
        if self.refusal_vector is None:
            return
        norm = float(np.linalg.norm(self.refusal_vector))
        if norm > 1e-8:
            self.refusal_vector = (self.refusal_vector / norm).astype(np.float32)

    def _sample_records(
        self,
        records: Sequence[Mapping[str, Any]],
        label: int,
        max_items: int,
    ) -> Sequence[Mapping[str, Any]]:
        subset = [r for r in records if int(r["label"]) == label]
        if len(subset) <= max_items:
            return subset
        rng = random.Random(self.seed + label)
        indices = list(range(len(subset)))
        rng.shuffle(indices)
        return [subset[i] for i in indices[:max_items]]

    def fit(self, records: Sequence[Mapping[str, Any]]) -> None:
        benign_records = self._sample_records(
            records,
            label=0,
            max_items=self.max_fit_samples_per_class,
        )
        jailbreak_records = self._sample_records(
            records,
            label=1,
            max_items=self.max_fit_samples_per_class,
        )

        if not benign_records or not jailbreak_records:
            raise ValueError("RefusalDirectionDetector.fit needs both benign and jailbreak records.")

        benign_acts = []
        for row in benign_records:
            benign_acts.append(self.runtime.prefill_hidden_state(str(row["prompt"]), self.layer))

        jailbreak_acts = []
        for row in jailbreak_records:
            jailbreak_acts.append(self.runtime.prefill_hidden_state(str(row["prompt"]), self.layer))

        benign_mean = np.mean(np.stack(benign_acts, axis=0), axis=0)
        jailbreak_mean = np.mean(np.stack(jailbreak_acts, axis=0), axis=0)

        self.refusal_vector = (jailbreak_mean - benign_mean).astype(np.float32)
        self._normalize_vector()

        if self.vector_path is not None:
            ensure_dir(self.vector_path.parent)
            np.save(self.vector_path, self.refusal_vector)

    def detect(self, prompt: str) -> float:
        if self.refusal_vector is None:
            raise RuntimeError(
                "Refusal direction vector is not available. Provide vector_path or call fit first."
            )

        activation = self.runtime.prefill_hidden_state(prompt, self.layer)
        return float(cosine_similarity(activation, self.refusal_vector))
