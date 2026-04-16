from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.methods.base_detector import BaseDetector
from src.utils import HiddenStateRuntime


class JailbreakingLeavesTraceDetector(BaseDetector):
    """JLT-style detector using multi-layer prefill activations and Mahalanobis distances."""

    def __init__(
        self,
        runtime: HiddenStateRuntime,
        layers: Sequence[int] = (15, 20, 25),
        threshold: float = 0.0,
        max_fit_samples: int = 256,
        covariance_epsilon: float = 1e-4,
        seed: int = 42,
    ) -> None:
        super().__init__(runtime=runtime, name="jlt", threshold=threshold)
        self.layers = tuple(layers)
        self.max_fit_samples = max_fit_samples
        self.covariance_epsilon = covariance_epsilon
        self.seed = seed
        self.layer_stats: Dict[int, Dict[str, np.ndarray]] = {}

    def _sample_benign(self, records: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
        benign = [r for r in records if int(r["label"]) == 0]
        if len(benign) <= self.max_fit_samples:
            return benign

        rng = random.Random(self.seed)
        indices = list(range(len(benign)))
        rng.shuffle(indices)
        return [benign[i] for i in indices[: self.max_fit_samples]]

    def fit(self, records: Sequence[Mapping[str, Any]]) -> None:
        benign = self._sample_benign(records)
        if not benign:
            raise ValueError("JLT fit requires benign records to build covariance statistics.")

        activations_by_layer: Dict[int, list[np.ndarray]] = {layer: [] for layer in self.layers}

        for row in benign:
            activations = self.runtime.prefill_hidden_states(str(row["prompt"]), self.layers)
            for layer in self.layers:
                activations_by_layer[layer].append(activations[layer])

        self.layer_stats = {}
        for layer in self.layers:
            matrix = np.stack(activations_by_layer[layer], axis=0).astype(np.float32)
            mean = np.mean(matrix, axis=0)

            # Full covariance is too expensive for 8B hidden size on 6GB hardware,
            # so we use diagonal covariance for a memory-safe Mahalanobis approximation.
            var = np.var(matrix, axis=0) + self.covariance_epsilon

            self.layer_stats[layer] = {
                "mean": mean.astype(np.float32),
                "var": var.astype(np.float32),
            }

    def _layer_distance(self, layer: int, activation: np.ndarray) -> float:
        stats = self.layer_stats.get(layer)
        if stats is None:
            raise RuntimeError("Layer statistics missing. Call fit before detect.")

        delta = activation - stats["mean"]
        inv_var = 1.0 / stats["var"]
        distance_sq = float(np.sum((delta * delta) * inv_var))
        return float(np.sqrt(max(distance_sq, 0.0)))

    def detect(self, prompt: str) -> float:
        if not self.layer_stats:
            raise RuntimeError("JLT detector is not calibrated. Call fit before detect.")

        activations = self.runtime.prefill_hidden_states(prompt, self.layers)
        distances = [self._layer_distance(layer, activations[layer]) for layer in self.layers]
        return float(np.mean(np.asarray(distances, dtype=np.float32)))
