from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.methods.base_detector import BaseDetector
from src.utils import HiddenStateRuntime, ensure_dir

TOXIC_PATTERN = re.compile(
    r"\b(weapon|bomb|explosive|kill|poison|bioweapon|malware|phishing|fraud)\b",
    flags=re.IGNORECASE,
)


class JBShieldDetector(BaseDetector):
    """Concept-vector detector that contrasts jailbreak and toxicity projections."""

    def __init__(
        self,
        runtime: HiddenStateRuntime,
        layer: int = 20,
        threshold: float = 0.0,
        jailbreak_vector_path: Optional[Path] = None,
        toxicity_vector_path: Optional[Path] = None,
        use_logistic_calibration: bool = True,
        max_fit_samples_per_class: int = 256,
        seed: int = 42,
    ) -> None:
        super().__init__(runtime=runtime, name="jbshield", threshold=threshold)
        self.layer = layer
        self.jailbreak_vector_path = jailbreak_vector_path
        self.toxicity_vector_path = toxicity_vector_path
        self.use_logistic_calibration = use_logistic_calibration
        self.max_fit_samples_per_class = max_fit_samples_per_class
        self.seed = seed

        self.jailbreak_vector: Optional[np.ndarray] = None
        self.toxicity_vector: Optional[np.ndarray] = None
        self.classifier: Optional[LogisticRegression] = None

        self._try_load_vectors()

    def _try_load_vectors(self) -> None:
        if self.jailbreak_vector_path is not None and self.jailbreak_vector_path.exists():
            self.jailbreak_vector = np.load(self.jailbreak_vector_path).astype(np.float32)
            self.jailbreak_vector = self._normalize(self.jailbreak_vector)

        if self.toxicity_vector_path is not None and self.toxicity_vector_path.exists():
            self.toxicity_vector = np.load(self.toxicity_vector_path).astype(np.float32)
            self.toxicity_vector = self._normalize(self.toxicity_vector)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)

    def _sample_class(
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

    def _toxic_subset(self, records: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
        matched = []
        for row in records:
            candidate = str(row.get("raw_prompt", row.get("prompt", "")))
            if TOXIC_PATTERN.search(candidate):
                matched.append(row)
        return matched

    def _extract_mean_activation(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not records:
            raise ValueError("Cannot compute mean activation for an empty record set.")

        activations = [
            self.runtime.prefill_hidden_state(str(record["prompt"]), self.layer)
            for record in records
        ]
        return np.mean(np.stack(activations, axis=0), axis=0).astype(np.float32)

    def fit(self, records: Sequence[Mapping[str, Any]]) -> None:
        benign = self._sample_class(records, label=0, max_items=self.max_fit_samples_per_class)
        jailbreak = self._sample_class(records, label=1, max_items=self.max_fit_samples_per_class)

        if not benign or not jailbreak:
            raise ValueError("JBShield fit requires both benign and jailbreak records.")

        benign_mean = self._extract_mean_activation(benign)
        jailbreak_mean = self._extract_mean_activation(jailbreak)

        if self.jailbreak_vector is None:
            self.jailbreak_vector = self._normalize(jailbreak_mean - benign_mean)

        if self.toxicity_vector is None:
            toxic_subset = self._toxic_subset(jailbreak)
            if toxic_subset:
                toxic_mean = self._extract_mean_activation(toxic_subset)
                self.toxicity_vector = self._normalize(toxic_mean - benign_mean)
            else:
                # Fallback keeps detector functional when explicit toxicity examples are absent.
                self.toxicity_vector = np.zeros_like(self.jailbreak_vector, dtype=np.float32)

        if self.jailbreak_vector_path is not None:
            ensure_dir(self.jailbreak_vector_path.parent)
            np.save(self.jailbreak_vector_path, self.jailbreak_vector)

        if self.toxicity_vector_path is not None:
            ensure_dir(self.toxicity_vector_path.parent)
            np.save(self.toxicity_vector_path, self.toxicity_vector)

        if self.use_logistic_calibration:
            X = []
            y = []
            for row in benign + jailbreak:
                activation = self.runtime.prefill_hidden_state(str(row["prompt"]), self.layer)
                jb_proj = float(np.dot(activation, self.jailbreak_vector))
                tox_proj = float(np.dot(activation, self.toxicity_vector))
                X.append([jb_proj, tox_proj])
                y.append(int(row["label"]))

            if len(set(y)) >= 2:
                self.classifier = LogisticRegression(random_state=self.seed, max_iter=500)
                self.classifier.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64))

    def _projection_features(self, prompt: str) -> tuple[float, float]:
        if self.jailbreak_vector is None or self.toxicity_vector is None:
            raise RuntimeError("JBShield vectors are unavailable. Call fit or provide vector paths.")

        activation = self.runtime.prefill_hidden_state(prompt, self.layer)
        jb_proj = float(np.dot(activation, self.jailbreak_vector))
        tox_proj = float(np.dot(activation, self.toxicity_vector))
        return jb_proj, tox_proj

    def detect(self, prompt: str) -> float:
        jb_proj, tox_proj = self._projection_features(prompt)

        if self.classifier is not None:
            score = float(
                self.classifier.predict_proba(np.asarray([[jb_proj, tox_proj]], dtype=np.float32))[0, 1]
            )
            return score

        return float(jb_proj - tox_proj)

    def detect_with_intervention(
        self,
        prompt: str,
        intervention_strength: float = 0.5,
        toxicity_reinforcement: float = 0.2,
    ) -> dict[str, Any]:
        if self.jailbreak_vector is None or self.toxicity_vector is None:
            raise RuntimeError("JBShield vectors are unavailable. Call fit or provide vector paths.")

        activation = self.runtime.prefill_hidden_state(prompt, self.layer)
        jb_proj = float(np.dot(activation, self.jailbreak_vector))
        tox_proj = float(np.dot(activation, self.toxicity_vector))

        adjusted = (
            activation
            - intervention_strength * jb_proj * self.jailbreak_vector
            + toxicity_reinforcement * tox_proj * self.toxicity_vector
        )

        score = self.detect(prompt)
        return {
            "score": float(score),
            "flagged": bool(score >= self.threshold),
            "intervention_applied": True,
            "intervention_layer": self.layer,
            "adjusted_activation_norm": float(np.linalg.norm(adjusted)),
            "jb_projection": jb_proj,
            "toxicity_projection": tox_proj,
        }
