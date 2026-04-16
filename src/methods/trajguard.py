from __future__ import annotations

import random
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from src.methods.base_detector import BaseDetector
from src.utils import HiddenStateRuntime, mean_pairwise_cosine_distance


class TrajGuardDetector(BaseDetector):
    """Trajectory-based detector over decode-time hidden-state drift."""

    def __init__(
        self,
        runtime: HiddenStateRuntime,
        layer: int = 20,
        window_size: int = 5,
        consecutive_windows: int = 2,
        threshold: float = 0.20,
        max_new_tokens: int = 24,
        max_fit_samples: int = 128,
        seed: int = 42,
    ) -> None:
        super().__init__(runtime=runtime, name="trajguard", threshold=threshold)
        self.layer = layer
        self.window_size = window_size
        self.consecutive_windows = consecutive_windows
        self.max_new_tokens = max_new_tokens
        self.max_fit_samples = max_fit_samples
        self.seed = seed

    def _window_drifts(self, trajectory: Sequence[np.ndarray]) -> Sequence[float]:
        if len(trajectory) < self.window_size:
            return []

        drifts = []
        for i in range(self.window_size - 1, len(trajectory)):
            window = trajectory[i - self.window_size + 1 : i + 1]
            drifts.append(float(mean_pairwise_cosine_distance(window)))
        return drifts

    def _score_and_flag(self, trajectory: Sequence[np.ndarray]) -> Tuple[float, bool]:
        drifts = self._window_drifts(trajectory)
        if not drifts:
            return 0.0, False

        max_drift = float(np.max(drifts))

        consec = 0
        flagged = False
        for drift in drifts:
            if drift >= self.threshold:
                consec += 1
            else:
                consec = 0

            if consec >= self.consecutive_windows:
                flagged = True
                break

        return max_drift, flagged

    def fit(self, records: Sequence[Mapping[str, Any]]) -> None:
        benign = [r for r in records if int(r["label"]) == 0]
        if not benign:
            return

        if len(benign) > self.max_fit_samples:
            rng = random.Random(self.seed)
            indices = list(range(len(benign)))
            rng.shuffle(indices)
            benign = [benign[i] for i in indices[: self.max_fit_samples]]

        benign_scores = []
        for row in benign:
            traj = self.runtime.generate_hidden_trajectory(
                str(row["prompt"]),
                layer=self.layer,
                max_new_tokens=self.max_new_tokens,
            )
            score, _ = self._score_and_flag(traj)
            benign_scores.append(score)

        if benign_scores:
            # 95th percentile on benign trajectories acts as a conservative threshold.
            self.threshold = float(np.percentile(np.asarray(benign_scores, dtype=np.float32), 95.0))

    def detect(self, prompt: str) -> float:
        trajectory = self.runtime.generate_hidden_trajectory(
            prompt,
            layer=self.layer,
            max_new_tokens=self.max_new_tokens,
        )
        score, _ = self._score_and_flag(trajectory)
        return float(score)
