"""
Mechanistic analysis: refusal direction extraction and RDPS/FPS computation.
"""

import numpy as np
from pathlib import Path
from typing import Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_hidden_states_for_prompts(
    model,
    tokenizer,
    prompts: list,
    model_id: str,
    layer: int,
    max_prompts: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Extract last-token hidden state at one layer for a prompt list."""
    import random
    import torch

    from src.models import format_prompt

    random.seed(seed)
    selected = prompts[:max_prompts]
    hidden_states = []

    print(f"  Extracting layer {layer} hidden states for {len(selected)} prompts...")

    for i, item in enumerate(selected):
        prompt_text = item["prompt"] if isinstance(item, dict) else item
        formatted = format_prompt(tokenizer, prompt_text, model_id)

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        hook_output = []

        def hook_fn(module, input_, output):
            _ = module
            _ = input_
            hs = output[0] if isinstance(output, tuple) else output
            hook_output.append(hs[:, -1, :].detach().cpu().float())

        layer_module = model.model.layers[layer]
        hook = layer_module.register_forward_hook(hook_fn)

        with torch.no_grad():
            model(**inputs, output_hidden_states=False)

        hook.remove()

        if hook_output:
            hidden_states.append(hook_output[0][0].numpy())

        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
            print(f"    {i+1}/{len(selected)} done")

    return np.array(hidden_states)


def compute_refusal_direction(
    hidden_states_refused: np.ndarray,
    hidden_states_complied: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Compute mean-difference refusal direction and PCA top-1 variance."""
    from sklearn.decomposition import PCA

    mean_refused = hidden_states_refused.mean(axis=0)
    mean_complied = hidden_states_complied.mean(axis=0)

    diff = mean_refused - mean_complied
    r_hat = diff / (np.linalg.norm(diff) + 1e-8)

    pca = PCA(n_components=min(10, hidden_states_refused.shape[0] - 1))
    pca.fit(hidden_states_refused - mean_refused)
    variance_explained = float(pca.explained_variance_ratio_[0])

    return r_hat, variance_explained


def compute_rdps(hidden_states: np.ndarray, r_hat: np.ndarray) -> np.ndarray:
    """Compute absolute cosine alignment with refusal direction."""
    norms = np.linalg.norm(hidden_states, axis=1, keepdims=True)
    h_normalized = hidden_states / (norms + 1e-8)
    cos_sim = h_normalized @ r_hat
    return np.abs(cos_sim)


def compute_fps(
    rdps_fp16: np.ndarray,
    behavioral_labels_fp16: np.ndarray,
    rdps_quantized: np.ndarray,
    behavioral_labels_quantized: np.ndarray,
) -> dict:
    """Compute Flip Predictability Score (AUC for RDPS-drop predictor)."""
    from sklearn.metrics import f1_score, roc_auc_score

    flips = (behavioral_labels_fp16 == 1) & (behavioral_labels_quantized == 0)
    n_flips = int(flips.sum())
    n_stable = int((~flips).sum())

    print(f"  Flips identified: {n_flips} (of {len(flips)} prompts)")

    if n_flips < 10:
        return {
            "auc": None,
            "n_flips": n_flips,
            "n_stable": n_stable,
            "error": "insufficient_flips",
            "message": (
                f"Only {n_flips} flips found. Need >=10 for reliable FPS. "
                "This means QAG is small - mechanistic claim cannot be made."
            ),
        }

    rdps_drop = rdps_fp16 - rdps_quantized
    auc = float(roc_auc_score(flips.astype(int), rdps_drop))

    thresholds = np.percentile(rdps_drop, np.arange(10, 90, 5))
    best_f1 = 0.0
    best_thresh = 0.0
    for t in thresholds:
        preds = (rdps_drop > t).astype(int)
        if preds.sum() > 0:
            f1 = float(f1_score(flips.astype(int), preds, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)

    return {
        "auc": round(auc, 4),
        "n_flips": n_flips,
        "n_stable": n_stable,
        "best_f1": round(best_f1, 4),
        "best_threshold": round(best_thresh, 4),
        "interpretation": (
            "strong"
            if auc >= 0.70
            else "moderate"
            if auc >= 0.60
            else "weak - report as null mechanistic result"
        ),
    }
