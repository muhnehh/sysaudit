from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def to_chat_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if "[INST]" in prompt and "[/INST]" in prompt:
        return prompt
    return f"[INST] {' '.join(prompt.split())} [/INST]"


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    denom = max(a_norm * b_norm, eps)
    return float(np.dot(a, b) / denom)


def mean_pairwise_cosine_distance(vectors: Sequence[np.ndarray], eps: float = 1e-8) -> float:
    if len(vectors) < 2:
        return 0.0

    distances: List[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = cosine_similarity(vectors[i], vectors[j], eps=eps)
            distances.append(1.0 - sim)

    if not distances:
        return 0.0
    return float(np.mean(distances))


@dataclass
class RuntimeConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_context_tokens: int = 2048
    max_memory_gb: int = 6
    device_map: str = "auto"
    trust_remote_code: bool = False


class HiddenStateRuntime:
    """Shared runtime for model loading, prefill hidden states, and token trajectory extraction."""

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig()
        self.tokenizer = None
        self.model = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
        }

        if torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["max_memory"] = {
                0: f"{self.config.max_memory_gb}GiB",
                "cpu": "48GiB",
            }
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        self.model.eval()

    def _tokenize(self, prompt: str) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")

        prompt = to_chat_prompt(prompt)
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.max_context_tokens,
            return_tensors="pt",
        )
        device = _get_input_device(self.model)
        return {k: v.to(device) for k, v in encoded.items()}

    def prefill_hidden_states(
        self,
        prompt: str,
        layers: Sequence[int],
    ) -> Dict[int, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        inputs = self._tokenize(prompt)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")

        extracted: Dict[int, np.ndarray] = {}
        for layer in layers:
            index = layer if layer >= 0 else len(hidden_states) + layer
            layer_tensor = hidden_states[index][0, -1, :]
            extracted[layer] = (
                layer_tensor.detach().float().cpu().numpy().astype(np.float32)
            )

        clear_cuda_cache()
        return extracted

    def prefill_hidden_state(self, prompt: str, layer: int) -> np.ndarray:
        return self.prefill_hidden_states(prompt, [layer])[layer]

    def generate_hidden_trajectory(
        self,
        prompt: str,
        layer: int,
        max_new_tokens: int = 32,
    ) -> List[np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        token_batch = self._tokenize(prompt)
        input_ids = token_batch["input_ids"]
        attention_mask = token_batch.get("attention_mask")

        trajectory: List[np.ndarray] = []

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if outputs.hidden_states is None:
                break

            index = layer if layer >= 0 else len(outputs.hidden_states) + layer
            token_hidden = outputs.hidden_states[index][0, -1, :]
            trajectory.append(token_hidden.detach().float().cpu().numpy().astype(np.float32))

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if attention_mask is not None:
                next_mask = torch.ones_like(next_token, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

            if input_ids.shape[-1] >= self.config.max_context_tokens:
                break

        clear_cuda_cache()
        return trajectory

    def cuda_memory_allocated_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.memory_allocated() / (1024 * 1024))

    def reset_peak_memory_stats(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def cuda_peak_memory_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
