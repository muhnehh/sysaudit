from __future__ import annotations

import json
import random
import re
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


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _strip_known_chat_wrappers(prompt: str) -> str:
    text = prompt.strip()

    inst_match = re.search(r"\[INST\](.*?)\[/INST\]", text, flags=re.IGNORECASE | re.DOTALL)
    if inst_match:
        text = inst_match.group(1)

    gemma_match = re.search(
        r"<start_of_turn>user\s*(.*?)\s*<end_of_turn>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if gemma_match:
        text = gemma_match.group(1)

    return _normalize_whitespace(text)


def to_chat_prompt(
    prompt: str,
    model_name: Optional[str] = None,
    tokenizer: Optional[Any] = None,
) -> str:
    raw_prompt = prompt.strip()
    user_text = _strip_known_chat_wrappers(raw_prompt)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass

    model_key = (model_name or "").lower()
    if "gemma" in model_key:
        if "<start_of_turn>user" in raw_prompt and "<start_of_turn>model" in raw_prompt:
            return raw_prompt
        return (
            f"<start_of_turn>user\n{user_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

    if "[INST]" in raw_prompt and "[/INST]" in raw_prompt:
        return raw_prompt

    return f"[INST] {user_text} [/INST]"


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
    quantize_4bit: bool = True
    trust_remote_code: bool = False
    hf_token: Optional[str] = None


class HiddenStateRuntime:
    """Shared runtime for model loading, prefill hidden states, and token trajectory extraction."""

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig()
        self.tokenizer = None
        self.model = None
        self._cpu_peak_mb = 0.0
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                token=self.config.hf_token,
            )
        except OSError as exc:
            message = str(exc).lower()
            if "gated repo" in message or "401" in message or "access" in message:
                raise RuntimeError(
                    "Model access denied. Authenticate with Hugging Face (huggingface-cli login) "
                    "or pass an accessible model via --model-name."
                ) from exc
            raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
            "low_cpu_mem_usage": True,
        }

        if torch.cuda.is_available() and self.config.quantize_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["dtype"] = torch.float16
            model_kwargs["max_memory"] = {
                0: f"{self.config.max_memory_gb}GiB",
                "cpu": "48GiB",
            }
        elif torch.cuda.is_available():
            model_kwargs["dtype"] = torch.float16
            model_kwargs["max_memory"] = {
                0: f"{self.config.max_memory_gb}GiB",
                "cpu": "48GiB",
            }
        else:
            cpu_cap_gb = max(1, int(self.config.max_memory_gb))
            offload_dir = Path(".cache") / "hf_offload"
            offload_dir.mkdir(parents=True, exist_ok=True)

            # On CPU-only machines, force auto placement so Accelerate can spill to disk.
            if str(self.config.device_map).lower() == "cpu":
                model_kwargs["device_map"] = "auto"

            model_kwargs["dtype"] = "auto"
            model_kwargs["offload_folder"] = str(offload_dir)
            model_kwargs["offload_state_dict"] = True
            model_kwargs["offload_buffers"] = True
            model_kwargs["max_memory"] = {
                "cpu": f"{cpu_cap_gb}GiB",
            }

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                token=self.config.hf_token,
                **model_kwargs,
            )
        except OSError as exc:
            message = str(exc).lower()
            if "gated repo" in message or "401" in message or "access" in message:
                raise RuntimeError(
                    "Model access denied. Authenticate with Hugging Face (huggingface-cli login) "
                    "or pass an accessible model via --model-name."
                ) from exc
            raise
        self.model.eval()

    def _tokenize(self, prompt: str) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")

        prompt = to_chat_prompt(
            prompt,
            model_name=self.config.model_name,
            tokenizer=self.tokenizer,
        )
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.max_context_tokens,
            return_tensors="pt",
        )
        device = _get_input_device(self.model)
        return {k: v.to(device) for k, v in encoded.items()}

    @staticmethod
    def _resolve_layer_index(layer: int, num_hidden_states: int) -> int:
        if num_hidden_states <= 0:
            raise ValueError("num_hidden_states must be positive.")

        index = layer if layer >= 0 else num_hidden_states + layer
        index = max(0, min(num_hidden_states - 1, index))
        return index

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
            index = self._resolve_layer_index(layer, len(hidden_states))
            layer_tensor = hidden_states[index][0, -1, :]
            extracted[layer] = (
                layer_tensor.detach().float().cpu().numpy().astype(np.float32)
            )

        clear_cuda_cache()
        return extracted

    def prefill_hidden_state(self, prompt: str, layer: int) -> np.ndarray:
        return self.prefill_hidden_states(prompt, [layer])[layer]

    def num_hidden_layers(self) -> int:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        model_config = getattr(self.model, "config", None)
        num_layers = getattr(model_config, "num_hidden_layers", None)
        if num_layers is None:
            raise RuntimeError("Unable to determine model hidden layer count from config.")
        return int(num_layers)

    @staticmethod
    def _process_rss_mb() -> float:
        try:
            import psutil  # Imported lazily to keep runtime dependency soft.

            process = psutil.Process()
            return float(process.memory_info().rss / (1024 * 1024))
        except Exception:
            return 0.0

    def memory_allocated_mb(self) -> float:
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024 * 1024))
        return self._process_rss_mb()

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

            index = self._resolve_layer_index(layer, len(outputs.hidden_states))
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
            return
        self._cpu_peak_mb = self._process_rss_mb()

    def peak_memory_mb(self) -> float:
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 * 1024))

        current_mb = self._process_rss_mb()
        if current_mb > self._cpu_peak_mb:
            self._cpu_peak_mb = current_mb
        return self._cpu_peak_mb

    def cuda_peak_memory_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
