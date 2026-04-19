"""
Standard autoregressive generation (D_0 baseline).
"""

from pathlib import Path
from typing import Optional
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models import format_prompt


def extract_last_token_hidden_states(
    model,
    tokenizer,
    prompt: str,
    model_id: str,
    capture_layers: list[int],
) -> dict[int, "torch.Tensor"]:
    """Run one forward pass and return last-token hidden states for selected layers."""
    if not capture_layers:
        return {}

    formatted = format_prompt(tokenizer, prompt, model_id)
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    by_layer = {}
    for layer in capture_layers:
        hidden_idx = layer + 1  # hidden_states[0] is embedding output
        if hidden_idx >= len(hidden_states):
            raise ValueError(
                f"Layer {layer} out of range for model with {len(hidden_states) - 1} transformer layers"
            )
        vec = hidden_states[hidden_idx][:, -1, :].detach().cpu().float()[0]
        by_layer[layer] = vec

    return by_layer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    model_id: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    seed: int = 42,
    capture_layer: Optional[int] = None,
) -> dict:
    """
    Generate one response and optionally capture hidden state at one layer.
    """
    _ = temperature
    torch.manual_seed(seed)

    formatted = format_prompt(tokenizer, prompt, model_id)
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    hidden_state = None

    with torch.no_grad():
        if capture_layer is not None:
            hook_output = []

            def hook_fn(module, input_, output):
                _ = module
                _ = input_
                hs = output[0] if isinstance(output, tuple) else output
                hook_output.append(hs[:, -1, :].detach().cpu().float())

            layer_module = model.model.layers[capture_layer]
            hook = layer_module.register_forward_hook(hook_fn)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)

        if capture_layer is not None:
            hook.remove()
            hidden_state = hook_output[0].numpy() if hook_output else None

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    result = {
        "prompt": prompt,
        "response": response.strip(),
        "latency_ms": round(latency_ms, 2),
        "input_token_count": inputs["input_ids"].shape[1],
        "output_token_count": len(new_tokens),
    }

    if hidden_state is not None:
        result["hidden_state"] = hidden_state.tolist()
        result["hidden_state_layer"] = capture_layer
        result["hidden_state_dim"] = hidden_state.shape[-1]

    return result
