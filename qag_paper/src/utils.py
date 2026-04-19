"""
Shared utilities: seeding, logging, hardware info, JSON serialization.
Import this in every script. Never set seeds outside this module.
"""

import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
	"""Set seeds for full reproducibility. Call at the start of every script."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	os.environ["PYTHONHASHSEED"] = str(seed)


def get_hardware_info() -> dict:
	"""Record hardware context for every run. Logged with all results."""
	info = {
		"timestamp": datetime.utcnow().isoformat(),
		"python_version": sys.version.split()[0],
		"torch_version": torch.__version__,
		"cuda_available": torch.cuda.is_available(),
	}
	if torch.cuda.is_available():
		props = torch.cuda.get_device_properties(0)
		info["gpu_name"] = props.name
		info["gpu_vram_gb"] = round(props.total_memory / 1e9, 2)
		info["cuda_version"] = torch.version.cuda
		info["driver_version"] = subprocess.check_output(
			["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
			text=True,
		).strip()
	return info


def get_git_hash() -> str:
	"""Get current git commit hash. Returns 'no-git' if not in a repo."""
	try:
		return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
	except Exception:
		return "no-git"


def make_run_dir(results_dir: str, experiment_name: str) -> Path:
	"""Create a timestamped run directory. Returns Path object."""
	ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
	run_dir = Path(results_dir) / f"{experiment_name}_{ts}"
	run_dir.mkdir(parents=True, exist_ok=True)
	return run_dir


def save_run_metadata(run_dir: Path, config: dict, extra: dict | None = None) -> None:
	"""Save config + hardware info to run directory. Call before any generation."""
	meta = {
		"config": config,
		"hardware": get_hardware_info(),
		"git_hash": get_git_hash(),
	}
	if extra:
		meta.update(extra)
	with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
		json.dump(meta, f, indent=2)


def sha256_file(path: str) -> str:
	"""Compute SHA256 hash of a file. Used to verify dataset integrity."""
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(8192), b""):
			h.update(chunk)
	return h.hexdigest()


def clear_vram() -> None:
	"""
	Aggressively clear VRAM between model loads.
	Call this every time you delete a model object.
	"""
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.synchronize()


def get_vram_used_gb() -> float:
	"""Return currently allocated VRAM in GB."""
	if not torch.cuda.is_available():
		return 0.0
	return torch.cuda.memory_allocated(0) / 1e9


def check_vram_headroom(required_gb: float) -> None:
	"""
	Assert that enough VRAM is free. Raises RuntimeError if not.
	required_gb examples: fp16 gemma-2-2b=4.2, int4=1.4, int8=2.2.
	"""
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA not available")
	total = torch.cuda.get_device_properties(0).total_memory / 1e9
	allocated = torch.cuda.memory_allocated(0) / 1e9
	free = total - allocated
	if free < required_gb:
		raise RuntimeError(
			f"Insufficient VRAM: {free:.1f}GB free, {required_gb:.1f}GB required. "
			"Call clear_vram() and delete any loaded models before proceeding."
		)


def save_jsonl(data: list, path: str) -> None:
	"""Save list of dicts to JSONL file."""
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		for item in data:
			f.write(json.dumps(item) + "\n")


def load_jsonl(path: str) -> list:
	"""Load JSONL file into list of dicts."""
	with open(path, encoding="utf-8") as f:
		return [json.loads(line) for line in f if line.strip()]


def load_yaml(path: str) -> dict:
	"""Load YAML config file."""
	import yaml

	with open(path, encoding="utf-8") as f:
		return yaml.safe_load(f)
