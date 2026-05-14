"""Small shared utilities for AIM-Flow."""

from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    """Map config dtype strings to torch dtypes."""

    normalized = dtype_name.lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timestamp() -> str:
    """Return a filesystem-friendly UTC timestamp."""

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def safe_slugify(text: str, max_length: int = 80) -> str:
    """Convert free text into a conservative filename slug."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].strip("-") or "prompt"


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write JSON with stable formatting."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to YAML."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def get_hf_token() -> str | None:
    """Read a Hugging Face token from common environment variables."""

    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

