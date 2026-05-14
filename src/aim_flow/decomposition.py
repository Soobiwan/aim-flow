"""Manual and template prompt decomposition helpers.

AIM-Flow v1 intentionally uses manually supplied decompositions. The fallback
function in this module is only for smoke tests and should not be interpreted as
a research-quality decomposer.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from aim_flow.prompt_schema import PrimitivePrompt, PromptDecomposition
from aim_flow.utils import read_yaml


def load_prompt_decomposition_from_yaml(path: str | Path, key: str) -> PromptDecomposition:
    """Load a named prompt decomposition from a YAML file."""

    data = read_yaml(path)
    if key not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(f"Prompt key '{key}' not found in {path}. Available keys: {available}")
    return PromptDecomposition.from_dict(data[key])


def save_prompt_decomposition(decomposition: PromptDecomposition, path: str | Path) -> None:
    """Save one decomposition to YAML."""

    decomposition.validate()
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(decomposition.to_dict(), f, sort_keys=False)


def simple_template_decompose(prompt: str) -> PromptDecomposition:
    """Return a tiny rule-based decomposition for smoke tests.

    This is not the main AIM-Flow method and does not use an LLM. It simply
    reuses the prompt as the anchor and creates short comma-separated primitives.
    """

    cleaned = prompt.strip()
    if not cleaned:
        raise ValueError("prompt must be non-empty")
    chunks = [part.strip() for part in cleaned.split(",") if part.strip()]
    primitives = [
        PrimitivePrompt(text=chunk, type="template_chunk", weight=1.0, schedule="constant")
        for chunk in (chunks[:5] or [cleaned])
    ]
    return PromptDecomposition(
        full_prompt=cleaned,
        anchor_prompt=chunks[-1] if len(chunks) > 1 else cleaned,
        primitive_prompts=primitives,
    )

