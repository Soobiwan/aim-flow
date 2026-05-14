"""Helpers for Sparse Primitive Flow Composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aim_flow.prompt_schema import PrimitiveFlowSet
from aim_flow.utils import read_yaml


def load_primitive_flow_set_from_yaml(
    path: str | Path,
    key: str,
    section: str = "primitive_flow_prompts",
) -> PrimitiveFlowSet:
    """Load one PrimitiveFlowSet from a YAML prompt file."""

    data = read_yaml(path)
    if section not in data:
        raise KeyError(f"Prompt section {section!r} not found in {path}.")
    section_data = data[section] or {}
    if key not in section_data:
        raise KeyError(f"Prompt key {key!r} not found in section {section!r}.")
    return PrimitiveFlowSet.from_dict(section_data[key])


def _clip_step(step: int, num_steps: int) -> int | None:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if step < 0 or step >= num_steps:
        return None
    return int(step)


def parse_aggregation_steps(
    num_steps: int,
    aggregation_steps: list[int] | None = None,
    aggregation_step_fractions: list[float] | None = None,
    final_only: bool = False,
    aggregate_every_n_steps: int | None = None,
) -> set[int]:
    """Resolve sparse aggregation controls into zero-based denoising step indices."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if final_only:
        return {num_steps - 1}

    resolved: set[int] = set()
    if aggregation_steps:
        for step in aggregation_steps:
            clipped = _clip_step(int(step), num_steps)
            if clipped is not None:
                resolved.add(clipped)
        return resolved

    if aggregation_step_fractions:
        for fraction in aggregation_step_fractions:
            step = round(float(fraction) * (num_steps - 1))
            clipped = _clip_step(int(step), num_steps)
            if clipped is not None:
                resolved.add(clipped)

    if aggregate_every_n_steps is not None:
        every = max(1, int(aggregate_every_n_steps))
        resolved.update(range(0, num_steps, every))
        resolved.add(num_steps - 1)

    return resolved


def build_condition_list(
    flow_set: PrimitiveFlowSet,
    include_source: bool = True,
    include_target: bool = True,
    source_weight: float = 0.7,
    target_weight: float = 1.2,
    max_primitives: int | None = None,
) -> list[dict[str, Any]]:
    """Build the complete prompt-conditioned flow list for VFA."""

    flow_set.validate()
    conditions: list[dict[str, Any]] = []
    if include_source:
        conditions.append(
            {
                "name": "source",
                "role": "source",
                "text": flow_set.source_prompt,
                "weight": float(source_weight),
            }
        )
    primitives = flow_set.get_enabled_primitives()
    if max_primitives is not None:
        primitives = primitives[: max(0, int(max_primitives))]
    for index, primitive in enumerate(primitives):
        conditions.append(
            {
                "name": primitive.name or f"primitive_{index}",
                "role": primitive.role,
                "text": primitive.text,
                "weight": float(primitive.weight),
            }
        )
    if include_target:
        conditions.append(
            {
                "name": "target",
                "role": "target",
                "text": flow_set.target_prompt,
                "weight": float(target_weight),
            }
        )
    if not any(condition["role"] == "primitive" for condition in conditions):
        raise ValueError("Primitive flow condition list requires at least one primitive condition.")
    return conditions
