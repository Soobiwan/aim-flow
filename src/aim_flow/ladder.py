"""Condition ladder helpers for LadderFlow / AIM-Flow v3."""

from __future__ import annotations

from pathlib import Path

from aim_flow.prompt_schema import ConditionLadder
from aim_flow.schedules import get_progressive_reference_index
from aim_flow.utils import read_yaml


def build_manual_ladder_from_dict(data: dict) -> ConditionLadder:
    """Build and validate a condition ladder from a mapping."""

    return ConditionLadder.from_dict(data)


def load_ladder_from_yaml(path: str | Path, key: str, section: str = "ladder_prompts") -> ConditionLadder:
    """Load a named condition ladder from YAML."""

    data = read_yaml(path)
    section_data = data.get(section, data if section == "" else None)
    if not isinstance(section_data, dict):
        available = ", ".join(sorted(data.keys()))
        raise KeyError(f"Prompt section '{section}' not found in {path}. Available top-level keys: {available}")
    if key not in section_data:
        available = ", ".join(sorted(section_data.keys()))
        raise KeyError(f"Ladder key '{key}' not found in section '{section}'. Available keys: {available}")
    return build_manual_ladder_from_dict(section_data[key])


def select_reference_condition_index(
    step_index: int,
    num_steps: int,
    num_conditions: int,
    policy: str = "progressive",
) -> int:
    """Select the scheduled ladder reference condition index."""

    if num_conditions <= 0:
        raise ValueError("num_conditions must be positive.")
    name = policy.lower()
    if name == "progressive":
        return get_progressive_reference_index(step_index, num_steps, num_conditions)
    if name == "full":
        return num_conditions - 1
    if name == "base":
        return 0
    if name == "piecewise":
        chunk = max(1, num_steps // num_conditions)
        return min(num_conditions - 1, max(0, step_index // chunk))
    raise ValueError(f"Unknown reference policy: {policy}")


def get_active_condition_indices(
    step_index: int,
    num_steps: int,
    num_conditions: int,
    active_policy: str = "all",
    reference_index: int | None = None,
) -> list[int]:
    """Select active ladder conditions for a sparse aggregation step."""

    del step_index, num_steps
    if num_conditions <= 0:
        raise ValueError("num_conditions must be positive.")
    ref = 0 if reference_index is None else max(0, min(num_conditions - 1, reference_index))
    final = num_conditions - 1
    name = active_policy.lower()
    if name == "all":
        indices = list(range(num_conditions))
    elif name == "upto_reference":
        indices = list(range(ref + 1))
    elif name == "around_reference":
        indices = [idx for idx in (ref - 1, ref, ref + 1) if 0 <= idx < num_conditions]
    elif name == "final_plus_reference":
        indices = [ref, final]
        if ref <= 1:
            indices.append(0)
    else:
        raise ValueError(f"Unknown active condition policy: {active_policy}")
    return sorted(set(indices))


def parse_aggregation_steps(
    aggregation_steps: list[int] | None,
    aggregation_step_fractions: list[float] | None,
    num_steps: int,
) -> set[int]:
    """Parse sparse aggregation step indices from explicit values or fractions."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    parsed: set[int] = set()
    if aggregation_steps:
        candidates = [int(step) for step in aggregation_steps]
    elif aggregation_step_fractions:
        candidates = [round(float(frac) * num_steps) for frac in aggregation_step_fractions]
    else:
        candidates = []
    for step in candidates:
        if 0 <= step < num_steps:
            parsed.add(int(step))
    return parsed
