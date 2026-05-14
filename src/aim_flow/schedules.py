"""Smooth timestep schedules for primitive residuals."""

from __future__ import annotations

import math


def _progress(step_index: int, num_steps: int) -> float:
    if num_steps <= 1:
        return 0.0
    if step_index < 0 or step_index >= num_steps:
        raise ValueError(f"step_index must be in [0, {num_steps - 1}], got {step_index}")
    return step_index / float(num_steps - 1)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def get_schedule_weight(schedule_name: str, step_index: int, num_steps: int) -> float:
    """Return a smooth primitive weight in [0, 1].

    `step_index=0` is the beginning of denoising at high noise. Early schedules
    are strongest there; late schedules rise near the final sampling steps.
    """

    name = schedule_name.lower()
    x = _progress(step_index, num_steps)

    if name == "constant":
        value = 1.0
    elif name == "early":
        value = 1.0 - _sigmoid(10.0 * (x - 0.35))
    elif name == "late":
        value = _sigmoid(10.0 * (x - 0.65))
    elif name == "middle":
        value = math.exp(-0.5 * ((x - 0.5) / 0.22) ** 2)
    elif name == "early_middle":
        early = 1.0 - _sigmoid(9.0 * (x - 0.45))
        middle = math.exp(-0.5 * ((x - 0.42) / 0.24) ** 2)
        value = 0.55 * early + 0.45 * middle
    elif name == "middle_late":
        middle = math.exp(-0.5 * ((x - 0.58) / 0.25) ** 2)
        late = _sigmoid(9.0 * (x - 0.55))
        value = 0.45 * middle + 0.55 * late
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    return max(0.0, min(1.0, float(value)))

