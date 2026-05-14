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
    elif name == "full_late":
        late = _sigmoid(10.0 * (x - 0.50))
        value = 0.25 + 0.75 * late
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    return max(0.0, min(1.0, float(value)))


def get_lambda_schedule_weight(schedule_name: str, step_index: int, num_steps: int) -> float:
    """Return the smooth schedule multiplier for global AIM-Flow strength."""

    return get_schedule_weight(schedule_name, step_index, num_steps)


def get_condition_schedule_weight(schedule_name: str, step_index: int, num_steps: int) -> float:
    """Return the smooth LadderFlow condition schedule weight."""

    return get_schedule_weight(schedule_name, step_index, num_steps)


def get_progressive_reference_index(step_index: int, num_steps: int, num_conditions: int) -> int:
    """Progressively move from C0 to the final ladder condition."""

    if num_conditions <= 0:
        raise ValueError("num_conditions must be positive.")
    if num_conditions == 1:
        return 0
    x = _progress(step_index, num_steps)
    if x >= 0.78:
        return num_conditions - 1
    scaled = x / 0.78
    return max(0, min(num_conditions - 1, int(round(scaled * (num_conditions - 1)))))


def get_ltp_radius_ratio(
    step_index: int,
    num_steps: int,
    early_radius_ratio: float,
    middle_radius_ratio: float,
    late_radius_ratio: float,
) -> float:
    """Smooth LTP radius: strict early, looser middle, slightly stricter late."""

    x = _progress(step_index, num_steps)
    early_w = 1.0 - _sigmoid(12.0 * (x - 0.30))
    late_w = _sigmoid(12.0 * (x - 0.70))
    middle_w = math.exp(-0.5 * ((x - 0.5) / 0.24) ** 2)
    total = early_w + middle_w + late_w
    if total <= 0.0:
        return float(middle_radius_ratio)
    value = (
        early_radius_ratio * early_w
        + middle_radius_ratio * middle_w
        + late_radius_ratio * late_w
    ) / total
    lo = min(early_radius_ratio, middle_radius_ratio, late_radius_ratio)
    hi = max(early_radius_ratio, middle_radius_ratio, late_radius_ratio)
    return max(lo, min(hi, float(value)))


def get_ltp_radius_ratio_for_step(
    step_index: int,
    num_steps: int,
    early_radius_ratio: float,
    middle_radius_ratio: float,
    late_radius_ratio: float,
) -> float:
    """Alias for LadderFlow LTP radius scheduling."""

    return get_ltp_radius_ratio(
        step_index,
        num_steps,
        early_radius_ratio,
        middle_radius_ratio,
        late_radius_ratio,
    )
