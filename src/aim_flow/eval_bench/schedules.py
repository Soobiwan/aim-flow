"""SPFC schedule helpers for benchmark configs."""

from __future__ import annotations

from aim_flow.eval_bench.constants import DEFAULT_NUM_INFERENCE_STEPS, SPFC_SCHEDULE_ABLATIONS_1_INDEXED


def one_indexed_to_zero_indexed(
    schedule: list[int],
    num_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    required_count: int | None = None,
) -> list[int]:
    """Convert user-facing denoising step numbers to sampler step indices."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    converted: list[int] = []
    seen: set[int] = set()
    for step in schedule:
        user_step = int(step)
        if user_step < 1 or user_step > num_steps:
            raise ValueError(f"Aggregation step {user_step} is outside the 1..{num_steps} range.")
        zero_step = user_step - 1
        if zero_step in seen:
            raise ValueError(f"Duplicate aggregation step after conversion: {user_step}.")
        seen.add(zero_step)
        converted.append(zero_step)
    if required_count is not None and len(converted) != required_count:
        raise ValueError(f"Expected {required_count} aggregation steps, got {len(converted)}.")
    return converted


def get_schedule_ablation(
    name: str,
    num_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    required_count: int = 16,
) -> list[int]:
    """Return a named SPFC ablation schedule as zero-based indices."""

    if name not in SPFC_SCHEDULE_ABLATIONS_1_INDEXED:
        available = ", ".join(sorted(SPFC_SCHEDULE_ABLATIONS_1_INDEXED))
        raise KeyError(f"Unknown SPFC schedule ablation {name!r}. Available: {available}")
    return one_indexed_to_zero_indexed(
        SPFC_SCHEDULE_ABLATIONS_1_INDEXED[name],
        num_steps=num_steps,
        required_count=required_count,
    )
