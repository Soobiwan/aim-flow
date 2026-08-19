"""Math helpers for contextual Marginal Flow steering."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def _working_dtype(tensor: torch.Tensor) -> torch.dtype:
    return torch.float32 if tensor.dtype in {torch.float16, torch.bfloat16} else tensor.dtype


def per_sample_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Return one vector norm per leading batch element."""

    if tensor.ndim < 1:
        raise ValueError("Expected a tensor with a batch dimension.")
    work = tensor.to(dtype=_working_dtype(tensor))
    return torch.linalg.vector_norm(work.reshape(work.shape[0], -1), dim=1)


def normalize_per_sample(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize every sample across all non-batch dimensions."""

    if eps <= 0:
        raise ValueError("eps must be positive.")
    work = tensor.to(dtype=_working_dtype(tensor))
    norms = per_sample_norm(work)
    shape = (work.shape[0],) + (1,) * (work.ndim - 1)
    return (work / (norms.reshape(shape) + float(eps))).to(dtype=tensor.dtype)


def project_to_simplex(values: torch.Tensor) -> torch.Tensor:
    """Project vectors along the last dimension onto the probability simplex."""

    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("Simplex projection requires a non-empty last dimension.")
    if not torch.isfinite(values).all():
        raise ValueError("Simplex projection input must be finite.")

    original_dtype = values.dtype if values.is_floating_point() else torch.float32
    work = values.to(dtype=_working_dtype(values) if values.is_floating_point() else torch.float32)
    sorted_values, _ = torch.sort(work, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=-1) - 1.0
    indices = torch.arange(1, work.shape[-1] + 1, device=work.device, dtype=work.dtype)
    active = sorted_values - cumulative / indices > 0
    rho = active.sum(dim=-1, keepdim=True).clamp_min(1) - 1
    theta = torch.gather(cumulative / indices, dim=-1, index=rho)
    projected = torch.clamp(work - theta, min=0.0)
    projected = projected / projected.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(work.dtype).eps)
    return projected.to(dtype=original_dtype)


def solve_balanced_marginal_direction(
    normalized_directions: torch.Tensor,
    solver_steps: int = 20,
    solver_lr: float = 0.1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the simplex min-norm problem for a small set of directions.

    Args:
        normalized_directions: Tensor shaped ``[batch, primitives, ...]``.

    Returns:
        Unit balanced direction, simplex coefficients, and the unnormalized
        convex-combination direction.
    """

    if normalized_directions.ndim < 3:
        raise ValueError("normalized_directions must have shape [batch, primitives, ...].")
    if normalized_directions.shape[1] < 1:
        raise ValueError("At least one marginal direction is required.")
    if solver_steps < 0:
        raise ValueError("solver_steps must be non-negative.")
    if solver_lr < 0:
        raise ValueError("solver_lr must be non-negative.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    work = normalized_directions.to(dtype=_working_dtype(normalized_directions))
    batch_size, num_primitives = work.shape[:2]
    flat = work.reshape(batch_size, num_primitives, -1)
    gram = torch.bmm(flat, flat.transpose(1, 2))
    alpha = torch.full(
        (batch_size, num_primitives),
        1.0 / num_primitives,
        device=work.device,
        dtype=work.dtype,
    )
    for _ in range(int(solver_steps)):
        gradient = 2.0 * torch.bmm(gram, alpha.unsqueeze(-1)).squeeze(-1)
        alpha = project_to_simplex(alpha - float(solver_lr) * gradient)

    raw_direction = torch.sum(
        alpha.reshape((batch_size, num_primitives) + (1,) * (work.ndim - 2)) * work,
        dim=1,
    )
    direction = normalize_per_sample(raw_direction, eps=eps)
    return direction, alpha, raw_direction


def remove_target_opposition(
    correction: torch.Tensor,
    full_prediction: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Remove only the component of a correction that opposes the target flow."""

    if correction.shape != full_prediction.shape:
        raise ValueError(
            f"correction and full_prediction shapes must match: "
            f"{tuple(correction.shape)} vs {tuple(full_prediction.shape)}."
        )
    work = correction.to(dtype=_working_dtype(correction))
    target = full_prediction.to(device=work.device, dtype=work.dtype)
    target_hat = normalize_per_sample(target, eps=eps)
    parallel = torch.sum(
        work.reshape(work.shape[0], -1) * target_hat.reshape(target_hat.shape[0], -1),
        dim=1,
    )
    negative_parallel = torch.minimum(parallel, torch.zeros_like(parallel))
    shape = (work.shape[0],) + (1,) * (work.ndim - 1)
    preserved = work - negative_parallel.reshape(shape) * target_hat
    return preserved.to(dtype=correction.dtype)


def clip_marginal_correction(
    correction: torch.Tensor,
    full_prediction: torch.Tensor,
    trust_ratio: float = 0.15,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Limit correction norms to ``trust_ratio * ||full_prediction||`` per sample."""

    if correction.shape != full_prediction.shape:
        raise ValueError(
            f"correction and full_prediction shapes must match: "
            f"{tuple(correction.shape)} vs {tuple(full_prediction.shape)}."
        )
    if trust_ratio < 0:
        raise ValueError("trust_ratio must be non-negative.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    work = correction.to(dtype=_working_dtype(correction))
    target = full_prediction.to(device=work.device, dtype=work.dtype)
    correction_norm = per_sample_norm(work)
    maximum_norm = float(trust_ratio) * per_sample_norm(target)
    scale = torch.minimum(
        torch.ones_like(correction_norm),
        maximum_norm / (correction_norm + float(eps)),
    )
    shape = (work.shape[0],) + (1,) * (work.ndim - 1)
    return (work * scale.reshape(shape)).to(dtype=correction.dtype)


def compute_marginal_flow_update(
    full_pred: torch.Tensor,
    ablated_preds: Sequence[torch.Tensor],
    trust_ratio: float = 0.15,
    solver_steps: int = 20,
    solver_lr: float = 0.1,
    eps: float = 1e-6,
    return_debug: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    """Compute the balanced, target-preserving Marginal Flow correction."""

    if not ablated_preds:
        raise ValueError("At least one ablated prediction is required.")
    for index, prediction in enumerate(ablated_preds):
        if prediction.shape != full_pred.shape:
            raise ValueError(
                f"Ablated prediction {index} shape must match full_pred: "
                f"{tuple(prediction.shape)} vs {tuple(full_pred.shape)}."
            )
        if prediction.device != full_pred.device:
            raise ValueError(
                f"Ablated prediction {index} device must match full_pred: "
                f"{prediction.device} vs {full_pred.device}."
            )

    work_full = full_pred.to(dtype=_working_dtype(full_pred))
    work_ablated = torch.stack(
        [prediction.to(device=full_pred.device, dtype=work_full.dtype) for prediction in ablated_preds],
        dim=1,
    )
    residuals = work_full.unsqueeze(1) - work_ablated
    batch_size, num_primitives = residuals.shape[:2]
    residuals_flat_batch = residuals.reshape(batch_size * num_primitives, *residuals.shape[2:])
    normalized = normalize_per_sample(residuals_flat_batch, eps=eps).reshape_as(residuals)

    direction, alpha, raw_direction = solve_balanced_marginal_direction(
        normalized,
        solver_steps=solver_steps,
        solver_lr=solver_lr,
        eps=eps,
    )
    preserved = remove_target_opposition(direction, work_full, eps=eps)
    correction = clip_marginal_correction(preserved, work_full, trust_ratio=trust_ratio, eps=eps)
    correction = correction.to(dtype=full_pred.dtype)

    debug: dict[str, Any] | None = None
    if return_debug:
        flat_normalized = normalized.reshape(batch_size, num_primitives, -1)
        pairwise_cosine = torch.bmm(flat_normalized, flat_normalized.transpose(1, 2))
        full_norm = per_sample_norm(work_full)
        correction_norm = per_sample_norm(correction)
        final_dot = torch.sum(
            correction.to(dtype=work_full.dtype).reshape(batch_size, -1) * work_full.reshape(batch_size, -1),
            dim=1,
        )
        final_cosine = final_dot / (correction_norm * full_norm + float(eps))
        debug = {
            "raw_marginal_norms": per_sample_norm(residuals_flat_batch).reshape(batch_size, num_primitives),
            "pairwise_cosine_matrix": pairwise_cosine,
            "alpha": alpha,
            "raw_balanced_direction_norm": per_sample_norm(raw_direction),
            "target_preserved_norm": per_sample_norm(preserved),
            "trust_clipped_norm": correction_norm,
            "full_prediction_norm": full_norm,
            "final_correction_target_cosine": final_cosine,
        }
    return correction, alpha, debug


def parse_intervention_steps(
    num_steps: int,
    intervention_steps: Sequence[int] | None = None,
    intervention_step_fractions: Sequence[float] | None = None,
) -> set[int]:
    """Resolve Marginal Flow's sparse, zero-based denoising step indices."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if intervention_steps:
        candidates = [int(step) for step in intervention_steps]
    elif intervention_step_fractions:
        candidates = [round(float(fraction) * (num_steps - 1)) for fraction in intervention_step_fractions]
    else:
        candidates = []
    return {step for step in candidates if 0 <= step < num_steps}
