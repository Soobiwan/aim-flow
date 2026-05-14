"""Tensor aggregation math for AIM-Flow."""

from __future__ import annotations

from typing import Any

import torch


def flatten_for_similarity(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten all non-batch dimensions for cosine computations."""

    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(1, -1)
    return tensor.reshape(tensor.shape[0], -1)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity over flattened non-batch dimensions."""

    a_flat = flatten_for_similarity(a.float())
    b_flat = flatten_for_similarity(b.float())
    numerator = (a_flat * b_flat).sum(dim=-1)
    denominator = a_flat.norm(dim=-1).clamp_min(eps) * b_flat.norm(dim=-1).clamp_min(eps)
    return numerator / denominator


def norm_clip(delta: torch.Tensor, reference: torch.Tensor, clip_ratio: float) -> torch.Tensor:
    """Clip delta norm relative to the reference prediction norm."""

    if clip_ratio is None or clip_ratio <= 0:
        return delta
    delta_flat = flatten_for_similarity(delta.float())
    ref_flat = flatten_for_similarity(reference.float())
    delta_norm = delta_flat.norm(dim=-1).clamp_min(1e-8)
    max_norm = ref_flat.norm(dim=-1).clamp_min(1e-8) * clip_ratio
    scale = torch.minimum(torch.ones_like(delta_norm), max_norm / delta_norm)
    view_shape = [delta.shape[0]] + [1] * (delta.ndim - 1) if delta.ndim > 1 else [1]
    return delta * scale.to(device=delta.device, dtype=delta.dtype).reshape(view_shape)


def conflict_gate(delta: torch.Tensor, anchor: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return a soft gate that downweights residuals opposing the anchor."""

    cos = cosine_similarity(delta, anchor)
    if threshold <= -1.0:
        return torch.ones_like(cos)
    gate = (cos - threshold) / (1.0 - threshold)
    return gate.clamp(0.0, 1.0).to(device=delta.device, dtype=delta.dtype)


def _mean_float(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().mean().cpu().item())


def aggregate_predictions(
    anchor_pred: torch.Tensor,
    primitive_preds: list[torch.Tensor],
    primitive_weights: list[float],
    primitive_schedule_weights: list[float],
    lambda_global: float,
    conflict_threshold: float,
    norm_clip_ratio: float,
    mode: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Aggregate anchor and primitive predictions.

    Modes:
      - anchor: return the anchor unchanged
      - naive: residual sum without conflict gating
      - full: residual sum with conflict gating and norm clipping
    """

    if mode == "anchor":
        return anchor_pred, {
            "primitive_norms": [],
            "primitive_cosines": [],
            "primitive_gates": [],
            "effective_weights": [],
            "total_correction_norm": 0.0,
            "anchor_norm": _mean_float(flatten_for_similarity(anchor_pred.float()).norm(dim=-1)),
        }
    if mode not in {"naive", "full"}:
        raise ValueError("aggregate_predictions handles modes 'anchor', 'naive', and 'full' only.")
    if not (len(primitive_preds) == len(primitive_weights) == len(primitive_schedule_weights)):
        raise ValueError("primitive predictions, weights, and schedule weights must have equal length.")

    correction = torch.zeros_like(anchor_pred)
    primitive_norms: list[float] = []
    primitive_cosines: list[float] = []
    primitive_gates: list[float] = []
    effective_weights: list[float] = []

    for primitive_pred, base_weight, schedule_weight in zip(
        primitive_preds, primitive_weights, primitive_schedule_weights
    ):
        if primitive_pred.shape != anchor_pred.shape:
            raise ValueError(f"Primitive shape {primitive_pred.shape} does not match anchor {anchor_pred.shape}.")
        delta = primitive_pred - anchor_pred
        cos = cosine_similarity(delta, anchor_pred)
        gate = torch.ones_like(cos, dtype=anchor_pred.dtype, device=anchor_pred.device)
        if mode == "full":
            gate = conflict_gate(delta, anchor_pred, conflict_threshold)
        clipped_delta = norm_clip(delta, anchor_pred, norm_clip_ratio)
        weight = float(base_weight) * float(schedule_weight)
        view_shape = [anchor_pred.shape[0]] + [1] * (anchor_pred.ndim - 1) if anchor_pred.ndim > 1 else [1]
        gated_weight = gate.reshape(view_shape) * weight
        correction = correction + gated_weight.to(anchor_pred.dtype) * clipped_delta

        primitive_norms.append(_mean_float(flatten_for_similarity(delta.float()).norm(dim=-1)))
        primitive_cosines.append(_mean_float(cos))
        primitive_gates.append(_mean_float(gate))
        effective_weights.append(weight * _mean_float(gate))

    total_correction = lambda_global * correction
    output = anchor_pred + total_correction
    debug = {
        "primitive_norms": primitive_norms,
        "primitive_cosines": primitive_cosines,
        "primitive_gates": primitive_gates,
        "effective_weights": effective_weights,
        "total_correction_norm": _mean_float(flatten_for_similarity(total_correction.float()).norm(dim=-1)),
        "anchor_norm": _mean_float(flatten_for_similarity(anchor_pred.float()).norm(dim=-1)),
    }
    return output, debug

