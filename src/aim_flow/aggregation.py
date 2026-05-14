"""Velocity Field Aggregation (VFA) math for AIM-Flow v2."""

from __future__ import annotations

from typing import Any

import torch


def flatten_for_similarity(x: torch.Tensor) -> torch.Tensor:
    """Flatten for per-sample similarity, preserving a batch dimension when present."""

    detached = x.detach().float()
    if detached.ndim == 0:
        return detached.reshape(1, 1)
    if detached.ndim == 1:
        return detached.reshape(1, -1)
    return detached.reshape(detached.shape[0], -1)


def cosine_similarity_tensor(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safely compute cosine similarity over flattened non-batch dimensions."""

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for cosine similarity: {a.shape} vs {b.shape}")
    a_flat = flatten_for_similarity(a)
    b_flat = flatten_for_similarity(b)
    numerator = (a_flat * b_flat).sum(dim=-1)
    denominator = a_flat.norm(dim=-1).clamp_min(eps) * b_flat.norm(dim=-1).clamp_min(eps)
    return numerator / denominator


def _broadcast_batch_scale(scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim <= 1:
        return scale.reshape(1).to(device=target.device, dtype=target.dtype)
    return scale.reshape([target.shape[0]] + [1] * (target.ndim - 1)).to(device=target.device, dtype=target.dtype)


def _mean_float(x: torch.Tensor) -> float:
    return float(x.detach().float().mean().cpu().item())


def _norm_per_sample(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return flatten_for_similarity(x).norm(dim=-1).clamp_min(eps)


def clip_tensor_norm(x: torch.Tensor, max_norm: float | torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Clip tensor norm per sample while preserving direction."""

    x_norm = _norm_per_sample(x, eps=eps)
    if isinstance(max_norm, torch.Tensor):
        max_norm_tensor = max_norm.detach().float().to(device=x.device).reshape(-1)
    else:
        max_norm_tensor = torch.full_like(x_norm, float(max_norm), device=x.device)
    max_norm_tensor = max_norm_tensor.clamp_min(0.0)
    scale = torch.minimum(torch.ones_like(x_norm), max_norm_tensor / x_norm.clamp_min(eps))
    return x * _broadcast_batch_scale(scale, x)


def compute_vfa_gate(
    delta_i: torch.Tensor,
    delta_full: torch.Tensor,
    conflict_threshold: float = -0.10,
    min_gate: float = 0.0,
    max_gate: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate primitive residuals by agreement with the model's full-prompt direction."""

    cosine = cosine_similarity_tensor(delta_i, delta_full)
    if conflict_threshold >= 1.0:
        raise ValueError("conflict_threshold must be < 1.0")
    gate = (cosine - conflict_threshold) / (1.0 - conflict_threshold)
    gate = gate.clamp(min_gate, max_gate)
    return gate.to(device=delta_i.device, dtype=delta_i.dtype), cosine


def aggregate_vfa(
    anchor_pred: torch.Tensor,
    full_pred: torch.Tensor,
    primitive_preds: list[torch.Tensor],
    primitive_base_weights: list[float],
    primitive_schedule_weights: list[float],
    lambda_global: float,
    conflict_threshold: float,
    velocity_clip_ratio: float,
    min_gate: float = 0.0,
    max_gate: float = 1.0,
    use_delta_full_gate: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Aggregate anchor-augmented primitive velocity residuals with VFA."""

    if full_pred.shape != anchor_pred.shape:
        raise ValueError(f"Full prediction shape {full_pred.shape} does not match anchor {anchor_pred.shape}.")
    if not (len(primitive_preds) == len(primitive_base_weights) == len(primitive_schedule_weights)):
        raise ValueError("primitive predictions, base weights, and schedule weights must have equal length.")

    delta_full = full_pred - anchor_pred
    delta_raw = torch.zeros_like(anchor_pred)
    primitive_cosines: list[float] = []
    primitive_gates: list[float] = []
    primitive_effective_weights: list[float] = []

    for primitive_pred, base_weight, schedule_weight in zip(
        primitive_preds, primitive_base_weights, primitive_schedule_weights
    ):
        if primitive_pred.shape != anchor_pred.shape:
            raise ValueError(f"Primitive prediction shape {primitive_pred.shape} does not match anchor {anchor_pred.shape}.")
        delta_i = primitive_pred - anchor_pred
        if use_delta_full_gate:
            gate, cosine = compute_vfa_gate(
                delta_i,
                delta_full,
                conflict_threshold=conflict_threshold,
                min_gate=min_gate,
                max_gate=max_gate,
            )
        else:
            gate = torch.ones(anchor_pred.shape[0] if anchor_pred.ndim > 1 else 1, device=anchor_pred.device, dtype=anchor_pred.dtype)
            cosine = cosine_similarity_tensor(delta_i, delta_full)
        scalar_weight = float(base_weight) * float(schedule_weight)
        eff = gate * scalar_weight
        delta_raw = delta_raw + _broadcast_batch_scale(eff, delta_i) * delta_i
        primitive_cosines.append(_mean_float(cosine))
        primitive_gates.append(_mean_float(gate))
        primitive_effective_weights.append(float(scalar_weight) * _mean_float(gate))

    anchor_norm = _norm_per_sample(anchor_pred)
    max_norm = anchor_norm * float(velocity_clip_ratio)
    delta_clipped = clip_tensor_norm(delta_raw, max_norm)
    aggregated_pred = anchor_pred + float(lambda_global) * delta_clipped

    debug = {
        "delta_full_norm": _mean_float(_norm_per_sample(delta_full)),
        "anchor_norm": _mean_float(anchor_norm),
        "raw_correction_norm": _mean_float(_norm_per_sample(delta_raw)),
        "clipped_correction_norm": _mean_float(_norm_per_sample(delta_clipped)),
        "primitive_cosines": primitive_cosines,
        "primitive_gates": primitive_gates,
        "primitive_base_weights": [float(w) for w in primitive_base_weights],
        "primitive_schedule_weights": [float(w) for w in primitive_schedule_weights],
        "primitive_effective_weights": primitive_effective_weights,
    }
    return aggregated_pred, debug


def _aggregate_naive_v1(
    anchor_pred: torch.Tensor,
    primitive_preds: list[torch.Tensor],
    primitive_weights: list[float],
    primitive_schedule_weights: list[float],
    lambda_global: float,
    norm_clip_ratio: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Original standalone-primitive residual aggregation, kept for comparison."""

    correction = torch.zeros_like(anchor_pred)
    primitive_norms: list[float] = []
    effective_weights: list[float] = []
    for primitive_pred, base_weight, schedule_weight in zip(primitive_preds, primitive_weights, primitive_schedule_weights):
        if primitive_pred.shape != anchor_pred.shape:
            raise ValueError(f"Primitive prediction shape {primitive_pred.shape} does not match anchor {anchor_pred.shape}.")
        delta = primitive_pred - anchor_pred
        weight = float(base_weight) * float(schedule_weight)
        correction = correction + weight * delta
        primitive_norms.append(_mean_float(_norm_per_sample(delta)))
        effective_weights.append(weight)
    max_norm = _norm_per_sample(anchor_pred) * float(norm_clip_ratio)
    clipped = clip_tensor_norm(correction, max_norm)
    output = anchor_pred + float(lambda_global) * clipped
    return output, {
        "primitive_norms": primitive_norms,
        "primitive_cosines": [],
        "primitive_gates": [1.0 for _ in primitive_preds],
        "effective_weights": effective_weights,
        "total_correction_norm": _mean_float(_norm_per_sample(float(lambda_global) * clipped)),
        "anchor_norm": _mean_float(_norm_per_sample(anchor_pred)),
        "raw_correction_norm": _mean_float(_norm_per_sample(correction)),
        "clipped_correction_norm": _mean_float(_norm_per_sample(clipped)),
        "method": "naive_v1_standalone_primitives",
    }


def aggregate_predictions(
    anchor_pred: torch.Tensor,
    primitive_preds: list[torch.Tensor],
    primitive_weights: list[float],
    primitive_schedule_weights: list[float],
    lambda_global: float,
    conflict_threshold: float,
    norm_clip_ratio: float,
    mode: str,
    full_pred: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Backward-compatible aggregation wrapper.

    `mode="naive_v1"` uses the original standalone primitive residuals.
    `mode="aim_v2"` or `mode="full"` requires `full_pred` and uses VFA.
    """

    if mode == "anchor":
        return anchor_pred, {
            "primitive_norms": [],
            "primitive_cosines": [],
            "primitive_gates": [],
            "effective_weights": [],
            "total_correction_norm": 0.0,
            "anchor_norm": _mean_float(_norm_per_sample(anchor_pred)),
        }
    if mode in {"naive", "naive_v1"}:
        return _aggregate_naive_v1(
            anchor_pred,
            primitive_preds,
            primitive_weights,
            primitive_schedule_weights,
            lambda_global,
            norm_clip_ratio,
        )
    if mode in {"full", "aim_v2"}:
        if full_pred is None:
            raise ValueError("full_pred is required for AIM-Flow v2 VFA aggregation.")
        return aggregate_vfa(
            anchor_pred=anchor_pred,
            full_pred=full_pred,
            primitive_preds=primitive_preds,
            primitive_base_weights=primitive_weights,
            primitive_schedule_weights=primitive_schedule_weights,
            lambda_global=lambda_global,
            conflict_threshold=conflict_threshold,
            velocity_clip_ratio=norm_clip_ratio,
        )
    raise ValueError(f"Unsupported aggregation mode: {mode}")


# Backward-compatible aliases used by older tests/imports.
cosine_similarity = cosine_similarity_tensor


def norm_clip(delta: torch.Tensor, reference: torch.Tensor, clip_ratio: float) -> torch.Tensor:
    return clip_tensor_norm(delta, _norm_per_sample(reference) * float(clip_ratio))


def conflict_gate(delta: torch.Tensor, anchor: torch.Tensor, threshold: float) -> torch.Tensor:
    gate, _ = compute_vfa_gate(delta, anchor, conflict_threshold=threshold)
    return gate
