"""Velocity Field Aggregation (VFA) math for AIM-Flow v2."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


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


def compute_pairwise_cosine_matrix(predictions: list[torch.Tensor]) -> torch.Tensor:
    """Return a K x K cosine matrix between complete condition velocity predictions."""

    if not predictions:
        raise ValueError("predictions must be non-empty.")
    base_shape = predictions[0].shape
    for prediction in predictions:
        if prediction.shape != base_shape:
            raise ValueError("All ladder predictions must have the same shape.")
    flattened = [flatten_for_similarity(prediction).mean(dim=0) for prediction in predictions]
    matrix = torch.stack(flattened, dim=0)
    matrix = F.normalize(matrix, dim=-1, eps=1e-8)
    return matrix @ matrix.T


def compute_consensus_gates(
    predictions: list[torch.Tensor],
    min_gate: float = 0.0,
    max_gate: float = 1.0,
) -> tuple[list[float], dict[str, Any]]:
    """Gate conditions by mean positive agreement with other condition velocities."""

    pairwise = compute_pairwise_cosine_matrix(predictions)
    gates: list[float] = []
    for idx in range(pairwise.shape[0]):
        others = [j for j in range(pairwise.shape[0]) if j != idx]
        if not others:
            gate = 1.0
        else:
            gate = float(pairwise[idx, others].clamp_min(0.0).mean().cpu().item())
        gates.append(max(min_gate, min(max_gate, gate)))
    return gates, {"pairwise_cosine_matrix": pairwise.detach().cpu().tolist(), "consensus_gates": gates}


def compute_final_consistency_gates(
    predictions: list[torch.Tensor],
    final_prediction: torch.Tensor,
    min_gate: float = 0.0,
    max_gate: float = 1.0,
) -> tuple[list[float], dict[str, Any]]:
    """Gate conditions by positive agreement with the final condition velocity."""

    gates: list[float] = []
    cosines: list[float] = []
    for prediction in predictions:
        cosine = _mean_float(cosine_similarity_tensor(prediction, final_prediction))
        cosines.append(cosine)
        gates.append(max(min_gate, min(max_gate, max(0.0, cosine))))
    return gates, {"final_consistency_cosines": cosines, "final_consistency_gates": gates}


def compute_target_consistency_gates(
    predictions: list[torch.Tensor],
    target_prediction: torch.Tensor,
    min_gate: float = 0.0,
    max_gate: float = 1.0,
) -> tuple[list[float], dict[str, Any]]:
    """Gate condition velocities by positive agreement with the target prompt velocity."""

    gates: list[float] = []
    cosines: list[float] = []
    for prediction in predictions:
        cosine = _mean_float(cosine_similarity_tensor(prediction, target_prediction))
        cosines.append(cosine)
        gates.append(max(min_gate, min(max_gate, max(0.0, cosine))))
    return gates, {"target_consistency_cosines": cosines, "target_consistency_gates": gates}


def aggregate_primitive_vfa(
    predictions: list[torch.Tensor],
    condition_names: list[str],
    condition_roles: list[str],
    condition_base_weights: list[float],
    target_index: int,
    vfa_temperature: float = 0.7,
    use_consensus_gating: bool = True,
    use_target_consistency_gating: bool = True,
    velocity_clip_ratio: float = 0.50,
    steering_strength: float = 1.0,
    min_gate: float = 0.0,
    max_gate: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Aggregate source, primitive, and target prompt velocities directly.

    This is intentionally not an anchor/source residual method. The source flow
    is one condition among the VFA inputs, while the target flow is retained as
    the stabilization reference for clipping.
    """

    if not predictions:
        raise ValueError("predictions must be non-empty.")
    count = len(predictions)
    if not (len(condition_names) == len(condition_roles) == len(condition_base_weights) == count):
        raise ValueError("prediction, condition name, role, and weight counts must match.")
    if not 0 <= target_index < count:
        raise ValueError(f"target_index out of range: {target_index}")
    if vfa_temperature <= 0.0:
        raise ValueError("vfa_temperature must be positive.")
    if steering_strength < 0.0:
        raise ValueError("steering_strength must be non-negative.")
    base_shape = predictions[0].shape
    for prediction in predictions:
        if prediction.shape != base_shape:
            raise ValueError("All primitive-flow predictions must have equal shape.")

    pairwise = compute_pairwise_cosine_matrix(predictions).detach().cpu().tolist()
    consensus_gates = [1.0] * count
    if use_consensus_gating:
        consensus_gates, _ = compute_consensus_gates(predictions, min_gate, max_gate)

    target_gates = [1.0] * count
    target_debug: dict[str, Any] = {"target_consistency_gates": target_gates, "target_consistency_cosines": None}
    if use_target_consistency_gating:
        target_gates, target_debug = compute_target_consistency_gates(
            predictions,
            predictions[target_index],
            min_gate,
            max_gate,
        )

    raw_scores: list[float] = []
    for base_weight, consensus_gate, target_gate in zip(condition_base_weights, consensus_gates, target_gates):
        raw_scores.append(float(base_weight) * float(consensus_gate) * float(target_gate))
    raw_scores[target_index] = max(raw_scores[target_index], 1e-4)

    score_tensor = torch.tensor(raw_scores, device=predictions[0].device, dtype=torch.float32)
    softmax_weights = torch.softmax(score_tensor / float(vfa_temperature), dim=0)
    v_raw = torch.zeros_like(predictions[0])
    for weight, prediction in zip(softmax_weights, predictions):
        v_raw = v_raw + prediction * weight.to(device=prediction.device, dtype=prediction.dtype)

    target_pred = predictions[target_index]
    correction = v_raw - target_pred
    max_norm = _norm_per_sample(target_pred) * float(velocity_clip_ratio)
    correction_clipped = clip_tensor_norm(correction, max_norm)
    correction_steered = correction_clipped * float(steering_strength)
    v_agg = target_pred + correction_steered

    debug = {
        "condition_names": list(condition_names),
        "condition_roles": list(condition_roles),
        "condition_base_weights": [float(w) for w in condition_base_weights],
        "target_index": int(target_index),
        "pairwise_cosine_matrix": pairwise,
        "consensus_gates": consensus_gates,
        "target_consistency_gates": target_gates,
        "target_consistency_cosines": target_debug.get("target_consistency_cosines"),
        "use_consensus_gating": bool(use_consensus_gating),
        "use_target_consistency_gating": bool(use_target_consistency_gating),
        "raw_scores": raw_scores,
        "softmax_weights": [float(w.detach().cpu().item()) for w in softmax_weights],
        "target_norm": _mean_float(_norm_per_sample(target_pred)),
        "raw_correction_norm": _mean_float(_norm_per_sample(correction)),
        "clipped_correction_norm": _mean_float(_norm_per_sample(correction_clipped)),
        "steered_correction_norm": _mean_float(_norm_per_sample(correction_steered)),
        "velocity_clip_ratio": float(velocity_clip_ratio),
        "steering_strength": float(steering_strength),
    }
    return v_agg, debug


def aggregate_ladder_vfa(
    predictions: list[torch.Tensor],
    condition_base_weights: list[float],
    condition_schedule_weights: list[float],
    reference_index: int,
    final_index: int,
    vfa_temperature: float = 0.7,
    use_consensus_gating: bool = True,
    use_final_consistency_gating: bool = True,
    velocity_clip_ratio: float = 0.50,
    min_gate: float = 0.0,
    max_gate: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Aggregate complete ladder condition velocities with VFA and reference clipping."""

    if not predictions:
        raise ValueError("predictions must be non-empty.")
    count = len(predictions)
    if not (len(condition_base_weights) == len(condition_schedule_weights) == count):
        raise ValueError("prediction, base weight, and schedule weight counts must match.")
    if not 0 <= reference_index < count:
        raise ValueError(f"reference_index out of range: {reference_index}")
    if not 0 <= final_index < count:
        raise ValueError(f"final_index out of range: {final_index}")
    for prediction in predictions:
        if prediction.shape != predictions[0].shape:
            raise ValueError("All ladder predictions must have equal shape.")
    if vfa_temperature <= 0.0:
        raise ValueError("vfa_temperature must be positive.")

    consensus_gates = [1.0] * count
    consensus_debug: dict[str, Any] = {"pairwise_cosine_matrix": compute_pairwise_cosine_matrix(predictions).detach().cpu().tolist()}
    if use_consensus_gating:
        consensus_gates, consensus_debug = compute_consensus_gates(predictions, min_gate, max_gate)

    final_gates = [1.0] * count
    final_debug: dict[str, Any] = {"final_consistency_gates": final_gates}
    if use_final_consistency_gating:
        final_gates, final_debug = compute_final_consistency_gates(
            predictions,
            predictions[final_index],
            min_gate,
            max_gate,
        )

    raw_scores = []
    for base_weight, schedule_weight, consensus_gate, final_gate in zip(
        condition_base_weights,
        condition_schedule_weights,
        consensus_gates,
        final_gates,
    ):
        raw_scores.append(float(base_weight) * float(schedule_weight) * float(consensus_gate) * float(final_gate))
    raw_scores[final_index] = max(raw_scores[final_index], 0.15 * max(max(raw_scores), 1.0))

    score_tensor = torch.tensor(raw_scores, device=predictions[0].device, dtype=torch.float32)
    softmax_weights = torch.softmax(score_tensor / float(vfa_temperature), dim=0)
    v_raw = torch.zeros_like(predictions[0])
    for weight, prediction in zip(softmax_weights, predictions):
        v_raw = v_raw + prediction * weight.to(device=prediction.device, dtype=prediction.dtype)

    ref_pred = predictions[reference_index]
    correction = v_raw - ref_pred
    max_norm = _norm_per_sample(ref_pred) * float(velocity_clip_ratio)
    correction_clipped = clip_tensor_norm(correction, max_norm)
    v_agg = ref_pred + correction_clipped

    debug = {
        "pairwise_cosine_matrix": consensus_debug.get("pairwise_cosine_matrix"),
        "consensus_gates": consensus_gates,
        "final_consistency_gates": final_gates,
        "final_consistency_cosines": final_debug.get("final_consistency_cosines"),
        "condition_base_weights": [float(w) for w in condition_base_weights],
        "condition_schedule_weights": [float(w) for w in condition_schedule_weights],
        "raw_scores": raw_scores,
        "softmax_weights": [float(w.detach().cpu().item()) for w in softmax_weights],
        "reference_index": int(reference_index),
        "final_index": int(final_index),
        "raw_correction_norm": _mean_float(_norm_per_sample(correction)),
        "clipped_correction_norm": _mean_float(_norm_per_sample(correction_clipped)),
        "reference_norm": _mean_float(_norm_per_sample(ref_pred)),
    }
    return v_agg, debug


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
