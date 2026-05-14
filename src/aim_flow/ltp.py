"""Latent Trajectory Projection (LTP) for AIM-Flow v2."""

from __future__ import annotations

from typing import Any

import torch

from aim_flow.aggregation import clip_tensor_norm, flatten_for_similarity


def _mean_norm(x: torch.Tensor) -> float:
    return float(flatten_for_similarity(x).norm(dim=-1).mean().cpu().item())


def apply_velocity_ltp(
    anchor_pred: torch.Tensor,
    candidate_pred: torch.Tensor,
    radius_ratio: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Clip candidate velocity correction around the anchor velocity."""

    if anchor_pred.shape != candidate_pred.shape:
        raise ValueError(f"Shape mismatch for velocity LTP: {anchor_pred.shape} vs {candidate_pred.shape}")
    correction = candidate_pred - anchor_pred
    max_norm = flatten_for_similarity(anchor_pred).norm(dim=-1).clamp_min(eps) * float(radius_ratio)
    projected = clip_tensor_norm(correction, max_norm, eps=eps)
    return anchor_pred + projected, {
        "velocity_correction_norm": _mean_norm(correction),
        "projected_velocity_correction_norm": _mean_norm(projected),
        "velocity_anchor_norm": _mean_norm(anchor_pred),
        "radius_ratio": float(radius_ratio),
        "velocity_ltp_active": True,
    }


def apply_latent_ltp(
    x_t: torch.Tensor,
    x_next_anchor: torch.Tensor,
    x_next_candidate: torch.Tensor,
    radius_ratio: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Project candidate latent update into a radius around the anchor trajectory."""

    if x_t.shape != x_next_anchor.shape or x_t.shape != x_next_candidate.shape:
        raise ValueError(
            "Latent LTP tensors must have matching shapes: "
            f"x_t={x_t.shape}, anchor={x_next_anchor.shape}, candidate={x_next_candidate.shape}"
        )
    anchor_step = x_next_anchor - x_t
    candidate_offset = x_next_candidate - x_next_anchor
    max_offset_norm = flatten_for_similarity(anchor_step).norm(dim=-1).clamp_min(eps) * float(radius_ratio)
    projected_offset = clip_tensor_norm(candidate_offset, max_offset_norm, eps=eps)
    x_next = x_next_anchor + projected_offset
    return x_next, {
        "anchor_step_norm": _mean_norm(anchor_step),
        "candidate_offset_norm": _mean_norm(candidate_offset),
        "projected_offset_norm": _mean_norm(projected_offset),
        "radius_ratio": float(radius_ratio),
        "latent_ltp_active": True,
    }


def apply_ladder_velocity_ltp(
    reference_pred: torch.Tensor,
    candidate_pred: torch.Tensor,
    radius_ratio: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Clip a ladder candidate velocity around the scheduled reference velocity."""

    if reference_pred.shape != candidate_pred.shape:
        raise ValueError(f"Shape mismatch for ladder velocity LTP: {reference_pred.shape} vs {candidate_pred.shape}")
    correction = candidate_pred - reference_pred
    max_norm = flatten_for_similarity(reference_pred).norm(dim=-1).clamp_min(eps) * float(radius_ratio)
    projected = clip_tensor_norm(correction, max_norm, eps=eps)
    return reference_pred + projected, {
        "reference_velocity_norm": _mean_norm(reference_pred),
        "candidate_correction_norm": _mean_norm(correction),
        "projected_correction_norm": _mean_norm(projected),
        "radius_ratio": float(radius_ratio),
        "ltp_mode": "velocity",
        "ltp_active": True,
    }


def apply_ladder_latent_ltp(
    x_t: torch.Tensor,
    x_next_reference: torch.Tensor,
    x_next_candidate: torch.Tensor,
    radius_ratio: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Project ladder candidate latent update around the scheduled reference trajectory."""

    if x_t.shape != x_next_reference.shape or x_t.shape != x_next_candidate.shape:
        raise ValueError(
            "Ladder latent LTP tensors must have matching shapes: "
            f"x_t={x_t.shape}, reference={x_next_reference.shape}, candidate={x_next_candidate.shape}"
        )
    reference_step = x_next_reference - x_t
    candidate_offset = x_next_candidate - x_next_reference
    max_offset_norm = flatten_for_similarity(reference_step).norm(dim=-1).clamp_min(eps) * float(radius_ratio)
    projected_offset = clip_tensor_norm(candidate_offset, max_offset_norm, eps=eps)
    x_next = x_next_reference + projected_offset
    return x_next, {
        "reference_step_norm": _mean_norm(reference_step),
        "candidate_offset_norm": _mean_norm(candidate_offset),
        "projected_offset_norm": _mean_norm(projected_offset),
        "radius_ratio": float(radius_ratio),
        "ltp_mode": "latent",
        "ltp_active": True,
    }


def apply_primitive_velocity_ltp(
    target_pred: torch.Tensor,
    candidate_pred: torch.Tensor,
    radius_ratio: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Clip primitive-flow candidate velocity around the full target velocity."""

    if target_pred.shape != candidate_pred.shape:
        raise ValueError(f"Shape mismatch for primitive velocity LTP: {target_pred.shape} vs {candidate_pred.shape}")
    correction = candidate_pred - target_pred
    max_norm = flatten_for_similarity(target_pred).norm(dim=-1).clamp_min(eps) * float(radius_ratio)
    projected = clip_tensor_norm(correction, max_norm, eps=eps)
    return target_pred + projected, {
        "target_velocity_norm": _mean_norm(target_pred),
        "candidate_correction_norm": _mean_norm(correction),
        "projected_correction_norm": _mean_norm(projected),
        "radius_ratio": float(radius_ratio),
        "ltp_mode": "velocity",
        "ltp_active": True,
    }


def apply_primitive_latent_ltp(
    x_t: torch.Tensor,
    x_next_target: torch.Tensor,
    x_next_candidate: torch.Tensor,
    radius_ratio: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Project candidate latent update around the full target prompt trajectory."""

    if x_t.shape != x_next_target.shape or x_t.shape != x_next_candidate.shape:
        raise ValueError(
            "Primitive latent LTP tensors must have matching shapes: "
            f"x_t={x_t.shape}, target={x_next_target.shape}, candidate={x_next_candidate.shape}"
        )
    target_step = x_next_target - x_t
    candidate_offset = x_next_candidate - x_next_target
    max_offset_norm = flatten_for_similarity(target_step).norm(dim=-1).clamp_min(eps) * float(radius_ratio)
    projected_offset = clip_tensor_norm(candidate_offset, max_offset_norm, eps=eps)
    x_next = x_next_target + projected_offset
    return x_next, {
        "target_step_norm": _mean_norm(target_step),
        "candidate_offset_norm": _mean_norm(candidate_offset),
        "projected_offset_norm": _mean_norm(projected_offset),
        "radius_ratio": float(radius_ratio),
        "ltp_mode": "latent",
        "ltp_active": True,
    }
