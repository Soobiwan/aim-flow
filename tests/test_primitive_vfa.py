import pytest
import torch

from aim_flow.aggregation import (
    aggregate_primitive_vfa,
    compute_consensus_gates,
    compute_target_consistency_gates,
)


def test_primitive_vfa_output_shape_and_weights_sum():
    target = torch.ones(1, 2, 2)
    preds = [target * 0.5, target * 1.1, target]
    output, debug = aggregate_primitive_vfa(
        predictions=preds,
        condition_names=["source", "primitive", "target"],
        condition_roles=["source", "primitive", "target"],
        condition_base_weights=[0.7, 1.0, 1.2],
        target_index=2,
    )
    assert output.shape == target.shape
    assert sum(debug["softmax_weights"]) == pytest.approx(1.0)


def test_consensus_gate_prefers_aligned_vectors():
    aligned_a = torch.tensor([1.0, 0.0])
    aligned_b = torch.tensor([0.9, 0.1])
    opposite = torch.tensor([-1.0, 0.0])
    gates, _ = compute_consensus_gates([aligned_a, aligned_b, opposite])
    assert gates[0] > gates[2]
    assert gates[1] > gates[2]


def test_target_consistency_gate_prefers_target_aligned_vectors():
    target = torch.tensor([1.0, 0.0])
    aligned = torch.tensor([0.5, 0.0])
    opposite = torch.tensor([-1.0, 0.0])
    gates, _ = compute_target_consistency_gates([aligned, opposite, target], target)
    assert gates[0] > gates[1]
    assert gates[2] > gates[1]


def test_velocity_clipping_reduces_large_deviation_around_target():
    target = torch.ones(1, 4)
    large = torch.ones(1, 4) * 100.0
    output, debug = aggregate_primitive_vfa(
        predictions=[large, target],
        condition_names=["primitive", "target"],
        condition_roles=["primitive", "target"],
        condition_base_weights=[10.0, 1.2],
        target_index=1,
        use_consensus_gating=False,
        use_target_consistency_gating=False,
        velocity_clip_ratio=0.1,
    )
    assert output.shape == target.shape
    assert debug["clipped_correction_norm"] < debug["raw_correction_norm"]
