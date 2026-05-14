import torch

from aim_flow.aggregation import (
    aggregate_ladder_vfa,
    compute_consensus_gates,
    compute_final_consistency_gates,
)


def test_ladder_vfa_shape_and_softmax_weights():
    preds = [
        torch.ones(1, 4),
        torch.ones(1, 4) * 2,
        torch.ones(1, 4) * 3,
    ]

    out, debug = aggregate_ladder_vfa(
        predictions=preds,
        condition_base_weights=[1.0, 1.0, 1.2],
        condition_schedule_weights=[1.0, 0.8, 1.0],
        reference_index=1,
        final_index=2,
    )

    assert out.shape == preds[0].shape
    assert abs(sum(debug["softmax_weights"]) - 1.0) < 1e-6


def test_consensus_gates_higher_for_aligned_vectors():
    preds = [
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.9, 0.1]]),
        torch.tensor([[-1.0, 0.0]]),
    ]

    gates, _ = compute_consensus_gates(preds)

    assert gates[0] > gates[2]
    assert gates[1] > gates[2]


def test_final_consistency_gates_higher_for_final_aligned_vectors():
    preds = [
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.5, 0.0]]),
        torch.tensor([[-1.0, 0.0]]),
    ]

    gates, _ = compute_final_consistency_gates(preds, preds[0])

    assert gates[0] > gates[2]
    assert gates[1] > gates[2]


def test_ladder_vfa_correction_clipping():
    ref = torch.ones(1, 8)
    preds = [ref, torch.ones(1, 8) * 100.0]

    out, debug = aggregate_ladder_vfa(
        predictions=preds,
        condition_base_weights=[1.0, 3.0],
        condition_schedule_weights=[1.0, 1.0],
        reference_index=0,
        final_index=1,
        velocity_clip_ratio=0.1,
        use_consensus_gating=False,
        use_final_consistency_gating=False,
    )

    correction_norm = (out - ref).norm().item()
    assert correction_norm <= 0.1 * ref.norm().item() + 1e-5
    assert debug["clipped_correction_norm"] <= debug["raw_correction_norm"]
