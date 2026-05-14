import torch

from aim_flow.aggregation import aggregate_predictions, conflict_gate, cosine_similarity, norm_clip


def test_aggregation_output_shape_equals_anchor_shape() -> None:
    anchor = torch.ones(1, 2, 4, 4)
    primitive = anchor + 0.5
    output, debug = aggregate_predictions(
        anchor_pred=anchor,
        primitive_preds=[primitive],
        primitive_weights=[1.0],
        primitive_schedule_weights=[1.0],
        lambda_global=0.5,
        conflict_threshold=-0.15,
        norm_clip_ratio=0.0,
        mode="naive",
    )
    assert output.shape == anchor.shape
    assert debug["total_correction_norm"] > 0


def test_conflict_gating_downweights_opposite_vectors() -> None:
    anchor = torch.ones(1, 8)
    delta = -torch.ones(1, 8)
    gate = conflict_gate(delta, anchor, threshold=-0.15)
    assert float(gate.item()) == 0.0


def test_full_aggregation_gates_conflicting_primitive() -> None:
    anchor = torch.ones(1, 4)
    primitive = torch.zeros(1, 4)
    full = anchor + torch.ones(1, 4)
    output, debug = aggregate_predictions(
        anchor_pred=anchor,
        full_pred=full,
        primitive_preds=[primitive],
        primitive_weights=[1.0],
        primitive_schedule_weights=[1.0],
        lambda_global=1.0,
        conflict_threshold=-0.15,
        norm_clip_ratio=0.0,
        mode="full",
    )
    assert torch.allclose(output, anchor)
    assert debug["primitive_gates"][0] == 0.0


def test_norm_clip_limits_delta_norm() -> None:
    reference = torch.ones(1, 10)
    delta = torch.ones(1, 10) * 10.0
    clipped = norm_clip(delta, reference, clip_ratio=0.25)
    ratio = clipped.norm() / reference.norm()
    assert ratio <= 0.251


def test_cosine_similarity_identity() -> None:
    tensor = torch.randn(2, 3, 4)
    cos = cosine_similarity(tensor, tensor)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)
