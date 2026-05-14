import torch

from aim_flow.aggregation import aggregate_vfa, compute_vfa_gate


def test_vfa_gate_is_high_when_primitive_aligns_with_full_direction() -> None:
    delta_full = torch.ones(1, 8)
    delta_i = torch.ones(1, 8)
    gate, cosine = compute_vfa_gate(delta_i, delta_full, conflict_threshold=-0.10)
    assert float(cosine.item()) > 0.99
    assert float(gate.item()) > 0.99


def test_vfa_gate_is_low_when_primitive_opposes_full_direction() -> None:
    delta_full = torch.ones(1, 8)
    delta_i = -torch.ones(1, 8)
    gate, cosine = compute_vfa_gate(delta_i, delta_full, conflict_threshold=-0.10)
    assert float(cosine.item()) < -0.99
    assert float(gate.item()) == 0.0


def test_aggregate_vfa_output_shape_matches_anchor() -> None:
    anchor = torch.ones(1, 2, 4, 4)
    full = anchor + 0.5
    primitive = anchor + 0.25
    output, debug = aggregate_vfa(
        anchor_pred=anchor,
        full_pred=full,
        primitive_preds=[primitive],
        primitive_base_weights=[1.0],
        primitive_schedule_weights=[1.0],
        lambda_global=0.75,
        conflict_threshold=-0.10,
        velocity_clip_ratio=0.35,
    )
    assert output.shape == anchor.shape
    assert debug["primitive_gates"][0] > 0.0


def test_aggregate_vfa_velocity_clipping_reduces_large_correction() -> None:
    anchor = torch.ones(1, 10)
    full = anchor + 10.0
    primitive = anchor + 10.0
    _, debug = aggregate_vfa(
        anchor_pred=anchor,
        full_pred=full,
        primitive_preds=[primitive],
        primitive_base_weights=[1.0],
        primitive_schedule_weights=[1.0],
        lambda_global=1.0,
        conflict_threshold=-0.10,
        velocity_clip_ratio=0.1,
    )
    assert debug["clipped_correction_norm"] < debug["raw_correction_norm"]

