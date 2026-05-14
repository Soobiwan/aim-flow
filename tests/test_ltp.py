import torch

from aim_flow.ltp import apply_latent_ltp, apply_velocity_ltp


def test_latent_ltp_leaves_small_candidate_offset_unchanged() -> None:
    x_t = torch.zeros(1, 4)
    x_next_anchor = torch.ones(1, 4)
    x_next_candidate = x_next_anchor + 0.1
    output, debug = apply_latent_ltp(x_t, x_next_anchor, x_next_candidate, radius_ratio=1.0)
    assert torch.allclose(output, x_next_candidate)
    assert debug["projected_offset_norm"] == debug["candidate_offset_norm"]


def test_latent_ltp_clips_large_candidate_offset() -> None:
    x_t = torch.zeros(1, 4)
    x_next_anchor = torch.ones(1, 4)
    x_next_candidate = x_next_anchor + 10.0
    output, debug = apply_latent_ltp(x_t, x_next_anchor, x_next_candidate, radius_ratio=0.25)
    assert output.shape == x_t.shape
    assert debug["projected_offset_norm"] < debug["candidate_offset_norm"]


def test_velocity_ltp_clips_correction_around_anchor_prediction() -> None:
    anchor_pred = torch.ones(1, 8)
    candidate_pred = anchor_pred + 10.0
    output, debug = apply_velocity_ltp(anchor_pred, candidate_pred, radius_ratio=0.1)
    assert output.shape == anchor_pred.shape
    assert debug["projected_velocity_correction_norm"] < debug["velocity_correction_norm"]

