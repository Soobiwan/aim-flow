import torch

from aim_flow.ltp import apply_primitive_latent_ltp, apply_primitive_velocity_ltp


def test_latent_ltp_leaves_small_candidate_offset_unchanged():
    x_t = torch.zeros(1, 4)
    x_next_target = torch.ones(1, 4)
    x_next_candidate = x_next_target + 0.1
    output, debug = apply_primitive_latent_ltp(x_t, x_next_target, x_next_candidate, radius_ratio=1.0)
    assert torch.allclose(output, x_next_candidate)
    assert debug["projected_offset_norm"] == debug["candidate_offset_norm"]


def test_latent_ltp_clips_large_candidate_offset():
    x_t = torch.zeros(1, 4)
    x_next_target = torch.ones(1, 4)
    x_next_candidate = x_next_target + 10.0
    output, debug = apply_primitive_latent_ltp(x_t, x_next_target, x_next_candidate, radius_ratio=0.25)
    assert output.shape == x_t.shape
    assert debug["projected_offset_norm"] < debug["candidate_offset_norm"]


def test_velocity_ltp_clips_candidate_around_target_prediction():
    target_pred = torch.ones(1, 4)
    candidate_pred = torch.ones(1, 4) * 100.0
    output, debug = apply_primitive_velocity_ltp(target_pred, candidate_pred, radius_ratio=0.1)
    assert output.shape == target_pred.shape
    assert debug["projected_correction_norm"] < debug["candidate_correction_norm"]
