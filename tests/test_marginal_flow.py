import torch

from aim_flow.marginal_flow import (
    clip_marginal_correction,
    compute_marginal_flow_update,
    normalize_per_sample,
    parse_intervention_steps,
    per_sample_norm,
    project_to_simplex,
    remove_target_opposition,
    solve_balanced_marginal_direction,
)


def test_simplex_projection_is_nonnegative_and_sums_to_one() -> None:
    values = torch.tensor([[2.0, -1.0, 0.5], [-4.0, 3.0, 9.0]])

    projected = project_to_simplex(values)

    assert torch.all(projected >= 0)
    assert torch.allclose(projected.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_balanced_solver_matches_identical_directions() -> None:
    direction = normalize_per_sample(torch.randn(2, 4, 3, 3))
    directions = torch.stack([direction, direction, direction], dim=1)

    balanced, alpha, _ = solve_balanced_marginal_direction(directions)

    assert torch.allclose(balanced, direction, atol=2e-5)
    assert torch.allclose(alpha, torch.full_like(alpha, 1.0 / 3.0), atol=1e-5)


def test_balanced_solver_balances_two_orthogonal_directions() -> None:
    directions = torch.tensor([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]])

    balanced, alpha, _ = solve_balanced_marginal_direction(directions)

    expected = torch.tensor([[[[2.0**-0.5, 2.0**-0.5]]]])
    assert torch.allclose(alpha, torch.tensor([[0.5, 0.5]]), atol=1e-5)
    assert torch.allclose(balanced, expected, atol=2e-5)


def test_balanced_solver_stays_finite_for_opposing_directions() -> None:
    directions = torch.tensor([[[[[1.0, 0.0]]], [[[-1.0, 0.0]]]]])

    balanced, alpha, raw = solve_balanced_marginal_direction(directions)

    assert torch.isfinite(balanced).all()
    assert torch.isfinite(alpha).all()
    assert torch.isfinite(raw).all()
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(balanced, torch.zeros_like(balanced), atol=1e-6)


def test_target_preservation_removes_only_negative_parallel_component() -> None:
    full = torch.tensor([[[[1.0, 0.0]]]])
    opposing = torch.tensor([[[[-1.0, 1.0]]]])
    aligned = torch.tensor([[[[1.0, 1.0]]]])

    preserved_opposing = remove_target_opposition(opposing, full)
    preserved_aligned = remove_target_opposition(aligned, full)

    assert torch.allclose(preserved_opposing, torch.tensor([[[[0.0, 1.0]]]]), atol=2e-6)
    assert torch.allclose(preserved_aligned, aligned, atol=1e-6)


def test_trust_clipping_limits_each_latent_sample() -> None:
    full = torch.stack([torch.ones(16, 8, 8), torch.full((16, 8, 8), 2.0)])
    correction = torch.full_like(full, 100.0)
    trust_ratio = 0.15

    clipped = clip_marginal_correction(correction, full, trust_ratio=trust_ratio)

    assert torch.all(per_sample_norm(clipped) <= trust_ratio * per_sample_norm(full) + 1e-5)


def test_complete_update_handles_expected_latent_shape_without_nonfinite_values() -> None:
    torch.manual_seed(7)
    full = torch.randn(2, 16, 8, 8)
    ablated = [full - torch.randn_like(full) * scale for scale in (0.1, 0.2, 0.3)]

    correction, alpha, debug = compute_marginal_flow_update(full, ablated, trust_ratio=0.15)

    assert correction.shape == full.shape
    assert alpha.shape == (2, 3)
    assert torch.isfinite(correction).all()
    assert torch.isfinite(alpha).all()
    assert torch.all(alpha >= 0)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.all(per_sample_norm(correction) <= 0.15 * per_sample_norm(full) + 1e-5)
    assert debug is not None
    assert debug["pairwise_cosine_matrix"].shape == (2, 3, 3)
    assert torch.all(debug["final_correction_target_cosine"] >= -1e-5)


def test_sparse_intervention_steps_support_indices_and_fractions() -> None:
    assert parse_intervention_steps(8, intervention_steps=[-1, 1, 3, 5, 8]) == {1, 3, 5}
    assert parse_intervention_steps(8, intervention_step_fractions=[0.0, 0.5, 0.75]) == {0, 4, 5}
