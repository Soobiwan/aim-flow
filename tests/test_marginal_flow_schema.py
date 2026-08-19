import pytest

from aim_flow.config import RunConfig
from aim_flow.decomposition import load_marginal_flow_prompt_set_from_yaml
from aim_flow.prompt_schema import MarginalFlowPromptSet


def test_loads_explicit_contextual_ablations_without_source_prompt() -> None:
    prompt_set = load_marginal_flow_prompt_set_from_yaml(
        "configs/marginal_flow_prompts.yaml",
        "red_cube_blue_sphere",
    )

    assert prompt_set.target_prompt == "A red cube underneath a blue sphere."
    assert prompt_set.get_ablated_prompts() == [
        "A cube underneath a blue sphere.",
        "A red cube underneath a sphere.",
        "A red cube and a blue sphere.",
    ]
    assert "source_prompt" not in prompt_set.to_dict()


def test_marginal_prompt_schema_requires_at_least_one_primitive() -> None:
    with pytest.raises(ValueError, match="at least one primitive"):
        MarginalFlowPromptSet.from_dict({"target_prompt": "A target", "primitives": []})


def test_marginal_prompt_schema_requires_explicit_ablation() -> None:
    with pytest.raises(ValueError, match="ablated_prompt"):
        MarginalFlowPromptSet.from_dict(
            {
                "target_prompt": "A target",
                "primitives": [{"name": "missing", "primitive": "object"}],
            }
        )


def test_marginal_config_loads_without_changing_primitive_flow_defaults() -> None:
    config = RunConfig.from_dict(
        {
            "marginal_flow": {
                "enabled": True,
                "intervention_steps": [4, 8, 12, 16],
                "steering_strength": 1.0,
                "trust_ratio": 0.15,
                "solver_steps": 20,
                "solver_lr": 0.1,
            }
        }
    )

    assert config.marginal_flow.enabled is True
    assert config.marginal_flow.intervention_steps == [4, 8, 12, 16]
    assert config.primitive_flow.enabled is True
    assert config.primitive_flow.mode == "sparse_primitive_flow"
