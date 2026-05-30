import pytest

from aim_flow.primitive_flow import build_condition_list, load_primitive_flow_set_from_yaml
from aim_flow.prompt_schema import PrimitiveFlowPrompt, PrimitiveFlowSet


def test_loads_german_shepherd_prompt_set():
    flow_set = load_primitive_flow_set_from_yaml("configs/sample_prompts.yaml", "german_shepherd")
    assert flow_set.target_prompt
    assert flow_set.source_prompt
    assert len(flow_set.get_enabled_primitives()) == 3


def test_get_all_condition_prompts_includes_source_and_target():
    flow_set = load_primitive_flow_set_from_yaml("configs/sample_prompts.yaml", "german_shepherd")
    prompts = flow_set.get_all_condition_prompts()
    names = flow_set.get_all_condition_names()
    assert prompts[0] == flow_set.source_prompt
    assert prompts[-1] == flow_set.target_prompt
    assert names[0] == "source"
    assert names[-1] == "target"


def test_build_condition_list_weights_and_roles():
    flow_set = load_primitive_flow_set_from_yaml("configs/sample_prompts.yaml", "german_shepherd")
    conditions = build_condition_list(flow_set, source_weight=0.7, target_weight=1.2)
    assert [condition["role"] for condition in conditions] == [
        "source",
        "primitive",
        "primitive",
        "primitive",
        "target",
    ]
    assert conditions[0]["weight"] == 0.7
    assert conditions[-1]["weight"] == 1.2


def test_build_condition_list_can_flatten_all_weights():
    flow_set = load_primitive_flow_set_from_yaml("configs/sample_prompts.yaml", "german_shepherd")
    conditions = build_condition_list(
        flow_set,
        source_weight=0.7,
        target_weight=1.2,
        uniform_weights=True,
    )
    assert [condition["weight"] for condition in conditions] == [1.0] * len(conditions)


def test_invalid_empty_primitive_fails():
    with pytest.raises(ValueError, match="text must be non-empty"):
        PrimitiveFlowSet(
            name="bad",
            target_prompt="target",
            source_prompt="source",
            primitive_prompts=[PrimitiveFlowPrompt(text="")],
        )
