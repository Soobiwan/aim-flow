import pytest

from aim_flow.prompt_schema import PrimitivePrompt, PromptDecomposition


def valid_dict() -> dict:
    return {
        "full_prompt": "three red spheres beside a blue cube",
        "anchor_prompt": "objects on a table",
        "primitive_prompts": [
            {"text": "three red spheres", "type": "count_entity", "weight": 0.8, "schedule": "early"},
            {"text": "blue cube", "type": "attribute_entity", "weight": 0.7, "schedule": "late"},
        ],
    }


def test_prompt_schema_loads_valid_dicts() -> None:
    decomposition = PromptDecomposition.from_dict(valid_dict())
    assert decomposition.full_prompt.startswith("three")
    assert len(decomposition.primitive_prompts) == 2
    assert decomposition.to_dict()["primitive_prompts"][0]["schedule"] == "early"


def test_prompt_schema_rejects_invalid_primitive_weights() -> None:
    with pytest.raises(ValueError):
        PrimitivePrompt(text="bad", type="test", weight=-0.1).validate()
    data = valid_dict()
    data["primitive_prompts"][0]["weight"] = 2.5
    with pytest.raises(ValueError):
        PromptDecomposition.from_dict(data)


def test_prompt_schema_rejects_unknown_schedule() -> None:
    data = valid_dict()
    data["primitive_prompts"][0]["schedule"] = "never"
    with pytest.raises(ValueError):
        PromptDecomposition.from_dict(data)

