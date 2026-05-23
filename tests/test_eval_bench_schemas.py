import pytest

from aim_flow.eval_bench.schemas import DecompositionManifest, PromptManifest, PromptSample, make_decomposition_template


def test_decomposition_manifest_converts_to_primitive_flow_set():
    manifest = DecompositionManifest.from_dict(
        {
            "schema_version": "spfc_decomposition_v1",
            "items": [
                {
                    "id": "sample_000000",
                    "target_prompt": "a green bench and a red car",
                    "source_prompt": "a bench and a car",
                    "negative_prompt": "blurry",
                    "primitive_prompts": [
                        {
                            "name": "S1_green_bench",
                            "text": "a green bench",
                            "role": "primitive",
                            "weight": 1.0,
                            "enabled": True,
                        }
                    ],
                }
            ],
        }
    )
    flow_set = manifest.items[0].to_flow_set()
    assert flow_set.target_prompt == "a green bench and a red car"
    assert flow_set.source_prompt == "a bench and a car"
    assert flow_set.get_enabled_primitives()[0].name == "S1_green_bench"


def test_decomposition_manifest_rejects_missing_primitives():
    with pytest.raises(ValueError, match="primitive_prompts"):
        DecompositionManifest.from_dict(
            {
                "schema_version": "spfc_decomposition_v1",
                "items": [
                    {
                        "id": "sample_000000",
                        "target_prompt": "target",
                        "source_prompt": "source",
                        "primitive_prompts": [],
                    }
                ],
            }
        )


def test_decomposition_template_covers_prompt_manifest():
    prompts = PromptManifest(
        benchmark="unit",
        subset_size=2,
        seed=13,
        samples=[
            PromptSample(id="a", category="color", prompt="a red cup", source="unit", split="unit"),
            PromptSample(id="b", category="shape", prompt="a square clock", source="unit", split="unit"),
        ],
    )
    template = make_decomposition_template(prompts)
    assert [item.id for item in template.items] == ["a", "b"]
    assert template.items[0].to_flow_set().target_prompt == "a red cup"
