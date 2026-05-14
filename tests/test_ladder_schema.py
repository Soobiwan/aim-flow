from aim_flow.decomposition import load_condition_ladder_from_yaml


def test_loads_german_shepherd_ladder():
    ladder = load_condition_ladder_from_yaml(
        "configs/sample_prompts.yaml",
        "german_shepherd",
        section="ladder_prompts",
    )

    enabled = ladder.get_enabled_conditions()
    assert len(enabled) >= 2
    assert ladder.get_base_condition().name == "C0_base_object_action"
    assert ladder.get_final_condition().name == "C3_full"
    assert all(0.0 <= condition.weight <= 3.0 for condition in enabled)
    assert ladder.full_prompt in ladder.get_final_condition().text
