from aim_flow.prompt_schema import PrimitivePrompt, PromptDecomposition


def test_build_anchor_augmented_text_uses_anchor_and_primitive() -> None:
    primitive = PrimitivePrompt(text="four cyborg dogs", type="count_entity")
    assert primitive.build_anchor_augmented_text("cyborg dogs on grass") == "cyborg dogs on grass, four cyborg dogs"


def test_explicit_anchor_augmented_text_overrides_automatic_text() -> None:
    primitive = PrimitivePrompt(
        text="orange hats",
        type="attribute",
        anchor_augmented_text="dogs on grass, all dogs wearing orange hats",
    )
    assert primitive.build_anchor_augmented_text("ignored anchor") == "dogs on grass, all dogs wearing orange hats"


def test_disabled_primitives_are_excluded() -> None:
    decomposition = PromptDecomposition(
        full_prompt="full",
        anchor_prompt="anchor",
        primitive_prompts=[
            PrimitivePrompt(text="enabled", type="test"),
            PrimitivePrompt(text="disabled", type="test", enabled=False),
        ],
    )
    assert [primitive.text for primitive in decomposition.get_enabled_primitives()] == ["enabled"]
    assert decomposition.build_anchor_augmented_primitive_prompts() == ["anchor, enabled"]

