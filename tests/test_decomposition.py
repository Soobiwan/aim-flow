from pathlib import Path

from aim_flow.decomposition import load_prompt_decomposition_from_yaml, simple_template_decompose


def test_load_prompt_decomposition_from_yaml(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.yaml"
    prompts.write_text(
        """
sample:
  full_prompt: "a red cube beside two blue spheres"
  anchor_prompt: "objects arranged on a table"
  primitive_prompts:
    - text: "red cube"
      type: "attribute_entity"
      weight: 0.8
      schedule: "late"
    - text: "two blue spheres"
      type: "count_entity"
      weight: 0.9
      schedule: "early"
""",
        encoding="utf-8",
    )
    decomposition = load_prompt_decomposition_from_yaml(prompts, "sample")
    assert decomposition.anchor_prompt == "objects arranged on a table"
    assert len(decomposition.primitive_prompts) == 2


def test_simple_template_decompose_for_smoke_tests() -> None:
    decomposition = simple_template_decompose("a red cube, beside a blue sphere, studio lighting")
    assert decomposition.full_prompt.startswith("a red cube")
    assert len(decomposition.primitive_prompts) == 3

