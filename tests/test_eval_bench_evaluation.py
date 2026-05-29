import json
from pathlib import Path

import pytest
from PIL import Image

from aim_flow.eval_bench.evaluation import run_t2i_compbench_official
from aim_flow.eval_bench.schemas import PromptManifest, PromptSample


def _write_manifest_and_image(tmp_path: Path, category: str) -> tuple[Path, Path]:
    sample = PromptSample(
        id=f"sample_{category}",
        category=category,
        prompt="a red cup" if category != "spatial" else "a cup on the left of a chair",
        source="unit",
        split="unit",
    )
    manifest_path = tmp_path / "manifest.json"
    PromptManifest(benchmark="t2i_compbench", subset_size=1, seed=13, samples=[sample]).save(manifest_path)
    run_root = tmp_path / "runs"
    image_dir = run_root / "t2i_compbench" / "base"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), color="red").save(image_dir / f"{sample.id}.png")
    return manifest_path, run_root


def _write_official_script(root, rel_path: str) -> None:
    script = root / rel_path
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('stub')\n", encoding="utf-8")


def test_t2i_compbench_commands_use_official_script_working_dirs(tmp_path):
    manifest_path, run_root = _write_manifest_and_image(tmp_path, "color")
    official = tmp_path / "official"
    _write_official_script(official, "BLIPvqa_eval/BLIP_vqa.py")

    output = run_t2i_compbench_official(
        manifest_path=manifest_path,
        run_root=run_root,
        methods=["base"],
        official_repo=official,
        output_dir=tmp_path / "eval",
        execute=False,
        categories=["color"],
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    command = data["commands"][0]
    assert command["cmd"][0].endswith("python") or "python" in command["cmd"][0]
    assert command["cmd"][1] == str(official / "BLIPvqa_eval" / "BLIP_vqa.py")
    assert command["cwd"] == str(official / "BLIPvqa_eval")


def test_t2i_compbench_preflights_missing_selected_official_scripts(tmp_path):
    manifest_path, run_root = _write_manifest_and_image(tmp_path, "color")
    official = tmp_path / "official"
    official.mkdir()

    with pytest.raises(FileNotFoundError, match="BLIP_vqa.py"):
        run_t2i_compbench_official(
            manifest_path=manifest_path,
            run_root=run_root,
            methods=["base"],
            official_repo=official,
            output_dir=tmp_path / "eval",
            execute=False,
            categories=["color"],
        )


def test_t2i_compbench_only_requires_scripts_for_selected_categories(tmp_path):
    manifest_path, run_root = _write_manifest_and_image(tmp_path, "spatial")
    official = tmp_path / "official"
    _write_official_script(official, "UniDet_eval/2D_spatial_eval.py")

    output = run_t2i_compbench_official(
        manifest_path=manifest_path,
        run_root=run_root,
        methods=["base"],
        official_repo=official,
        output_dir=tmp_path / "eval",
        execute=False,
        categories=["spatial"],
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    command = data["commands"][0]
    assert command["cmd"][1] == str(official / "UniDet_eval" / "2D_spatial_eval.py")
    assert command["cwd"] == str(official / "UniDet_eval")
