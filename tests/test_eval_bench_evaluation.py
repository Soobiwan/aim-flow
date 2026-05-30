import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.bench_evaluate import _merge_scores
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


def test_t2i_compbench_parallel_jobs_receive_exclusive_gpu_ids(tmp_path):
    samples = [
        PromptSample(id="sample_color", category="color", prompt="a red cup", source="unit", split="unit"),
        PromptSample(id="sample_shape", category="shape", prompt="a round cup", source="unit", split="unit"),
    ]
    manifest_path = tmp_path / "manifest.json"
    PromptManifest(benchmark="t2i_compbench", subset_size=2, seed=13, samples=samples).save(manifest_path)
    image_dir = tmp_path / "runs" / "t2i_compbench" / "base"
    image_dir.mkdir(parents=True)
    for sample in samples:
        Image.new("RGB", (16, 16), color="red").save(image_dir / f"{sample.id}.png")

    official = tmp_path / "official"
    script = official / "BLIPvqa_eval" / "BLIP_vqa.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

out = Path(sys.argv[sys.argv.index("--out_dir") + 1])
time.sleep(0.1)
result = out / "annotation_blip" / "vqa_result.json"
result.parent.mkdir(parents=True, exist_ok=True)
result.write_text(json.dumps([{"answer": "1.0"}]), encoding="utf-8")
(out / "gpu.txt").write_text(os.environ["CUDA_VISIBLE_DEVICES"], encoding="utf-8")
""",
        encoding="utf-8",
    )

    output = run_t2i_compbench_official(
        manifest_path=manifest_path,
        run_root=tmp_path / "runs",
        methods=["base"],
        official_repo=official,
        output_dir=tmp_path / "eval",
        execute=True,
        categories=["color", "shape"],
        gpu_ids=["0", "1"],
        parallel_workers=2,
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    assert {command["gpu"] for command in data["commands"]} == {"0", "1"}
    assert data["scores"]["base"]["mean"] == 1.0


def test_append_replaces_refreshed_method_commands_without_duplication(tmp_path):
    scores_path = tmp_path / "scores.json"
    previous = {
        "benchmark": "t2i_compbench",
        "scores": {"base": {"mean": 0.1}, "spfc_target_only_uniform": {"mean": 0.2}},
        "commands": [
            {"method": "base", "category": "color"},
            {"method": "spfc_target_only_uniform", "category": "old"},
        ],
    }
    current = {
        "benchmark": "t2i_compbench",
        "scores": {"spfc_target_only_uniform": {"mean": 0.3}},
        "commands": [{"method": "spfc_target_only_uniform", "category": "new"}],
    }
    scores_path.write_text(json.dumps(current), encoding="utf-8")

    _merge_scores(scores_path, scores_path, previous)

    merged = json.loads(scores_path.read_text(encoding="utf-8"))
    assert merged["scores"] == {"base": {"mean": 0.1}, "spfc_target_only_uniform": {"mean": 0.3}}
    assert merged["commands"] == [
        {"method": "base", "category": "color"},
        {"method": "spfc_target_only_uniform", "category": "new"},
    ]
