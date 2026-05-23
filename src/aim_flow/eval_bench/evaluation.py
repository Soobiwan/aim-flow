"""Evaluator adapters and score aggregation for benchmark outputs."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from aim_flow.eval_bench.schemas import PromptManifest
from aim_flow.utils import ensure_dir, safe_slugify, write_json


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def average_vqa_result(path: str | Path) -> float:
    """Average a T2I-CompBench vqa_result.json file."""

    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}.")
    scores = []
    for item in data:
        try:
            scores.append(float(item["answer"]))
        except Exception:
            continue
    if not scores:
        return float("nan")
    return float(sum(scores) / len(scores))


def _generated_image_path(run_root: Path, benchmark: str, method: str, sample_id: str) -> Path:
    return run_root / benchmark / method / f"{sample_id}.png"


def stage_t2i_compbench_inputs(
    manifest: PromptManifest,
    run_root: str | Path,
    method: str,
    category: str,
    output_dir: str | Path,
) -> Path:
    """Copy generated images into T2I-CompBench's filename convention."""

    run_dir = Path(run_root)
    stage = ensure_dir(Path(output_dir) / method / category / "examples")
    samples_dir = ensure_dir(stage / "samples")
    prompts = [sample for sample in manifest.samples if sample.category == category]
    for question_id, sample in enumerate(prompts):
        source = _generated_image_path(run_dir, manifest.benchmark, method, sample.id)
        if not source.exists():
            raise FileNotFoundError(source)
        prompt_name = safe_slugify(sample.prompt, max_length=150).replace("-", " ")
        target = samples_dir / f"{prompt_name}_{question_id:06d}.png"
        shutil.copy2(source, target)
    write_json(
        {
            "method": method,
            "category": category,
            "count": len(prompts),
            "samples": [sample.to_dict() for sample in prompts],
        },
        stage / "bench_stage_index.json",
    )
    return stage


def run_t2i_compbench_official(
    manifest_path: str | Path,
    run_root: str | Path,
    methods: list[str],
    official_repo: str | Path,
    output_dir: str | Path,
    execute: bool = True,
) -> Path:
    """Run or stage official T2I-CompBench metrics for generated images."""

    manifest = PromptManifest.load(manifest_path)
    official = Path(official_repo)
    if not official.exists():
        raise FileNotFoundError(f"T2I-CompBench repo not found: {official}")
    out = ensure_dir(output_dir)
    scores: dict[str, dict[str, float | None]] = {}
    commands: list[dict[str, Any]] = []
    category_eval = {
        "color": ["python", str(official / "BLIPvqa_eval" / "BLIP_vqa.py")],
        "shape": ["python", str(official / "BLIPvqa_eval" / "BLIP_vqa.py")],
        "texture": ["python", str(official / "BLIPvqa_eval" / "BLIP_vqa.py")],
        "spatial": ["python", str(official / "UniDet_eval" / "2D_spatial_eval.py")],
    }
    for method in methods:
        method_scores: dict[str, float | None] = {}
        for category, base_cmd in category_eval.items():
            stage = stage_t2i_compbench_inputs(manifest, run_root, method, category, out / "staged")
            if category == "spatial":
                cmd = base_cmd + ["--outpath", str(stage)]
                result_path = stage / "labels" / "annotation_obj_detection_2d" / "vqa_result.json"
                cwd = official / "UniDet_eval"
            else:
                cmd = base_cmd + ["--out_dir", str(stage)]
                result_path = stage / "annotation_blip" / "vqa_result.json"
                cwd = official
            commands.append({"method": method, "category": category, "cmd": cmd, "cwd": str(cwd)})
            if execute:
                subprocess.run(cmd, cwd=str(cwd), check=True)
                method_scores[category] = average_vqa_result(result_path)
            else:
                method_scores[category] = None
        numeric = [value for value in method_scores.values() if isinstance(value, float) and not math.isnan(value)]
        method_scores["mean"] = float(sum(numeric) / len(numeric)) if numeric else None
        scores[method] = method_scores
    result = {"benchmark": "t2i_compbench", "official_repo": str(official), "scores": scores, "commands": commands}
    output_path = out / "t2i_compbench_scores.json"
    write_json(result, output_path)
    return output_path


def evaluate_coco_clipscore(
    manifest_path: str | Path,
    run_root: str | Path,
    methods: list[str],
    output_dir: str | Path,
    model_id: str = "openai/clip-vit-base-patch32",
) -> Path:
    """Compute COCO CLIPScore-style image/text similarity for generated images."""

    manifest = PromptManifest.load(manifest_path)
    from transformers import CLIPModel, CLIPProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    run_dir = Path(run_root)
    scores: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for method in methods:
            sample_scores: list[float] = []
            for sample in manifest.samples:
                image_path = _generated_image_path(run_dir, manifest.benchmark, method, sample.id)
                image = Image.open(image_path).convert("RGB")
                inputs = processor(text=[sample.prompt], images=[image], return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                sample_scores.append(float((image_embeds * text_embeds).sum(dim=-1).item()))
            scores[method] = {
                "CLIPScore": float(sum(sample_scores) / len(sample_scores)),
                "failures": 0.0,
            }
    output = Path(output_dir) / "coco_scores.json"
    write_json({"benchmark": "coco", "scores": scores, "clip_model_id": model_id}, output)
    return output


def summarize_runtime(run_root: str | Path, manifest_path: str | Path, methods: list[str]) -> dict[str, float]:
    manifest = PromptManifest.load(manifest_path)
    root = Path(run_root)
    runtimes: dict[str, float] = {}
    for method in methods:
        values = []
        for sample in manifest.samples:
            metadata_path = root / manifest.benchmark / method / f"{sample.id}.json"
            if metadata_path.exists():
                metadata = _read_json(metadata_path)
                if "runtime_sec" in metadata:
                    values.append(float(metadata["runtime_sec"]))
        runtimes[method] = float(sum(values) / len(values)) if values else float("nan")
    return runtimes
