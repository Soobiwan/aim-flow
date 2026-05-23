"""Report and qualitative grid helpers for the evaluation bench."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from aim_flow.eval_bench.evaluation import summarize_runtime
from aim_flow.eval_bench.schemas import PromptManifest
from aim_flow.utils import ensure_dir, write_json


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def write_score_tables(
    t2i_scores_path: str | Path | None,
    coco_scores_path: str | Path | None,
    output_dir: str | Path,
    run_root: str | Path | None = None,
    coco_manifest_path: str | Path | None = None,
) -> dict[str, str]:
    """Write compact CSV/Markdown tables from evaluator JSON outputs."""

    out = ensure_dir(output_dir)
    written: dict[str, str] = {}
    if t2i_scores_path:
        data = _read_json(t2i_scores_path)
        rows = []
        for method, scores in data.get("scores", {}).items():
            row = {
                "method": method,
                "color": scores.get("color"),
                "shape": scores.get("shape"),
                "texture": scores.get("texture"),
                "spatial": scores.get("spatial"),
                "mean": scores.get("mean"),
            }
            rows.append(row)
        written["t2i_csv"] = str(_write_csv(out / "t2i_compbench_table.csv", rows))
        written["t2i_md"] = str(_write_markdown(out / "t2i_compbench_table.md", rows))
    if coco_scores_path:
        data = _read_json(coco_scores_path)
        runtimes = summarize_runtime(run_root, coco_manifest_path, list(data.get("scores", {}))) if run_root and coco_manifest_path else {}
        rows = []
        for method, scores in data.get("scores", {}).items():
            rows.append(
                {
                    "method": method,
                    "CLIPScore": scores.get("CLIPScore"),
                    "BLIP-VQA caption-match": scores.get("BLIP-VQA caption-match", ""),
                    "mean_runtime_sec": runtimes.get(method, ""),
                    "failures": scores.get("failures", 0),
                }
            )
        written["coco_csv"] = str(_write_csv(out / "coco_table.csv", rows))
        written["coco_md"] = str(_write_markdown(out / "coco_table.md", rows))
    write_json(written, out / "report_index.json")
    return written


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_qualitative_grid(
    manifest_path: str | Path,
    run_root: str | Path,
    output_path: str | Path,
    methods: list[str] | None = None,
    sample_ids: list[str] | None = None,
    max_prompts: int = 8,
) -> Path:
    """Create a prompt/base/rectified/SPFC qualitative comparison grid."""

    manifest = PromptManifest.load(manifest_path)
    selected_methods = methods or ["base", "rectified_cfgpp", "spfc"]
    selected = [sample for sample in manifest.samples if sample_ids is None or sample.id in sample_ids][:max_prompts]
    if not selected:
        raise ValueError("No samples selected for qualitative grid.")

    root = Path(run_root)
    cell_w = 220
    image_w = 180
    image_h = 180
    row_h = 230
    header_h = 46
    grid_w = cell_w * (len(selected_methods) + 1)
    grid_h = header_h + row_h * len(selected)
    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = _font(18)
    text_font = _font(14)
    labels = ["prompt"] + selected_methods
    for col, label in enumerate(labels):
        draw.text((col * cell_w + 10, 14), label, fill=(15, 15, 15), font=header_font)
    for row, sample in enumerate(selected):
        y = header_h + row * row_h
        draw.line((0, y, grid_w, y), fill=(220, 220, 220))
        prompt_text = _wrap(sample.prompt, 28, 9)
        draw.multiline_text((10, y + 12), prompt_text, fill=(20, 20, 20), font=text_font, spacing=3)
        for col, method in enumerate(selected_methods, start=1):
            image_path = root / manifest.benchmark / method / f"{sample.id}.png"
            if image_path.exists():
                image = Image.open(image_path).convert("RGB").resize((image_w, image_h), Image.Resampling.LANCZOS)
                canvas.paste(image, (col * cell_w + 20, y + 28))
            else:
                draw.rectangle((col * cell_w + 20, y + 28, col * cell_w + 20 + image_w, y + 28 + image_h), outline=(200, 40, 40))
                draw.text((col * cell_w + 34, y + 104), "missing", fill=(200, 40, 40), font=text_font)
    output = Path(output_path)
    ensure_dir(output.parent)
    canvas.save(output)
    return output


def _wrap(text: str, width: int, max_lines: int) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(" ".join(current))
    return "\n".join(lines)
