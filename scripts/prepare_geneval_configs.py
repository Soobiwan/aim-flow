"""Convert exported GenEval SPFC prompt blocks into benchmark JSON configs."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.eval_bench.schemas import DecompositionManifest, PromptManifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GenEval prompt and SPFC decomposition JSON files.")
    parser.add_argument("--input", required=True, type=Path, help="Flat primitive_flow_prompts export.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--category", default="color")
    parser.add_argument("--split", default="color_attr")
    return parser.parse_args()


def parse_scalar(key: str, value: str) -> Any:
    if key in {"name", "target_prompt", "source_prompt", "negative_prompt", "text", "role"}:
        return ast.literal_eval(value)
    if key == "weight":
        return float(value)
    if key == "enabled":
        return value.lower() == "true"
    raise ValueError(f"Unsupported field: {key}")


def parse_flat_export(path: Path) -> list[dict[str, Any]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0] != "primitive_flow_prompts:":
        raise ValueError("Expected primitive_flow_prompts: as the first non-empty line.")

    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    primitive: dict[str, Any] | None = None
    for line in lines[1:]:
        if re.fullmatch(r"[a-z0-9_]+:", line):
            if line == "primitive_prompts:":
                continue
            if current is not None:
                entries.append(current)
            current = {"key": line[:-1], "primitive_prompts": []}
            primitive = None
            continue
        if current is None:
            raise ValueError(f"Field appears before an entry key: {line}")
        if line.startswith("- name:"):
            primitive = {"name": parse_scalar("name", line.split(":", 1)[1].strip())}
            current["primitive_prompts"].append(primitive)
            continue
        if ":" not in line:
            raise ValueError(f"Expected key: value line, got: {line}")
        key, value = line.split(":", 1)
        parsed = parse_scalar(key, value.strip())
        target = primitive if primitive is not None and key in {"text", "role", "weight", "enabled"} else current
        target[key] = parsed
    if current is not None:
        entries.append(current)
    return entries


def build_configs(entries: list[dict[str, Any]], category: str, split: str, seed: int) -> tuple[dict, dict]:
    samples = []
    items = []
    for index, entry in enumerate(entries):
        sample_id = f"geneval_{category}_{index:06d}"
        metadata = {"category": category, "decomposition_key": entry["key"], "original_index": index}
        samples.append(
            {
                "id": sample_id,
                "category": category,
                "prompt": entry["target_prompt"],
                "source": "GenEval",
                "split": split,
                "metadata": {"decomposition_key": entry["key"], "original_index": index},
            }
        )
        items.append(
            {
                "id": sample_id,
                "target_prompt": entry["target_prompt"],
                "source_prompt": entry["source_prompt"],
                "negative_prompt": entry["negative_prompt"],
                "primitive_prompts": entry["primitive_prompts"],
                "metadata": {**metadata, "template": False},
            }
        )
    manifest = {
        "benchmark": "geneval",
        "manifest_version": "1.0",
        "samples": samples,
        "seed": seed,
        "subset_size": len(samples),
    }
    decompositions = {"items": items, "schema_version": "spfc_decomposition_v1"}
    return manifest, decompositions


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    entries = parse_flat_export(args.input)
    if len(entries) != args.subset_size:
        raise ValueError(f"Expected {args.subset_size} decompositions, found {len(entries)}.")
    manifest, decompositions = build_configs(entries, args.category, args.split, args.seed)
    PromptManifest.from_dict(manifest)
    DecompositionManifest.from_dict(decompositions)

    stem = f"geneval_{args.subset_size}_seed{args.seed}"
    manifest_path = args.output_dir / f"{stem}.json"
    decomposition_path = args.output_dir / f"{stem}_spfc.json"
    write_json(manifest_path, manifest)
    write_json(decomposition_path, decompositions)
    print(f"manifest: {manifest_path}")
    print(f"decompositions: {decomposition_path}")
    print(f"items: {len(entries)}")


if __name__ == "__main__":
    main()
