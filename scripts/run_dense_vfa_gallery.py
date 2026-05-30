"""Generate dense PrimitiveFlow VFA comparisons and a combined gallery."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.config import load_config
from aim_flow.eval_bench.schemas import DecompositionItem, DecompositionManifest
from aim_flow.primitive_flow import load_primitive_flow_set_from_yaml
from aim_flow.sampler import AIMFlowSampler
from aim_flow.sd3_backend import SD3Backend
from aim_flow.utils import ensure_dir, write_json
from aim_flow.visualize import make_comparison_gallery, make_image_grid, save_metadata_json


COLUMN_LABELS = [
    "Base SD3",
    "Default VFA: w * g_cons * g_tgt",
    "Your method: w=1, g_tgt only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dense VFA comparisons and one combined gallery.")
    parser.add_argument("--config", default="configs/sd3_medium_kaggle.yaml")
    parser.add_argument("--decompositions", default="configs/t2i_compbench_100_seed13_spfc.json")
    parser.add_argument("--prompts", default="configs/sample_prompts.yaml")
    parser.add_argument("--prompt-section", default="primitive_flow_prompts")
    parser.add_argument("--prompt-keys", nargs="+", help="Append named YAML prompt examples.")
    parser.add_argument("--output-dir", default="outputs/dense_vfa_gallery")
    parser.add_argument("--limit", type=int, default=10, help="Use the first N decompositions unless --sample-ids is set.")
    parser.add_argument("--sample-ids", nargs="+", help="Optional explicit decomposition IDs.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing images.")
    parser.add_argument("--gallery-only", action="store_true", help="Build grids from existing images without loading SD3.")
    return parser.parse_args()


def select_items(manifest: DecompositionManifest, sample_ids: list[str] | None, limit: int) -> list[DecompositionItem]:
    if sample_ids:
        items = manifest.item_by_id()
        missing = [sample_id for sample_id in sample_ids if sample_id not in items]
        if missing:
            raise KeyError(f"Unknown decomposition IDs: {missing}")
        return [items[sample_id] for sample_id in sample_ids]
    if limit <= 0:
        raise ValueError("--limit must be positive.")
    return manifest.items[:limit]


def load_yaml_items(prompts_path: str, prompt_section: str, prompt_keys: list[str] | None) -> list[DecompositionItem]:
    items = []
    for prompt_key in prompt_keys or []:
        flow_set = load_primitive_flow_set_from_yaml(prompts_path, prompt_key, section=prompt_section)
        items.append(
            DecompositionItem(
                id=prompt_key,
                target_prompt=flow_set.target_prompt,
                source_prompt=flow_set.source_prompt,
                negative_prompt=flow_set.negative_prompt,
                primitive_prompts=[primitive.to_dict() for primitive in flow_set.primitive_prompts],
                metadata={"source": prompts_path, "prompt_section": prompt_section},
            )
        )
    return items


def append_unique_items(items: list[DecompositionItem], additions: list[DecompositionItem]) -> list[DecompositionItem]:
    seen = {item.id for item in items}
    duplicates = [item.id for item in additions if item.id in seen]
    if duplicates:
        raise ValueError(f"Duplicate comparison IDs: {duplicates}")
    return items + additions


def output_paths(output_dir: Path, sample_id: str) -> dict[str, Path]:
    root = output_dir / sample_id
    return {
        "root": root,
        "base": root / "base_sd3.png",
        "base_metadata": root / "metadata_base_sd3.json",
        "default_vfa": root / "default_vfa.png",
        "default_vfa_metadata": root / "metadata_default_vfa.json",
        "target_only_uniform": root / "target_only_uniform.png",
        "target_only_uniform_metadata": root / "metadata_target_only_uniform.json",
        "row_grid": root / "comparison_grid.png",
    }


def should_generate(image_path: Path, metadata_path: Path, overwrite: bool) -> bool:
    return overwrite or not (image_path.exists() and metadata_path.exists())


def save_metadata(metadata: dict[str, Any], item: DecompositionItem, variant: str, path: Path) -> None:
    metadata.update({"comparison_sample_id": item.id, "comparison_variant": variant})
    save_metadata_json(metadata, path)


def generate_base(backend: SD3Backend, item: DecompositionItem, paths: dict[str, Path], overwrite: bool) -> None:
    if not should_generate(paths["base"], paths["base_metadata"], overwrite):
        return
    config = backend.config
    image = backend.generate_base(
        prompt=item.target_prompt,
        negative_prompt=item.negative_prompt,
        seed=config.sampler.seed,
        num_inference_steps=config.sampler.num_inference_steps,
        guidance_scale=config.sampler.guidance_scale,
        height=config.sampler.height,
        width=config.sampler.width,
    )
    backend.save_image(image, paths["base"])
    save_metadata(
        {
            "method": "base_sd3",
            "target_prompt": item.target_prompt,
            "negative_prompt": item.negative_prompt,
            "runtime_config": config.to_dict(),
        },
        item,
        "base_sd3",
        paths["base_metadata"],
    )


def generate_dense_variant(
    backend: SD3Backend,
    sampler: AIMFlowSampler,
    item: DecompositionItem,
    paths: dict[str, Path],
    variant: str,
    use_consensus_gating: bool,
    uniform_condition_weights: bool,
    overwrite: bool,
) -> None:
    image_path = paths[variant]
    metadata_path = paths[f"{variant}_metadata"]
    if not should_generate(image_path, metadata_path, overwrite):
        return
    config = backend.config.primitive_flow
    config.use_consensus_gating = use_consensus_gating
    config.use_target_consistency_gating = True
    config.uniform_condition_weights = uniform_condition_weights
    image, metadata = sampler.generate_sparse_primitive_flow(item.to_flow_set(), mode="primitive_flow_dense")
    backend.save_image(image, image_path)
    save_metadata(metadata, item, variant, metadata_path)


def main() -> None:
    args = parse_args()
    manifest = DecompositionManifest.load(args.decompositions)
    items = select_items(manifest, args.sample_ids, args.limit)
    items = append_unique_items(items, load_yaml_items(args.prompts, args.prompt_section, args.prompt_keys))
    output_dir = ensure_dir(args.output_dir)
    rows = [output_paths(output_dir, item.id) for item in items]
    required = [
        path
        for row in rows
        for path in (row["base"], row["default_vfa"], row["target_only_uniform"])
        if not path.exists()
    ]
    if args.gallery_only and required:
        raise FileNotFoundError(f"Gallery inputs are missing, including: {required[0]}")

    backend: SD3Backend | None = None
    if not args.gallery_only:
        config = load_config(args.config)
        backend = SD3Backend(config).load()
        sampler = AIMFlowSampler(backend, config)
        for index, (item, paths) in enumerate(zip(items, rows), start=1):
            ensure_dir(paths["root"])
            print(f"[{index}/{len(items)}] {item.id}: base", flush=True)
            generate_base(backend, item, paths, args.overwrite)
            print(f"[{index}/{len(items)}] {item.id}: default VFA", flush=True)
            generate_dense_variant(
                backend,
                sampler,
                item,
                paths,
                variant="default_vfa",
                use_consensus_gating=True,
                uniform_condition_weights=False,
                overwrite=args.overwrite,
            )
            print(f"[{index}/{len(items)}] {item.id}: target-only uniform", flush=True)
            generate_dense_variant(
                backend,
                sampler,
                item,
                paths,
                variant="target_only_uniform",
                use_consensus_gating=False,
                uniform_condition_weights=True,
                overwrite=args.overwrite,
            )

    image_rows: list[list[Path]] = []
    row_labels: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    for item, paths in zip(items, rows):
        images = [paths["base"], paths["default_vfa"], paths["target_only_uniform"]]
        make_image_grid(images, COLUMN_LABELS, paths["row_grid"])
        image_rows.append(images)
        row_labels.append(f"{item.id}\n{item.target_prompt}")
        summary_rows.append({"sample_id": item.id, "target_prompt": item.target_prompt, "images": [str(path) for path in images]})
    gallery_path = make_comparison_gallery(image_rows, COLUMN_LABELS, row_labels, output_dir / "comparison_gallery.png")
    write_json({"columns": COLUMN_LABELS, "rows": summary_rows}, output_dir / "comparison_gallery.json")
    print(f"gallery: {gallery_path}")


if __name__ == "__main__":
    main()
