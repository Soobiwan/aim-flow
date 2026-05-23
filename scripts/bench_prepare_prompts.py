"""Prepare deterministic benchmark prompt manifests and SPFC decomposition templates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.eval_bench.constants import DEFAULT_SEED
from aim_flow.eval_bench.prompt_sources import build_coco_manifest, build_custom_prompt_manifest, build_t2i_compbench_manifest
from aim_flow.eval_bench.schemas import make_decomposition_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SPFC evaluation prompt manifests.")
    parser.add_argument("--benchmark", choices=["t2i_compbench", "coco", "custom", "both"], default="both")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--t2i-subset-size", default="100", help="T2I subset size integer or 'full'.")
    parser.add_argument("--coco-subset-size", type=int, default=100)
    parser.add_argument("--t2i-dataset-root", help="Path to T2I-CompBench examples/dataset.")
    parser.add_argument("--coco-captions-json", help="Path to COCO captions_val*.json.")
    parser.add_argument("--coco-prompt-file", help="Plain text COCO prompt file, one prompt per line.")
    parser.add_argument("--custom-prompt-file", help="Plain text custom difficult prompt file, one prompt per line.")
    parser.add_argument("--custom-subset-size", default="full", help="Custom prompt subset size integer or 'full'.")
    parser.add_argument("--output-dir", default="benchmarks/manifests")
    parser.add_argument("--decomposition-dir", default="benchmarks/decompositions")
    parser.add_argument("--write-decomposition-template", action="store_true")
    return parser.parse_args()


def _subset_label(value: str | int) -> str:
    return "full" if str(value).lower() == "full" else str(int(value))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    decomposition_dir = Path(args.decomposition_dir)
    manifests = []
    if args.benchmark in {"t2i_compbench", "both"}:
        t2i_subset = "full" if str(args.t2i_subset_size).lower() == "full" else int(args.t2i_subset_size)
        manifest = build_t2i_compbench_manifest(t2i_subset, args.seed, args.t2i_dataset_root)
        path = output_dir / f"t2i_compbench_{_subset_label(t2i_subset)}_seed{args.seed}.json"
        manifest.save(path)
        print(f"t2i_manifest: {path}")
        manifests.append((manifest, "t2i_compbench", _subset_label(t2i_subset)))
    if args.benchmark in {"coco", "both"}:
        manifest = build_coco_manifest(args.coco_subset_size, args.seed, args.coco_captions_json, args.coco_prompt_file)
        path = output_dir / f"coco_{args.coco_subset_size}_seed{args.seed}.json"
        manifest.save(path)
        print(f"coco_manifest: {path}")
        manifests.append((manifest, "coco", str(args.coco_subset_size)))
    if args.benchmark == "custom":
        if not args.custom_prompt_file:
            raise SystemExit("--custom-prompt-file is required when --benchmark custom.")
        custom_subset = "full" if str(args.custom_subset_size).lower() == "full" else int(args.custom_subset_size)
        manifest = build_custom_prompt_manifest(args.custom_prompt_file, custom_subset, args.seed)
        path = output_dir / f"custom_difficult_{_subset_label(custom_subset)}_seed{args.seed}.json"
        manifest.save(path)
        print(f"custom_manifest: {path}")
        manifests.append((manifest, "custom_difficult", _subset_label(custom_subset)))
    if args.write_decomposition_template:
        for manifest, name, subset in manifests:
            template = make_decomposition_template(manifest)
            path = decomposition_dir / f"{name}_{subset}_seed{args.seed}_spfc.json"
            template.save(path)
            print(f"decomposition_template: {path}")


if __name__ == "__main__":
    main()
