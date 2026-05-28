"""Evaluate generated benchmark images after generation models are unloaded."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.eval_bench.evaluation import evaluate_coco_clipscore, run_t2i_compbench_official


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SPFC benchmark outputs.")
    parser.add_argument("--benchmark", choices=["t2i_compbench", "coco"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-root", default="benchmarks/runs/spfc_eval_seed13")
    parser.add_argument("--methods", nargs="+", default=["base", "rectified_cfgpp", "spfc"])
    parser.add_argument("--output-dir", default="benchmarks/reports/spfc_eval_seed13/eval")
    parser.add_argument("--t2i-repo-dir", default="external/T2I-CompBench")
    parser.add_argument("--execute-official", action="store_true", help="Run official T2I scripts instead of staging commands only.")
    parser.add_argument(
        "--t2i-categories",
        nargs="+",
        choices=["color", "shape", "texture", "spatial"],
        help="T2I-CompBench categories to evaluate. Defaults to all categories.",
    )
    parser.add_argument("--append", action="store_true", help="Merge this method's scores into an existing score JSON.")
    return parser.parse_args()


def _merge_scores(existing_path: Path, new_path: Path, previous: dict | None) -> None:
    with new_path.open("r", encoding="utf-8") as f:
        current = json.load(f)
    merged = previous or {key: value for key, value in current.items() if key not in {"scores", "commands"}}
    merged.setdefault("scores", {})
    merged["scores"].update(current.get("scores", {}))
    if "commands" in current:
        merged.setdefault("commands", [])
        merged["commands"].extend(current.get("commands", []))
    for key, value in current.items():
        if key not in {"scores", "commands"}:
            merged[key] = value
    with existing_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    canonical_name = "t2i_compbench_scores.json" if args.benchmark == "t2i_compbench" else "coco_scores.json"
    canonical_path = output_dir / canonical_name
    previous = None
    if args.append and canonical_path.exists():
        with canonical_path.open("r", encoding="utf-8") as f:
            previous = json.load(f)
    if args.benchmark == "t2i_compbench":
        output = run_t2i_compbench_official(
            manifest_path=args.manifest,
            run_root=args.run_root,
            methods=args.methods,
            official_repo=args.t2i_repo_dir,
            output_dir=output_dir,
            execute=args.execute_official,
            categories=args.t2i_categories,
        )
    else:
        output = evaluate_coco_clipscore(
            manifest_path=args.manifest,
            run_root=args.run_root,
            methods=args.methods,
            output_dir=output_dir,
        )
    if args.append:
        _merge_scores(canonical_path, Path(output), previous)
        output = canonical_path
    print(f"eval_scores: {output}")


if __name__ == "__main__":
    main()
