"""Create benchmark tables and qualitative grids."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.eval_bench.reporting import make_qualitative_grid, write_score_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SPFC benchmark reports.")
    parser.add_argument("--t2i-scores")
    parser.add_argument("--coco-scores")
    parser.add_argument("--output-dir", default="benchmarks/reports/spfc_eval_seed13")
    parser.add_argument("--run-root", default="benchmarks/runs/spfc_eval_seed13")
    parser.add_argument("--coco-manifest")
    parser.add_argument("--qualitative-manifest")
    parser.add_argument("--qualitative-output", default="benchmarks/reports/spfc_eval_seed13/qualitative_grid.png")
    parser.add_argument("--qualitative-methods", nargs="+", default=["base", "rectified_cfgpp", "spfc"])
    parser.add_argument("--qualitative-sample-ids", nargs="*")
    parser.add_argument("--qualitative-max-prompts", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_score_tables(
        t2i_scores_path=args.t2i_scores,
        coco_scores_path=args.coco_scores,
        output_dir=args.output_dir,
        run_root=args.run_root,
        coco_manifest_path=args.coco_manifest,
    )
    if args.qualitative_manifest:
        grid = make_qualitative_grid(
            manifest_path=args.qualitative_manifest,
            run_root=args.run_root,
            output_path=args.qualitative_output,
            methods=args.qualitative_methods,
            sample_ids=args.qualitative_sample_ids,
            max_prompts=args.qualitative_max_prompts,
        )
        outputs["qualitative_grid"] = str(grid)
    print("report_outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
