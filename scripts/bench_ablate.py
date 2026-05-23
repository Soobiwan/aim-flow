"""Generate SPFC ablation and sweep suites."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.eval_bench.constants import DEFAULT_SEED
from aim_flow.eval_bench.generation import generate_spfc_ablation_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPFC ablation/sweep generation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--decompositions", required=True)
    parser.add_argument("--run-root", default="benchmarks/runs/spfc_eval_seed13")
    parser.add_argument("--suite", choices=["components", "steering", "primitive_count", "schedule"], required=True)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_spfc_ablation_suite(
        manifest_path=args.manifest,
        decomposition_path=args.decompositions,
        run_root=args.run_root,
        config_path=args.config,
        seed=args.seed,
        suite=args.suite,
    )
    print("ablation_indices:")
    for method, path in outputs.items():
        print(f"  {method}: {path}")


if __name__ == "__main__":
    main()
