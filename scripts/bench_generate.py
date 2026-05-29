"""Generate benchmark images for SPFC, Rectified-CFG++, CFG, and base SD3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.eval_bench.constants import DEFAULT_GUIDANCE_SCALE, DEFAULT_SEED
from aim_flow.eval_bench.generation import generate_methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPFC benchmark generation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-root", default="benchmarks/runs/spfc_eval_seed13")
    parser.add_argument("--methods", nargs="+", default=["spfc", "rectified_cfgpp", "base"])
    parser.add_argument("--decompositions", help="Required when methods includes spfc.")
    parser.add_argument("--config", help="Optional base RunConfig YAML.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--rectified-repo-dir", default="external/Rectified-CFGpp")
    parser.add_argument("--spfc-variant", help="Optional SPFC variant, e.g. no_ltp or steering_0.75.")
    parser.add_argument("--spfc-method-label", help="Output method directory name for an SPFC variant.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_methods(
        manifest_path=args.manifest,
        run_root=args.run_root,
        methods=args.methods,
        config_path=args.config,
        decomposition_path=args.decompositions,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        rectified_repo_dir=args.rectified_repo_dir,
        spfc_variant=args.spfc_variant,
        spfc_method_label=args.spfc_method_label,
    )
    print("generation_indices:")
    for method, path in outputs.items():
        print(f"  {method}: {path}")


if __name__ == "__main__":
    main()
