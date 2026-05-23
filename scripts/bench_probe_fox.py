"""Run the fox/rainboots SPFC timestep probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.config import load_config
from aim_flow.decomposition import load_primitive_flow_set_from_yaml
from aim_flow.eval_bench.generation import load_bench_config
from aim_flow.eval_bench.probes import run_fox_timestep_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SPFC fox timestep probe.")
    parser.add_argument("--config", default="configs/sd3_medium_kaggle.yaml")
    parser.add_argument("--prompts", default="configs/sample_prompts.yaml")
    parser.add_argument("--prompt-section", default="primitive_flow_prompts")
    parser.add_argument("--prompt-key", default="reflection_shadow_fox")
    parser.add_argument("--output-dir", default="benchmarks/probes/fox_rainboots_seed13")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    config = load_bench_config(config_path=args.config, seed=13)
    config.primitive_flow.max_primitives = base.primitive_flow.max_primitives
    flow_set = load_primitive_flow_set_from_yaml(args.prompts, args.prompt_key, section=args.prompt_section)
    outputs = run_fox_timestep_probe(flow_set, args.output_dir, config=config)
    print("fox_probe_outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
