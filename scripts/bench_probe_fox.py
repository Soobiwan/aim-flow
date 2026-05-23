"""Run the fox/rainboots SPFC timestep probe."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.config import load_config
from aim_flow.decomposition import load_primitive_flow_set_from_yaml
from aim_flow.eval_bench.generation import load_bench_config
from aim_flow.eval_bench.probes import run_fox_timestep_probe
from aim_flow.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SPFC fox timestep probe.")
    parser.add_argument("--config", default="configs/sd3_medium_kaggle.yaml")
    parser.add_argument("--prompts", default="configs/sample_prompts.yaml")
    parser.add_argument("--prompt-section", default="primitive_flow_prompts")
    parser.add_argument("--prompt-key", default="marble_fox_rainboots_mirror")
    parser.add_argument("--output-dir", default="benchmarks/probes/fox_rainboots_seed13")
    parser.add_argument("--probe-mode", choices=["cutoff", "rollout", "both"], default="both")
    parser.add_argument("--max-probe-steps", type=int, default=None)
    parser.add_argument("--max-primitives", type=int, default=None)
    parser.add_argument("--ltp-mode", choices=["latent", "velocity", "off"], default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    base = load_config(args.config)
    config = load_bench_config(config_path=args.config, seed=13)
    config.primitive_flow.max_primitives = base.primitive_flow.max_primitives
    if args.max_primitives is not None:
        config.primitive_flow.max_primitives = args.max_primitives
    if args.ltp_mode is not None:
        config.primitive_flow.ltp_mode = args.ltp_mode
        config.primitive_flow.ltp_enabled = args.ltp_mode != "off"
    flow_set = load_primitive_flow_set_from_yaml(args.prompts, args.prompt_key, section=args.prompt_section)
    outputs = run_fox_timestep_probe(
        flow_set,
        args.output_dir,
        config=config,
        probe_mode=args.probe_mode,
        max_probe_steps=args.max_probe_steps,
    )
    print("fox_probe_outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception:
        output_dir = ensure_dir(args.output_dir)
        traceback_text = traceback.format_exc()
        error_path = output_dir / "probe_error.txt"
        error_path.write_text(traceback_text, encoding="utf-8")
        print(traceback_text, file=sys.stderr)
        print(f"fox_probe_error: {error_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
