"""Run base, anchor, naive v1, and AIM-Flow v2 comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.compare import run_prompt_key
from aim_flow.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIM-Flow comparison modes.")
    parser.add_argument("--config", required=True, help="Path to run config YAML.")
    parser.add_argument("--prompts", required=True, help="Path to prompt decompositions YAML.")
    parser.add_argument("--prompt-key", required=True, help="Prompt key inside the prompts YAML.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated outputs.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["base", "anchor", "naive_v1", "aim_v2"],
        choices=["base", "anchor", "naive_v1", "aim_v2", "full"],
        help="Generation modes to run.",
    )
    parser.add_argument("--disable-ltp", action="store_true", help="Disable latent trajectory projection.")
    parser.add_argument("--ltp-mode", choices=["velocity", "latent", "off"], help="Override LTP mode.")
    parser.add_argument("--lambda-global", type=float, help="Override AIM-Flow global lambda.")
    parser.add_argument("--velocity-clip-ratio", type=float, help="Override VFA velocity clip ratio.")
    parser.add_argument("--conflict-threshold", type=float, help="Override VFA conflict threshold.")
    parser.add_argument("--num-inference-steps", type=int, help="Override number of inference steps.")
    parser.add_argument("--height", type=int, help="Override image height.")
    parser.add_argument("--width", type=int, help="Override image width.")
    return parser.parse_args()


def apply_overrides(config, args: argparse.Namespace) -> None:
    if args.disable_ltp:
        config.aim_flow.ltp.enabled = False
        config.aim_flow.ltp.mode = "off"
    if args.ltp_mode:
        config.aim_flow.ltp.mode = args.ltp_mode
        config.aim_flow.ltp.enabled = args.ltp_mode != "off"
    if args.lambda_global is not None:
        config.aim_flow.lambda_global = args.lambda_global
    if args.velocity_clip_ratio is not None:
        config.aim_flow.vfa.velocity_clip_ratio = args.velocity_clip_ratio
    if args.conflict_threshold is not None:
        config.aim_flow.vfa.conflict_threshold = args.conflict_threshold
    if args.num_inference_steps is not None:
        config.sampler.num_inference_steps = args.num_inference_steps
    if args.height is not None:
        config.sampler.height = args.height
    if args.width is not None:
        config.sampler.width = args.width


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    paths = run_prompt_key(
        config=config,
        prompts_path=args.prompts,
        prompt_key=args.prompt_key,
        output_dir=args.output_dir,
        modes=args.modes,
    )
    print("Generated outputs:")
    for name, path in paths.items():
        kind = "grid" if name == "comparison_grid" else name
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
