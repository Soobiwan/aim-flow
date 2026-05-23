"""Run AIM-Flow / LadderFlow / PrimitiveFlow comparison modes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.compare import run_prompt_key
from aim_flow.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIM-Flow / PrimitiveFlow comparison modes.")
    parser.add_argument("--config", required=True, help="Path to run config YAML.")
    parser.add_argument("--prompts", required=True, help="Path to prompt decompositions YAML.")
    parser.add_argument("--prompt-section", default="primitive_flow_prompts", help="Prompt section, e.g. primitive_flow_prompts.")
    parser.add_argument("--prompt-key", required=True, help="Prompt key inside the prompts YAML.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated outputs.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["base", "source_only", "primitive_flow_final_only", "primitive_flow_sparse"],
        choices=[
            "base",
            "anchor",
            "naive_v1",
            "aim_v2",
            "full",
            "ladder_c0",
            "ladder_progressive_noagg",
            "ladder_v3_sparse",
            "ladder_v3_dense",
            "source_only",
            "primitive_flow_final_only",
            "primitive_flow_sparse",
            "primitive_flow_dense",
            "primitive_flow_dense_optional",
        ],
        help="Generation modes to run.",
    )
    parser.add_argument("--aggregation-steps", nargs="*", type=int, help="Explicit aggregation step indices.")
    parser.add_argument("--aggregation-step-fractions", nargs="*", type=float, help="Aggregation fractions.")
    parser.add_argument("--aggregate-every-n-steps", type=int, help="Aggregate every N steps.")
    parser.add_argument("--final-only", action="store_true", help="Aggregate only on the final denoising step.")
    parser.add_argument("--disable-ltp", action="store_true", help="Disable latent trajectory projection.")
    parser.add_argument("--ltp-mode", choices=["velocity", "latent", "off"], help="Override LTP mode.")
    parser.add_argument("--reference-policy", choices=["progressive", "full", "base", "piecewise"], help="Ladder reference policy.")
    parser.add_argument("--active-policy", choices=["all", "upto_reference", "around_reference", "final_plus_reference"], help="Ladder active condition policy.")
    parser.add_argument("--non-aggregation-policy", choices=["reference", "full", "base"], help="Ladder non-aggregation policy.")
    parser.add_argument("--lambda-global", type=float, help="Override AIM-Flow global lambda.")
    parser.add_argument("--vfa-temperature", type=float, help="Override LadderFlow VFA softmax temperature.")
    parser.add_argument("--velocity-clip-ratio", type=float, help="Override VFA velocity clip ratio.")
    parser.add_argument("--steering-strength", type=float, help="Override PrimitiveFlow correction strength.")
    parser.add_argument("--ltp-radius-ratio", type=float, help="Override PrimitiveFlow LTP radius ratio.")
    parser.add_argument("--conflict-threshold", type=float, help="Override VFA conflict threshold.")
    parser.add_argument("--disable-consensus-gating", action="store_true", help="Disable LadderFlow consensus gating.")
    parser.add_argument("--disable-final-consistency-gating", action="store_true", help="Disable LadderFlow final consistency gating.")
    parser.add_argument("--disable-target-consistency-gating", action="store_true", help="Disable PrimitiveFlow target consistency gating.")
    parser.add_argument("--max-primitives", type=int, help="Override max primitive prompts for PrimitiveFlow/AIM-Flow.")
    parser.add_argument("--num-inference-steps", type=int, help="Override number of inference steps.")
    parser.add_argument("--height", type=int, help="Override image height.")
    parser.add_argument("--width", type=int, help="Override image width.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    return parser.parse_args()


def apply_overrides(config, args: argparse.Namespace) -> None:
    if args.disable_ltp:
        config.aim_flow.ltp.enabled = False
        config.aim_flow.ltp.mode = "off"
        config.ladder_flow.ltp_enabled = False
        config.ladder_flow.ltp_mode = "off"
        config.primitive_flow.ltp_enabled = False
        config.primitive_flow.ltp_mode = "off"
    if args.ltp_mode:
        config.aim_flow.ltp.mode = args.ltp_mode
        config.aim_flow.ltp.enabled = args.ltp_mode != "off"
        config.ladder_flow.ltp_mode = args.ltp_mode
        config.ladder_flow.ltp_enabled = args.ltp_mode != "off"
        config.primitive_flow.ltp_mode = args.ltp_mode
        config.primitive_flow.ltp_enabled = args.ltp_mode != "off"
    if args.final_only:
        config.primitive_flow.final_only = True
        config.primitive_flow.aggregation_steps = None
        config.primitive_flow.aggregation_step_fractions = None
        config.primitive_flow.aggregate_every_n_steps = None
    if args.aggregation_steps is not None:
        config.ladder_flow.aggregation_steps = args.aggregation_steps
        config.primitive_flow.aggregation_steps = args.aggregation_steps
        config.primitive_flow.final_only = False
    if args.aggregation_step_fractions is not None:
        config.ladder_flow.aggregation_step_fractions = args.aggregation_step_fractions
        config.primitive_flow.aggregation_step_fractions = args.aggregation_step_fractions
        if args.aggregation_steps is None:
            config.ladder_flow.aggregation_steps = None
            config.primitive_flow.aggregation_steps = None
        config.primitive_flow.final_only = False
    if args.aggregate_every_n_steps is not None:
        config.ladder_flow.aggregate_every_n_steps = args.aggregate_every_n_steps
        config.primitive_flow.aggregate_every_n_steps = args.aggregate_every_n_steps
        config.primitive_flow.final_only = False
    if args.reference_policy:
        config.ladder_flow.reference_policy = args.reference_policy
    if args.active_policy:
        config.ladder_flow.active_policy = args.active_policy
    if args.non_aggregation_policy:
        config.ladder_flow.non_aggregation_policy = args.non_aggregation_policy
    if args.lambda_global is not None:
        config.aim_flow.lambda_global = args.lambda_global
    if args.vfa_temperature is not None:
        config.ladder_flow.vfa_temperature = args.vfa_temperature
        config.primitive_flow.vfa_temperature = args.vfa_temperature
    if args.velocity_clip_ratio is not None:
        config.aim_flow.vfa.velocity_clip_ratio = args.velocity_clip_ratio
        config.ladder_flow.velocity_clip_ratio = args.velocity_clip_ratio
        config.primitive_flow.velocity_clip_ratio = args.velocity_clip_ratio
    if args.steering_strength is not None:
        config.primitive_flow.steering_strength = args.steering_strength
    if args.ltp_radius_ratio is not None:
        config.primitive_flow.ltp_radius_ratio = args.ltp_radius_ratio
    if args.conflict_threshold is not None:
        config.aim_flow.vfa.conflict_threshold = args.conflict_threshold
    if args.disable_consensus_gating:
        config.ladder_flow.use_consensus_gating = False
        config.primitive_flow.use_consensus_gating = False
    if args.disable_final_consistency_gating:
        config.ladder_flow.use_final_consistency_gating = False
    if args.disable_target_consistency_gating:
        config.primitive_flow.use_target_consistency_gating = False
    if args.max_primitives is not None:
        config.aim_flow.max_primitives = args.max_primitives
        config.primitive_flow.max_primitives = args.max_primitives
    if args.num_inference_steps is not None:
        config.sampler.num_inference_steps = args.num_inference_steps
    if args.height is not None:
        config.sampler.height = args.height
    if args.width is not None:
        config.sampler.width = args.width
    if args.seed is not None:
        config.sampler.seed = args.seed


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
        prompt_section=args.prompt_section,
    )
    print("Generated outputs:")
    for name, path in paths.items():
        kind = "grid" if name == "comparison_grid" else name
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
