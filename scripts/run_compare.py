"""Run base, anchor, naive AIM-Flow, and full AIM-Flow comparisons."""

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
        default=["base", "anchor", "naive", "full"],
        choices=["base", "anchor", "naive", "full"],
        help="Generation modes to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = run_prompt_key(
        config=config,
        prompts_path=args.prompts,
        prompt_key=args.prompt_key,
        output_dir=args.output_dir,
        modes=args.modes,
    )
    print("Generated outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

