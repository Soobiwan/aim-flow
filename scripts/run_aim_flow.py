"""Generate one AIM-Flow image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.config import load_config
from aim_flow.decomposition import load_prompt_decomposition_from_yaml
from aim_flow.sampler import AIMFlowSampler
from aim_flow.sd3_backend import SD3Backend
from aim_flow.utils import ensure_dir
from aim_flow.visualize import save_metadata_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIM-Flow generation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--prompt-key", required=True)
    parser.add_argument("--mode", default="full", choices=["anchor", "naive", "full"])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    decomposition = load_prompt_decomposition_from_yaml(args.prompts, args.prompt_key)
    output_dir = ensure_dir(args.output_dir)
    backend = SD3Backend(config).load()
    sampler = AIMFlowSampler(backend, config)
    image, metadata = sampler.generate(decomposition, mode=args.mode)
    image_path = output_dir / f"{args.mode}.png"
    metadata_path = output_dir / f"{args.mode}_metadata.json"
    backend.save_image(image, image_path)
    save_metadata_json(metadata, metadata_path)
    print(f"{args.mode}_image: {image_path}")
    print(f"{args.mode}_metadata: {metadata_path}")


if __name__ == "__main__":
    main()

