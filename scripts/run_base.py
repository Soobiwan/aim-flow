"""Generate only the normal SD3 full-prompt baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.config import load_config
from aim_flow.decomposition import load_prompt_decomposition_from_yaml
from aim_flow.sd3_backend import SD3Backend
from aim_flow.utils import ensure_dir
from aim_flow.visualize import save_metadata_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run base SD3 full-prompt generation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--prompt-key", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    decomposition = load_prompt_decomposition_from_yaml(args.prompts, args.prompt_key)
    output_dir = ensure_dir(args.output_dir)
    backend = SD3Backend(config).load()
    image = backend.generate_base(
        prompt=decomposition.full_prompt,
        negative_prompt=decomposition.negative_prompt,
        seed=config.sampler.seed,
        num_inference_steps=config.sampler.num_inference_steps,
        guidance_scale=config.sampler.guidance_scale,
        height=config.sampler.height,
        width=config.sampler.width,
    )
    image_path = output_dir / "base.png"
    metadata_path = output_dir / "base_metadata.json"
    backend.save_image(image, image_path)
    save_metadata_json(
        {
            "prompt_decomposition": decomposition.to_dict(),
            "mode": "base",
            "model_id": config.model.model_id,
            "runtime_config": config.to_dict(),
            "seed": config.sampler.seed,
            "num_inference_steps": config.sampler.num_inference_steps,
            "height": config.sampler.height,
            "width": config.sampler.width,
            "guidance_scale": config.sampler.guidance_scale,
        },
        metadata_path,
    )
    print(f"base_image: {image_path}")
    print(f"base_metadata: {metadata_path}")


if __name__ == "__main__":
    main()
