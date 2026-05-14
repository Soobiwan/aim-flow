"""High-level generation pipeline for AIM-Flow comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aim_flow.config import RunConfig
from aim_flow.prompt_schema import PromptDecomposition
from aim_flow.sampler import AIMFlowSampler
from aim_flow.sd3_backend import SD3Backend
from aim_flow.utils import ensure_dir
from aim_flow.visualize import make_image_grid, save_metadata_json


def run_aim_flow_comparison(
    prompt_decomposition: PromptDecomposition,
    config: RunConfig,
    output_dir: str | Path,
    modes: list[str] | None = None,
) -> dict[str, Path]:
    """Generate selected baseline/AIM-Flow modes and save outputs."""

    selected_modes = modes or ["base", "anchor", "naive", "full"]
    output = ensure_dir(output_dir)
    backend = SD3Backend(config).load()
    sampler = AIMFlowSampler(backend, config)
    paths: dict[str, Path] = {}
    grid_image_paths: list[Path] = []
    grid_labels: list[str] = []

    for mode in selected_modes:
        if mode == "base":
            image = backend.generate_base(
                prompt=prompt_decomposition.full_prompt,
                negative_prompt=prompt_decomposition.negative_prompt,
                seed=config.sampler.seed,
                num_inference_steps=config.sampler.num_inference_steps,
                guidance_scale=config.sampler.guidance_scale,
                height=config.sampler.height,
                width=config.sampler.width,
            )
            metadata: dict[str, Any] = {
                "prompt_decomposition": prompt_decomposition.to_dict(),
                "mode": "base",
                "model_id": config.model.model_id,
                "runtime_config": config.to_dict(),
                "seed": config.sampler.seed,
                "num_inference_steps": config.sampler.num_inference_steps,
                "height": config.sampler.height,
                "width": config.sampler.width,
                "guidance_scale": config.sampler.guidance_scale,
            }
        elif mode in {"anchor", "naive", "full"}:
            image, metadata = sampler.generate(prompt_decomposition, mode=mode)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        image_path = output / f"{mode}.png"
        metadata_path = output / f"{mode}_metadata.json"
        backend.save_image(image, image_path)
        save_metadata_json(metadata, metadata_path)
        paths[f"{mode}_image"] = image_path
        paths[f"{mode}_metadata"] = metadata_path
        grid_image_paths.append(image_path)
        grid_labels.append(mode)

    if grid_image_paths:
        grid_path = output / "comparison_grid.png"
        make_image_grid(grid_image_paths, grid_labels, grid_path)
        paths["comparison_grid"] = grid_path
    return paths
