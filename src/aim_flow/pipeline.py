"""High-level generation pipeline for AIM-Flow comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aim_flow.config import RunConfig
from aim_flow.prompt_schema import ConditionLadder, MarginalFlowPromptSet, PrimitiveFlowSet, PromptDecomposition
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

    selected_modes = modes or ["base", "anchor", "naive_v1", "aim_v2"]
    output = ensure_dir(output_dir)
    backend = SD3Backend(config).load()
    sampler = AIMFlowSampler(backend, config)
    paths: dict[str, Path] = {}
    grid_image_paths: list[Path] = []
    grid_labels: list[str] = []

    names = {
        "base": ("base_full_prompt.png", "metadata_base.json", "Base SD3: full prompt"),
        "anchor": ("anchor_only.png", "metadata_anchor.json", "Anchor only"),
        "naive_v1": ("naive_v1_standalone_primitives.png", "metadata_naive_v1.json", "Naive v1: standalone primitives"),
        "aim_v2": ("aim_v2_anchor_augmented_vfa_ltp.png", "metadata_aim_v2.json", "AIM-Flow v2: anchor + primitive VFA/LTP"),
        "full": ("aim_v2_anchor_augmented_vfa_ltp.png", "metadata_aim_v2.json", "AIM-Flow v2: anchor + primitive VFA/LTP"),
    }

    for mode in selected_modes:
        normalized_mode = "aim_v2" if mode == "full" else mode
        if normalized_mode not in {"base", "anchor", "naive_v1", "aim_v2"}:
            raise ValueError(f"Unknown mode: {mode}")
        image, metadata = sampler.generate(prompt_decomposition, mode=normalized_mode)

        image_name, metadata_name, label = names[normalized_mode]
        image_path = output / image_name
        metadata_path = output / metadata_name
        backend.save_image(image, image_path)
        save_metadata_json(metadata, metadata_path)
        paths[f"{normalized_mode}_image"] = image_path
        paths[f"{normalized_mode}_metadata"] = metadata_path
        grid_image_paths.append(image_path)
        grid_labels.append(label)

    if grid_image_paths:
        grid_path = output / "comparison_grid.png"
        make_image_grid(grid_image_paths, grid_labels, grid_path)
        paths["comparison_grid"] = grid_path
    return paths


def run_ladder_flow_comparison(
    condition_ladder: ConditionLadder,
    config: RunConfig,
    output_dir: str | Path,
    modes: list[str] | None = None,
) -> dict[str, Path]:
    """Generate LadderFlow v3 baselines/comparisons and save outputs."""

    selected_modes = modes or ["base", "ladder_c0", "ladder_progressive_noagg", "ladder_v3_sparse"]
    output = ensure_dir(output_dir)
    backend = SD3Backend(config).load()
    sampler = AIMFlowSampler(backend, config)
    paths: dict[str, Path] = {}
    grid_image_paths: list[Path] = []
    grid_labels: list[str] = []

    names = {
        "base": ("base_full_prompt.png", "metadata_base.json", "Base SD3: full prompt"),
        "ladder_c0": ("ladder_c0.png", "metadata_ladder_c0.json", "C0 only"),
        "ladder_progressive_noagg": (
            "ladder_progressive_noagg.png",
            "metadata_ladder_progressive_noagg.json",
            "Progressive ladder, no aggregation",
        ),
        "ladder_v3_sparse": (
            "ladder_v3_sparse.png",
            "metadata_ladder_v3_sparse.json",
            "LadderFlow v3 sparse VFA/LTP",
        ),
        "ladder_v3_dense": (
            "ladder_v3_dense.png",
            "metadata_ladder_v3_dense.json",
            "LadderFlow v3 dense VFA/LTP",
        ),
    }

    for mode in selected_modes:
        if mode not in names:
            raise ValueError(f"Unknown LadderFlow comparison mode: {mode}")
        if mode == "base":
            image = backend.generate_base(
                prompt=condition_ladder.full_prompt,
                negative_prompt=condition_ladder.negative_prompt,
                seed=config.sampler.seed,
                num_inference_steps=config.sampler.num_inference_steps,
                guidance_scale=config.sampler.guidance_scale,
                height=config.sampler.height,
                width=config.sampler.width,
            )
            metadata: dict[str, Any] = {
                "method": "base",
                "full_prompt": condition_ladder.full_prompt,
                "negative_prompt": condition_ladder.negative_prompt,
                "condition_ladder": condition_ladder.to_dict(),
                "runtime_config": config.to_dict(),
            }
        elif mode == "ladder_c0":
            image = backend.generate_base(
                prompt=condition_ladder.get_base_condition().text,
                negative_prompt=condition_ladder.negative_prompt,
                seed=config.sampler.seed,
                num_inference_steps=config.sampler.num_inference_steps,
                guidance_scale=config.sampler.guidance_scale,
                height=config.sampler.height,
                width=config.sampler.width,
            )
            metadata = {
                "method": "ladder_c0",
                "condition_text": condition_ladder.get_base_condition().text,
                "condition_ladder": condition_ladder.to_dict(),
                "runtime_config": config.to_dict(),
            }
        else:
            image, metadata = sampler.generate_ladder_v3(condition_ladder, mode=mode)

        image_name, metadata_name, label = names[mode]
        image_path = output / image_name
        metadata_path = output / metadata_name
        backend.save_image(image, image_path)
        save_metadata_json(metadata, metadata_path)
        paths[f"{mode}_image"] = image_path
        paths[f"{mode}_metadata"] = metadata_path
        grid_image_paths.append(image_path)
        grid_labels.append(label)

    if grid_image_paths:
        grid_path = output / "comparison_grid.png"
        make_image_grid(grid_image_paths, grid_labels, grid_path)
        paths["comparison_grid"] = grid_path
    return paths


def run_primitive_flow_comparison(
    flow_set: PrimitiveFlowSet,
    config: RunConfig,
    output_dir: str | Path,
    modes: list[str] | None = None,
) -> dict[str, Path]:
    """Generate Sparse Primitive Flow baselines/comparisons and save outputs."""

    selected_modes = modes or ["base", "source_only", "primitive_flow_final_only", "primitive_flow_sparse"]
    output = ensure_dir(output_dir)
    backend = SD3Backend(config).load()
    sampler = AIMFlowSampler(backend, config)
    paths: dict[str, Path] = {}
    grid_image_paths: list[Path] = []
    grid_labels: list[str] = []

    names = {
        "base": ("base_full_target.png", "metadata_base.json", "Base SD3: full target"),
        "source_only": ("source_only.png", "metadata_source_only.json", "Source prompt only"),
        "primitive_flow_final_only": (
            "primitive_flow_final_only.png",
            "metadata_primitive_flow_final_only.json",
            "PrimitiveFlow: final step only",
        ),
        "primitive_flow_sparse": (
            "primitive_flow_sparse.png",
            "metadata_primitive_flow_sparse.json",
            "PrimitiveFlow: sparse aggregation",
        ),
        "primitive_flow_dense": (
            "primitive_flow_dense.png",
            "metadata_primitive_flow_dense.json",
            "PrimitiveFlow: dense",
        ),
        "primitive_flow_dense_optional": (
            "primitive_flow_dense.png",
            "metadata_primitive_flow_dense.json",
            "PrimitiveFlow: dense",
        ),
    }

    execution_modes = list(selected_modes)
    if config.model.defer_model_cpu_offload and "primitive_flow_sparse" in execution_modes:
        execution_modes = ["primitive_flow_sparse"] + [
            mode for mode in execution_modes if mode != "primitive_flow_sparse"
        ]

    for mode in execution_modes:
        if mode not in names:
            raise ValueError(f"Unknown primitive-flow comparison mode: {mode}")
        if mode == "base":
            image = backend.generate_base(
                prompt=flow_set.target_prompt,
                negative_prompt=flow_set.negative_prompt,
                seed=config.sampler.seed,
                num_inference_steps=config.sampler.num_inference_steps,
                guidance_scale=config.sampler.guidance_scale,
                height=config.sampler.height,
                width=config.sampler.width,
            )
            metadata: dict[str, Any] = {
                "method": "base",
                "target_prompt": flow_set.target_prompt,
                "source_prompt": flow_set.source_prompt,
                "primitive_flow_set": flow_set.to_dict(),
                "runtime_config": config.to_dict(),
            }
        elif mode == "source_only":
            image = backend.generate_base(
                prompt=flow_set.source_prompt,
                negative_prompt=flow_set.negative_prompt,
                seed=config.sampler.seed,
                num_inference_steps=config.sampler.num_inference_steps,
                guidance_scale=config.sampler.guidance_scale,
                height=config.sampler.height,
                width=config.sampler.width,
            )
            metadata = {
                "method": "source_only",
                "target_prompt": flow_set.target_prompt,
                "source_prompt": flow_set.source_prompt,
                "primitive_flow_set": flow_set.to_dict(),
                "runtime_config": config.to_dict(),
            }
        else:
            image, metadata = sampler.generate_sparse_primitive_flow(flow_set, mode=mode)

        image_name, metadata_name, label = names[mode]
        image_path = output / image_name
        metadata_path = output / metadata_name
        backend.save_image(image, image_path)
        save_metadata_json(metadata, metadata_path)
        paths[f"{mode}_image"] = image_path
        paths[f"{mode}_metadata"] = metadata_path
        grid_image_paths.append(image_path)
        grid_labels.append(label)

    if grid_image_paths:
        grid_path = output / "comparison_grid.png"
        make_image_grid(grid_image_paths, grid_labels, grid_path)
        paths["comparison_grid"] = grid_path
    return paths


def run_marginal_flow_comparison(
    prompt_set: MarginalFlowPromptSet,
    config: RunConfig,
    output_dir: str | Path,
    modes: list[str] | None = None,
) -> dict[str, Path]:
    """Generate the target-only baseline and/or contextual Marginal Flow output."""

    selected_modes = modes or ["base", "marginal_flow"]
    output = ensure_dir(output_dir)
    backend = SD3Backend(config).load()
    sampler = AIMFlowSampler(backend, config)
    paths: dict[str, Path] = {}
    grid_image_paths: list[Path] = []
    grid_labels: list[str] = []
    names = {
        "base": ("base_full_target.png", "metadata_base.json", "Base SD3: full target"),
        "marginal_flow": (
            "marginal_flow.png",
            "metadata_marginal_flow.json",
            "Marginal Flow: contextual ablations",
        ),
    }

    execution_modes = list(selected_modes)
    if config.model.defer_model_cpu_offload and "marginal_flow" in execution_modes:
        execution_modes = ["marginal_flow"] + [mode for mode in execution_modes if mode != "marginal_flow"]

    for mode in execution_modes:
        if mode not in names:
            raise ValueError(f"Unknown Marginal Flow comparison mode: {mode}")
        if mode == "base":
            image = backend.generate_base(
                prompt=prompt_set.target_prompt,
                negative_prompt=prompt_set.negative_prompt,
                seed=config.sampler.seed,
                num_inference_steps=config.sampler.num_inference_steps,
                guidance_scale=config.sampler.guidance_scale,
                height=config.sampler.height,
                width=config.sampler.width,
            )
            metadata: dict[str, Any] = {
                "method": "base",
                "target_prompt": prompt_set.target_prompt,
                "marginal_flow_prompt_set": prompt_set.to_dict(),
                "runtime_config": config.to_dict(),
            }
        else:
            image, metadata = sampler.generate_marginal_flow(prompt_set, mode=mode)

        image_name, metadata_name, label = names[mode]
        image_path = output / image_name
        metadata_path = output / metadata_name
        backend.save_image(image, image_path)
        save_metadata_json(metadata, metadata_path)
        paths[f"{mode}_image"] = image_path
        paths[f"{mode}_metadata"] = metadata_path
        grid_image_paths.append(image_path)
        grid_labels.append(label)

    if grid_image_paths:
        grid_path = output / "comparison_grid.png"
        make_image_grid(grid_image_paths, grid_labels, grid_path)
        paths["comparison_grid"] = grid_path
    return paths
