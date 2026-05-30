"""SPFC timestep probes for the fox/rainboots qualitative analysis."""

from __future__ import annotations

import copy
import gc
from pathlib import Path
from typing import Any

import torch

from aim_flow.aggregation import aggregate_primitive_vfa
from aim_flow.config import RunConfig
from aim_flow.eval_bench.constants import DEFAULT_SPFC_SCHEDULE_1_INDEXED
from aim_flow.eval_bench.generation import load_bench_config, unload_model
from aim_flow.eval_bench.schedules import one_indexed_to_zero_indexed
from aim_flow.ltp import apply_primitive_latent_ltp, apply_primitive_velocity_ltp
from aim_flow.prompt_schema import PrimitiveFlowSet
from aim_flow.sampler import AIMFlowSampler
from aim_flow.sd3_backend import SD3Backend, TextCondition
from aim_flow.utils import ensure_dir, write_json
from aim_flow.visualize import make_image_grid, save_metadata_json


def run_fox_timestep_probe(
    flow_set: PrimitiveFlowSet,
    output_dir: str | Path,
    config: RunConfig | None = None,
    schedule_1_indexed: list[int] | None = None,
    probe_mode: str = "both",
    max_probe_steps: int | None = None,
) -> dict[str, Any]:
    """Create cutoff-final and step-rollout grids for one SPFC prompt."""

    base_config = config or load_bench_config()
    if probe_mode not in {"cutoff", "rollout", "both"}:
        raise ValueError("probe_mode must be one of: cutoff, rollout, both")
    schedule = schedule_1_indexed or DEFAULT_SPFC_SCHEDULE_1_INDEXED
    zero_schedule = one_indexed_to_zero_indexed(schedule, num_steps=base_config.sampler.num_inference_steps, required_count=16)
    if max_probe_steps is not None:
        if max_probe_steps <= 0:
            raise ValueError("max_probe_steps must be positive when provided.")
        zero_schedule = zero_schedule[:max_probe_steps]
    out = ensure_dir(output_dir)

    index = {
        "probe_mode": probe_mode,
        "schedule_1_indexed": schedule,
        "schedule_zero_indexed": zero_schedule,
        "selected_schedule_1_indexed": [step + 1 for step in zero_schedule],
    }
    if probe_mode in {"cutoff", "both"}:
        cutoff_paths = _generate_cutoff_finals(flow_set, out / "cutoff_finals", base_config, zero_schedule)
        cutoff_grid = make_image_grid(
            [path for _, path in cutoff_paths],
            [f"stop@{step + 1}" for step, _ in cutoff_paths],
            out / "fox_cutoff_final_grid.png",
        )
        index["cutoff_grid"] = str(cutoff_grid)
    if probe_mode in {"rollout", "both"}:
        rollout_paths = _generate_step_rollouts(flow_set, out / "step_rollouts", base_config, zero_schedule)
        rollout_grid = make_image_grid(
            [path for _, path in rollout_paths],
            [f"rollout@{step + 1}" for step, _ in rollout_paths],
            out / "fox_step_rollout_grid.png",
        )
        index["rollout_grid"] = str(rollout_grid)
    write_json(index, out / "probe_index.json")
    return index


def _generate_cutoff_finals(
    flow_set: PrimitiveFlowSet,
    output_dir: Path,
    base_config: RunConfig,
    zero_schedule: list[int],
) -> list[tuple[int, Path]]:
    output_dir = ensure_dir(output_dir)
    config = copy.deepcopy(base_config)
    backend = SD3Backend(config).load()
    sampler = AIMFlowSampler(backend, config)
    paths: list[tuple[int, Path]] = []
    try:
        for index, cutoff in enumerate(zero_schedule, start=1):
            print(f"[fox_probe] cutoff final {index}/{len(zero_schedule)}: stop@{cutoff + 1}", flush=True)
            config.primitive_flow.aggregation_steps = [step for step in zero_schedule if step <= cutoff]
            image, metadata = sampler.generate_sparse_primitive_flow(flow_set, mode="primitive_flow_sparse")
            image_path = output_dir / f"stop_at_step_{cutoff + 1:02d}.png"
            metadata_path = output_dir / f"stop_at_step_{cutoff + 1:02d}.json"
            backend.save_image(image, image_path)
            metadata["probe"] = "cutoff_final"
            metadata["cutoff_step_1_indexed"] = cutoff + 1
            save_metadata_json(metadata, metadata_path)
            paths.append((cutoff, image_path))
            del image
            _clear_memory()
    finally:
        unload_model(backend)
    return paths


def _finish_target_only(
    sampler: AIMFlowSampler,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    start_index: int,
    target_condition: TextCondition,
) -> Any:
    scheduler = sampler.backend.require_pipe().scheduler
    state = sampler._snapshot_scheduler_state(scheduler)
    final_latents: torch.Tensor | None = None
    try:
        rollout_latents = latents.clone()
        sampler_cfg = sampler.config.sampler
        autocast_device = sampler.backend.execution_device.type
        use_autocast = autocast_device == "cuda" and sampler.backend.dtype in {torch.float16, torch.bfloat16}
        with torch.inference_mode():
            for step_index in range(start_index, len(timesteps)):
                timestep = timesteps[step_index]
                with torch.autocast(device_type=autocast_device, dtype=sampler.backend.dtype, enabled=use_autocast):
                    target_pred = sampler._compatible_prediction(
                        sampler.backend.predict_with_condition(
                            rollout_latents,
                            timestep,
                            target_condition,
                            guidance_scale=sampler_cfg.guidance_scale,
                        ),
                        rollout_latents,
                        "probe_rollout_target",
                    )
                rollout_latents = sampler.safe_scheduler_step(scheduler, target_pred, timestep, rollout_latents)
                del target_pred
        final_latents = rollout_latents.detach()
    finally:
        sampler._restore_scheduler_state(scheduler, state)
    if final_latents is None:
        raise RuntimeError("Target-only rollout did not produce final latents.")
    return sampler.backend.decode_latents(final_latents)


def _generate_step_rollouts(
    flow_set: PrimitiveFlowSet,
    output_dir: Path,
    config: RunConfig,
    zero_schedule: list[int],
) -> list[tuple[int, Path]]:
    output_dir = ensure_dir(output_dir)
    backend = SD3Backend(config).load()
    sampler = AIMFlowSampler(backend, config)
    paths: list[tuple[int, Path]] = []
    try:
        pipe = backend.require_pipe()
        sampler_cfg = config.sampler
        primitive_cfg = config.primitive_flow
        encoded = backend.encode_primitive_flow_conditions(flow_set, primitive_cfg)
        condition_embeddings: list[TextCondition] = encoded["conditions"]
        condition_texts: list[str] = encoded["condition_texts"]
        condition_names: list[str] = encoded["condition_names"]
        condition_roles: list[str] = encoded["condition_roles"]
        condition_weights: list[float] = encoded["condition_weights"]
        target_index: int | None = encoded["target_index"]
        target_condition: TextCondition = encoded["target_condition"]
        if target_index is None:
            target_index = len(condition_embeddings)
            condition_embeddings = condition_embeddings + [target_condition]
            condition_names = condition_names + ["target"]
            condition_roles = condition_roles + ["target"]
            condition_texts = condition_texts + [flow_set.target_prompt]
            condition_weights = condition_weights + [
                1.0 if primitive_cfg.uniform_condition_weights else float(primitive_cfg.target_weight)
            ]

        latents = backend.prepare_latents(sampler_cfg.seed, sampler_cfg.height, sampler_cfg.width)
        timesteps = backend.set_timesteps(sampler_cfg.num_inference_steps)
        latent_ltp_available = True
        latent_ltp_probe: dict[str, Any] | None = None
        if primitive_cfg.ltp_enabled and primitive_cfg.ltp_mode == "latent" and zero_schedule:
            latent_ltp_available, latent_ltp_probe = sampler._probe_latent_ltp_scheduler(pipe.scheduler, timesteps[0], latents)

        autocast_device = backend.execution_device.type
        use_autocast = autocast_device == "cuda" and backend.dtype in {torch.float16, torch.bfloat16}
        with torch.inference_mode():
            for step_index, timestep in enumerate(timesteps):
                do_aggregate = step_index in zero_schedule
                with torch.autocast(device_type=autocast_device, dtype=backend.dtype, enabled=use_autocast):
                    if do_aggregate:
                        predictions = [
                            sampler._compatible_prediction(
                                backend.predict_with_condition(
                                    latents,
                                    timestep,
                                    condition,
                                    guidance_scale=sampler_cfg.guidance_scale,
                                ),
                                latents,
                                f"probe_primitive_flow_{index}",
                            ).detach()
                            for index, condition in enumerate(condition_embeddings)
                        ]
                        if primitive_cfg.sequential_condition_forward and latents.is_cuda:
                            torch.cuda.empty_cache()
                        target_pred = predictions[target_index]
                        candidate_pred, _ = aggregate_primitive_vfa(
                            predictions=predictions,
                            condition_names=condition_names,
                            condition_roles=condition_roles,
                            condition_base_weights=condition_weights,
                            target_index=target_index,
                            vfa_temperature=primitive_cfg.vfa_temperature,
                            use_consensus_gating=primitive_cfg.use_consensus_gating,
                            use_target_consistency_gating=primitive_cfg.use_target_consistency_gating,
                            velocity_clip_ratio=primitive_cfg.velocity_clip_ratio,
                            steering_strength=primitive_cfg.steering_strength,
                            min_gate=primitive_cfg.min_gate,
                            max_gate=primitive_cfg.max_gate,
                        )
                        candidate_pred = sampler._compatible_prediction(candidate_pred, latents, "probe_candidate")
                    else:
                        target_pred = sampler._compatible_prediction(
                            backend.predict_with_condition(
                                latents,
                                timestep,
                                target_condition,
                                guidance_scale=sampler_cfg.guidance_scale,
                            ),
                            latents,
                            "probe_target",
                        )

                if do_aggregate:
                    print(
                        f"[fox_probe] step rollout {len(paths) + 1}/{len(zero_schedule)}: rollout@{step_index + 1}",
                        flush=True,
                    )
                    ltp_mode = "off" if not primitive_cfg.ltp_enabled else primitive_cfg.ltp_mode
                    if ltp_mode == "latent" and not latent_ltp_available:
                        ltp_mode = "velocity"
                    if ltp_mode == "latent":
                        try:
                            x_next_target, x_next_candidate = sampler._compute_anchor_and_candidate_steps(
                                pipe.scheduler,
                                target_pred,
                                candidate_pred,
                                timestep,
                                latents,
                            )
                            latents, _ = apply_primitive_latent_ltp(
                                latents,
                                x_next_target,
                                x_next_candidate,
                                primitive_cfg.ltp_radius_ratio,
                            )
                        except Exception:
                            candidate_pred, _ = apply_primitive_velocity_ltp(
                                target_pred,
                                candidate_pred,
                                primitive_cfg.ltp_radius_ratio,
                            )
                            candidate_pred = sampler._compatible_prediction(candidate_pred, latents, "probe_velocity_ltp")
                            latents = sampler.safe_scheduler_step(pipe.scheduler, candidate_pred, timestep, latents)
                    elif ltp_mode == "velocity":
                        candidate_pred, _ = apply_primitive_velocity_ltp(
                            target_pred,
                            candidate_pred,
                            primitive_cfg.ltp_radius_ratio,
                        )
                        candidate_pred = sampler._compatible_prediction(candidate_pred, latents, "probe_velocity_ltp")
                        latents = sampler.safe_scheduler_step(pipe.scheduler, candidate_pred, timestep, latents)
                    else:
                        latents = sampler.safe_scheduler_step(pipe.scheduler, candidate_pred, timestep, latents)
                    image = _finish_target_only(sampler, latents, timesteps, step_index + 1, target_condition)
                    image_path = output_dir / f"rollout_from_step_{step_index + 1:02d}.png"
                    metadata_path = output_dir / f"rollout_from_step_{step_index + 1:02d}.json"
                    backend.save_image(image, image_path)
                    write_json(
                        {
                            "probe": "step_rollout",
                            "rollout_step_1_indexed": step_index + 1,
                            "target_prompt": flow_set.target_prompt,
                            "latent_ltp_probe": latent_ltp_probe,
                        },
                        metadata_path,
                    )
                    paths.append((step_index, image_path))
                    del image, predictions, candidate_pred
                    _clear_memory()
                else:
                    latents = sampler.safe_scheduler_step(pipe.scheduler, target_pred, timestep, latents)
                del target_pred
    finally:
        unload_model(backend)
    return paths


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
