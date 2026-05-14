"""AIM-Flow v2 sampler."""

from __future__ import annotations

import copy
from typing import Any

import torch
from PIL import Image
from tqdm.auto import tqdm

from aim_flow.aggregation import aggregate_ladder_vfa, aggregate_predictions, aggregate_primitive_vfa, aggregate_vfa
from aim_flow.config import RunConfig
from aim_flow.ladder import get_active_condition_indices, parse_aggregation_steps, select_reference_condition_index
from aim_flow.ltp import (
    apply_ladder_latent_ltp,
    apply_ladder_velocity_ltp,
    apply_latent_ltp,
    apply_primitive_latent_ltp,
    apply_primitive_velocity_ltp,
    apply_velocity_ltp,
)
from aim_flow.primitive_flow import parse_aggregation_steps as parse_primitive_aggregation_steps
from aim_flow.prompt_schema import ConditionLadder, PrimitiveFlowSet, PrimitivePrompt, PromptDecomposition
from aim_flow.schedules import (
    get_condition_schedule_weight,
    get_lambda_schedule_weight,
    get_ltp_radius_ratio,
    get_ltp_radius_ratio_for_step,
    get_schedule_weight,
)
from aim_flow.sd3_backend import SD3Backend, TextCondition


class AIMFlowSampler:
    """Generate images with AIM-Flow v2: anchor-preserved primitive steering."""

    def __init__(self, backend: SD3Backend, run_config: RunConfig):
        self.backend = backend
        self.config = run_config

    def generate(self, prompt_decomposition: PromptDecomposition, mode: str = "aim_v2") -> tuple[Image.Image, dict[str, Any]]:
        """Generate one image and metadata for the selected mode."""

        prompt_decomposition.validate()
        normalized_mode = "aim_v2" if mode == "full" else mode
        if normalized_mode == "base":
            return self._generate_base(prompt_decomposition)
        if normalized_mode == "anchor":
            return self._generate_anchor(prompt_decomposition)
        if normalized_mode not in {"naive_v1", "aim_v2"}:
            raise ValueError("AIMFlowSampler.generate supports modes: base, anchor, naive_v1, aim_v2, full")
        return self._generate_custom(prompt_decomposition, normalized_mode)

    def _generate_base(self, prompt_decomposition: PromptDecomposition) -> tuple[Image.Image, dict[str, Any]]:
        sampler_cfg = self.config.sampler
        image = self.backend.generate_base(
            prompt=prompt_decomposition.full_prompt,
            negative_prompt=prompt_decomposition.negative_prompt,
            seed=sampler_cfg.seed,
            num_inference_steps=sampler_cfg.num_inference_steps,
            guidance_scale=sampler_cfg.guidance_scale,
            height=sampler_cfg.height,
            width=sampler_cfg.width,
        )
        return image, self._base_metadata(prompt_decomposition, "base")

    def _generate_anchor(self, prompt_decomposition: PromptDecomposition) -> tuple[Image.Image, dict[str, Any]]:
        sampler_cfg = self.config.sampler
        image = self.backend.generate_anchor(
            anchor_prompt=prompt_decomposition.anchor_prompt,
            negative_prompt=prompt_decomposition.negative_prompt,
            seed=sampler_cfg.seed,
            num_inference_steps=sampler_cfg.num_inference_steps,
            guidance_scale=sampler_cfg.guidance_scale,
            height=sampler_cfg.height,
            width=sampler_cfg.width,
        )
        return image, self._base_metadata(prompt_decomposition, "anchor")

    def _generate_custom(self, prompt_decomposition: PromptDecomposition, mode: str) -> tuple[Image.Image, dict[str, Any]]:
        pipe = self.backend.require_pipe()
        sampler_cfg = self.config.sampler
        aim_cfg = self.config.aim_flow
        vfa_cfg = aim_cfg.vfa
        ltp_cfg = aim_cfg.ltp
        primitives = prompt_decomposition.get_enabled_primitives()[: aim_cfg.max_primitives]
        if not primitives:
            raise ValueError("AIM-Flow requires at least one enabled primitive prompt.")
        use_cfg = sampler_cfg.guidance_scale > 1.0

        self.backend.validate_custom_sampling_compatibility()
        self.backend.validate_image_size(sampler_cfg.height, sampler_cfg.width)

        full_condition = self.backend.encode_text_condition(
            prompt_decomposition.full_prompt,
            prompt_decomposition.negative_prompt,
            do_classifier_free_guidance=use_cfg,
        )
        anchor_condition = self.backend.encode_text_condition(
            prompt_decomposition.anchor_prompt,
            prompt_decomposition.negative_prompt,
            do_classifier_free_guidance=use_cfg,
        )
        primitive_conditions, primitive_condition_texts = self._encode_primitive_conditions(
            prompt_decomposition,
            primitives,
            mode,
            use_cfg,
        )
        if hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()

        latents = self.backend.prepare_latents(
            seed=sampler_cfg.seed,
            height=sampler_cfg.height,
            width=sampler_cfg.width,
        )
        timesteps = self.backend.set_timesteps(sampler_cfg.num_inference_steps)
        debug_steps: list[dict[str, Any]] = []
        ltp_warning: str | None = None
        latent_ltp_available = True
        latent_ltp_probe: dict[str, Any] | None = None
        if ltp_cfg.enabled and ltp_cfg.mode == "latent" and mode == "aim_v2":
            latent_ltp_available, latent_ltp_probe = self._probe_latent_ltp_scheduler(
                pipe.scheduler,
                timesteps[0],
                latents,
            )
            if not latent_ltp_available:
                ltp_warning = (
                    "Latent LTP disabled before sampling; falling back to velocity LTP because "
                    f"the scheduler is not safely replayable: {latent_ltp_probe}"
                )

        autocast_device = self.backend.execution_device.type
        use_autocast = autocast_device == "cuda" and self.backend.dtype in {torch.float16, torch.bfloat16}

        with torch.inference_mode():
            for step_index, timestep in enumerate(tqdm(timesteps, desc=f"AIM-Flow {mode}")):
                with torch.autocast(device_type=autocast_device, dtype=self.backend.dtype, enabled=use_autocast):
                    anchor_pred = self.backend.predict_with_condition(
                        latents,
                        timestep,
                        anchor_condition,
                        guidance_scale=sampler_cfg.guidance_scale,
                    )
                    anchor_pred = self._compatible_prediction(anchor_pred, latents, "anchor")
                    primitive_preds = [
                        self._compatible_prediction(
                            self.backend.predict_with_condition(
                                latents,
                                timestep,
                                condition,
                                guidance_scale=sampler_cfg.guidance_scale,
                            ),
                            latents,
                            f"primitive_{index}",
                        ).detach()
                        for index, condition in enumerate(primitive_conditions)
                    ]
                    if aim_cfg.sequential_primitive_forward and latents.is_cuda:
                        torch.cuda.empty_cache()
                    schedule_weights = [
                        get_schedule_weight(primitive.schedule, step_index, len(timesteps))
                        for primitive in primitives
                    ]
                    lambda_t = aim_cfg.lambda_global
                    if mode == "aim_v2":
                        lambda_t *= get_lambda_schedule_weight(aim_cfg.lambda_schedule, step_index, len(timesteps))
                        full_pred = self._compatible_prediction(
                            self.backend.predict_with_condition(
                                latents,
                                timestep,
                                full_condition,
                                guidance_scale=sampler_cfg.guidance_scale,
                            ),
                            latents,
                            "full",
                        )
                        candidate_pred, vfa_debug = aggregate_vfa(
                            anchor_pred=anchor_pred,
                            full_pred=full_pred,
                            primitive_preds=primitive_preds,
                            primitive_base_weights=[primitive.weight for primitive in primitives],
                            primitive_schedule_weights=schedule_weights,
                            lambda_global=lambda_t,
                            conflict_threshold=vfa_cfg.conflict_threshold,
                            velocity_clip_ratio=vfa_cfg.velocity_clip_ratio,
                            min_gate=vfa_cfg.min_gate,
                            max_gate=vfa_cfg.max_gate,
                            use_delta_full_gate=vfa_cfg.use_delta_full_gate,
                        )
                    else:
                        full_pred = None
                        candidate_pred, vfa_debug = aggregate_predictions(
                            anchor_pred=anchor_pred,
                            primitive_preds=primitive_preds,
                            primitive_weights=[primitive.weight for primitive in primitives],
                            primitive_schedule_weights=schedule_weights,
                            lambda_global=lambda_t,
                            conflict_threshold=vfa_cfg.conflict_threshold,
                            norm_clip_ratio=vfa_cfg.velocity_clip_ratio,
                            mode="naive_v1",
                        )
                    candidate_pred = self._compatible_prediction(candidate_pred, latents, "candidate")

                radius_ratio_t = get_ltp_radius_ratio(
                    step_index,
                    len(timesteps),
                    ltp_cfg.early_radius_ratio,
                    ltp_cfg.middle_radius_ratio,
                    ltp_cfg.late_radius_ratio,
                )
                latents_dtype = latents.dtype
                ltp_mode = "off" if not ltp_cfg.enabled or mode == "naive_v1" else ltp_cfg.mode
                if ltp_mode == "latent" and not latent_ltp_available:
                    ltp_mode = "velocity"
                if ltp_mode == "velocity":
                    candidate_pred, ltp_debug = apply_velocity_ltp(anchor_pred, candidate_pred, radius_ratio_t)
                    ltp_debug["latent_ltp_fallback"] = not latent_ltp_available and ltp_cfg.mode == "latent"
                    if latent_ltp_probe is not None and ltp_debug["latent_ltp_fallback"]:
                        ltp_debug["fallback_reason"] = latent_ltp_probe
                    candidate_pred = self._compatible_prediction(candidate_pred, latents, "candidate_after_velocity_ltp")
                    latents = self._scheduler_step(pipe.scheduler, candidate_pred, timestep, latents)
                elif ltp_mode == "latent":
                    try:
                        x_next_anchor, x_next_candidate = self._compute_anchor_and_candidate_steps(
                            pipe.scheduler,
                            anchor_pred,
                            candidate_pred,
                            timestep,
                            latents,
                        )
                        latents, ltp_debug = apply_latent_ltp(latents, x_next_anchor, x_next_candidate, radius_ratio_t)
                    except Exception as exc:
                        if not ltp_cfg.fallback_to_velocity_ltp:
                            raise
                        ltp_warning = f"Latent LTP disabled; fell back to velocity LTP because scheduler stepping failed: {exc}"
                        latent_ltp_available = False
                        latent_ltp_probe = {
                            "scheduler_class": pipe.scheduler.__class__.__name__,
                            "checked": True,
                            "runtime_error": str(exc),
                        }
                        candidate_pred, ltp_debug = apply_velocity_ltp(anchor_pred, candidate_pred, radius_ratio_t)
                        ltp_debug["latent_ltp_fallback"] = True
                        ltp_debug["fallback_reason"] = str(exc)
                        candidate_pred = self._compatible_prediction(candidate_pred, latents, "candidate_after_velocity_ltp")
                        latents = self._scheduler_step(pipe.scheduler, candidate_pred, timestep, latents)
                elif ltp_mode == "off":
                    ltp_debug = {"ltp_active": False, "mode": "off", "radius_ratio": float(radius_ratio_t)}
                    latents = self._scheduler_step(pipe.scheduler, candidate_pred, timestep, latents)
                else:
                    raise ValueError(f"Unknown LTP mode: {ltp_mode}")

                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
                debug_steps.append(
                    {
                        "step_index": step_index,
                        "timestep": float(timestep.detach().float().cpu().item()),
                        "lambda_t": float(lambda_t),
                        "radius_ratio_t": float(radius_ratio_t),
                        "primitive_texts": [primitive.text for primitive in primitives],
                        "primitive_anchor_augmented_texts": primitive_condition_texts,
                        "primitive_schedules": [primitive.schedule for primitive in primitives],
                        "primitive_base_weights": [primitive.weight for primitive in primitives],
                        "primitive_schedule_weights": schedule_weights,
                        "vfa": vfa_debug,
                        "ltp": ltp_debug,
                    }
                )

                del anchor_pred, primitive_preds, candidate_pred
                if full_pred is not None:
                    del full_pred

        image = self.backend.decode_latents(latents)
        metadata = self._base_metadata(prompt_decomposition, mode)
        metadata["method"] = (
            "AIM-Flow v2: Anchor-Preserved Primitive Steering"
            if mode == "aim_v2"
            else "AIM-Flow v1 naive standalone primitive residuals"
        )
        metadata["primitive_prompts_used"] = [primitive.to_dict() for primitive in primitives]
        metadata["primitive_original_texts"] = [primitive.text for primitive in primitives]
        metadata["primitive_condition_texts"] = primitive_condition_texts
        metadata["primitive_conditioning"] = "anchor_augmented" if mode == "aim_v2" else "standalone"
        metadata["debug_steps"] = debug_steps
        metadata["ltp_warning"] = ltp_warning
        metadata["latent_ltp_available"] = latent_ltp_available
        metadata["latent_ltp_probe"] = latent_ltp_probe
        metadata["custom_loop_audit"] = {
            "classifier_free_guidance": use_cfg,
            "guidance_scale": sampler_cfg.guidance_scale,
            "sequential_cfg_forward": True,
            "sequential_primitive_forward": aim_cfg.sequential_primitive_forward,
            "latents_dtype": str(latents.dtype),
            "latents_device": str(latents.device),
            "embedding_shapes": {
                "full": self._condition_shapes(full_condition),
                "anchor": self._condition_shapes(anchor_condition),
                "primitives": [self._condition_shapes(condition) for condition in primitive_conditions],
            },
        }
        metadata["note"] = (
            "AIM-Flow v2 uses only the model's own conditional velocity fields. "
            "No VQA, reward model, external judge, training, or image feedback is used."
        )
        return image, metadata

    def generate_ladder_v3(
        self,
        condition_ladder: ConditionLadder,
        mode: str = "ladder_v3_sparse",
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Generate one image with LadderFlow v3 condition-ladder aggregation."""

        condition_ladder.validate()
        pipe = self.backend.require_pipe()
        sampler_cfg = self.config.sampler
        ladder_cfg = self.config.ladder_flow
        conditions = condition_ladder.get_enabled_conditions()[: ladder_cfg.max_conditions]
        if len(conditions) < 2:
            raise ValueError("LadderFlow v3 requires at least two enabled conditions.")
        use_cfg = sampler_cfg.guidance_scale > 1.0

        self.backend.validate_custom_sampling_compatibility()
        self.backend.validate_image_size(sampler_cfg.height, sampler_cfg.width)
        encoded = self.backend.encode_ladder_conditions(condition_ladder)
        condition_embeddings: list[TextCondition] = encoded["conditions"][: len(conditions)]
        condition_texts: list[str] = encoded["condition_texts"][: len(conditions)]
        if hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()

        latents = self.backend.prepare_latents(sampler_cfg.seed, sampler_cfg.height, sampler_cfg.width)
        timesteps = self.backend.set_timesteps(sampler_cfg.num_inference_steps)
        num_steps = len(timesteps)
        num_conditions = len(conditions)
        final_index = num_conditions - 1
        aggregation_steps = parse_aggregation_steps(
            ladder_cfg.aggregation_steps,
            ladder_cfg.aggregation_step_fractions,
            num_steps,
        )
        if ladder_cfg.aggregate_every_n_steps is not None:
            every = max(1, int(ladder_cfg.aggregate_every_n_steps))
            aggregation_steps.update(range(0, num_steps, every))
        if mode == "ladder_progressive_noagg":
            aggregation_steps = set()
        elif mode == "ladder_v3_dense":
            aggregation_steps = set(range(num_steps))

        latent_ltp_available = True
        latent_ltp_probe: dict[str, Any] | None = None
        latent_ltp_disabled_reason: str | None = None
        if ladder_cfg.ltp_enabled and ladder_cfg.ltp_mode == "latent" and aggregation_steps:
            latent_ltp_available, latent_ltp_probe = self._probe_latent_ltp_scheduler(pipe.scheduler, timesteps[0], latents)
            if not latent_ltp_available:
                latent_ltp_disabled_reason = "scheduler step not safely repeatable"
                if not ladder_cfg.fallback_to_velocity_ltp:
                    raise RuntimeError(
                        "Latent LTP is not safe with this scheduler and fallback_to_velocity_ltp is disabled."
                    )

        debug_steps: list[dict[str, Any]] = []
        autocast_device = self.backend.execution_device.type
        use_autocast = autocast_device == "cuda" and self.backend.dtype in {torch.float16, torch.bfloat16}

        with torch.inference_mode():
            for step_index, timestep in enumerate(tqdm(timesteps, desc=f"LadderFlow {mode}")):
                reference_index = select_reference_condition_index(step_index, num_steps, num_conditions, ladder_cfg.reference_policy)
                do_aggregate = step_index in aggregation_steps
                radius_ratio_t = get_ltp_radius_ratio_for_step(
                    step_index,
                    num_steps,
                    ladder_cfg.ltp_early_radius_ratio,
                    ladder_cfg.ltp_middle_radius_ratio,
                    ladder_cfg.ltp_late_radius_ratio,
                )
                step_debug: dict[str, Any] = {
                    "step_index": step_index,
                    "timestep": float(timestep.detach().float().cpu().item()),
                    "do_aggregate": bool(do_aggregate),
                    "reference_index": reference_index,
                    "reference_text": condition_texts[reference_index],
                    "radius_ratio": float(radius_ratio_t),
                }

                with torch.autocast(device_type=autocast_device, dtype=self.backend.dtype, enabled=use_autocast):
                    if do_aggregate:
                        active_indices = get_active_condition_indices(
                            step_index,
                            num_steps,
                            num_conditions,
                            ladder_cfg.active_policy,
                            reference_index,
                        )
                        active_indices = sorted(set(active_indices + [reference_index, final_index]))
                        predictions = [
                            self._compatible_prediction(
                                self.backend.predict_with_condition(
                                    latents,
                                    timestep,
                                    condition_embeddings[index],
                                    guidance_scale=sampler_cfg.guidance_scale,
                                ),
                                latents,
                                f"ladder_condition_{index}",
                            ).detach()
                            for index in active_indices
                        ]
                        if ladder_cfg.sequential_condition_forward and latents.is_cuda:
                            torch.cuda.empty_cache()
                        local_reference_index = active_indices.index(reference_index)
                        local_final_index = active_indices.index(final_index)
                        schedule_weights = [
                            get_condition_schedule_weight(conditions[index].schedule, step_index, num_steps)
                            for index in active_indices
                        ]
                        candidate_pred, vfa_debug = aggregate_ladder_vfa(
                            predictions=predictions,
                            condition_base_weights=[conditions[index].weight for index in active_indices],
                            condition_schedule_weights=schedule_weights,
                            reference_index=local_reference_index,
                            final_index=local_final_index,
                            vfa_temperature=ladder_cfg.vfa_temperature,
                            use_consensus_gating=ladder_cfg.use_consensus_gating,
                            use_final_consistency_gating=ladder_cfg.use_final_consistency_gating,
                            velocity_clip_ratio=ladder_cfg.velocity_clip_ratio,
                            min_gate=ladder_cfg.min_gate,
                            max_gate=ladder_cfg.max_gate,
                        )
                        candidate_pred = self._compatible_prediction(candidate_pred, latents, "ladder_candidate")
                        reference_pred = predictions[local_reference_index]
                    else:
                        selected_index = self._select_non_aggregation_index(ladder_cfg.non_aggregation_policy, reference_index, final_index)
                        selected_pred = self._compatible_prediction(
                            self.backend.predict_with_condition(
                                latents,
                                timestep,
                                condition_embeddings[selected_index],
                                guidance_scale=sampler_cfg.guidance_scale,
                            ),
                            latents,
                            f"ladder_selected_{selected_index}",
                        )

                latents_dtype = latents.dtype
                if do_aggregate:
                    ltp_mode = "off" if not ladder_cfg.ltp_enabled else ladder_cfg.ltp_mode
                    if ltp_mode == "latent" and not latent_ltp_available:
                        ltp_mode = "velocity"
                    fallback_to_velocity = False
                    if ltp_mode == "velocity":
                        candidate_pred, ltp_debug = apply_ladder_velocity_ltp(reference_pred, candidate_pred, radius_ratio_t)
                        candidate_pred = self._compatible_prediction(candidate_pred, latents, "ladder_candidate_velocity_ltp")
                        latents = self.safe_scheduler_step(pipe.scheduler, candidate_pred, timestep, latents)
                    elif ltp_mode == "latent":
                        try:
                            x_next_reference, x_next_candidate = self._compute_anchor_and_candidate_steps(
                                pipe.scheduler,
                                reference_pred,
                                candidate_pred,
                                timestep,
                                latents,
                            )
                            latents, ltp_debug = apply_ladder_latent_ltp(latents, x_next_reference, x_next_candidate, radius_ratio_t)
                        except Exception as exc:
                            if not ladder_cfg.fallback_to_velocity_ltp:
                                raise
                            fallback_to_velocity = True
                            latent_ltp_available = False
                            latent_ltp_disabled_reason = "scheduler step not safely repeatable"
                            candidate_pred, ltp_debug = apply_ladder_velocity_ltp(reference_pred, candidate_pred, radius_ratio_t)
                            ltp_debug["fallback_reason"] = str(exc)
                            candidate_pred = self._compatible_prediction(candidate_pred, latents, "ladder_candidate_velocity_ltp")
                            latents = self.safe_scheduler_step(pipe.scheduler, candidate_pred, timestep, latents)
                    elif ltp_mode == "off":
                        ltp_debug = {"ltp_active": False, "ltp_mode": "off", "radius_ratio": float(radius_ratio_t)}
                        latents = self.safe_scheduler_step(pipe.scheduler, candidate_pred, timestep, latents)
                    else:
                        raise ValueError(f"Unknown LadderFlow LTP mode: {ltp_mode}")
                    ltp_debug["fallback_to_velocity_ltp"] = bool(fallback_to_velocity or (ladder_cfg.ltp_mode == "latent" and not latent_ltp_available))
                    step_debug.update(
                        {
                            "active_indices": active_indices,
                            "active_texts": [condition_texts[index] for index in active_indices],
                            "vfa": vfa_debug,
                            "ltp": ltp_debug,
                            "vfa_weights": vfa_debug.get("softmax_weights"),
                            "consensus_gates": vfa_debug.get("consensus_gates"),
                            "final_consistency_gates": vfa_debug.get("final_consistency_gates"),
                            "pairwise_cosine_matrix": vfa_debug.get("pairwise_cosine_matrix"),
                        }
                    )
                    del predictions, candidate_pred, reference_pred
                else:
                    latents = self.safe_scheduler_step(pipe.scheduler, selected_pred, timestep, latents)
                    step_debug.update(
                        {
                            "selected_condition_index": selected_index,
                            "selected_condition_text": condition_texts[selected_index],
                        }
                    )
                    del selected_pred
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
                debug_steps.append(step_debug)

        image = self.backend.decode_latents(latents)
        metadata = {
            "method": mode,
            "full_prompt": condition_ladder.full_prompt,
            "negative_prompt": condition_ladder.negative_prompt,
            "condition_ladder": condition_ladder.to_dict(),
            "condition_texts": condition_texts,
            "aggregation_steps": sorted(aggregation_steps),
            "reference_policy": ladder_cfg.reference_policy,
            "active_policy": ladder_cfg.active_policy,
            "non_aggregation_policy": ladder_cfg.non_aggregation_policy,
            "ltp_mode": ladder_cfg.ltp_mode,
            "latent_ltp_available": latent_ltp_available,
            "latent_ltp_probe": latent_ltp_probe,
            "latent_ltp_disabled_reason": latent_ltp_disabled_reason,
            "vfa_temperature": ladder_cfg.vfa_temperature,
            "seed": sampler_cfg.seed,
            "num_inference_steps": sampler_cfg.num_inference_steps,
            "height": sampler_cfg.height,
            "width": sampler_cfg.width,
            "guidance_scale": sampler_cfg.guidance_scale,
            "runtime_config": self.config.to_dict(),
            "debug_steps": debug_steps,
            "note": (
                "LadderFlow v3 aggregates complete condition velocity fields. "
                "No VQA, reward model, external judge, training, fine-tuning, or image feedback is used."
            ),
        }
        return image, metadata

    def generate_sparse_primitive_flow(
        self,
        flow_set: PrimitiveFlowSet,
        mode: str = "primitive_flow_sparse",
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Generate one image with Sparse Primitive Flow Composition."""

        flow_set.validate()
        pipe = self.backend.require_pipe()
        sampler_cfg = self.config.sampler
        primitive_cfg = self.config.primitive_flow
        use_cfg = sampler_cfg.guidance_scale > 1.0

        self.backend.validate_custom_sampling_compatibility()
        self.backend.validate_image_size(sampler_cfg.height, sampler_cfg.width)
        encoded = self.backend.encode_primitive_flow_conditions(flow_set, primitive_cfg)
        condition_embeddings: list[TextCondition] = encoded["conditions"]
        condition_texts: list[str] = encoded["condition_texts"]
        condition_names: list[str] = encoded["condition_names"]
        condition_roles: list[str] = encoded["condition_roles"]
        condition_weights: list[float] = encoded["condition_weights"]
        target_index: int | None = encoded["target_index"]
        target_condition: TextCondition = encoded["target_condition"]

        if target_index is None:
            condition_embeddings = condition_embeddings + [target_condition]
            condition_texts = condition_texts + [flow_set.target_prompt]
            condition_names = condition_names + ["target"]
            condition_roles = condition_roles + ["target"]
            condition_weights = condition_weights + [float(primitive_cfg.target_weight)]
            target_index = len(condition_embeddings) - 1
        if not condition_embeddings:
            raise ValueError("Sparse primitive flow requires at least one encoded condition.")
        if condition_roles[target_index] != "target":
            raise ValueError("target_index must point to the target prompt condition.")
        if hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()

        latents = self.backend.prepare_latents(sampler_cfg.seed, sampler_cfg.height, sampler_cfg.width)
        timesteps = self.backend.set_timesteps(sampler_cfg.num_inference_steps)
        num_steps = len(timesteps)
        aggregation_steps = parse_primitive_aggregation_steps(
            num_steps=num_steps,
            aggregation_steps=primitive_cfg.aggregation_steps,
            aggregation_step_fractions=primitive_cfg.aggregation_step_fractions,
            final_only=primitive_cfg.final_only,
            aggregate_every_n_steps=primitive_cfg.aggregate_every_n_steps,
        )
        if mode == "primitive_flow_final_only":
            aggregation_steps = {num_steps - 1}
        elif mode in {"primitive_flow_dense", "primitive_flow_dense_optional"}:
            aggregation_steps = set(range(num_steps))

        latent_ltp_available = True
        latent_ltp_probe: dict[str, Any] | None = None
        latent_ltp_disabled_reason: str | None = None
        if primitive_cfg.ltp_enabled and primitive_cfg.ltp_mode == "latent" and aggregation_steps:
            latent_ltp_available, latent_ltp_probe = self._probe_latent_ltp_scheduler(
                pipe.scheduler,
                timesteps[0],
                latents,
            )
            if not latent_ltp_available:
                latent_ltp_disabled_reason = "scheduler step not safely repeatable"
                if not primitive_cfg.fallback_to_velocity_ltp:
                    raise RuntimeError(
                        "Latent LTP is not safe with this scheduler and fallback_to_velocity_ltp is disabled."
                    )

        debug_steps: list[dict[str, Any]] = []
        autocast_device = self.backend.execution_device.type
        use_autocast = autocast_device == "cuda" and self.backend.dtype in {torch.float16, torch.bfloat16}

        with torch.inference_mode():
            for step_index, timestep in enumerate(tqdm(timesteps, desc=f"PrimitiveFlow {mode}")):
                do_aggregate = step_index in aggregation_steps
                step_debug: dict[str, Any] = {
                    "step_index": step_index,
                    "timestep": float(timestep.detach().float().cpu().item()),
                    "do_aggregate": bool(do_aggregate),
                }

                with torch.autocast(device_type=autocast_device, dtype=self.backend.dtype, enabled=use_autocast):
                    if do_aggregate:
                        predictions = [
                            self._compatible_prediction(
                                self.backend.predict_with_condition(
                                    latents,
                                    timestep,
                                    condition,
                                    guidance_scale=sampler_cfg.guidance_scale,
                                ),
                                latents,
                                f"primitive_flow_{index}_{condition_roles[index]}",
                            ).detach()
                            for index, condition in enumerate(condition_embeddings)
                        ]
                        if primitive_cfg.sequential_condition_forward and latents.is_cuda:
                            torch.cuda.empty_cache()
                        target_pred = predictions[target_index]
                        candidate_pred, vfa_debug = aggregate_primitive_vfa(
                            predictions=predictions,
                            condition_names=condition_names,
                            condition_roles=condition_roles,
                            condition_base_weights=condition_weights,
                            target_index=target_index,
                            vfa_temperature=primitive_cfg.vfa_temperature,
                            use_consensus_gating=primitive_cfg.use_consensus_gating,
                            use_target_consistency_gating=primitive_cfg.use_target_consistency_gating,
                            velocity_clip_ratio=primitive_cfg.velocity_clip_ratio,
                            min_gate=primitive_cfg.min_gate,
                            max_gate=primitive_cfg.max_gate,
                        )
                        candidate_pred = self._compatible_prediction(candidate_pred, latents, "primitive_flow_candidate")
                    else:
                        target_pred = self._compatible_prediction(
                            self.backend.predict_with_condition(
                                latents,
                                timestep,
                                target_condition,
                                guidance_scale=sampler_cfg.guidance_scale,
                            ),
                            latents,
                            "primitive_flow_target",
                        )

                latents_dtype = latents.dtype
                if do_aggregate:
                    ltp_mode = "off" if not primitive_cfg.ltp_enabled else primitive_cfg.ltp_mode
                    if ltp_mode == "latent" and not latent_ltp_available:
                        ltp_mode = "velocity"
                    ltp_fallback = False
                    ltp_fallback_reason: str | None = None
                    if ltp_mode == "latent":
                        try:
                            x_next_target, x_next_candidate = self._compute_anchor_and_candidate_steps(
                                pipe.scheduler,
                                target_pred,
                                candidate_pred,
                                timestep,
                                latents,
                            )
                            latents, ltp_debug = apply_primitive_latent_ltp(
                                latents,
                                x_next_target,
                                x_next_candidate,
                                primitive_cfg.ltp_radius_ratio,
                            )
                        except Exception as exc:
                            if not primitive_cfg.fallback_to_velocity_ltp:
                                raise
                            latent_ltp_available = False
                            latent_ltp_disabled_reason = "scheduler step not safely repeatable"
                            ltp_fallback = True
                            ltp_fallback_reason = str(exc)
                            candidate_pred, ltp_debug = apply_primitive_velocity_ltp(
                                target_pred,
                                candidate_pred,
                                primitive_cfg.ltp_radius_ratio,
                            )
                            candidate_pred = self._compatible_prediction(
                                candidate_pred,
                                latents,
                                "primitive_flow_candidate_velocity_ltp",
                            )
                            latents = self.safe_scheduler_step(pipe.scheduler, candidate_pred, timestep, latents)
                    elif ltp_mode == "velocity":
                        ltp_fallback = primitive_cfg.ltp_mode == "latent" and not latent_ltp_available
                        if ltp_fallback and latent_ltp_disabled_reason:
                            ltp_fallback_reason = latent_ltp_disabled_reason
                        candidate_pred, ltp_debug = apply_primitive_velocity_ltp(
                            target_pred,
                            candidate_pred,
                            primitive_cfg.ltp_radius_ratio,
                        )
                        candidate_pred = self._compatible_prediction(
                            candidate_pred,
                            latents,
                            "primitive_flow_candidate_velocity_ltp",
                        )
                        latents = self.safe_scheduler_step(pipe.scheduler, candidate_pred, timestep, latents)
                    elif ltp_mode == "off":
                        ltp_debug = {
                            "ltp_active": False,
                            "ltp_mode": "off",
                            "radius_ratio": float(primitive_cfg.ltp_radius_ratio),
                        }
                        latents = self.safe_scheduler_step(pipe.scheduler, candidate_pred, timestep, latents)
                    else:
                        raise ValueError(f"Unknown primitive-flow LTP mode: {ltp_mode}")
                    ltp_debug["ltp_fallback"] = bool(ltp_fallback)
                    if ltp_fallback_reason:
                        ltp_debug["ltp_fallback_reason"] = ltp_fallback_reason
                    step_debug.update(
                        {
                            "condition_names": condition_names,
                            "condition_roles": condition_roles,
                            "condition_texts": condition_texts,
                            "condition_weights": condition_weights,
                            "target_index": target_index,
                            "vfa": vfa_debug,
                            "pairwise_cosine_matrix": vfa_debug.get("pairwise_cosine_matrix"),
                            "consensus_gates": vfa_debug.get("consensus_gates"),
                            "target_consistency_gates": vfa_debug.get("target_consistency_gates"),
                            "raw_scores": vfa_debug.get("raw_scores"),
                            "softmax_weights": vfa_debug.get("softmax_weights"),
                            "target_norm": vfa_debug.get("target_norm"),
                            "raw_correction_norm": vfa_debug.get("raw_correction_norm"),
                            "clipped_correction_norm": vfa_debug.get("clipped_correction_norm"),
                            "ltp_debug": ltp_debug,
                            "ltp_fallback": bool(ltp_fallback),
                            "ltp_fallback_reason": ltp_fallback_reason,
                        }
                    )
                    del predictions, candidate_pred
                else:
                    latents = self.safe_scheduler_step(pipe.scheduler, target_pred, timestep, latents)
                    step_debug.update(
                        {
                            "selected_condition": "target",
                            "selected_condition_text": flow_set.target_prompt,
                        }
                    )
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
                debug_steps.append(step_debug)
                del target_pred

        image = self.backend.decode_latents(latents)
        metadata = {
            "method": "sparse_primitive_flow",
            "mode": mode,
            "target_prompt": flow_set.target_prompt,
            "source_prompt": flow_set.source_prompt,
            "primitive_prompts": [primitive.to_dict() for primitive in flow_set.get_enabled_primitives()],
            "negative_prompt": flow_set.negative_prompt,
            "aggregation_steps": sorted(aggregation_steps),
            "final_only": primitive_cfg.final_only or mode == "primitive_flow_final_only",
            "aggregate_every_n_steps": primitive_cfg.aggregate_every_n_steps,
            "model_id": self.config.model.model_id,
            "seed": sampler_cfg.seed,
            "num_inference_steps": sampler_cfg.num_inference_steps,
            "height": sampler_cfg.height,
            "width": sampler_cfg.width,
            "guidance_scale": sampler_cfg.guidance_scale,
            "ltp_mode": primitive_cfg.ltp_mode,
            "ltp_radius_ratio": primitive_cfg.ltp_radius_ratio,
            "ltp_enabled": primitive_cfg.ltp_enabled,
            "ltp_fallback_to_velocity": primitive_cfg.fallback_to_velocity_ltp,
            "latent_ltp_available": latent_ltp_available,
            "latent_ltp_probe": latent_ltp_probe,
            "latent_ltp_disabled_reason": latent_ltp_disabled_reason,
            "vfa_temperature": primitive_cfg.vfa_temperature,
            "source_weight": primitive_cfg.source_weight,
            "target_weight": primitive_cfg.target_weight,
            "velocity_clip_ratio": primitive_cfg.velocity_clip_ratio,
            "condition_names": condition_names,
            "condition_roles": condition_roles,
            "condition_texts": condition_texts,
            "condition_weights": condition_weights,
            "target_index": target_index,
            "runtime_config": self.config.to_dict(),
            "custom_loop_audit": {
                "classifier_free_guidance": use_cfg,
                "guidance_scale": sampler_cfg.guidance_scale,
                "sequential_cfg_forward": True,
                "sequential_condition_forward": primitive_cfg.sequential_condition_forward,
                "normal_steps_use": "target_prompt_only",
                "aggregation_reference": "target_prompt",
                "latents_dtype": str(latents.dtype),
                "latents_device": str(latents.device),
                "embedding_shapes": {
                    "conditions": [self._condition_shapes(condition) for condition in condition_embeddings],
                    "target": self._condition_shapes(target_condition),
                },
            },
            "debug_steps": debug_steps,
            "note": (
                "Sparse Primitive Flow Composition directly aggregates complete prompt-conditioned "
                "velocity fields at selected timesteps. It uses no VQA, reward model, external judge, "
                "training, fine-tuning, image feedback, anchor residuals, or primitive schedules."
            ),
        }
        return image, metadata

    def _encode_primitive_conditions(
        self,
        prompt_decomposition: PromptDecomposition,
        primitives: list[PrimitivePrompt],
        mode: str,
        do_classifier_free_guidance: bool,
    ) -> tuple[list[TextCondition], list[str]]:
        if mode == "aim_v2":
            texts = [
                primitive.build_anchor_augmented_text(prompt_decomposition.anchor_prompt)
                for primitive in primitives
            ]
        else:
            texts = [primitive.text for primitive in primitives]
        return (
            [
                self.backend.encode_text_condition(
                    text,
                    prompt_decomposition.negative_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )
                for text in texts
            ],
            texts,
        )

    @staticmethod
    def _scheduler_output_to_latents(output: Any) -> torch.Tensor:
        if hasattr(output, "prev_sample"):
            return output.prev_sample
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        if isinstance(output, torch.Tensor):
            return output
        raise TypeError(f"Could not extract prev_sample from scheduler output type {type(output)!r}")

    def _scheduler_step(self, scheduler: Any, prediction: torch.Tensor, timestep: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        return self._scheduler_output_to_latents(scheduler.step(prediction, timestep, latents, return_dict=True))

    def safe_scheduler_step(self, scheduler: Any, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """Return scheduler prev_sample through a single non-branching step call."""

        return self._scheduler_step(scheduler, model_output, timestep, sample)

    @staticmethod
    def _select_non_aggregation_index(policy: str, reference_index: int, final_index: int) -> int:
        name = policy.lower()
        if name == "reference":
            return reference_index
        if name == "full":
            return final_index
        if name == "base":
            return 0
        raise ValueError(f"Unknown non-aggregation policy: {policy}")

    @staticmethod
    def _condition_shapes(condition: TextCondition) -> dict[str, Any]:
        return {
            "prompt_embeds": list(condition.prompt_embeds.shape),
            "pooled_prompt_embeds": list(condition.pooled_prompt_embeds.shape),
            "negative_prompt_embeds": (
                list(condition.negative_prompt_embeds.shape)
                if condition.negative_prompt_embeds is not None
                else None
            ),
            "negative_pooled_prompt_embeds": (
                list(condition.negative_pooled_prompt_embeds.shape)
                if condition.negative_pooled_prompt_embeds is not None
                else None
            ),
            "device": str(condition.prompt_embeds.device),
            "dtype": str(condition.prompt_embeds.dtype),
        }

    @staticmethod
    def _compatible_prediction(prediction: torch.Tensor, latents: torch.Tensor, name: str) -> torch.Tensor:
        if prediction.shape != latents.shape:
            raise ValueError(
                f"{name} prediction shape must match latents: {tuple(prediction.shape)} vs {tuple(latents.shape)}."
            )
        if prediction.device != latents.device:
            raise ValueError(
                f"{name} prediction device must match latents: {prediction.device} vs {latents.device}."
            )
        if prediction.dtype != latents.dtype:
            prediction = prediction.to(dtype=latents.dtype)
        return prediction

    @staticmethod
    def _snapshot_scheduler_state(scheduler: Any) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for key, value in scheduler.__dict__.items():
            if key.startswith("_") or key in {"timesteps", "sigmas"}:
                try:
                    state[key] = value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                except Exception:
                    state[key] = value
        return state

    @staticmethod
    def _restore_scheduler_state(scheduler: Any, state: dict[str, Any]) -> None:
        for key, value in state.items():
            setattr(scheduler, key, value)

    @staticmethod
    def _scheduler_states_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
        if set(left) != set(right):
            return False
        for key in left:
            l_value = left[key]
            r_value = right[key]
            if isinstance(l_value, torch.Tensor) or isinstance(r_value, torch.Tensor):
                if not (isinstance(l_value, torch.Tensor) and isinstance(r_value, torch.Tensor)):
                    return False
                if l_value.shape != r_value.shape or l_value.dtype != r_value.dtype:
                    return False
                if not torch.equal(l_value.detach().cpu(), r_value.detach().cpu()):
                    return False
            else:
                try:
                    if l_value != r_value:
                        return False
                except Exception:
                    if repr(l_value) != repr(r_value):
                        return False
        return True

    def _probe_latent_ltp_scheduler(
        self,
        scheduler: Any,
        timestep: torch.Tensor,
        latents: torch.Tensor,
    ) -> tuple[bool, dict[str, Any]]:
        """Check whether scheduler.step can be replayed safely for latent LTP."""

        before = self._snapshot_scheduler_state(scheduler)
        prediction = torch.zeros_like(latents)
        try:
            first = self._scheduler_step(scheduler, prediction, timestep, latents)
            after_first = self._snapshot_scheduler_state(scheduler)
            self._restore_scheduler_state(scheduler, before)
            second = self._scheduler_step(scheduler, prediction, timestep, latents)
            after_second = self._snapshot_scheduler_state(scheduler)
            outputs_match = torch.allclose(first, second, rtol=1e-5, atol=1e-5)
            states_match = self._scheduler_states_equivalent(after_first, after_second)
            return outputs_match and states_match, {
                "scheduler_class": scheduler.__class__.__name__,
                "outputs_match": bool(outputs_match),
                "post_step_states_match": bool(states_match),
                "checked": True,
            }
        except Exception as exc:
            return False, {
                "scheduler_class": scheduler.__class__.__name__,
                "checked": True,
                "error": str(exc),
            }
        finally:
            self._restore_scheduler_state(scheduler, before)

    def _compute_anchor_and_candidate_steps(
        self,
        scheduler: Any,
        anchor_pred: torch.Tensor,
        candidate_pred: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        before = self._snapshot_scheduler_state(scheduler)
        x_next_anchor = self._scheduler_step(scheduler, anchor_pred, timestep, latents)
        after_one = self._snapshot_scheduler_state(scheduler)
        self._restore_scheduler_state(scheduler, before)
        x_next_candidate = self._scheduler_step(scheduler, candidate_pred, timestep, latents)
        self._restore_scheduler_state(scheduler, after_one)
        return x_next_anchor, x_next_candidate

    def _base_metadata(self, prompt_decomposition: PromptDecomposition, mode: str) -> dict[str, Any]:
        sampler_cfg = self.config.sampler
        aim_cfg = self.config.aim_flow
        return {
            "prompt_decomposition": prompt_decomposition.to_dict(),
            "mode": mode,
            "model_id": self.config.model.model_id,
            "runtime_config": self.config.to_dict(),
            "seed": sampler_cfg.seed,
            "num_inference_steps": sampler_cfg.num_inference_steps,
            "height": sampler_cfg.height,
            "width": sampler_cfg.width,
            "guidance_scale": sampler_cfg.guidance_scale,
            "lambda_global": aim_cfg.lambda_global,
            "lambda_schedule": aim_cfg.lambda_schedule,
            "conflict_threshold": aim_cfg.vfa.conflict_threshold,
            "velocity_clip_ratio": aim_cfg.vfa.velocity_clip_ratio,
            "ltp_mode": aim_cfg.ltp.mode,
        }
