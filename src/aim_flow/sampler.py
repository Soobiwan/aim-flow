"""AIM-Flow v2 sampler."""

from __future__ import annotations

import copy
from typing import Any

import torch
from PIL import Image
from tqdm.auto import tqdm

from aim_flow.aggregation import aggregate_predictions, aggregate_vfa
from aim_flow.config import RunConfig
from aim_flow.ltp import apply_latent_ltp, apply_velocity_ltp
from aim_flow.prompt_schema import PrimitivePrompt, PromptDecomposition
from aim_flow.schedules import get_lambda_schedule_weight, get_ltp_radius_ratio, get_schedule_weight
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
