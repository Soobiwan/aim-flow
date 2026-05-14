"""AIM-Flow custom sampler."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from tqdm.auto import tqdm

from aim_flow.aggregation import aggregate_predictions
from aim_flow.config import RunConfig
from aim_flow.prompt_schema import PromptDecomposition
from aim_flow.schedules import get_schedule_weight
from aim_flow.sd3_backend import SD3Backend, get_sd3_transformer_prediction


class AIMFlowSampler:
    """Generate images using anchor-informed modular flow guidance."""

    def __init__(self, backend: SD3Backend, run_config: RunConfig):
        self.backend = backend
        self.config = run_config

    def generate(self, prompt_decomposition: PromptDecomposition, mode: str = "full") -> tuple[Image.Image, dict[str, Any]]:
        """Generate one image and metadata for the selected mode."""

        prompt_decomposition.validate()
        if mode == "anchor":
            return self._generate_anchor(prompt_decomposition)
        if mode not in {"naive", "full"}:
            raise ValueError("AIMFlowSampler.generate supports modes: anchor, naive, full")
        return self._generate_custom(prompt_decomposition, mode)

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
        primitives = prompt_decomposition.primitive_prompts[: aim_cfg.max_primitives]
        if not primitives:
            raise ValueError("AIM-Flow requires at least one primitive prompt.")

        self.backend.validate_custom_sampling_compatibility()
        self.backend.validate_image_size(sampler_cfg.height, sampler_cfg.width)
        encoded_anchor = self.backend.encode_single_prompt(
            prompt_decomposition.anchor_prompt,
            prompt_decomposition.negative_prompt,
        )
        encoded_primitives = [
            self.backend.encode_single_prompt(primitive.text, prompt_decomposition.negative_prompt)
            for primitive in primitives
        ]
        if hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()

        latents = self.backend.prepare_latents(
            seed=sampler_cfg.seed,
            height=sampler_cfg.height,
            width=sampler_cfg.width,
        )
        timesteps = self.backend.set_timesteps(sampler_cfg.num_inference_steps)
        debug_steps: list[dict[str, Any]] = []

        autocast_device = self.backend.execution_device.type
        use_autocast = autocast_device == "cuda" and self.backend.dtype in {torch.float16, torch.bfloat16}

        with torch.no_grad():
            for step_index, timestep in enumerate(tqdm(timesteps, desc=f"AIM-Flow {mode}")):
                with torch.autocast(device_type=autocast_device, dtype=self.backend.dtype, enabled=use_autocast):
                    anchor_pred = get_sd3_transformer_prediction(
                        pipe=pipe,
                        latents=latents,
                        timestep=timestep,
                        prompt_embeds=encoded_anchor["prompt_embeds"],
                        pooled_prompt_embeds=encoded_anchor["pooled_prompt_embeds"],
                        guidance_scale=None,
                    )

                    primitive_preds = []
                    for encoded in encoded_primitives:
                        primitive_preds.append(
                            get_sd3_transformer_prediction(
                                pipe=pipe,
                                latents=latents,
                                timestep=timestep,
                                prompt_embeds=encoded["prompt_embeds"],
                                pooled_prompt_embeds=encoded["pooled_prompt_embeds"],
                                guidance_scale=None,
                            )
                        )

                    schedule_weights = [
                        get_schedule_weight(primitive.schedule, step_index, len(timesteps))
                        for primitive in primitives
                    ]
                    aggregated_pred, debug = aggregate_predictions(
                        anchor_pred=anchor_pred,
                        primitive_preds=primitive_preds,
                        primitive_weights=[primitive.weight for primitive in primitives],
                        primitive_schedule_weights=schedule_weights,
                        lambda_global=aim_cfg.lambda_global,
                        conflict_threshold=aim_cfg.conflict_threshold,
                        norm_clip_ratio=aim_cfg.norm_clip_ratio,
                        mode=mode,
                    )

                latents_dtype = latents.dtype
                step_output = pipe.scheduler.step(aggregated_pred, timestep, latents, return_dict=False)
                latents = step_output[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
                debug_steps.append(
                    {
                        "step_index": step_index,
                        "timestep": float(timestep.detach().float().cpu().item()),
                        "schedule_weights": schedule_weights,
                        **debug,
                    }
                )

                del anchor_pred, primitive_preds, aggregated_pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        image = self.backend.decode_latents(latents)
        metadata = self._base_metadata(prompt_decomposition, mode)
        metadata["primitive_prompts_used"] = [primitive.to_dict() for primitive in primitives]
        metadata["debug_steps"] = debug_steps
        metadata["note"] = (
            "AIM-Flow v1 uses conditional SD3 predictions in the custom loop. "
            "TODO: add exact CFG-compatible multi-condition aggregation."
        )
        return image, metadata

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
            "conflict_threshold": aim_cfg.conflict_threshold,
            "norm_clip_ratio": aim_cfg.norm_clip_ratio,
        }
