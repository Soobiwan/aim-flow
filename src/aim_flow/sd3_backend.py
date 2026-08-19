"""Diffusers Stable Diffusion 3 backend for AIM-Flow."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from aim_flow.config import MarginalFlowConfig, PrimitiveFlowConfig, RunConfig
from aim_flow.primitive_flow import build_condition_list
from aim_flow.prompt_schema import ConditionLadder, MarginalFlowPromptSet, PrimitiveFlowSet, PromptDecomposition
from aim_flow.utils import get_device, get_hf_token, get_torch_dtype


def _model_load_error_message(model_id: str, token: str | None) -> str:
    message = (
        f"Failed to load {model_id!r} from Hugging Face. "
        "This benchmark uses the gated SD3 Medium Diffusers model, so the runtime "
        "needs a Hugging Face token whose account has accepted the model license."
    )
    if not token:
        message += " No HF_TOKEN/HUGGINGFACE_TOKEN environment variable is set."
    else:
        message += " HF_TOKEN is set, but the token may be invalid or missing access to the gated model."
    return message


def _load_pipeline_with_fallback(pipeline_cls: Any, model_id: str, kwargs: dict[str, Any]) -> Any:
    """Load a Diffusers pipeline while tolerating older keyword names."""

    try:
        return pipeline_cls.from_pretrained(model_id, **kwargs)
    except TypeError:
        fallback = dict(kwargs)
        changed = False
        if "token" in fallback:
            fallback["use_auth_token"] = fallback.pop("token")
            changed = True
        if "low_cpu_mem_usage" in fallback:
            fallback.pop("low_cpu_mem_usage")
            changed = True
        if not changed:
            raise
        return pipeline_cls.from_pretrained(model_id, **fallback)


@dataclass
class TextCondition:
    """Prompt embedding bundle for a single SD3 condition."""

    prompt: str
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None = None
    negative_pooled_prompt_embeds: torch.Tensor | None = None

    def to(self, device: torch.device, dtype: torch.dtype) -> "TextCondition":
        """Move embeddings to the execution device/dtype used by latents."""

        return TextCondition(
            prompt=self.prompt,
            prompt_embeds=self.prompt_embeds.to(device=device, dtype=dtype),
            pooled_prompt_embeds=self.pooled_prompt_embeds.to(device=device, dtype=dtype),
            negative_prompt_embeds=(
                self.negative_prompt_embeds.to(device=device, dtype=dtype)
                if self.negative_prompt_embeds is not None
                else None
            ),
            negative_pooled_prompt_embeds=(
                self.negative_pooled_prompt_embeds.to(device=device, dtype=dtype)
                if self.negative_pooled_prompt_embeds is not None
                else None
            ),
        )

    def validate(self) -> None:
        """Validate SD3 prompt and pooled embedding ranks/shapes."""

        if self.prompt_embeds.ndim != 3:
            raise ValueError(
                f"SD3 prompt_embeds for {self.prompt!r} must be rank 3 "
                f"[batch, sequence, channels], got shape {tuple(self.prompt_embeds.shape)}."
            )
        if self.pooled_prompt_embeds.ndim != 2:
            raise ValueError(
                f"SD3 pooled_prompt_embeds for {self.prompt!r} must be rank 2 "
                f"[batch, channels], got shape {tuple(self.pooled_prompt_embeds.shape)}."
            )
        if self.prompt_embeds.shape[0] != self.pooled_prompt_embeds.shape[0]:
            raise ValueError(
                "Prompt and pooled prompt batch sizes differ for "
                f"{self.prompt!r}: {self.prompt_embeds.shape[0]} vs {self.pooled_prompt_embeds.shape[0]}."
            )
        if self.negative_prompt_embeds is not None:
            if self.negative_prompt_embeds.shape != self.prompt_embeds.shape:
                raise ValueError(
                    "Negative prompt embeds must match positive prompt embeds for CFG: "
                    f"{tuple(self.negative_prompt_embeds.shape)} vs {tuple(self.prompt_embeds.shape)}."
                )
            if self.negative_pooled_prompt_embeds is None:
                raise ValueError("negative_pooled_prompt_embeds is required when negative_prompt_embeds is set.")
            if self.negative_pooled_prompt_embeds.shape != self.pooled_prompt_embeds.shape:
                raise ValueError(
                    "Negative pooled prompt embeds must match positive pooled prompt embeds for CFG: "
                    f"{tuple(self.negative_pooled_prompt_embeds.shape)} vs {tuple(self.pooled_prompt_embeds.shape)}."
                )


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Could not extract tensor from transformer output type {type(output)!r}")


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    """Read from Diffusers FrozenDict/config objects and simple test doubles."""

    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def get_sd3_transformer_prediction(
    pipe: Any,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance_scale: float | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    negative_pooled_prompt_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Call the SD3 transformer and return a prediction tensor.

    Diffusers SD3 versions have changed minor argument details over time. This
    helper centralizes the call and raises a focused error if the installed
    version no longer matches the expected public pipeline pattern.
    """

    attempted: list[str] = []
    try:
        timestep_in = timestep
        if timestep_in.ndim == 0:
            timestep_in = timestep_in.expand(latents.shape[0])
        if prompt_embeds.shape[0] != latents.shape[0]:
            raise ValueError(
                f"Prompt batch size must match latent batch size: {prompt_embeds.shape[0]} vs {latents.shape[0]}."
            )
        if pooled_prompt_embeds.shape[0] != latents.shape[0]:
            raise ValueError(
                f"Pooled prompt batch size must match latent batch size: "
                f"{pooled_prompt_embeds.shape[0]} vs {latents.shape[0]}."
            )

        common_kwargs = {
            "timestep": timestep_in,
            "return_dict": False,
        }
        forward = getattr(pipe.transformer, "forward", pipe.transformer)
        signature = inspect.signature(forward)
        if "joint_attention_kwargs" in signature.parameters:
            common_kwargs["joint_attention_kwargs"] = None
        if "guidance" in signature.parameters:
            guidance_value = 1.0 if guidance_scale is None else float(guidance_scale)
            common_kwargs["guidance"] = torch.full(
                (latents.shape[0],),
                guidance_value,
                device=latents.device,
                dtype=torch.float32,
            )

        def call_transformer(
            latent_model_input: torch.Tensor,
            embeds: torch.Tensor,
            pooled_embeds: torch.Tensor,
        ) -> torch.Tensor:
            call_errors: list[str] = []
            output = None
            for latent_arg_name in ("hidden_states", "sample"):
                kwargs = dict(common_kwargs)
                kwargs[latent_arg_name] = latent_model_input
                kwargs["encoder_hidden_states"] = embeds
                kwargs["pooled_projections"] = pooled_embeds
                attempted.append(
                    f"{latent_arg_name}=latents, timestep=timestep, "
                    "encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds"
                )
                try:
                    output = pipe.transformer(**kwargs)
                    break
                except TypeError as exc:
                    call_errors.append(f"{latent_arg_name}: {exc}")
            if output is None:
                raise TypeError("; ".join(call_errors))
            prediction = _first_tensor(output)
            if prediction.shape != latents.shape:
                raise ValueError(
                    f"SD3 transformer prediction shape must match latents: "
                    f"{tuple(prediction.shape)} vs {tuple(latents.shape)}."
                )
            return prediction

        if guidance_scale and guidance_scale > 1.0 and negative_prompt_embeds is not None:
            if negative_pooled_prompt_embeds is None:
                raise ValueError("negative_pooled_prompt_embeds is required for CFG prediction.")
            if negative_prompt_embeds.shape != prompt_embeds.shape:
                raise ValueError(
                    f"Negative prompt embeds shape mismatch: "
                    f"{tuple(negative_prompt_embeds.shape)} vs {tuple(prompt_embeds.shape)}."
                )
            if negative_pooled_prompt_embeds.shape != pooled_prompt_embeds.shape:
                raise ValueError(
                    f"Negative pooled prompt embeds shape mismatch: "
                    f"{tuple(negative_pooled_prompt_embeds.shape)} vs {tuple(pooled_prompt_embeds.shape)}."
                )
            # Sequential CFG avoids doubling transformer batch memory on 16 GB Kaggle GPUs.
            noise_uncond = call_transformer(latents, negative_prompt_embeds, negative_pooled_prompt_embeds)
            noise_text = call_transformer(latents, prompt_embeds, pooled_prompt_embeds)
            return noise_uncond + float(guidance_scale) * (noise_text - noise_uncond)

        return call_transformer(latents, prompt_embeds, pooled_prompt_embeds)
    except Exception as exc:
        raise RuntimeError(
            "Failed to call the SD3 transformer with any supported Diffusers signature. "
            f"Attempted signatures: {attempted}. Original error: {exc}"
        ) from exc


def validate_torch_cuda_build_for_current_gpu() -> None:
    """Fail before model download if the installed Torch wheel cannot run on the GPU."""

    if not torch.cuda.is_available():
        return
    capability = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = set(torch.cuda.get_arch_list())
    if supported_arches and required_arch not in supported_arches:
        device_name = torch.cuda.get_device_name(0)
        raise RuntimeError(
            f"Installed PyTorch cannot execute CUDA kernels on {device_name} ({required_arch}). "
            f"This PyTorch build supports {sorted(supported_arches)}. On Kaggle P100, install a "
            "Pascal-compatible wheel before loading SD3, for example:\n"
            "  pip uninstall -y torch torchvision torchaudio\n"
            "  pip install --no-cache-dir --force-reinstall torch==2.4.1+cu118 "
            "--index-url https://download.pytorch.org/whl/cu118\n"
            "Then restart the Kaggle runtime/kernel and rerun the GPU check cell."
        )


class SD3Backend:
    """Thin wrapper around Diffusers StableDiffusion3Pipeline."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.pipe: Any | None = None
        self.device = get_device()
        self.dtype = get_torch_dtype(config.model.dtype)
        self._cpu_offload_deferred = False
        self._cpu_offload_active = False

    def load(self) -> "SD3Backend":
        """Load Stable Diffusion 3 Medium through Diffusers."""

        validate_torch_cuda_build_for_current_gpu()
        try:
            from diffusers import StableDiffusion3Pipeline
        except Exception as exc:
            raise RuntimeError("diffusers with StableDiffusion3Pipeline is required for SD3 inference.") from exc

        token = get_hf_token()
        kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }
        if not self.config.model.load_t5_text_encoder:
            kwargs["text_encoder_3"] = None
            kwargs["tokenizer_3"] = None
        variant = os.environ.get("SD3_MODEL_VARIANT")
        if variant:
            kwargs["variant"] = variant
        if token:
            kwargs["token"] = token

        try:
            self.pipe = _load_pipeline_with_fallback(StableDiffusion3Pipeline, self.config.model.model_id, kwargs)
        except Exception as exc:
            raise RuntimeError(_model_load_error_message(self.config.model.model_id, token)) from exc

        if self.config.model.defer_model_cpu_offload and (
            self.config.model.enable_model_cpu_offload or self.config.model.enable_sequential_cpu_offload
        ):
            self._cpu_offload_deferred = True
        elif self.config.model.enable_sequential_cpu_offload and hasattr(self.pipe, "enable_sequential_cpu_offload"):
            self.pipe.enable_sequential_cpu_offload()
            self._cpu_offload_active = True
        elif self.config.model.enable_model_cpu_offload:
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
                self._cpu_offload_active = True
            else:
                self.pipe.to(self.device)
        else:
            self.pipe.to(self.device)

        if self.config.model.enable_attention_slicing and hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()

        if self.config.model.enable_vae_slicing and hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)
        return self

    def require_pipe(self) -> Any:
        if self.pipe is None:
            raise RuntimeError("SD3Backend.load() must be called before inference.")
        return self.pipe

    def activate_model_cpu_offload(self) -> None:
        """Enable deferred model CPU offload after prompt embeddings are prepared."""

        pipe = self.require_pipe()
        if not self._cpu_offload_deferred or self._cpu_offload_active:
            return
        if self.config.model.enable_sequential_cpu_offload and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
            self._cpu_offload_active = True
        elif hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            self._cpu_offload_active = True
        else:
            pipe.to(self.device)
        self._cpu_offload_deferred = False

    @property
    def execution_device(self) -> torch.device:
        """Return the pipeline execution device, respecting Diffusers CPU offload hooks."""

        pipe = self.require_pipe()
        if self._cpu_offload_deferred and not self._cpu_offload_active:
            return torch.device("cpu")
        execution_device = getattr(pipe, "_execution_device", None) or self.device
        return torch.device(execution_device)

    def validate_custom_sampling_compatibility(self) -> None:
        """Fail early if the installed Diffusers SD3 API is incompatible."""

        pipe = self.require_pipe()
        required_attrs = ["transformer", "scheduler", "vae", "image_processor", "encode_prompt", "prepare_latents"]
        missing_attrs = [name for name in required_attrs if not hasattr(pipe, name)]
        if missing_attrs:
            raise RuntimeError(
                "Installed Diffusers StableDiffusion3Pipeline is missing required attributes for "
                f"AIM-Flow custom sampling: {missing_attrs}."
            )

        encode_params = inspect.signature(pipe.encode_prompt).parameters
        required_encode = {"prompt", "prompt_2", "prompt_3", "device", "num_images_per_prompt"}
        missing_encode = sorted(required_encode.difference(encode_params))
        if missing_encode:
            raise RuntimeError(
                "Installed StableDiffusion3Pipeline.encode_prompt has an incompatible signature. "
                f"Missing parameters: {missing_encode}."
            )

        prepare_params = inspect.signature(pipe.prepare_latents).parameters
        required_prepare = {"batch_size", "num_channels_latents", "height", "width", "dtype", "device", "generator"}
        missing_prepare = sorted(required_prepare.difference(prepare_params))
        if missing_prepare:
            raise RuntimeError(
                "Installed StableDiffusion3Pipeline.prepare_latents has an incompatible signature. "
                f"Missing parameters: {missing_prepare}."
            )

        forward = getattr(pipe.transformer, "forward", pipe.transformer)
        transformer_params = inspect.signature(forward).parameters
        required_common = {"timestep", "encoder_hidden_states", "pooled_projections"}
        has_latent_arg = "hidden_states" in transformer_params or "sample" in transformer_params
        missing_transformer = sorted(required_common.difference(transformer_params))
        if missing_transformer or not has_latent_arg:
            raise RuntimeError(
                "Installed SD3 transformer has an incompatible forward signature for AIM-Flow. "
                f"Missing parameters: {missing_transformer}; requires either `hidden_states` or `sample`."
            )

    def validate_image_size(self, height: int, width: int) -> None:
        """Mirror SD3's height/width divisibility requirement."""

        divisor = self.vae_scale_factor * self.transformer_patch_size
        if height % divisor != 0 or width % divisor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {divisor} for this SD3 pipeline, "
                f"but got height={height}, width={width}."
            )

    def encode_text_condition(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        device: torch.device | None = None,
        do_classifier_free_guidance: bool = False,
    ) -> TextCondition:
        """Encode one prompt using the pipeline's public encode_prompt method."""

        pipe = self.require_pipe()
        self.validate_custom_sampling_compatibility()
        target_device = device or self.execution_device
        try:
            encode_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "prompt_2": None,
                "prompt_3": None,
                "negative_prompt": negative_prompt,
                "negative_prompt_2": None,
                "negative_prompt_3": None,
                "do_classifier_free_guidance": do_classifier_free_guidance,
                "device": target_device,
                "num_images_per_prompt": 1,
            }
            if "max_sequence_length" in inspect.signature(pipe.encode_prompt).parameters:
                encode_kwargs["max_sequence_length"] = 256
            encoded = pipe.encode_prompt(
                **encode_kwargs,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Failed to encode prompt with StableDiffusion3Pipeline.encode_prompt. "
                "The installed Diffusers signature may have changed."
            ) from exc

        if isinstance(encoded, dict):
            prompt_embeds = encoded["prompt_embeds"]
            pooled_prompt_embeds = encoded["pooled_prompt_embeds"]
            negative_prompt_embeds = encoded.get("negative_prompt_embeds")
            negative_pooled_prompt_embeds = encoded.get("negative_pooled_prompt_embeds")
        else:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encoded[:4]

        condition = TextCondition(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
        condition.validate()
        return condition

    def encode_single_prompt(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Backward-compatible dict wrapper around encode_text_condition."""

        condition = self.encode_text_condition(prompt, negative_prompt, device)
        return {
            "prompt_embeds": condition.prompt_embeds,
            "pooled_prompt_embeds": condition.pooled_prompt_embeds,
        }

    def encode_prompts(self, prompt_decomposition: PromptDecomposition) -> dict[str, Any]:
        """Encode anchor and primitive prompts once for custom sampling."""

        negative = prompt_decomposition.negative_prompt
        return {
            "anchor": self.encode_single_prompt(prompt_decomposition.anchor_prompt, negative),
            "primitives": [
                self.encode_single_prompt(primitive.text, negative)
                for primitive in prompt_decomposition.primitive_prompts
            ],
        }

    def encode_aim_flow_conditions(self, prompt_decomposition: PromptDecomposition) -> dict[str, Any]:
        """Encode full, anchor, and anchor-augmented primitive conditions for AIM-Flow v2."""

        negative = prompt_decomposition.negative_prompt
        use_cfg = self.config.sampler.guidance_scale > 1.0
        enabled_primitives = prompt_decomposition.get_enabled_primitives()
        anchor_augmented_texts = prompt_decomposition.build_anchor_augmented_primitive_prompts()
        return {
            "full": self.encode_text_condition(prompt_decomposition.full_prompt, negative, do_classifier_free_guidance=use_cfg),
            "anchor": self.encode_text_condition(prompt_decomposition.anchor_prompt, negative, do_classifier_free_guidance=use_cfg),
            "primitives": [
                self.encode_text_condition(text, negative, do_classifier_free_guidance=use_cfg)
                for text in anchor_augmented_texts
            ],
            "primitive_texts": anchor_augmented_texts,
            "primitive_original_texts": [primitive.text for primitive in enabled_primitives],
        }

    def encode_ladder_conditions(self, condition_ladder: ConditionLadder) -> dict[str, Any]:
        """Encode complete scene prompts for LadderFlow v3."""

        negative = condition_ladder.negative_prompt
        use_cfg = self.config.sampler.guidance_scale > 1.0
        conditions = condition_ladder.get_enabled_conditions()[: self.config.ladder_flow.max_conditions]
        return {
            "conditions": [
                self.encode_text_condition(
                    condition.text,
                    negative,
                    do_classifier_free_guidance=use_cfg,
                )
                for condition in conditions
            ],
            "condition_texts": [condition.text for condition in conditions],
            "condition_names": [condition.name for condition in conditions],
            "condition_types": [condition.type for condition in conditions],
        }

    def encode_primitive_flow_conditions(
        self,
        flow_set: PrimitiveFlowSet,
        config: PrimitiveFlowConfig,
    ) -> dict[str, Any]:
        """Encode source, primitive, and target prompts for sparse primitive flow."""

        negative = flow_set.negative_prompt
        use_cfg = self.config.sampler.guidance_scale > 1.0
        condition_dicts = build_condition_list(
            flow_set,
            include_source=config.include_source_flow,
            include_target=config.include_target_flow,
            source_weight=config.source_weight,
            target_weight=config.target_weight,
            uniform_weights=config.uniform_condition_weights,
            max_primitives=config.max_primitives,
        )
        target_indices = [index for index, item in enumerate(condition_dicts) if item["role"] == "target"]
        target_index = target_indices[0] if target_indices else None
        target_condition = self.encode_text_condition(
            flow_set.target_prompt,
            negative,
            do_classifier_free_guidance=use_cfg,
        )
        conditions = [
            self.encode_text_condition(
                item["text"],
                negative,
                do_classifier_free_guidance=use_cfg,
            )
            for item in condition_dicts
        ]
        return {
            "conditions": conditions,
            "condition_texts": [item["text"] for item in condition_dicts],
            "condition_names": [item["name"] for item in condition_dicts],
            "condition_roles": [item["role"] for item in condition_dicts],
            "condition_weights": [float(item["weight"]) for item in condition_dicts],
            "target_index": target_index,
            "target_condition": target_condition if target_index is None else conditions[target_index],
        }

    def encode_marginal_flow_conditions(
        self,
        prompt_set: MarginalFlowPromptSet,
        config: MarginalFlowConfig,
    ) -> dict[str, Any]:
        """Encode the full target and explicit target-ablation prompts."""

        negative = prompt_set.negative_prompt
        use_cfg = self.config.sampler.guidance_scale > 1.0
        primitives = prompt_set.get_enabled_primitives()[: config.max_primitives]
        if not primitives:
            raise ValueError("Marginal Flow requires at least one enabled primitive.")
        target_condition = self.encode_text_condition(
            prompt_set.target_prompt,
            negative,
            do_classifier_free_guidance=use_cfg,
        )
        ablated_conditions = [
            self.encode_text_condition(
                primitive.ablated_prompt,
                negative,
                do_classifier_free_guidance=use_cfg,
            )
            for primitive in primitives
        ]
        return {
            "target_condition": target_condition,
            "ablated_conditions": ablated_conditions,
            "ablated_prompts": [primitive.ablated_prompt for primitive in primitives],
            "primitive_names": [primitive.name or f"primitive_{index}" for index, primitive in enumerate(primitives)],
            "primitive_descriptions": [primitive.primitive for primitive in primitives],
            "primitives": primitives,
        }

    def predict_with_condition(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        condition: TextCondition,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        """Predict SD3 velocity/noise for a latent under one text condition."""

        pipe = self.require_pipe()
        condition = condition.to(device=latents.device, dtype=latents.dtype)
        condition.validate()
        return get_sd3_transformer_prediction(
            pipe=pipe,
            latents=latents,
            timestep=timestep,
            prompt_embeds=condition.prompt_embeds,
            pooled_prompt_embeds=condition.pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            negative_prompt_embeds=condition.negative_prompt_embeds,
            negative_pooled_prompt_embeds=condition.negative_pooled_prompt_embeds,
        )

    def generate_base(
        self,
        prompt: str,
        negative_prompt: str | None,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
    ) -> Image.Image:
        """Generate a normal full-prompt SD3 baseline image."""

        pipe = self.require_pipe()
        self.validate_image_size(height, width)
        self.activate_model_cpu_offload()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]

    def generate_anchor(
        self,
        anchor_prompt: str,
        negative_prompt: str | None,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
    ) -> Image.Image:
        """Generate a regular SD3 image from only the anchor prompt."""

        return self.generate_base(
            prompt=anchor_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )

    def set_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        """Set scheduler timesteps, including SD3 dynamic-shift kwargs when present."""

        pipe = self.require_pipe()
        self.validate_custom_sampling_compatibility()
        scheduler_signature = inspect.signature(pipe.scheduler.set_timesteps)
        kwargs: dict[str, Any] = {"device": self.execution_device}
        scheduler_config = getattr(pipe.scheduler, "config", None)
        if _config_get(scheduler_config, "use_dynamic_shifting", None):
            if "mu" not in scheduler_signature.parameters:
                raise RuntimeError(
                    "The SD3 scheduler config enables dynamic shifting, but scheduler.set_timesteps "
                    "does not accept `mu`. This Diffusers scheduler API is incompatible with AIM-Flow."
                )
            latent_h = self.config.sampler.height // self.vae_scale_factor
            latent_w = self.config.sampler.width // self.vae_scale_factor
            image_seq_len = (latent_h // self.transformer_patch_size) * (latent_w // self.transformer_patch_size)
            kwargs["mu"] = self._calculate_shift(image_seq_len)
        try:
            pipe.scheduler.set_timesteps(num_inference_steps, **kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "Failed to call scheduler.set_timesteps like Diffusers SD3 does. Expected "
                "set_timesteps(num_inference_steps, device=..., optional mu=...). "
                f"Original error: {exc}"
            ) from exc
        return pipe.scheduler.timesteps

    @property
    def vae_scale_factor(self) -> int:
        pipe = self.require_pipe()
        if hasattr(pipe, "vae_scale_factor"):
            return int(pipe.vae_scale_factor)
        return 8

    @property
    def transformer_patch_size(self) -> int:
        pipe = self.require_pipe()
        return int(getattr(pipe.transformer.config, "patch_size", 2))

    def _calculate_shift(self, image_seq_len: int) -> float:
        """Match Diffusers SD3 dynamic timestep shift defaults when available."""

        pipe = self.require_pipe()
        scheduler_config = getattr(pipe.scheduler, "config", None)
        base_image_seq_len = float(_config_get(scheduler_config, "base_image_seq_len", 256))
        max_image_seq_len = float(_config_get(scheduler_config, "max_image_seq_len", 4096))
        base_shift = float(_config_get(scheduler_config, "base_shift", 0.5))
        max_shift = float(_config_get(scheduler_config, "max_shift", 1.16))
        slope = (max_shift - base_shift) / max(max_image_seq_len - base_image_seq_len, 1.0)
        intercept = base_shift - slope * base_image_seq_len
        return image_seq_len * slope + intercept

    def prepare_latents(self, seed: int, height: int, width: int) -> torch.Tensor:
        """Prepare initial latent noise using the pipeline helper when possible."""

        pipe = self.require_pipe()
        self.validate_custom_sampling_compatibility()
        self.validate_image_size(height, width)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        num_channels = int(getattr(pipe.transformer.config, "in_channels", 16))

        if hasattr(pipe, "prepare_latents"):
            try:
                return pipe.prepare_latents(
                    batch_size=1,
                    num_channels_latents=num_channels,
                    height=height,
                    width=width,
                    dtype=self.dtype,
                    device=self.execution_device,
                    generator=generator,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "StableDiffusion3Pipeline.prepare_latents signature changed; "
                    "AIM-Flow cannot safely initialize custom sampling latents."
                ) from exc

        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        return torch.randn(
            (1, num_channels, latent_h, latent_w),
            generator=generator,
            dtype=self.dtype,
        ).to(self.execution_device)

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Decode final latents into a PIL image."""

        pipe = self.require_pipe()
        vae_config = getattr(pipe.vae, "config", None)
        scaling_factor = float(_config_get(vae_config, "scaling_factor", 1.0))
        shift_factor = float(_config_get(vae_config, "shift_factor", 0.0))
        latents = (latents / scaling_factor) + shift_factor
        with torch.no_grad():
            image = pipe.vae.decode(latents, return_dict=False)[0]
        image = image.detach()
        image = pipe.image_processor.postprocess(image, output_type="pil")
        if hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        return image[0]

    @staticmethod
    def save_image(image: Image.Image, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
