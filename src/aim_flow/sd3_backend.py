"""Diffusers Stable Diffusion 3 backend for AIM-Flow."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from aim_flow.config import RunConfig
from aim_flow.prompt_schema import PromptDecomposition
from aim_flow.utils import get_device, get_hf_token, get_torch_dtype


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

    try:
        timestep_in = timestep
        if timestep_in.ndim == 0:
            timestep_in = timestep_in.expand(latents.shape[0])

        if guidance_scale and guidance_scale > 1.0 and negative_prompt_embeds is not None:
            if negative_pooled_prompt_embeds is None:
                raise ValueError("negative_pooled_prompt_embeds is required for CFG prediction.")
            latent_model_input = torch.cat([latents, latents], dim=0)
            timestep_in = torch.cat([timestep_in, timestep_in], dim=0)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        else:
            latent_model_input = latents

        kwargs = {
            "hidden_states": latent_model_input,
            "timestep": timestep_in,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "return_dict": False,
        }

        forward = getattr(pipe.transformer, "forward", pipe.transformer)
        signature = inspect.signature(forward)
        required = {"hidden_states", "timestep", "encoder_hidden_states", "pooled_projections"}
        missing = sorted(required.difference(signature.parameters))
        if missing:
            raise TypeError(f"SD3 transformer.forward is missing required parameters: {missing}")
        if "joint_attention_kwargs" in signature.parameters:
            kwargs["joint_attention_kwargs"] = None

        noise_pred = _first_tensor(pipe.transformer(**kwargs))
        if guidance_scale and guidance_scale > 1.0 and negative_prompt_embeds is not None:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        return noise_pred
    except Exception as exc:
        raise RuntimeError(
            "Failed to call the SD3 transformer with the expected Diffusers signature. "
            "AIM-Flow expects a StableDiffusion3Pipeline whose transformer accepts "
            "hidden_states, timestep, encoder_hidden_states, and pooled_projections. "
            f"Original error: {exc}"
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

    def load(self) -> "SD3Backend":
        """Load Stable Diffusion 3 Medium through Diffusers."""

        validate_torch_cuda_build_for_current_gpu()
        try:
            from diffusers import StableDiffusion3Pipeline
        except Exception as exc:
            raise RuntimeError("diffusers with StableDiffusion3Pipeline is required for SD3 inference.") from exc

        token = get_hf_token()
        kwargs: dict[str, Any] = {"torch_dtype": self.dtype}
        if token:
            kwargs["token"] = token

        try:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(self.config.model.model_id, **kwargs)
        except TypeError:
            if token:
                kwargs.pop("token", None)
                kwargs["use_auth_token"] = token
            self.pipe = StableDiffusion3Pipeline.from_pretrained(self.config.model.model_id, **kwargs)

        if self.config.model.enable_model_cpu_offload:
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
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

    @property
    def execution_device(self) -> torch.device:
        """Return the pipeline execution device, respecting Diffusers CPU offload hooks."""

        pipe = self.require_pipe()
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
        required_transformer = {"hidden_states", "timestep", "encoder_hidden_states", "pooled_projections"}
        missing_transformer = sorted(required_transformer.difference(transformer_params))
        if missing_transformer:
            raise RuntimeError(
                "Installed SD3 transformer has an incompatible forward signature for AIM-Flow. "
                f"Missing parameters: {missing_transformer}."
            )

    def validate_image_size(self, height: int, width: int) -> None:
        """Mirror SD3's height/width divisibility requirement."""

        divisor = self.vae_scale_factor * self.transformer_patch_size
        if height % divisor != 0 or width % divisor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {divisor} for this SD3 pipeline, "
                f"but got height={height}, width={width}."
            )

    def encode_single_prompt(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
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
                "do_classifier_free_guidance": False,
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
        else:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encoded[:4]
            del negative_prompt_embeds, negative_pooled_prompt_embeds

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
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
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")
        if hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        return image[0]

    @staticmethod
    def save_image(image: Image.Image, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
