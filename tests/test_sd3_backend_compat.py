from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from aim_flow.config import RunConfig
from aim_flow.sd3_backend import SD3Backend, get_sd3_transformer_prediction


class FakeTransformer:
    config = SimpleNamespace(in_channels=16, patch_size=2)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        pooled_projections,
        joint_attention_kwargs=None,
        return_dict=False,
    ):
        assert timestep.shape[0] == hidden_states.shape[0]
        assert encoder_hidden_states.shape[0] == hidden_states.shape[0]
        assert pooled_projections.shape[0] == hidden_states.shape[0]
        return (torch.ones_like(hidden_states),)

    __call__ = forward


class BadTransformer:
    config = SimpleNamespace(in_channels=16, patch_size=2)

    def forward(self, hidden_states):
        return (hidden_states,)

    __call__ = forward


class FakeScheduler:
    def __init__(self) -> None:
        self.config = {
            "use_dynamic_shifting": True,
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.16,
        }
        self.timesteps = torch.tensor([])
        self.last_mu = None
        self.last_device = None

    def set_timesteps(self, num_inference_steps, device=None, mu=None):
        self.last_mu = mu
        self.last_device = device
        self.timesteps = torch.arange(num_inference_steps, device=device)


class FakeVAE:
    config = SimpleNamespace(scaling_factor=2.0, shift_factor=0.5, block_out_channels=[1, 2, 3, 4])

    def decode(self, latents, return_dict=False):
        self.last_latents = latents
        return (torch.zeros(1, 3, 8, 8),)


class FakeImageProcessor:
    def postprocess(self, image, output_type="pil"):
        return [Image.new("RGB", (8, 8), "white")]


class FakePipe:
    def __init__(self, transformer=None):
        self.transformer = transformer or FakeTransformer()
        self.scheduler = FakeScheduler()
        self.vae = FakeVAE()
        self.image_processor = FakeImageProcessor()
        self.vae_scale_factor = 8
        self._execution_device = torch.device("cpu")
        self.hooks_freed = False

    def encode_prompt(
        self,
        prompt,
        prompt_2,
        prompt_3,
        device=None,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        max_sequence_length=256,
    ):
        return (
            torch.zeros(1, 4, 6, device=device),
            None,
            torch.zeros(1, 3, device=device),
            None,
        )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        return torch.zeros(shape, dtype=dtype, device=device)

    def maybe_free_model_hooks(self):
        self.hooks_freed = True


def backend_with_pipe(pipe: FakePipe) -> SD3Backend:
    backend = SD3Backend(RunConfig())
    backend.pipe = pipe
    return backend


def test_transformer_prediction_matches_sd3_argument_shapes() -> None:
    pipe = FakePipe()
    latents = torch.zeros(1, 16, 8, 8)
    pred = get_sd3_transformer_prediction(
        pipe=pipe,
        latents=latents,
        timestep=torch.tensor(1.0),
        prompt_embeds=torch.zeros(1, 4, 6),
        pooled_prompt_embeds=torch.zeros(1, 3),
    )
    assert pred.shape == latents.shape


def test_incompatible_transformer_fails_precisely() -> None:
    pipe = FakePipe(transformer=BadTransformer())
    with pytest.raises(RuntimeError, match="pooled_projections"):
        get_sd3_transformer_prediction(
            pipe=pipe,
            latents=torch.zeros(1, 16, 8, 8),
            timestep=torch.tensor(1.0),
            prompt_embeds=torch.zeros(1, 4, 6),
            pooled_prompt_embeds=torch.zeros(1, 3),
        )


def test_set_timesteps_uses_dynamic_shift_mu() -> None:
    pipe = FakePipe()
    backend = backend_with_pipe(pipe)
    timesteps = backend.set_timesteps(4)
    assert len(timesteps) == 4
    assert pipe.scheduler.last_mu is not None
    assert pipe.scheduler.last_device == torch.device("cpu")


def test_prepare_latents_uses_pipeline_shape_and_dtype() -> None:
    backend = backend_with_pipe(FakePipe())
    latents = backend.prepare_latents(seed=7, height=512, width=512)
    assert latents.shape == (1, 16, 64, 64)
    assert latents.dtype == torch.float16


def test_decode_latents_matches_sd3_scaling_and_frees_hooks() -> None:
    pipe = FakePipe()
    backend = backend_with_pipe(pipe)
    image = backend.decode_latents(torch.ones(1, 16, 8, 8))
    assert image.size == (8, 8)
    assert torch.allclose(pipe.vae.last_latents, torch.ones(1, 16, 8, 8))
    assert pipe.hooks_freed

