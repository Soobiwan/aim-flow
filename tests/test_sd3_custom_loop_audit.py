import torch

from aim_flow.sampler import AIMFlowSampler
from aim_flow.sd3_backend import TextCondition


class ReplayableScheduler:
    def __init__(self):
        self._step_index = 0
        self.timesteps = torch.tensor([1.0])
        self.sigmas = torch.tensor([1.0])

    def step(self, prediction, timestep, latents, return_dict=True):
        self._step_index += 1

        class Output:
            prev_sample = latents - prediction

        return Output()


class NonReplayableScheduler(ReplayableScheduler):
    calls = 0

    def step(self, prediction, timestep, latents, return_dict=True):
        NonReplayableScheduler.calls += 1
        self._step_index += 1

        class Output:
            prev_sample = latents + float(NonReplayableScheduler.calls)

        return Output()


def test_text_condition_validates_positive_and_negative_shapes():
    condition = TextCondition(
        prompt="anchor",
        prompt_embeds=torch.zeros(1, 77, 4096),
        pooled_prompt_embeds=torch.zeros(1, 2048),
        negative_prompt_embeds=torch.zeros(1, 77, 4096),
        negative_pooled_prompt_embeds=torch.zeros(1, 2048),
    )

    condition.validate()


def test_text_condition_rejects_bad_negative_shape():
    condition = TextCondition(
        prompt="anchor",
        prompt_embeds=torch.zeros(1, 77, 4096),
        pooled_prompt_embeds=torch.zeros(1, 2048),
        negative_prompt_embeds=torch.zeros(1, 76, 4096),
        negative_pooled_prompt_embeds=torch.zeros(1, 2048),
    )

    try:
        condition.validate()
    except ValueError as exc:
        assert "Negative prompt embeds" in str(exc)
    else:
        raise AssertionError("Expected shape validation to fail")


def test_latent_ltp_scheduler_probe_accepts_replayable_scheduler():
    sampler = object.__new__(AIMFlowSampler)
    ok, debug = sampler._probe_latent_ltp_scheduler(
        ReplayableScheduler(),
        torch.tensor(1.0),
        torch.zeros(1, 4, 4, 4),
    )

    assert ok
    assert debug["outputs_match"]
    assert debug["post_step_states_match"]


def test_latent_ltp_scheduler_probe_rejects_non_replayable_scheduler():
    NonReplayableScheduler.calls = 0
    sampler = object.__new__(AIMFlowSampler)
    ok, debug = sampler._probe_latent_ltp_scheduler(
        NonReplayableScheduler(),
        torch.tensor(1.0),
        torch.zeros(1, 4, 4, 4),
    )

    assert not ok
    assert not debug["outputs_match"]
