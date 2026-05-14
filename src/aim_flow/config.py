"""Configuration dataclasses and YAML helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aim_flow.utils import read_yaml, write_yaml


@dataclass
class ModelConfig:
    model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    dtype: str = "float16"
    enable_model_cpu_offload: bool = True
    enable_attention_slicing: bool = False
    enable_vae_slicing: bool = True


@dataclass
class SamplerConfig:
    height: int = 512
    width: int = 512
    num_inference_steps: int = 24
    guidance_scale: float = 4.5
    seed: int = 42


@dataclass
class AimFlowConfig:
    max_primitives: int = 5
    lambda_global: float = 0.75
    conflict_threshold: float = -0.15
    norm_clip_ratio: float = 0.35
    sequential_primitive_forward: bool = True
    mode: str = "full"


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    aim_flow: AimFlowConfig = field(default_factory=AimFlowConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        """Build a RunConfig from a nested dictionary."""

        return cls(
            model=ModelConfig(**(data.get("model") or {})),
            sampler=SamplerConfig(**(data.get("sampler") or {})),
            aim_flow=AimFlowConfig(**(data.get("aim_flow") or {})),
        )

    @classmethod
    def load_yaml(cls, path: str | Path) -> "RunConfig":
        """Load a RunConfig from YAML."""

        return cls.from_dict(read_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a plain nested dictionary."""

        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        """Save this config to YAML."""

        write_yaml(self.to_dict(), path)


def load_config(path: str | Path) -> RunConfig:
    """Convenience wrapper for RunConfig.load_yaml."""

    return RunConfig.load_yaml(path)


def save_config(config: RunConfig, path: str | Path) -> None:
    """Convenience wrapper for RunConfig.save_yaml."""

    config.save_yaml(path)

