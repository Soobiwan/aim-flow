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
    load_t5_text_encoder: bool = False


@dataclass
class SamplerConfig:
    height: int = 512
    width: int = 512
    num_inference_steps: int = 24
    guidance_scale: float = 4.5
    seed: int = 42


@dataclass
class VFAConfig:
    enabled: bool = True
    use_full_prompt_consistency: bool = True
    conflict_threshold: float = -0.10
    velocity_clip_ratio: float = 0.35
    use_anchor_augmented_primitives: bool = True
    use_delta_full_gate: bool = True
    min_gate: float = 0.0
    max_gate: float = 1.0


@dataclass
class LTPConfig:
    enabled: bool = True
    mode: str = "latent"
    radius_ratio: float = 0.30
    early_radius_ratio: float = 0.20
    middle_radius_ratio: float = 0.35
    late_radius_ratio: float = 0.25
    fallback_to_velocity_ltp: bool = True


@dataclass
class LadderFlowConfig:
    enabled: bool = True
    mode: str = "ladder_v3"
    reference_policy: str = "progressive"
    active_policy: str = "all"
    aggregation_steps: list[int] | None = None
    aggregation_step_fractions: list[float] | None = None
    aggregate_every_n_steps: int | None = None
    non_aggregation_policy: str = "reference"
    vfa_temperature: float = 0.7
    use_consensus_gating: bool = True
    use_final_consistency_gating: bool = True
    min_gate: float = 0.0
    max_gate: float = 1.0
    velocity_clip_ratio: float = 0.50
    ltp_enabled: bool = True
    ltp_mode: str = "latent"
    ltp_radius_ratio: float = 0.35
    ltp_early_radius_ratio: float = 0.25
    ltp_middle_radius_ratio: float = 0.45
    ltp_late_radius_ratio: float = 0.30
    fallback_to_velocity_ltp: bool = True
    max_conditions: int = 6
    sequential_condition_forward: bool = True


@dataclass
class PrimitiveFlowConfig:
    enabled: bool = True
    mode: str = "sparse_primitive_flow"
    final_only: bool = False
    aggregation_steps: list[int] | None = None
    aggregation_step_fractions: list[float] | None = None
    aggregate_every_n_steps: int | None = None
    include_source_flow: bool = True
    include_target_flow: bool = True
    target_reference: bool = True
    vfa_temperature: float = 0.7
    use_consensus_gating: bool = True
    use_target_consistency_gating: bool = True
    uniform_condition_weights: bool = False
    min_gate: float = 0.0
    max_gate: float = 1.0
    source_weight: float = 0.7
    target_weight: float = 1.2
    velocity_clip_ratio: float = 0.50
    steering_strength: float = 1.0
    ltp_enabled: bool = True
    ltp_mode: str = "latent"
    ltp_radius_ratio: float = 0.35
    fallback_to_velocity_ltp: bool = True
    max_primitives: int = 5
    sequential_condition_forward: bool = True


@dataclass
class AimFlowConfig:
    lambda_global: float = 0.75
    lambda_schedule: str = "middle_late"
    max_primitives: int = 5
    sequential_primitive_forward: bool = True
    mode: str = "aim_v2"
    vfa: VFAConfig = field(default_factory=VFAConfig)
    ltp: LTPConfig = field(default_factory=LTPConfig)

    @property
    def conflict_threshold(self) -> float:
        """Backward-compatible access to the VFA conflict threshold."""

        return self.vfa.conflict_threshold

    @conflict_threshold.setter
    def conflict_threshold(self, value: float) -> None:
        self.vfa.conflict_threshold = value

    @property
    def norm_clip_ratio(self) -> float:
        """Backward-compatible access to the VFA velocity clip ratio."""

        return self.vfa.velocity_clip_ratio

    @norm_clip_ratio.setter
    def norm_clip_ratio(self, value: float) -> None:
        self.vfa.velocity_clip_ratio = value


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    aim_flow: AimFlowConfig = field(default_factory=AimFlowConfig)
    ladder_flow: LadderFlowConfig = field(default_factory=LadderFlowConfig)
    primitive_flow: PrimitiveFlowConfig = field(default_factory=PrimitiveFlowConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        """Build a RunConfig from a nested dictionary."""

        aim_raw = dict(data.get("aim_flow") or {})
        vfa_raw = dict(aim_raw.pop("vfa", {}) or {})
        ltp_raw = dict(aim_raw.pop("ltp", {}) or {})
        if "conflict_threshold" in aim_raw and "conflict_threshold" not in vfa_raw:
            vfa_raw["conflict_threshold"] = aim_raw.pop("conflict_threshold")
        if "norm_clip_ratio" in aim_raw and "velocity_clip_ratio" not in vfa_raw:
            vfa_raw["velocity_clip_ratio"] = aim_raw.pop("norm_clip_ratio")
        aim_config = AimFlowConfig(**aim_raw)
        aim_config.vfa = VFAConfig(**vfa_raw)
        aim_config.ltp = LTPConfig(**ltp_raw)
        return cls(
            model=ModelConfig(**(data.get("model") or {})),
            sampler=SamplerConfig(**(data.get("sampler") or {})),
            aim_flow=aim_config,
            ladder_flow=LadderFlowConfig(**(data.get("ladder_flow") or {})),
            primitive_flow=PrimitiveFlowConfig(**(data.get("primitive_flow") or {})),
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
