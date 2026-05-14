"""Prompt decomposition schema for AIM-Flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any


VALID_SCHEDULES = {
    "constant",
    "early",
    "middle",
    "late",
    "early_middle",
    "middle_late",
}


@dataclass
class PrimitivePrompt:
    text: str
    type: str
    weight: float = 1.0
    schedule: str = "constant"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrimitivePrompt":
        primitive = cls(
            text=str(data.get("text", "")).strip(),
            type=str(data.get("type", "primitive")).strip(),
            weight=float(data.get("weight", 1.0)),
            schedule=str(data.get("schedule", "constant")).strip(),
        )
        primitive.validate()
        return primitive

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.text:
            raise ValueError("PrimitivePrompt.text must be non-empty.")
        if not self.type:
            raise ValueError("PrimitivePrompt.type must be non-empty.")
        if not 0.0 <= self.weight <= 2.0:
            raise ValueError(f"PrimitivePrompt.weight must be in [0, 2], got {self.weight}.")
        if self.schedule not in VALID_SCHEDULES:
            raise ValueError(f"Unknown schedule '{self.schedule}'. Expected one of {sorted(VALID_SCHEDULES)}.")


@dataclass
class PromptDecomposition:
    full_prompt: str
    anchor_prompt: str
    primitive_prompts: list[PrimitivePrompt] = field(default_factory=list)
    negative_prompt: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptDecomposition":
        primitives_raw = data.get("primitive_prompts", data.get("primitives", []))
        decomposition = cls(
            full_prompt=str(data.get("full_prompt", data.get("full", ""))).strip(),
            anchor_prompt=str(data.get("anchor_prompt", data.get("anchor", ""))).strip(),
            primitive_prompts=[PrimitivePrompt.from_dict(item) for item in primitives_raw],
            negative_prompt=data.get("negative_prompt", data.get("negative")),
        )
        if decomposition.negative_prompt is not None:
            decomposition.negative_prompt = str(decomposition.negative_prompt)
        decomposition.validate()
        return decomposition

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_prompt": self.full_prompt,
            "anchor_prompt": self.anchor_prompt,
            "primitive_prompts": [primitive.to_dict() for primitive in self.primitive_prompts],
            "negative_prompt": self.negative_prompt,
        }

    def validate(self) -> None:
        if not self.full_prompt:
            raise ValueError("PromptDecomposition.full_prompt must be non-empty.")
        if not self.anchor_prompt:
            raise ValueError("PromptDecomposition.anchor_prompt must be non-empty.")
        if not self.primitive_prompts:
            raise ValueError("PromptDecomposition.primitive_prompts must contain at least one primitive.")
        for primitive in self.primitive_prompts:
            primitive.validate()

    def pretty(self) -> str:
        """Return a readable representation for logging."""

        return pformat(self.to_dict(), sort_dicts=False)

