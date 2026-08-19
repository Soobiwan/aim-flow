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
    "full_late",
}


@dataclass
class PrimitivePrompt:
    text: str
    type: str
    weight: float = 1.0
    schedule: str = "constant"
    enabled: bool = True
    anchor_augmented_text: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrimitivePrompt":
        primitive = cls(
            text=str(data.get("text", "")).strip(),
            type=str(data.get("type", "primitive")).strip(),
            weight=float(data.get("weight", 1.0)),
            schedule=str(data.get("schedule", "constant")).strip(),
            enabled=bool(data.get("enabled", True)),
            anchor_augmented_text=data.get("anchor_augmented_text"),
        )
        if primitive.anchor_augmented_text is not None:
            primitive.anchor_augmented_text = str(primitive.anchor_augmented_text)
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

    def build_anchor_augmented_text(self, anchor_prompt: str) -> str:
        """Return q_i = anchor + primitive, unless an explicit q_i is supplied."""

        if self.anchor_augmented_text is not None and self.anchor_augmented_text.strip():
            return self.anchor_augmented_text.strip()
        anchor = anchor_prompt.strip()
        if not anchor:
            raise ValueError("anchor_prompt must be non-empty when building anchor-augmented text.")
        return f"{anchor}, {self.text.strip()}"


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

    def get_enabled_primitives(self) -> list[PrimitivePrompt]:
        """Return primitives that are enabled for sampling."""

        return [primitive for primitive in self.primitive_prompts if primitive.enabled]

    def build_anchor_augmented_primitive_prompts(self) -> list[str]:
        """Build all enabled q_i = A + p_i primitive prompts."""

        return [
            primitive.build_anchor_augmented_text(self.anchor_prompt)
            for primitive in self.get_enabled_primitives()
        ]

    def pretty(self) -> str:
        """Return a readable representation for logging."""

        return pformat(self.to_dict(), sort_dicts=False)


@dataclass
class LadderCondition:
    """One complete scene condition in a progressive prompt ladder."""

    text: str
    name: str = ""
    type: str = "generic"
    weight: float = 1.0
    schedule: str = "constant"
    enabled: bool = True
    priority: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LadderCondition":
        condition = cls(
            text=str(data.get("text", "")).strip(),
            name=str(data.get("name", "")).strip(),
            type=str(data.get("type", "generic")).strip(),
            weight=float(data.get("weight", 1.0)),
            schedule=str(data.get("schedule", "constant")).strip(),
            enabled=bool(data.get("enabled", True)),
            priority=int(data.get("priority", 0)),
        )
        condition.validate()
        return condition

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.text:
            raise ValueError("LadderCondition.text must be non-empty.")
        if not 0.0 <= self.weight <= 3.0:
            raise ValueError(f"LadderCondition.weight must be in [0, 3], got {self.weight}.")
        if self.schedule not in VALID_SCHEDULES:
            raise ValueError(f"Unknown schedule '{self.schedule}'. Expected one of {sorted(VALID_SCHEDULES)}.")


@dataclass
class ConditionLadder:
    """A sequence of complete scene prompts for LadderFlow v3."""

    full_prompt: str
    conditions: list[LadderCondition]
    negative_prompt: str | None = None
    name: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConditionLadder":
        ladder = cls(
            full_prompt=str(data.get("full_prompt", data.get("full", ""))).strip(),
            conditions=[LadderCondition.from_dict(item) for item in data.get("conditions", [])],
            negative_prompt=data.get("negative_prompt", data.get("negative")),
            name=str(data.get("name", "")).strip(),
        )
        if ladder.negative_prompt is not None:
            ladder.negative_prompt = str(ladder.negative_prompt)
        ladder.validate()
        return ladder

    def validate(self) -> None:
        if not self.full_prompt:
            raise ValueError("ConditionLadder.full_prompt must be non-empty.")
        if not self.conditions:
            raise ValueError("ConditionLadder.conditions must contain at least two conditions.")
        for condition in self.conditions:
            condition.validate()
        enabled = self.get_enabled_conditions()
        if len(enabled) < 2:
            raise ValueError("ConditionLadder requires at least two enabled conditions.")
        if not enabled[0].text:
            raise ValueError("First enabled ladder condition must be a non-empty C0/base condition.")

    def get_enabled_conditions(self) -> list[LadderCondition]:
        return [condition for condition in self.conditions if condition.enabled]

    def get_condition_texts(self) -> list[str]:
        return [condition.text for condition in self.get_enabled_conditions()]

    def get_final_condition(self) -> LadderCondition:
        return self.get_enabled_conditions()[-1]

    def get_base_condition(self) -> LadderCondition:
        return self.get_enabled_conditions()[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "full_prompt": self.full_prompt,
            "negative_prompt": self.negative_prompt,
            "conditions": [condition.to_dict() for condition in self.conditions],
        }

    def pretty_print(self) -> str:
        return pformat(self.to_dict(), sort_dicts=False)


VALID_PRIMITIVE_FLOW_ROLES = {"source", "primitive", "target"}


@dataclass
class PrimitiveFlowPrompt:
    """One prompt-conditioned flow used by sparse primitive composition."""

    text: str
    name: str = ""
    role: str = "primitive"
    weight: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        self.text = self.text.strip()
        self.name = self.name.strip()
        self.role = self.role.strip()
        self.validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrimitiveFlowPrompt":
        prompt = cls(
            text=str(data.get("text", "")).strip(),
            name=str(data.get("name", "")).strip(),
            role=str(data.get("role", "primitive")).strip(),
            weight=float(data.get("weight", 1.0)),
            enabled=bool(data.get("enabled", True)),
        )
        prompt.validate()
        return prompt

    def validate(self) -> None:
        if not self.text:
            raise ValueError("PrimitiveFlowPrompt.text must be non-empty.")
        if self.role not in VALID_PRIMITIVE_FLOW_ROLES:
            raise ValueError(
                f"PrimitiveFlowPrompt.role must be one of {sorted(VALID_PRIMITIVE_FLOW_ROLES)}, got {self.role!r}."
            )
        if not 0.0 <= self.weight <= 3.0:
            raise ValueError(f"PrimitiveFlowPrompt.weight must be in [0, 3], got {self.weight}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrimitiveFlowSet:
    """Source, primitive, and target prompts for Sparse Primitive Flow Composition."""

    name: str
    target_prompt: str
    source_prompt: str
    primitive_prompts: list[PrimitiveFlowPrompt] = field(default_factory=list)
    negative_prompt: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrimitiveFlowSet":
        flow_set = cls(
            name=str(data.get("name", "")).strip(),
            target_prompt=str(data.get("target_prompt", data.get("full_prompt", ""))).strip(),
            source_prompt=str(data.get("source_prompt", data.get("base_prompt", ""))).strip(),
            primitive_prompts=[
                PrimitiveFlowPrompt.from_dict(item)
                for item in data.get("primitive_prompts", data.get("primitives", []))
            ],
            negative_prompt=data.get("negative_prompt", data.get("negative")),
        )
        if flow_set.negative_prompt is not None:
            flow_set.negative_prompt = str(flow_set.negative_prompt)
        flow_set.validate()
        return flow_set

    def validate(self) -> None:
        if not self.target_prompt:
            raise ValueError("PrimitiveFlowSet.target_prompt must be non-empty.")
        if not self.source_prompt:
            raise ValueError("PrimitiveFlowSet.source_prompt must be non-empty.")
        if not self.primitive_prompts:
            raise ValueError("PrimitiveFlowSet.primitive_prompts must contain at least one primitive.")
        for primitive in self.primitive_prompts:
            primitive.validate()
        if not self.get_enabled_primitives():
            raise ValueError("PrimitiveFlowSet requires at least one enabled primitive prompt.")

    def get_enabled_primitives(self) -> list[PrimitiveFlowPrompt]:
        return [primitive for primitive in self.primitive_prompts if primitive.enabled]

    def get_all_condition_prompts(self, include_source: bool = True, include_target: bool = True) -> list[str]:
        prompts: list[str] = []
        if include_source:
            prompts.append(self.source_prompt)
        prompts.extend(primitive.text for primitive in self.get_enabled_primitives())
        if include_target:
            prompts.append(self.target_prompt)
        return prompts

    def get_all_condition_names(self, include_source: bool = True, include_target: bool = True) -> list[str]:
        names: list[str] = []
        if include_source:
            names.append("source")
        names.extend(primitive.name or f"primitive_{index}" for index, primitive in enumerate(self.get_enabled_primitives()))
        if include_target:
            names.append("target")
        return names

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_prompt": self.target_prompt,
            "source_prompt": self.source_prompt,
            "negative_prompt": self.negative_prompt,
            "primitive_prompts": [primitive.to_dict() for primitive in self.primitive_prompts],
        }

    def pretty_print(self) -> str:
        return pformat(self.to_dict(), sort_dicts=False)


@dataclass
class MarginalPrimitive:
    """Human-readable primitive metadata paired with an explicit target ablation."""

    ablated_prompt: str
    primitive: str
    name: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        self.ablated_prompt = self.ablated_prompt.strip()
        self.primitive = self.primitive.strip()
        self.name = self.name.strip()
        self.validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarginalPrimitive":
        return cls(
            name=str(data.get("name", "")).strip(),
            primitive=str(data.get("primitive", data.get("text", ""))).strip(),
            ablated_prompt=str(data.get("ablated_prompt", "")).strip(),
            enabled=bool(data.get("enabled", True)),
        )

    def validate(self) -> None:
        if not self.primitive:
            raise ValueError("MarginalPrimitive.primitive must be non-empty metadata.")
        if not self.ablated_prompt:
            raise ValueError("MarginalPrimitive.ablated_prompt must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarginalFlowPromptSet:
    """Full target prompt and explicit contextual ablations for Marginal Flow."""

    target_prompt: str
    primitives: list[MarginalPrimitive] = field(default_factory=list)
    negative_prompt: str | None = None
    name: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarginalFlowPromptSet":
        prompt_set = cls(
            name=str(data.get("name", "")).strip(),
            target_prompt=str(data.get("target_prompt", data.get("full_prompt", ""))).strip(),
            primitives=[MarginalPrimitive.from_dict(item) for item in data.get("primitives", [])],
            negative_prompt=data.get("negative_prompt", data.get("negative")),
        )
        if prompt_set.negative_prompt is not None:
            prompt_set.negative_prompt = str(prompt_set.negative_prompt)
        prompt_set.validate()
        return prompt_set

    def validate(self) -> None:
        if not self.target_prompt:
            raise ValueError("MarginalFlowPromptSet.target_prompt must be non-empty.")
        if not self.primitives:
            raise ValueError("MarginalFlowPromptSet.primitives must contain at least one primitive.")
        for primitive in self.primitives:
            primitive.validate()
        if not self.get_enabled_primitives():
            raise ValueError("MarginalFlowPromptSet requires at least one enabled primitive.")

    def get_enabled_primitives(self) -> list[MarginalPrimitive]:
        return [primitive for primitive in self.primitives if primitive.enabled]

    def get_ablated_prompts(self) -> list[str]:
        return [primitive.ablated_prompt for primitive in self.get_enabled_primitives()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_prompt": self.target_prompt,
            "negative_prompt": self.negative_prompt,
            "primitives": [primitive.to_dict() for primitive in self.primitives],
        }

    def pretty_print(self) -> str:
        return pformat(self.to_dict(), sort_dicts=False)
