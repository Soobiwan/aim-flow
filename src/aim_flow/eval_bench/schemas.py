"""JSON schemas and conversion helpers for benchmark prompts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aim_flow.prompt_schema import PrimitiveFlowPrompt, PrimitiveFlowSet
from aim_flow.utils import ensure_dir


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


def _write_json(data: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return output


@dataclass
class PromptSample:
    id: str
    category: str
    prompt: str
    source: str
    split: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptSample":
        sample = cls(
            id=str(data.get("id", "")).strip(),
            category=str(data.get("category", "")).strip(),
            prompt=str(data.get("prompt", "")).strip(),
            source=str(data.get("source", "")).strip(),
            split=str(data.get("split", "")).strip(),
            metadata=dict(data.get("metadata") or {}),
        )
        sample.validate()
        return sample

    def validate(self) -> None:
        if not self.id:
            raise ValueError("PromptSample.id must be non-empty.")
        if not self.category:
            raise ValueError(f"PromptSample.category must be non-empty for {self.id}.")
        if not self.prompt:
            raise ValueError(f"PromptSample.prompt must be non-empty for {self.id}.")
        if not self.source:
            raise ValueError(f"PromptSample.source must be non-empty for {self.id}.")
        if not self.split:
            raise ValueError(f"PromptSample.split must be non-empty for {self.id}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptManifest:
    benchmark: str
    subset_size: int | str
    seed: int
    samples: list[PromptSample]
    manifest_version: str = "1.0"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptManifest":
        manifest = cls(
            manifest_version=str(data.get("manifest_version", "1.0")),
            benchmark=str(data.get("benchmark", "")).strip(),
            subset_size=data.get("subset_size"),
            seed=int(data.get("seed")),
            samples=[PromptSample.from_dict(item) for item in data.get("samples", [])],
        )
        manifest.validate()
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "PromptManifest":
        return cls.from_dict(_read_json(path))

    def validate(self) -> None:
        if not self.benchmark:
            raise ValueError("PromptManifest.benchmark must be non-empty.")
        if not self.samples:
            raise ValueError("PromptManifest.samples must be non-empty.")
        ids = [sample.id for sample in self.samples]
        duplicates = sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate sample ids in prompt manifest: {duplicates[:5]}")
        for sample in self.samples:
            sample.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "benchmark": self.benchmark,
            "subset_size": self.subset_size,
            "seed": self.seed,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    def save(self, path: str | Path) -> Path:
        self.validate()
        return _write_json(self.to_dict(), path)

    def sample_by_id(self) -> dict[str, PromptSample]:
        return {sample.id: sample for sample in self.samples}


@dataclass
class DecompositionItem:
    id: str
    target_prompt: str
    source_prompt: str
    primitive_prompts: list[dict[str, Any]]
    negative_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecompositionItem":
        item = cls(
            id=str(data.get("id", "")).strip(),
            target_prompt=str(data.get("target_prompt", "")).strip(),
            source_prompt=str(data.get("source_prompt", "")).strip(),
            negative_prompt=data.get("negative_prompt"),
            primitive_prompts=list(data.get("primitive_prompts") or []),
            metadata=dict(data.get("metadata") or {}),
        )
        if item.negative_prompt is not None:
            item.negative_prompt = str(item.negative_prompt)
        item.validate()
        return item

    def validate(self) -> None:
        if not self.id:
            raise ValueError("DecompositionItem.id must be non-empty.")
        if not self.target_prompt:
            raise ValueError(f"target_prompt must be non-empty for {self.id}.")
        if not self.source_prompt:
            raise ValueError(f"source_prompt must be non-empty for {self.id}.")
        if not self.primitive_prompts:
            raise ValueError(f"primitive_prompts must be non-empty for {self.id}.")
        for primitive in self.primitive_prompts:
            PrimitiveFlowPrompt.from_dict(primitive)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_flow_set(self) -> PrimitiveFlowSet:
        return PrimitiveFlowSet(
            name=self.id,
            target_prompt=self.target_prompt,
            source_prompt=self.source_prompt,
            negative_prompt=self.negative_prompt,
            primitive_prompts=[PrimitiveFlowPrompt.from_dict(item) for item in self.primitive_prompts],
        )


@dataclass
class DecompositionManifest:
    items: list[DecompositionItem]
    schema_version: str = "spfc_decomposition_v1"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecompositionManifest":
        manifest = cls(
            schema_version=str(data.get("schema_version", "spfc_decomposition_v1")),
            items=[DecompositionItem.from_dict(item) for item in data.get("items", [])],
        )
        manifest.validate()
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "DecompositionManifest":
        return cls.from_dict(_read_json(path))

    def validate(self) -> None:
        if self.schema_version != "spfc_decomposition_v1":
            raise ValueError(f"Unsupported decomposition schema_version: {self.schema_version}")
        if not self.items:
            raise ValueError("DecompositionManifest.items must be non-empty.")
        ids = [item.id for item in self.items]
        duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate decomposition ids: {duplicates[:5]}")
        for item in self.items:
            item.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "items": [item.to_dict() for item in self.items],
        }

    def save(self, path: str | Path) -> Path:
        self.validate()
        return _write_json(self.to_dict(), path)

    def item_by_id(self) -> dict[str, DecompositionItem]:
        return {item.id: item for item in self.items}


def make_decomposition_template(manifest: PromptManifest) -> DecompositionManifest:
    """Create a conservative template for manual/LLM decomposition filling."""

    items = []
    for sample in manifest.samples:
        items.append(
            DecompositionItem(
                id=sample.id,
                target_prompt=sample.prompt,
                source_prompt=sample.prompt,
                negative_prompt=(
                    "blurry, low quality, distorted, missing object, wrong attribute binding, "
                    "wrong spatial relation, text, watermark"
                ),
                primitive_prompts=[
                    {
                        "name": "S1_target_prompt_template",
                        "text": sample.prompt,
                        "role": "primitive",
                        "weight": 1.0,
                        "enabled": True,
                    }
                ],
                metadata={"template": True, "category": sample.category},
            )
        )
    return DecompositionManifest(items=items)
