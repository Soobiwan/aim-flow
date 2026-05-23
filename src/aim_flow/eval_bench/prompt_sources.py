"""Prompt manifest builders for T2I-CompBench and COCO."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

from aim_flow.eval_bench.constants import DEFAULT_SEED, T2I_CATEGORY_FILES
from aim_flow.eval_bench.schemas import PromptManifest, PromptSample


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_prompt_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _balanced_counts(total: int, categories: list[str]) -> dict[str, int]:
    base = total // len(categories)
    remainder = total % len(categories)
    return {category: base + (1 if index < remainder else 0) for index, category in enumerate(categories)}


def build_t2i_compbench_manifest(
    subset_size: int | str = 100,
    seed: int = DEFAULT_SEED,
    dataset_root: str | Path | None = None,
) -> PromptManifest:
    """Build a deterministic T2I-CompBench prompt manifest."""

    root = Path(dataset_root) if dataset_root else repo_root() / "external" / "T2I-CompBench" / "examples" / "dataset"
    categories = list(T2I_CATEGORY_FILES)
    rng = random.Random(seed)
    samples: list[PromptSample] = []
    full = str(subset_size).lower() == "full"
    counts = None if full else _balanced_counts(int(subset_size), categories)

    for category in categories:
        filename = T2I_CATEGORY_FILES[category]
        prompts = _read_prompt_lines(root / filename)
        if full:
            selected_indices = list(range(len(prompts)))
        else:
            count = min(int(counts[category]), len(prompts))
            selected_indices = sorted(rng.sample(range(len(prompts)), count))
        for rank, original_index in enumerate(selected_indices):
            samples.append(
                PromptSample(
                    id=f"t2i_{category}_{rank:06d}",
                    category=category,
                    prompt=prompts[original_index],
                    source="T2I-CompBench",
                    split=filename,
                    metadata={"original_index": original_index},
                )
            )

    return PromptManifest(
        benchmark="t2i_compbench",
        subset_size="full" if full else int(subset_size),
        seed=int(seed),
        samples=samples,
    )


def _caption_from_record(record: dict[str, Any]) -> str | None:
    for key in ("caption", "sentences", "text", "prompt"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
            if isinstance(first, dict):
                text = first.get("raw") or first.get("text") or first.get("caption")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    captions = record.get("captions")
    if isinstance(captions, list) and captions:
        first = captions[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return None


def _load_coco_captions_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(f"Expected COCO captions JSON with an annotations list: {path}")
    by_image: OrderedDict[int, dict[str, Any]] = OrderedDict()
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_id not in by_image:
            caption = str(annotation.get("caption", "")).strip()
            if caption:
                by_image[image_id] = {"image_id": image_id, "caption": caption, "annotation_id": annotation.get("id")}
    return list(by_image.values())


def _load_coco_from_datasets() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - exercised only in full benchmark environments
        raise RuntimeError(
            "COCO prompt loading requires either --coco-captions-json, --prompt-file, or the datasets package."
        ) from exc

    candidates = [
        ("lmms-lab/COCO-Caption2017", "val"),
        ("HuggingFaceM4/COCO", "validation"),
        ("nlphuji/mscoco_2014_5k_test_image_text_retrieval", "test"),
    ]
    last_error: Exception | None = None
    for dataset_name, split in candidates:
        try:
            dataset = load_dataset(dataset_name, split=split)
            rows = []
            for index, record in enumerate(dataset):
                caption = _caption_from_record(dict(record))
                if caption:
                    rows.append({"image_id": record.get("image_id", index), "caption": caption, "dataset": dataset_name})
            if rows:
                return rows
        except Exception as exc:  # pragma: no cover - network/data availability dependent
            last_error = exc
    raise RuntimeError(f"Could not load a supported COCO caption dataset. Last error: {last_error}")


def build_coco_manifest(
    subset_size: int = 100,
    seed: int = DEFAULT_SEED,
    captions_json: str | Path | None = None,
    prompt_file: str | Path | None = None,
) -> PromptManifest:
    """Build a deterministic COCO caption prompt manifest."""

    if prompt_file:
        rows = [
            {"image_id": index, "caption": prompt}
            for index, prompt in enumerate(_read_prompt_lines(Path(prompt_file)))
        ]
        source = str(prompt_file)
    elif captions_json:
        rows = _load_coco_captions_json(Path(captions_json))
        source = str(captions_json)
    else:
        rows = _load_coco_from_datasets()
        source = "COCO caption dataset"

    rng = random.Random(seed)
    count = min(int(subset_size), len(rows))
    selected_indices = sorted(rng.sample(range(len(rows)), count))
    samples = [
        PromptSample(
            id=f"coco_{rank:06d}",
            category="coco",
            prompt=str(rows[index]["caption"]).strip(),
            source=source,
            split="validation",
            metadata={"original_index": index, "image_id": rows[index].get("image_id")},
        )
        for rank, index in enumerate(selected_indices)
    ]
    return PromptManifest(
        benchmark="coco",
        subset_size=int(subset_size),
        seed=int(seed),
        samples=samples,
    )


def build_custom_prompt_manifest(
    prompt_file: str | Path,
    subset_size: int | str = "full",
    seed: int = DEFAULT_SEED,
    benchmark: str = "custom_difficult",
) -> PromptManifest:
    """Build a manifest from a plain text file of custom difficult prompts."""

    prompts = _read_prompt_lines(Path(prompt_file))
    full = str(subset_size).lower() == "full"
    if full:
        selected_indices = list(range(len(prompts)))
    else:
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(range(len(prompts)), min(int(subset_size), len(prompts))))
    samples = [
        PromptSample(
            id=f"custom_{rank:06d}",
            category="custom_difficult",
            prompt=prompts[index],
            source=str(prompt_file),
            split="custom",
            metadata={"original_index": index},
        )
        for rank, index in enumerate(selected_indices)
    ]
    return PromptManifest(
        benchmark=benchmark,
        subset_size="full" if full else int(subset_size),
        seed=int(seed),
        samples=samples,
    )
