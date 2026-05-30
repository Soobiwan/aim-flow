"""Visualization and metadata helpers."""

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from aim_flow.utils import ensure_dir, write_json


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def make_image_grid(image_paths: list[str | Path], labels: list[str], output_path: str | Path) -> Path:
    """Create a labeled side-by-side comparison grid."""

    if len(image_paths) != len(labels):
        raise ValueError("image_paths and labels must have equal length.")
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        raise ValueError("At least one image is required.")

    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    label_h = 52
    grid = Image.new("RGB", (cell_w * len(images), cell_h + label_h), "white")
    draw = ImageDraw.Draw(grid)
    font = _load_font(18)

    for idx, (image, label) in enumerate(zip(images, labels)):
        x = idx * cell_w
        resized = image.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        grid.paste(resized, (x, label_h))
        if len(label) > 34:
            midpoint = len(label) // 2
            split_at = label.rfind(" ", 0, midpoint + 8)
            if split_at > 0:
                label = label[:split_at] + "\n" + label[split_at + 1 :]
        draw.multiline_text((x + 10, 8), label, fill=(20, 20, 20), font=font, spacing=2)

    output = Path(output_path)
    ensure_dir(output.parent)
    grid.save(output)
    return output


def make_comparison_gallery(
    image_rows: list[list[str | Path]],
    column_labels: list[str],
    row_labels: list[str],
    output_path: str | Path,
) -> Path:
    """Create a labeled gallery with one comparison row per prompt."""

    if not image_rows:
        raise ValueError("At least one image row is required.")
    if len(image_rows) != len(row_labels):
        raise ValueError("image_rows and row_labels must have equal length.")
    if not column_labels:
        raise ValueError("At least one column label is required.")
    if any(len(row) != len(column_labels) for row in image_rows):
        raise ValueError("Every image row must match the column label count.")

    images: list[list[Image.Image]] = []
    for row in image_rows:
        images.append([Image.open(path).convert("RGB") for path in row])
    cell_w = max(image.width for row in images for image in row)
    cell_h = max(image.height for row in images for image in row)
    row_label_w = 250
    header_h = 58
    grid = Image.new(
        "RGB",
        (row_label_w + cell_w * len(column_labels), header_h + cell_h * len(images)),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    font = _load_font(18)

    for column_index, label in enumerate(column_labels):
        x = row_label_w + column_index * cell_w
        draw.multiline_text((x + 10, 10), textwrap.fill(label, width=32), fill=(20, 20, 20), font=font, spacing=2)

    for row_index, (row, label) in enumerate(zip(images, row_labels)):
        y = header_h + row_index * cell_h
        draw.multiline_text((10, y + 12), textwrap.fill(label, width=34), fill=(20, 20, 20), font=font, spacing=2)
        for column_index, image in enumerate(row):
            x = row_label_w + column_index * cell_w
            resized = image.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            grid.paste(resized, (x, y))

    output = Path(output_path)
    ensure_dir(output.parent)
    grid.save(output)
    return output


def save_metadata_json(metadata: dict[str, Any], output_path: str | Path) -> Path:
    """Save generation metadata to JSON."""

    output = Path(output_path)
    write_json(metadata, output)
    return output
