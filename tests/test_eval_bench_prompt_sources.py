import json

from aim_flow.eval_bench.prompt_sources import build_coco_manifest, build_t2i_compbench_manifest


def test_t2i_compbench_manifest_selects_balanced_seeded_subset(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for filename in ["color_val.txt", "shape_val.txt", "texture_val.txt", "spatial_val.txt"]:
        (dataset_root / filename).write_text("\n".join(f"{filename} prompt {i}" for i in range(40)), encoding="utf-8")

    first = build_t2i_compbench_manifest(100, seed=13, dataset_root=dataset_root)
    second = build_t2i_compbench_manifest(100, seed=13, dataset_root=dataset_root)

    assert len(first.samples) == 100
    assert {category: sum(sample.category == category for sample in first.samples) for category in ["color", "shape", "texture", "spatial"]} == {
        "color": 25,
        "shape": 25,
        "texture": 25,
        "spatial": 25,
    }
    assert [sample.prompt for sample in first.samples] == [sample.prompt for sample in second.samples]


def test_coco_manifest_selects_one_caption_per_image_id(tmp_path):
    captions_path = tmp_path / "captions.json"
    captions_path.write_text(
        json.dumps(
            {
                "annotations": [
                    {"id": 1, "image_id": 10, "caption": "first image first caption"},
                    {"id": 2, "image_id": 10, "caption": "first image second caption"},
                    {"id": 3, "image_id": 11, "caption": "second image caption"},
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = build_coco_manifest(2, seed=13, captions_json=captions_path)

    assert len(manifest.samples) == 2
    assert {sample.metadata["image_id"] for sample in manifest.samples} == {10, 11}
