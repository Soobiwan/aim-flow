"""Shared local runner for the GenEval generation scripts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.eval_bench.constants import DEFAULT_GUIDANCE_SCALE, DEFAULT_SEED
from aim_flow.eval_bench.generation import (
    generate_base,
    generate_cfg,
    generate_rectified_cfgpp,
    generate_spfc,
    load_bench_config,
)
from aim_flow.eval_bench.schemas import DecompositionManifest, PromptManifest
from aim_flow.primitive_flow import build_condition_list

BENCHMARK = "geneval"
RUN_SLUG = "geneval_seed13"
MANIFEST_REL = Path("configs/geneval_100_seed13.json")
DECOMP_REL = Path("configs/geneval_100_seed13_spfc.json")
EXPECTED_SAMPLE_COUNT = 100
RESTORED_SPFC_FORMULA = "r_i = w_i * g_i^cons * g_i^tgt"


def parse_args(description: str, default_guidance_scale: float, archive_filename: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repo-dir", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, help=f"Defaults to <repo>/{RUN_SLUG}.")
    parser.add_argument("--archive-path", type=Path, help=f"Defaults to <repo>/data/{archive_filename}.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--guidance-scale", type=float, default=default_guidance_scale)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate PNG/JSON pairs that already exist.")
    parser.add_argument("--install-deps", action="store_true", help="Install local requirements before generation.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print settings without loading SD3.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def install_dependencies(repo_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"],
        cwd=repo_dir,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."],
        cwd=repo_dir,
        check=True,
    )


def load_inputs(repo_dir: Path, method: str) -> tuple[PromptManifest, DecompositionManifest | None]:
    manifest = PromptManifest.load(repo_dir / MANIFEST_REL)
    if len(manifest.samples) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"Expected {EXPECTED_SAMPLE_COUNT} GenEval samples, found {len(manifest.samples)}.")
    decompositions = None
    if method == "spfc":
        decompositions = DecompositionManifest.load(repo_dir / DECOMP_REL)
        if len(decompositions.items) != len(manifest.samples):
            raise ValueError(f"Expected {len(manifest.samples)} decompositions, found {len(decompositions.items)}.")
    return manifest, decompositions


def validate_restored_spfc_config(config, decompositions: DecompositionManifest) -> None:
    primitive_flow = config.primitive_flow
    assert primitive_flow.uniform_condition_weights is False
    assert primitive_flow.use_consensus_gating is True
    assert primitive_flow.use_target_consistency_gating is True
    assert primitive_flow.source_weight == 0.7
    assert primitive_flow.target_weight == 1.2
    conditions = build_condition_list(
        decompositions.items[0].to_flow_set(),
        source_weight=primitive_flow.source_weight,
        target_weight=primitive_flow.target_weight,
        uniform_weights=primitive_flow.uniform_condition_weights,
        max_primitives=primitive_flow.max_primitives,
    )
    expected_weights = [
        primitive_flow.source_weight,
        *[
            primitive.weight
            for primitive in decompositions.items[0].to_flow_set().get_enabled_primitives()[
                : primitive_flow.max_primitives
            ]
        ],
        primitive_flow.target_weight,
    ]
    assert [condition["weight"] for condition in conditions] == expected_weights


def metadata_matches(method: str, metadata: dict[str, Any], guidance_scale: float) -> bool:
    if metadata.get("bench_method") != method:
        return False
    runtime_config = metadata.get("runtime_config", {})
    sampler = runtime_config.get("sampler", {})
    if float(sampler.get("guidance_scale", -1.0)) != float(guidance_scale):
        return False
    if method != "spfc":
        return True
    primitive_flow = runtime_config.get("primitive_flow", {})
    weights = metadata.get("condition_weights")
    return (
        primitive_flow.get("uniform_condition_weights") is False
        and primitive_flow.get("use_consensus_gating") is True
        and primitive_flow.get("use_target_consistency_gating") is True
        and float(primitive_flow.get("source_weight", 0.0)) == 0.7
        and float(primitive_flow.get("target_weight", 0.0)) == 1.2
        and isinstance(weights, list)
        and weights[:1] == [0.7]
        and weights[-1:] == [1.2]
        and 1.1 in weights
    )


def incompatible_metadata(method_dir: Path, method: str, guidance_scale: float) -> list[Path]:
    incompatible = []
    for path in sorted(method_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            metadata = load_json(path)
        except (json.JSONDecodeError, OSError):
            incompatible.append(path)
            continue
        if not metadata_matches(method, metadata, guidance_scale):
            incompatible.append(path)
    return incompatible


def write_readme(
    output_root: Path,
    method_dir: Path,
    method: str,
    title: str,
    seed: int,
    guidance_scale: float,
) -> Path:
    readme = output_root / f"README_{method}.txt"
    details = [
        "GenEval 100 generated images",
        f"method: {method}",
        f"title: {title}",
        f"seed: {seed}",
        f"guidance_scale: {guidance_scale}",
        f"image_dir: {method_dir}",
    ]
    if method == "spfc":
        details.extend(
            [
                f"vfa: {RESTORED_SPFC_FORMULA}",
                "uniform_condition_weights: false",
                "use_consensus_gating: true",
                "use_target_consistency_gating: true",
                "source_weight: 0.7",
                "target_weight: 1.2",
                "primitive_weights: authored decomposition weights",
            ]
        )
    readme.write_text("\n".join(details) + "\n", encoding="utf-8")
    return readme


def create_method_archive(output_root: Path, method_dir: Path, readme: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(readme, arcname=str(readme.relative_to(output_root.parent)))
        for path in sorted(method_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(output_root.parent)))


def generate_method(
    method: str,
    title: str,
    default_guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    archive_filename: str | None = None,
) -> None:
    archive_filename = archive_filename or f"{method}.zip"
    args = parse_args(
        description=f"Generate local {title} outputs for the GenEval subset.",
        default_guidance_scale=default_guidance_scale,
        archive_filename=archive_filename,
    )
    repo_dir = args.repo_dir.expanduser().resolve()
    output_root = (args.output_root or repo_dir / RUN_SLUG).expanduser().resolve()
    archive_path = (args.archive_path or repo_dir / "data" / archive_filename).expanduser().resolve()
    run_root = output_root / "runs"
    method_dir = run_root / BENCHMARK / method

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.install_deps:
        install_dependencies(repo_dir)
    manifest, decompositions = load_inputs(repo_dir, method)
    config = load_bench_config(seed=args.seed, guidance_scale=args.guidance_scale)
    if method == "spfc":
        assert decompositions is not None
        validate_restored_spfc_config(config, decompositions)

    print("repo:", repo_dir)
    print("manifest:", repo_dir / MANIFEST_REL)
    print("samples:", len(manifest.samples))
    print("output root:", output_root)
    print("method:", method)
    print("guidance_scale:", args.guidance_scale)
    if method == "spfc":
        print("decompositions:", repo_dir / DECOMP_REL)
        print("vfa:", RESTORED_SPFC_FORMULA)
        print("uniform_condition_weights: false")
        print("use_consensus_gating: true")
        print("use_target_consistency_gating: true")
        print("source_weight: 0.7")
        print("target_weight: 1.2")
    if args.dry_run:
        print("dry run: inputs and locked settings are valid")
        return

    incompatible = incompatible_metadata(method_dir, method, args.guidance_scale)
    if incompatible and not args.overwrite:
        raise RuntimeError(
            f"Found {len(incompatible)} existing {method} metadata file(s) from a different configuration. "
            "Rerun with --overwrite to regenerate them."
        )

    if method == "base":
        index_path = generate_base(manifest, run_root, config, skip_existing=not args.overwrite)
    elif method == "cfg":
        index_path = generate_cfg(manifest, run_root, config, skip_existing=not args.overwrite)
    elif method == "rectified_cfgpp":
        index_path = generate_rectified_cfgpp(
            manifest,
            run_root,
            config,
            repo_dir=repo_dir / "external" / "Rectified-CFGpp",
            skip_existing=not args.overwrite,
        )
    elif method == "spfc":
        assert decompositions is not None
        index_path = generate_spfc(
            manifest,
            decompositions,
            run_root,
            config,
            variant="full",
            method_label="spfc",
            skip_existing=not args.overwrite,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    images = sorted(method_dir.glob("*.png"))
    metadata = [path for path in method_dir.glob("*.json") if path.name != "index.json"]
    if len(images) != len(manifest.samples) or len(metadata) != len(manifest.samples):
        raise RuntimeError(
            f"Expected {len(manifest.samples)} images and metadata files, "
            f"found {len(images)} images and {len(metadata)} metadata files."
        )
    incompatible = incompatible_metadata(method_dir, method, args.guidance_scale)
    if incompatible:
        raise RuntimeError(f"Generated outputs failed {method} validation: {incompatible[:3]}")

    readme = write_readme(output_root, method_dir, method, title, args.seed, args.guidance_scale)
    create_method_archive(output_root, method_dir, readme, archive_path)
    print("index:", index_path)
    print("readme:", readme)
    print("zip archive:", archive_path)
