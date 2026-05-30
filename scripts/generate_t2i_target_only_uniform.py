"""Generate the GenEval SPFC subset with uniform target-only VFA."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.eval_bench.constants import DEFAULT_GUIDANCE_SCALE, DEFAULT_SEED
from aim_flow.eval_bench.generation import apply_spfc_variant, generate_methods, load_bench_config

BENCHMARK = "geneval"
RUN_SLUG = "geneval_seed13_target_only_uniform"
METHOD_LABEL = "spfc_target_only_uniform"
SPFC_VARIANT = "target_only_uniform"
MANIFEST_REL = Path("configs/geneval_100_seed13.json")
DECOMP_REL = Path("configs/geneval_100_seed13_spfc.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate local SPFC outputs with w=1 and target-consistency-only VFA."
    )
    parser.add_argument("--repo-dir", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, help=f"Defaults to <repo>/{RUN_SLUG}.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate PNG/JSON pairs that already exist.")
    parser.add_argument("--install-deps", action="store_true", help="Install local requirements before generation.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print settings without loading SD3.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_inputs(repo_dir: Path) -> tuple[Path, Path, int]:
    manifest_path = repo_dir / MANIFEST_REL
    decomposition_path = repo_dir / DECOMP_REL
    manifest = load_json(manifest_path)
    decompositions = load_json(decomposition_path)
    sample_count = len(manifest["samples"])
    decomposition_count = len(decompositions["items"])
    if sample_count != 100:
        raise ValueError(f"Expected 100 GenEval samples, found {sample_count}.")
    if decomposition_count != sample_count:
        raise ValueError(f"Expected {sample_count} decompositions, found {decomposition_count}.")
    return manifest_path, decomposition_path, sample_count


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


def validate_locked_variant(seed: int, guidance_scale: float) -> None:
    config = load_bench_config(seed=seed, guidance_scale=guidance_scale)
    config = apply_spfc_variant(config, SPFC_VARIANT)
    primitive_flow = config.primitive_flow
    assert primitive_flow.uniform_condition_weights is True
    assert primitive_flow.use_consensus_gating is False
    assert primitive_flow.use_target_consistency_gating is True
    assert primitive_flow.source_weight == 1.0
    assert primitive_flow.target_weight == 1.0


def metadata_matches_locked_variant(metadata: dict) -> bool:
    weights = metadata.get("condition_weights")
    primitive_flow = metadata.get("runtime_config", {}).get("primitive_flow", {})
    return (
        isinstance(weights, list)
        and bool(weights)
        and all(float(weight) == 1.0 for weight in weights)
        and primitive_flow.get("uniform_condition_weights") is True
        and primitive_flow.get("use_consensus_gating") is False
        and primitive_flow.get("use_target_consistency_gating") is True
        and float(primitive_flow.get("source_weight", 0.0)) == 1.0
        and float(primitive_flow.get("target_weight", 0.0)) == 1.0
    )


def incompatible_metadata(method_dir: Path) -> list[Path]:
    incompatible = []
    for path in sorted(method_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            metadata = load_json(path)
        except (json.JSONDecodeError, OSError):
            incompatible.append(path)
            continue
        if not metadata_matches_locked_variant(metadata):
            incompatible.append(path)
    return incompatible


def write_readme(output_root: Path, method_dir: Path, seed: int, guidance_scale: float) -> Path:
    readme = output_root / f"README_{METHOD_LABEL}.txt"
    readme.write_text(
        "GenEval 100 generated images\n"
        f"method: {METHOD_LABEL}\n"
        f"variant: {SPFC_VARIANT}\n"
        "vfa: r_i = 1 * g_i^tgt\n"
        "uniform_condition_weights: true\n"
        "use_consensus_gating: false\n"
        "use_target_consistency_gating: true\n"
        f"seed: {seed}\n"
        f"guidance_scale: {guidance_scale}\n"
        f"image_dir: {method_dir}\n",
        encoding="utf-8",
    )
    return readme


def main() -> None:
    args = parse_args()
    repo_dir = args.repo_dir.expanduser().resolve()
    output_root = (args.output_root or repo_dir / RUN_SLUG).expanduser().resolve()
    run_root = output_root / "runs"

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.install_deps:
        install_dependencies(repo_dir)

    manifest_path, decomposition_path, sample_count = validate_inputs(repo_dir)
    validate_locked_variant(args.seed, args.guidance_scale)
    method_dir = run_root / BENCHMARK / METHOD_LABEL

    print("repo:", repo_dir)
    print("manifest:", manifest_path)
    print("decompositions:", decomposition_path)
    print("samples:", sample_count)
    print("output root:", output_root)
    print("method:", METHOD_LABEL)
    print("vfa: r_i = 1 * g_i^tgt")
    print("uniform_condition_weights: true")
    print("use_consensus_gating: false")
    print("use_target_consistency_gating: true")
    if args.dry_run:
        print("dry run: inputs and locked variant are valid")
        return

    incompatible = incompatible_metadata(method_dir)
    if incompatible and not args.overwrite:
        raise RuntimeError(
            f"Found {len(incompatible)} existing output metadata file(s) from a different configuration. "
            "Rerun with --overwrite to regenerate them."
        )

    generate_methods(
        manifest_path=manifest_path,
        run_root=run_root,
        methods=["spfc"],
        decomposition_path=decomposition_path,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        spfc_variant=SPFC_VARIANT,
        spfc_method_label=METHOD_LABEL,
        skip_existing=not args.overwrite,
    )

    index_path = method_dir / "index.json"
    images = sorted(method_dir.glob("*.png"))
    metadata = [path for path in method_dir.glob("*.json") if path.name != "index.json"]
    if not index_path.exists():
        raise FileNotFoundError(f"Missing generation index: {index_path}")
    if len(images) != sample_count or len(metadata) != sample_count:
        raise RuntimeError(
            f"Expected {sample_count} images and metadata files, found {len(images)} images and {len(metadata)} metadata files."
        )
    incompatible = incompatible_metadata(method_dir)
    if incompatible:
        raise RuntimeError(f"Generated outputs failed locked-variant validation: {incompatible[:3]}")

    readme = write_readme(output_root, method_dir, args.seed, args.guidance_scale)
    archive = Path(
        shutil.make_archive(
            str(output_root),
            "zip",
            root_dir=output_root.parent,
            base_dir=output_root.name,
        )
    )
    print("index:", index_path)
    print("readme:", readme)
    print("zip archive:", archive)


if __name__ == "__main__":
    main()
