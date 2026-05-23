"""Generation orchestration for the SPFC evaluation bench."""

from __future__ import annotations

import copy
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from aim_flow.config import RunConfig, load_config
from aim_flow.eval_bench.constants import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_MODEL_ID,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_SPFC_SCHEDULE_1_INDEXED,
    DEFAULT_WIDTH,
    RECTIFIED_CFGPP_COMMIT,
    RECTIFIED_CFGPP_REPO_URL,
    SPFC_SCHEDULE_ABLATIONS_1_INDEXED,
)
from aim_flow.eval_bench.schemas import DecompositionManifest, PromptManifest
from aim_flow.eval_bench.schedules import one_indexed_to_zero_indexed
from aim_flow.prompt_schema import PrimitiveFlowSet
from aim_flow.sampler import AIMFlowSampler
from aim_flow.sd3_backend import SD3Backend
from aim_flow.utils import ensure_dir, get_device, get_hf_token, get_torch_dtype, write_json
from aim_flow.visualize import save_metadata_json


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def unload_model(obj: Any | None = None) -> None:
    """Release model references and empty CUDA cache between phases."""

    if obj is not None:
        pipe = getattr(obj, "pipe", obj)
        if hasattr(pipe, "maybe_free_model_hooks"):
            try:
                pipe.maybe_free_model_hooks()
            except Exception:
                pass
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_bench_config(
    config_path: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    aggregation_steps_1_indexed: list[int] | None = None,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
) -> RunConfig:
    """Load a RunConfig and apply benchmark defaults."""

    config = load_config(config_path) if config_path else RunConfig()
    config.model.model_id = DEFAULT_MODEL_ID
    config.model.dtype = "float16"
    config.model.enable_model_cpu_offload = True
    config.model.load_t5_text_encoder = False
    config.sampler.seed = int(seed)
    config.sampler.num_inference_steps = int(num_inference_steps)
    config.sampler.height = int(height)
    config.sampler.width = int(width)
    config.sampler.guidance_scale = float(guidance_scale)
    schedule = aggregation_steps_1_indexed or DEFAULT_SPFC_SCHEDULE_1_INDEXED
    config.primitive_flow.aggregation_steps = one_indexed_to_zero_indexed(
        schedule,
        num_steps=config.sampler.num_inference_steps,
        required_count=16,
    )
    config.primitive_flow.aggregation_step_fractions = None
    config.primitive_flow.aggregate_every_n_steps = None
    config.primitive_flow.final_only = False
    config.primitive_flow.steering_strength = 1.0
    return config


def apply_spfc_variant(config: RunConfig, variant: str | None) -> RunConfig:
    """Apply one benchmark ablation/sweep variant to a config copy."""

    tuned = copy.deepcopy(config)
    if not variant or variant == "full":
        return tuned
    if variant == "no_consensus_gating":
        tuned.primitive_flow.use_consensus_gating = False
    elif variant == "no_target_consistency_gating":
        tuned.primitive_flow.use_target_consistency_gating = False
    elif variant == "no_ltp":
        tuned.primitive_flow.ltp_enabled = False
        tuned.primitive_flow.ltp_mode = "off"
    elif variant == "no_source_flow":
        tuned.primitive_flow.include_source_flow = False
    elif variant.startswith("steering_"):
        tuned.primitive_flow.steering_strength = float(variant.removeprefix("steering_"))
    elif variant.startswith("max_primitives_"):
        tuned.primitive_flow.max_primitives = int(variant.removeprefix("max_primitives_"))
    else:
        raise ValueError(f"Unknown SPFC variant: {variant}")
    return tuned


def ensure_rectified_cfgpp_repo(path: str | Path | None = None) -> Path:
    """Clone and pin the Rectified-CFG++ repository if needed."""

    repo_dir = Path(path) if path else repo_root() / "external" / "Rectified-CFGpp"
    if not repo_dir.exists():
        ensure_dir(repo_dir.parent)
        subprocess.run(["git", "clone", RECTIFIED_CFGPP_REPO_URL, str(repo_dir)], check=True)
    current = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    if current != RECTIFIED_CFGPP_COMMIT:
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin", RECTIFIED_CFGPP_COMMIT], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "checkout", RECTIFIED_CFGPP_COMMIT], check=True)
    return repo_dir


class RectifiedCFGPPBackend:
    """Thin adapter around the Rectified-CFG++ SD3 custom pipeline."""

    def __init__(
        self,
        config: RunConfig,
        repo_dir: str | Path | None = None,
        sigma_noise: float = 0.005,
    ) -> None:
        self.config = config
        self.repo_dir = ensure_rectified_cfgpp_repo(repo_dir)
        self.sigma_noise = float(sigma_noise)
        self.pipe: Any | None = None
        self.device = get_device()
        self.dtype = get_torch_dtype(config.model.dtype)

    def load(self) -> "RectifiedCFGPPBackend":
        from diffusers import StableDiffusion3Pipeline

        token = get_hf_token()
        kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "custom_pipeline": str(self.repo_dir / "rect-cfg-SD3-pipeline"),
            "text_encoder_3": None,
            "tokenizer_3": None,
        }
        if token:
            kwargs["token"] = token
        try:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(self.config.model.model_id, **kwargs)
        except TypeError:
            if token:
                kwargs.pop("token", None)
                kwargs["use_auth_token"] = token
            self.pipe = StableDiffusion3Pipeline.from_pretrained(self.config.model.model_id, **kwargs)

        if self.config.model.enable_model_cpu_offload and hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        if self.config.model.enable_vae_slicing and hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)
        return self

    def generate(self, prompt: str, negative_prompt: str | None = None):
        if self.pipe is None:
            raise RuntimeError("RectifiedCFGPPBackend.load() must be called before generation.")
        generator = torch.Generator(device="cpu").manual_seed(self.config.sampler.seed)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            num_inference_steps=self.config.sampler.num_inference_steps,
            true_cfg=self.config.sampler.guidance_scale,
            sigma_noise=self.sigma_noise,
            width=self.config.sampler.width,
            height=self.config.sampler.height,
            generator=generator,
        )
        return result.images[0]

    @staticmethod
    def save_image(image: Any, path: str | Path) -> None:
        output = Path(path)
        ensure_dir(output.parent)
        image.save(output)


def _sample_output_paths(run_root: Path, benchmark: str, method: str, sample_id: str) -> tuple[Path, Path]:
    base = run_root / benchmark / method
    return base / f"{sample_id}.png", base / f"{sample_id}.json"


def _write_run_index(run_root: Path, benchmark: str, method: str, paths: list[dict[str, Any]]) -> Path:
    output = run_root / benchmark / method / "index.json"
    write_json({"benchmark": benchmark, "method": method, "outputs": paths}, output)
    return output


def generate_spfc(
    manifest: PromptManifest,
    decompositions: DecompositionManifest,
    run_root: str | Path,
    config: RunConfig,
    variant: str | None = None,
    method_label: str | None = None,
) -> Path:
    """Generate all SPFC images for a manifest."""

    tuned = apply_spfc_variant(config, variant)
    items = decompositions.item_by_id()
    missing = [sample.id for sample in manifest.samples if sample.id not in items]
    if missing:
        raise KeyError(f"Missing decompositions for sample ids: {missing[:5]}")
    backend = SD3Backend(tuned).load()
    sampler = AIMFlowSampler(backend, tuned)
    run_dir = Path(run_root)
    method_name = method_label or ("spfc" if not variant or variant == "full" else f"spfc_{variant}")
    paths: list[dict[str, Any]] = []
    try:
        for sample in manifest.samples:
            flow_set: PrimitiveFlowSet = items[sample.id].to_flow_set()
            start = time.perf_counter()
            image, metadata = sampler.generate_sparse_primitive_flow(flow_set, mode="primitive_flow_sparse")
            runtime_sec = time.perf_counter() - start
            image_path, metadata_path = _sample_output_paths(run_dir, manifest.benchmark, method_name, sample.id)
            backend.save_image(image, image_path)
            metadata.update(
                {
                    "benchmark_sample": sample.to_dict(),
                    "bench_method": method_name,
                    "runtime_sec": runtime_sec,
                    "variant": variant or "full",
                }
            )
            save_metadata_json(metadata, metadata_path)
            paths.append({"sample_id": sample.id, "image": str(image_path), "metadata": str(metadata_path)})
    finally:
        unload_model(backend)
    return _write_run_index(run_dir, manifest.benchmark, method_name, paths)


def generate_base(manifest: PromptManifest, run_root: str | Path, config: RunConfig) -> Path:
    backend = SD3Backend(config).load()
    run_dir = Path(run_root)
    paths: list[dict[str, Any]] = []
    try:
        for sample in manifest.samples:
            start = time.perf_counter()
            image = backend.generate_base(
                prompt=sample.prompt,
                negative_prompt=None,
                seed=config.sampler.seed,
                num_inference_steps=config.sampler.num_inference_steps,
                guidance_scale=config.sampler.guidance_scale,
                height=config.sampler.height,
                width=config.sampler.width,
            )
            runtime_sec = time.perf_counter() - start
            image_path, metadata_path = _sample_output_paths(run_dir, manifest.benchmark, "base", sample.id)
            backend.save_image(image, image_path)
            write_json(
                {
                    "bench_method": "base",
                    "benchmark_sample": sample.to_dict(),
                    "runtime_sec": runtime_sec,
                    "runtime_config": config.to_dict(),
                },
                metadata_path,
            )
            paths.append({"sample_id": sample.id, "image": str(image_path), "metadata": str(metadata_path)})
    finally:
        unload_model(backend)
    return _write_run_index(run_dir, manifest.benchmark, "base", paths)


def generate_rectified_cfgpp(
    manifest: PromptManifest,
    run_root: str | Path,
    config: RunConfig,
    repo_dir: str | Path | None = None,
    sigma_noise: float = 0.005,
) -> Path:
    backend = RectifiedCFGPPBackend(config, repo_dir=repo_dir, sigma_noise=sigma_noise).load()
    run_dir = Path(run_root)
    paths: list[dict[str, Any]] = []
    try:
        for sample in manifest.samples:
            start = time.perf_counter()
            image = backend.generate(sample.prompt)
            runtime_sec = time.perf_counter() - start
            image_path, metadata_path = _sample_output_paths(run_dir, manifest.benchmark, "rectified_cfgpp", sample.id)
            backend.save_image(image, image_path)
            write_json(
                {
                    "bench_method": "rectified_cfgpp",
                    "benchmark_sample": sample.to_dict(),
                    "runtime_sec": runtime_sec,
                    "runtime_config": config.to_dict(),
                    "rectified_cfgpp_repo": str(backend.repo_dir),
                    "rectified_cfgpp_commit": RECTIFIED_CFGPP_COMMIT,
                    "sigma_noise": sigma_noise,
                },
                metadata_path,
            )
            paths.append({"sample_id": sample.id, "image": str(image_path), "metadata": str(metadata_path)})
    finally:
        unload_model(backend)
    return _write_run_index(run_dir, manifest.benchmark, "rectified_cfgpp", paths)


def generate_methods(
    manifest_path: str | Path,
    run_root: str | Path,
    methods: list[str],
    config_path: str | Path | None = None,
    decomposition_path: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    rectified_repo_dir: str | Path | None = None,
    spfc_variant: str | None = None,
    spfc_method_label: str | None = None,
) -> dict[str, str]:
    """Run benchmark methods sequentially, unloading between phases."""

    manifest = PromptManifest.load(manifest_path)
    config = load_bench_config(config_path=config_path, seed=seed)
    outputs: dict[str, str] = {}
    for method in methods:
        if method == "spfc":
            if not decomposition_path:
                raise ValueError("--decompositions is required for SPFC generation.")
            decompositions = DecompositionManifest.load(decomposition_path)
            outputs[spfc_method_label or "spfc"] = str(
                generate_spfc(
                    manifest,
                    decompositions,
                    run_root,
                    config,
                    variant=spfc_variant,
                    method_label=spfc_method_label,
                )
            )
        elif method == "rectified_cfgpp":
            outputs[method] = str(generate_rectified_cfgpp(manifest, run_root, config, repo_dir=rectified_repo_dir))
        elif method == "base":
            outputs[method] = str(generate_base(manifest, run_root, config))
        else:
            raise ValueError(f"Unknown generation method: {method}")
    write_json(outputs, Path(run_root) / manifest.benchmark / "generation_indices.json")
    return outputs


def generate_spfc_ablation_suite(
    manifest_path: str | Path,
    decomposition_path: str | Path,
    run_root: str | Path,
    config_path: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    suite: str = "components",
) -> dict[str, str]:
    """Generate a named SPFC ablation/sweep suite."""

    manifest = PromptManifest.load(manifest_path)
    decompositions = DecompositionManifest.load(decomposition_path)
    base_config = load_bench_config(config_path=config_path, seed=seed)
    outputs: dict[str, str] = {}
    if suite == "components":
        variants = [
            ("spfc", "full"),
            ("spfc_no_consensus_gating", "no_consensus_gating"),
            ("spfc_no_target_consistency_gating", "no_target_consistency_gating"),
            ("spfc_no_ltp", "no_ltp"),
            ("spfc_no_source_flow", "no_source_flow"),
        ]
        for label, variant in variants:
            outputs[label] = str(generate_spfc(manifest, decompositions, run_root, base_config, variant=variant, method_label=label))
        outputs["base"] = str(generate_base(manifest, run_root, base_config))
    elif suite == "steering":
        for strength in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
            label = f"spfc_steering_{str(strength).replace('.', 'p')}"
            variant = f"steering_{strength}"
            outputs[label] = str(generate_spfc(manifest, decompositions, run_root, base_config, variant=variant, method_label=label))
    elif suite == "primitive_count":
        for count in [1, 2, 3, 4, 5]:
            label = f"spfc_max_primitives_{count}"
            variant = f"max_primitives_{count}"
            outputs[label] = str(generate_spfc(manifest, decompositions, run_root, base_config, variant=variant, method_label=label))
    elif suite == "schedule":
        for name, schedule in SPFC_SCHEDULE_ABLATIONS_1_INDEXED.items():
            config = copy.deepcopy(base_config)
            config.primitive_flow.aggregation_steps = one_indexed_to_zero_indexed(
                schedule,
                num_steps=config.sampler.num_inference_steps,
                required_count=16,
            )
            label = f"spfc_schedule_{name}"
            outputs[label] = str(generate_spfc(manifest, decompositions, run_root, config, method_label=label))
    else:
        raise ValueError("suite must be one of: components, steering, primitive_count, schedule")
    write_json(outputs, Path(run_root) / manifest.benchmark / f"spfc_{suite}_indices.json")
    return outputs


def load_index(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
