"""Generation orchestration for the SPFC evaluation bench."""

from __future__ import annotations

import copy
import gc
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

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
from aim_flow.sd3_backend import SD3Backend, _load_pipeline_with_fallback, _model_load_error_message
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
    if variant == "target_only_uniform":
        tuned.primitive_flow.use_consensus_gating = False
        tuned.primitive_flow.use_target_consistency_gating = True
        tuned.primitive_flow.uniform_condition_weights = True
        tuned.primitive_flow.source_weight = 1.0
        tuned.primitive_flow.target_weight = 1.0
    elif variant == "no_consensus_gating":
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


def _install_rectified_cfgpp_diffusers_compat() -> None:
    """Install small compatibility shims needed by the pinned Rectified-CFG++ pipeline."""

    import diffusers.loaders as loaders

    if hasattr(loaders, "SD3IPAdapterMixin"):
        return

    class SD3IPAdapterMixin:
        @property
        def is_ip_adapter_active(self) -> bool:
            return False

        def load_ip_adapter(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError(
                "This Diffusers version does not provide SD3 IP-Adapter support; "
                "Rectified-CFG++ generation does not use IP-Adapters."
            )

        def unload_ip_adapter(self) -> None:
            return None

        def set_ip_adapter_scale(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError(
                "This Diffusers version does not provide SD3 IP-Adapter support; "
                "Rectified-CFG++ generation does not use IP-Adapters."
            )

    loaders.SD3IPAdapterMixin = SD3IPAdapterMixin


def _validate_rectified_cfgpp_pipeline_import(repo_dir: Path) -> None:
    """Fail early with a focused message if the external custom pipeline cannot import."""

    pipeline_path = repo_dir / "rect-cfg-SD3-pipeline" / "pipeline.py"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Rectified-CFG++ custom pipeline not found: {pipeline_path}")

    module_name = f"_aim_flow_rectified_cfgpp_pipeline_{abs(hash(str(pipeline_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create an import spec for Rectified-CFG++ pipeline: {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(f"Failed to import Rectified-CFG++ custom pipeline at {pipeline_path}: {exc}") from exc


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
        _install_rectified_cfgpp_diffusers_compat()
        _validate_rectified_cfgpp_pipeline_import(self.repo_dir)

        from diffusers import StableDiffusion3Pipeline

        token = get_hf_token()
        kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
            "custom_pipeline": str(self.repo_dir / "rect-cfg-SD3-pipeline"),
            "text_encoder_3": None,
            "tokenizer_3": None,
        }
        variant = os.environ.get("SD3_MODEL_VARIANT")
        if variant:
            kwargs["variant"] = variant
        if token:
            kwargs["token"] = token
        try:
            self.pipe = _load_pipeline_with_fallback(StableDiffusion3Pipeline, self.config.model.model_id, kwargs)
        except Exception as exc:
            raise RuntimeError(_model_load_error_message(self.config.model.model_id, token)) from exc

        if self.config.model.defer_model_cpu_offload and self.config.model.enable_model_cpu_offload:
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(self.device)
        elif self.config.model.enable_sequential_cpu_offload and hasattr(self.pipe, "enable_sequential_cpu_offload"):
            self.pipe.enable_sequential_cpu_offload()
        elif self.config.model.enable_model_cpu_offload and hasattr(self.pipe, "enable_model_cpu_offload"):
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


def _existing_output_index(manifest: PromptManifest, run_root: Path, method: str) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    for sample in manifest.samples:
        image_path, metadata_path = _sample_output_paths(run_root, manifest.benchmark, method, sample.id)
        if not image_path.exists() or not metadata_path.exists():
            return []
        paths.append({"sample_id": sample.id, "image": str(image_path), "metadata": str(metadata_path)})
    return paths


def _iter_samples_with_progress(manifest: PromptManifest, method: str):
    total = len(manifest.samples)
    progress = tqdm(manifest.samples, total=total, unit="image", dynamic_ncols=True)
    try:
        for index, sample in enumerate(progress, start=1):
            progress.set_description(f"{manifest.benchmark}/{method} image {index}/{total}")
            progress.set_postfix_str(f"left={total - index}", refresh=True)
            yield sample
    finally:
        progress.close()


def generate_spfc(
    manifest: PromptManifest,
    decompositions: DecompositionManifest,
    run_root: str | Path,
    config: RunConfig,
    variant: str | None = None,
    method_label: str | None = None,
    skip_existing: bool = False,
) -> Path:
    """Generate all SPFC images for a manifest."""

    tuned = apply_spfc_variant(config, variant)
    items = decompositions.item_by_id()
    missing = [sample.id for sample in manifest.samples if sample.id not in items]
    if missing:
        raise KeyError(f"Missing decompositions for sample ids: {missing[:5]}")
    run_dir = Path(run_root)
    method_name = method_label or ("spfc" if not variant or variant == "full" else f"spfc_{variant}")
    if skip_existing:
        existing = _existing_output_index(manifest, run_dir, method_name)
        if existing:
            return _write_run_index(run_dir, manifest.benchmark, method_name, existing)
    backend = SD3Backend(tuned).load()
    sampler = AIMFlowSampler(backend, tuned)
    paths: list[dict[str, Any]] = []
    try:
        for sample in _iter_samples_with_progress(manifest, method_name):
            image_path, metadata_path = _sample_output_paths(run_dir, manifest.benchmark, method_name, sample.id)
            if skip_existing and image_path.exists() and metadata_path.exists():
                paths.append({"sample_id": sample.id, "image": str(image_path), "metadata": str(metadata_path)})
                continue
            flow_set: PrimitiveFlowSet = items[sample.id].to_flow_set()
            start = time.perf_counter()
            image, metadata = sampler.generate_sparse_primitive_flow(flow_set, mode="primitive_flow_sparse")
            runtime_sec = time.perf_counter() - start
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


def generate_sd3_baseline(
    manifest: PromptManifest,
    run_root: str | Path,
    config: RunConfig,
    method_name: str,
    skip_existing: bool = False,
) -> Path:
    """Generate all images for a plain SD3 prompt-only baseline."""

    run_dir = Path(run_root)
    if skip_existing:
        existing = _existing_output_index(manifest, run_dir, method_name)
        if existing:
            return _write_run_index(run_dir, manifest.benchmark, method_name, existing)
    backend = SD3Backend(config).load()
    paths: list[dict[str, Any]] = []
    try:
        for sample in _iter_samples_with_progress(manifest, method_name):
            image_path, metadata_path = _sample_output_paths(run_dir, manifest.benchmark, method_name, sample.id)
            if skip_existing and image_path.exists() and metadata_path.exists():
                paths.append({"sample_id": sample.id, "image": str(image_path), "metadata": str(metadata_path)})
                continue
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
            backend.save_image(image, image_path)
            write_json(
                {
                    "bench_method": method_name,
                    "benchmark_sample": sample.to_dict(),
                    "runtime_sec": runtime_sec,
                    "runtime_config": config.to_dict(),
                },
                metadata_path,
            )
            paths.append({"sample_id": sample.id, "image": str(image_path), "metadata": str(metadata_path)})
    finally:
        unload_model(backend)
    return _write_run_index(run_dir, manifest.benchmark, method_name, paths)


def generate_cfg(manifest: PromptManifest, run_root: str | Path, config: RunConfig, skip_existing: bool = False) -> Path:
    """Generate standard SD3 images with classifier-free guidance."""

    return generate_sd3_baseline(manifest, run_root, config, "cfg", skip_existing=skip_existing)


def generate_base(manifest: PromptManifest, run_root: str | Path, config: RunConfig, skip_existing: bool = False) -> Path:
    """Generate standard SD3 images under the provided config as the base label."""

    return generate_sd3_baseline(manifest, run_root, config, "base", skip_existing=skip_existing)


def generate_rectified_cfgpp(
    manifest: PromptManifest,
    run_root: str | Path,
    config: RunConfig,
    repo_dir: str | Path | None = None,
    sigma_noise: float = 0.005,
    skip_existing: bool = False,
) -> Path:
    run_dir = Path(run_root)
    if skip_existing:
        existing = _existing_output_index(manifest, run_dir, "rectified_cfgpp")
        if existing:
            return _write_run_index(run_dir, manifest.benchmark, "rectified_cfgpp", existing)
    backend = RectifiedCFGPPBackend(config, repo_dir=repo_dir, sigma_noise=sigma_noise).load()
    paths: list[dict[str, Any]] = []
    try:
        for sample in _iter_samples_with_progress(manifest, "rectified_cfgpp"):
            image_path, metadata_path = _sample_output_paths(run_dir, manifest.benchmark, "rectified_cfgpp", sample.id)
            if skip_existing and image_path.exists() and metadata_path.exists():
                paths.append({"sample_id": sample.id, "image": str(image_path), "metadata": str(metadata_path)})
                continue
            start = time.perf_counter()
            image = backend.generate(sample.prompt)
            runtime_sec = time.perf_counter() - start
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
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    skip_existing: bool = False,
) -> dict[str, str]:
    """Run benchmark methods sequentially, unloading between phases."""

    manifest = PromptManifest.load(manifest_path)
    config = load_bench_config(config_path=config_path, seed=seed, guidance_scale=guidance_scale)
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
                    skip_existing=skip_existing,
                )
            )
        elif method == "rectified_cfgpp":
            outputs[method] = str(
                generate_rectified_cfgpp(
                    manifest,
                    run_root,
                    config,
                    repo_dir=rectified_repo_dir,
                    skip_existing=skip_existing,
                )
            )
        elif method == "cfg":
            outputs[method] = str(generate_cfg(manifest, run_root, config, skip_existing=skip_existing))
        elif method == "base":
            outputs[method] = str(generate_base(manifest, run_root, config, skip_existing=skip_existing))
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
