from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the locked CFG versus Rectified-CFG++ T2I-CompBench reproduction."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--rectified-repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--embedding-index", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    return parser.parse_args()


ARGS = parse_args()
sys.path.insert(0, str(ARGS.repo_root / "src"))

from aim_flow.eval_bench.generation import (  # noqa: E402
    RectifiedCFGPPBackend,
    _install_rectified_cfgpp_diffusers_compat,
    _validate_rectified_cfgpp_pipeline_import,
    load_bench_config,
    unload_model,
)
from aim_flow.sd3_backend import SD3Backend  # noqa: E402


RECORD = json.loads(ARGS.protocol.read_text(encoding="utf-8"))
P = RECORD["protocol"]
PROTOCOL_HASH = RECORD["protocol_hash"]
TASKS = [json.loads(line) for line in ARGS.tasks.read_text(encoding="utf-8").splitlines() if line]
EMBEDDING_RECORD = json.loads(ARGS.embedding_index.read_text(encoding="utf-8"))
CACHE_IDENTITY = EMBEDDING_RECORD.get("cache_identity", {})
CACHE_AUDIT = EMBEDDING_RECORD.get("singleton_cfg_exact_equivalence_audit")
EMBEDDINGS = EMBEDDING_RECORD["entries"]
EMBEDDING_ROOT = ARGS.embedding_index.parent
VALID_METHODS = {"cfg", "rectified_cfgpp"}
OVERWRITE_INVALID = os.environ.get("AIM_FLOW_OVERWRITE_INVALID", "0") == "1"

if set(ARGS.methods) - VALID_METHODS:
    raise ValueError(f"Unknown methods: {ARGS.methods}; expected a subset of {sorted(VALID_METHODS)}")
if int(P.get("t5_cache_batch_size", 0)) != 1:
    raise RuntimeError("The locked reproduction requires singleton T5 cache topology")
if CACHE_IDENTITY.get("encoding_batch_size") != 1:
    raise RuntimeError("Embedding cache was not encoded one prompt at a time")
if CACHE_IDENTITY.get("encoding_topology") != "singleton_per_exact_text":
    raise RuntimeError("Embedding cache topology does not match the locked reproduction")
if not CACHE_AUDIT or not CACHE_AUDIT.get("passed", False):
    raise RuntimeError("Embedding cache lacks a passing exact CFG-equivalence audit")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Each worker must see exactly one CUDA GPU through CUDA_VISIBLE_DEVICES")

torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.png")
    image.save(temporary, format="PNG")
    with Image.open(temporary) as check:
        check.load()
        if check.size != (int(P["width"]), int(P["height"])):
            raise RuntimeError(f"Wrong generated image size: {check.size}")
    os.replace(temporary, path)


class EmbeddingStore:
    def __init__(self) -> None:
        self.cache: dict[str, dict[str, torch.Tensor]] = {}

    def get(self, text: str) -> dict[str, torch.Tensor]:
        if text not in EMBEDDINGS:
            raise KeyError(f"Text absent from full-T5 cache: {text!r}")
        if text not in self.cache:
            path = EMBEDDING_ROOT / EMBEDDINGS[text]["file"]
            expected_sha256 = EMBEDDINGS[text].get("sha256")
            if expected_sha256 and sha256_file(path) != expected_sha256:
                raise RuntimeError(f"Cached embedding checksum mismatch: {path}")
            tensors = load_safetensors(str(path), device="cpu")
            if tuple(tensors["prompt_embeds"].shape) != (1, 333, 4096):
                raise RuntimeError(f"Invalid cached prompt shape: {path}")
            if tuple(tensors["pooled_prompt_embeds"].shape) != (1, 2048):
                raise RuntimeError(f"Invalid cached pooled-prompt shape: {path}")
            if any(tensor.dtype != torch.float16 for tensor in tensors.values()):
                raise RuntimeError(f"Cached embeddings are not FP16: {path}")
            self.cache[text] = tensors
        return self.cache[text]


STORE = EmbeddingStore()


def method_scale(method: str) -> float:
    if method == "cfg":
        return float(P["cfg_guidance_scale"])
    if method == "rectified_cfgpp":
        return float(P["rectified_guidance_scale"])
    raise ValueError(method)


def make_config(method: str, seed: int):
    cfg = load_bench_config(
        seed=seed,
        num_inference_steps=int(P["num_inference_steps"]),
        height=int(P["height"]),
        width=int(P["width"]),
        guidance_scale=method_scale(method),
    )
    cfg.model.model_id = str(ARGS.model_snapshot)
    cfg.model.dtype = P["dtype"]
    cfg.model.enable_model_cpu_offload = True
    cfg.model.load_t5_text_encoder = False
    cfg.model.enable_vae_slicing = True
    return cfg


def scheduler_record(pipe: Any) -> dict[str, Any]:
    config = dict(pipe.scheduler.config)
    semantic_config = dict(config)
    semantic_config.pop("_use_default_values", None)
    return {
        "class": f"{pipe.scheduler.__class__.__module__}.{pipe.scheduler.__class__.__name__}",
        "config": config,
        "semantic_config": semantic_config,
        "fingerprint": canonical_hash(semantic_config),
    }


def load_generator_only_pipeline(custom_pipeline: Path | None = None):
    from diffusers import StableDiffusion3Pipeline

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "text_encoder": None,
        "tokenizer": None,
        "text_encoder_2": None,
        "tokenizer_2": None,
        "text_encoder_3": None,
        "tokenizer_3": None,
    }
    if custom_pipeline is not None:
        kwargs["custom_pipeline"] = str(custom_pipeline)
    pipe = StableDiffusion3Pipeline.from_pretrained(str(ARGS.model_snapshot), **kwargs)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def expected_metadata(method: str, task: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": "t2i_compbench_rectified_cfgpp_paper_reproduction_seed25",
        "method": method,
        "prompt": task["prompt"],
        "subset_sample_id": task["subset_sample_id"],
        "category": task["category"],
        "official_prompt_index": task["official_prompt_index"],
        "selected_category_rank": task["selected_category_rank"],
        "sample_index": task["sample_index"],
        "question_id": task["question_id"],
        "seed": task["seed"],
        "protocol_hash": PROTOCOL_HASH,
        "model_id": P["model_id"],
        "model_revision": P["model_revision"],
        "height": P["height"],
        "width": P["width"],
        "num_inference_steps": P["num_inference_steps"],
        "guidance_parameter_name": "omega" if method == "cfg" else "lambda",
        "guidance_scale": method_scale(method),
        "negative_prompt": P["negative_prompt"],
        "full_text_encoders": P["full_text_encoders"],
        "full_t5_conditioning": True,
        "physical_gpu": ARGS.physical_gpu,
    }


def output_paths(method: str, task: dict[str, Any]) -> tuple[Path, Path]:
    stage = ARGS.artifact_root / "t2i_compbench" / method / task["category"]
    stem = f"{task['prompt']}_{int(task['question_id']):06d}"
    image_path = stage / "samples" / f"{stem}.png"
    metadata_path = stage / "sample_metadata" / f"{int(task['question_id']):06d}.json"
    return image_path, metadata_path


def output_valid(image_path: Path, metadata_path: Path, expected: dict[str, Any]) -> bool:
    if not image_path.exists() and not metadata_path.exists():
        return False
    if image_path.exists() != metadata_path.exists():
        image_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            valid = image.size == (int(P["width"]), int(P["height"]))
        valid = valid and all(metadata.get(key) == value for key, value in expected.items())
        valid = valid and metadata.get("image_sha256") == sha256_file(image_path)
    except Exception:
        valid = False
    if valid:
        return True
    if OVERWRITE_INVALID:
        image_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        return False
    raise RuntimeError(f"Existing output is corrupt or belongs to another protocol: {image_path}")


def pipeline_embeddings(prompt: str, backend: Any) -> dict[str, torch.Tensor]:
    positive = STORE.get(prompt)
    negative = STORE.get(P["negative_prompt"])
    device = backend.execution_device if hasattr(backend, "execution_device") else torch.device("cuda:0")
    dtype = backend.dtype
    return {
        "prompt_embeds": positive["prompt_embeds"].to(device=device, dtype=dtype),
        "pooled_prompt_embeds": positive["pooled_prompt_embeds"].to(device=device, dtype=dtype),
        "negative_prompt_embeds": negative["prompt_embeds"].to(device=device, dtype=dtype),
        "negative_pooled_prompt_embeds": negative["pooled_prompt_embeds"].to(device=device, dtype=dtype),
    }


def run_method(method: str) -> None:
    seed_everything(int(P["generation_seed_start"]))
    cfg = make_config(method, int(P["generation_seed_start"]))
    if method == "cfg":
        backend: Any = SD3Backend(cfg)
        backend.pipe = load_generator_only_pipeline()
    else:
        backend = RectifiedCFGPPBackend(
            cfg,
            repo_dir=ARGS.rectified_repo,
            sigma_noise=float(P["rectified_sigma_noise"]),
        )
        _install_rectified_cfgpp_diffusers_compat()
        _validate_rectified_cfgpp_pipeline_import(ARGS.rectified_repo)
        backend.pipe = load_generator_only_pipeline(ARGS.rectified_repo / "rect-cfg-SD3-pipeline")

    scheduler = scheduler_record(backend.pipe)
    if "FlowMatchEulerDiscreteScheduler" not in scheduler["class"]:
        raise RuntimeError(f"Unexpected scheduler: {scheduler['class']}")

    completed = 0
    started_method = time.perf_counter()
    try:
        for task in tqdm(TASKS, desc=f"GPU{ARGS.physical_gpu}/{method}", unit="image"):
            image_path, metadata_path = output_paths(method, task)
            expected = expected_metadata(method, task)
            if output_valid(image_path, metadata_path, expected):
                completed += 1
                continue

            seed_everything(int(task["seed"]))
            started = time.perf_counter()
            embeds = pipeline_embeddings(task["prompt"], backend)
            generator = torch.Generator(device="cpu").manual_seed(int(task["seed"]))
            common = {
                "prompt": None,
                "negative_prompt": None,
                "height": int(P["height"]),
                "width": int(P["width"]),
                "num_inference_steps": int(P["num_inference_steps"]),
                "num_images_per_prompt": 1,
                "generator": generator,
                **embeds,
            }
            if method == "cfg":
                result = backend.pipe(guidance_scale=float(P["cfg_guidance_scale"]), **common)
            else:
                result = backend.pipe(
                    true_cfg=float(P["rectified_guidance_scale"]),
                    sigma_noise=float(P["rectified_sigma_noise"]),
                    **common,
                )
            image = result.images[0]
            runtime_seconds = time.perf_counter() - started
            del result, embeds

            atomic_save_png(image, image_path)
            metadata = {
                **expected,
                "runtime_seconds": runtime_seconds,
                "image_sha256": sha256_file(image_path),
                "scheduler": scheduler,
                "embedding_key_positive": EMBEDDINGS[task["prompt"]]["key"],
                "embedding_key_negative": EMBEDDINGS[P["negative_prompt"]]["key"],
                "rectified_cfgpp_commit": (
                    RECORD["repository_pins"]["rectified_cfgpp"]["commit"]
                    if method == "rectified_cfgpp"
                    else None
                ),
                "rectified_sigma_noise": (
                    P["rectified_sigma_noise"] if method == "rectified_cfgpp" else None
                ),
                "rectified_implementation": (
                    RECORD["reproduction_scope"]["rectified_implementation"]
                    if method == "rectified_cfgpp"
                    else None
                ),
            }
            write_json(metadata, metadata_path)
            completed += 1
            del image
            torch.cuda.empty_cache()
    finally:
        write_json(
            {
                "method": method,
                "physical_gpu": ARGS.physical_gpu,
                "completed": completed,
                "expected": len(TASKS),
                "elapsed_seconds": time.perf_counter() - started_method,
                "guidance_scale": method_scale(method),
                "scheduler": scheduler,
                "protocol_hash": PROTOCOL_HASH,
            },
            ARGS.artifact_root / "t2i_compbench" / method / "generation_run.json",
        )
        unload_model(backend)
        del backend
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


for requested_method in ARGS.methods:
    run_method(requested_method)
