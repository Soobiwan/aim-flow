"""Helpers for running multiple configured decompositions."""

from __future__ import annotations

from pathlib import Path

from aim_flow.config import RunConfig
from aim_flow.decomposition import load_prompt_decomposition_from_yaml
from aim_flow.pipeline import run_aim_flow_comparison
from aim_flow.utils import ensure_dir


def run_prompt_key(
    config: RunConfig,
    prompts_path: str | Path,
    prompt_key: str,
    output_dir: str | Path,
    modes: list[str] | None = None,
) -> dict[str, Path]:
    """Run comparison for one prompt key from a YAML file."""

    decomposition = load_prompt_decomposition_from_yaml(prompts_path, prompt_key)
    return run_aim_flow_comparison(decomposition, config, ensure_dir(output_dir), modes=modes)


def run_all_prompt_keys(
    config: RunConfig,
    prompts_path: str | Path,
    prompt_keys: list[str],
    output_root: str | Path,
    modes: list[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """Run comparison for several prompt keys."""

    root = ensure_dir(output_root)
    return {
        key: run_prompt_key(config, prompts_path, key, root / key, modes=modes)
        for key in prompt_keys
    }

