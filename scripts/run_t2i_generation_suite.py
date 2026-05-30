"""Run GenEval comparison generation across two local GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GenEval generation two jobs at a time.")
    parser.add_argument("--repo-dir", type=Path, default=ROOT)
    parser.add_argument("--gpu-0", default="0")
    parser.add_argument("--gpu-1", default="1")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing method outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate every child script without loading models.")
    parser.add_argument("--skip-target-only", action="store_true", help="Skip the final target-only uniform method.")
    parser.add_argument("--logs-dir", type=Path, help="Defaults to <repo>/logs/geneval_generation.")
    return parser.parse_args()


def child_command(repo_dir: Path, script_name: str, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(repo_dir / "scripts" / script_name), "--repo-dir", str(repo_dir)]
    if args.overwrite:
        command.append("--overwrite")
    if args.dry_run:
        command.append("--dry-run")
    return command


def run_pair(
    repo_dir: Path,
    logs_dir: Path,
    jobs: list[tuple[str, str, str]],
    args: argparse.Namespace,
) -> None:
    processes = []
    for label, script_name, gpu in jobs:
        command = child_command(repo_dir, script_name, args)
        log_path = logs_dir / f"{label}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTHONUNBUFFERED", "1")
        print(f"[start] {label} on GPU {gpu}")
        print(f"        log: {log_path}")
        processes.append(
            (
                label,
                log_path,
                log_handle,
                subprocess.Popen(
                    command,
                    cwd=repo_dir,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                ),
            )
        )

    failures = []
    for label, log_path, log_handle, process in processes:
        returncode = process.wait()
        log_handle.close()
        if returncode:
            failures.append((label, returncode, log_path))
        else:
            print(f"[done]  {label}")
    if failures:
        details = ", ".join(f"{label} exited {returncode}; see {log_path}" for label, returncode, log_path in failures)
        raise RuntimeError(details)


def main() -> None:
    args = parse_args()
    repo_dir = args.repo_dir.expanduser().resolve()
    logs_dir = (args.logs_dir or repo_dir / "logs" / "geneval_generation").expanduser().resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_pair(
        repo_dir,
        logs_dir,
        [
            ("base", "generate_t2i_base.py", args.gpu_0),
            ("cfg", "generate_t2i_cfg.py", args.gpu_1),
        ],
        args,
    )
    run_pair(
        repo_dir,
        logs_dir,
        [
            ("spfc", "generate_t2i_spfc.py", args.gpu_0),
            ("rectified_cfgpp", "generate_t2i_rectified_cfgpp.py", args.gpu_1),
        ],
        args,
    )
    if not args.skip_target_only:
        run_pair(
            repo_dir,
            logs_dir,
            [("spfc_target_only_uniform", "generate_t2i_target_only_uniform.py", args.gpu_0)],
            args,
        )
    print("All requested GenEval generation jobs completed.")


if __name__ == "__main__":
    main()
