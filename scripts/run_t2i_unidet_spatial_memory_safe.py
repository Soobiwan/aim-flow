from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned official T2I-CompBench 2D spatial scorer with a "
            "smaller inference batch, without modifying its checkout."
        )
    )
    parser.add_argument("--official-script", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--outpath", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    official_script = args.official_script.resolve()
    source_bytes = official_script.read_bytes()
    actual_sha256 = sha256_bytes(source_bytes)
    if actual_sha256 != args.expected_source_sha256:
        raise RuntimeError(
            "Official 2D spatial scorer source hash mismatch: "
            f"expected {args.expected_source_sha256}, found {actual_sha256}"
        )

    source = source_bytes.decode("utf-8")
    # The pinned upstream file uses CRLF. Replace only the audited statement so
    # the launcher is independent of a checkout's newline normalization.
    original = "    batch_size = 64"
    replacement = f"    batch_size = {args.batch_size}"
    if source.count(original) != 1:
        raise RuntimeError(
            "Pinned scorer no longer contains exactly one audited batch-size literal"
        )
    adapted_source = source.replace(original, replacement, 1)

    outpath = args.outpath.resolve()
    scorer_root = official_script.parent
    os.chdir(scorer_root)
    sys.path.insert(0, str(scorer_root))
    sys.argv = [str(official_script), "--outpath", str(outpath)]
    print(
        "Running pinned official UniDet scorer with memory-only adaptation: "
        f"batch_size=64 -> {args.batch_size}; source_sha256={actual_sha256}",
        flush=True,
    )
    namespace = {
        "__name__": "__main__",
        "__file__": str(official_script),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(adapted_source, str(official_script), "exec"), namespace)


if __name__ == "__main__":
    main()
