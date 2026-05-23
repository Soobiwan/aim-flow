"""Validate an SPFC decomposition JSON file against a prompt manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.eval_bench.schemas import DecompositionManifest, PromptManifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SPFC decomposition JSON.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--decompositions", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = PromptManifest.load(args.manifest)
    decompositions = DecompositionManifest.load(args.decompositions)
    items = decompositions.item_by_id()
    missing = [sample.id for sample in manifest.samples if sample.id not in items]
    extra = [item_id for item_id in items if item_id not in manifest.sample_by_id()]
    if missing:
        raise SystemExit(f"Missing decompositions for {len(missing)} sample(s): {missing[:5]}")
    if extra:
        raise SystemExit(f"Decompositions contain {len(extra)} extra sample id(s): {extra[:5]}")
    print(f"valid_decompositions: {args.decompositions}")
    print(f"items: {len(decompositions.items)}")


if __name__ == "__main__":
    main()
