"""Make a labeled image grid from existing image files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim_flow.visualize import make_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a comparison grid.")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths in display order.")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels matching image paths.")
    parser.add_argument("--output", required=True, help="Output grid path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = make_image_grid(args.images, args.labels, args.output)
    print(f"grid: {output}")


if __name__ == "__main__":
    main()

