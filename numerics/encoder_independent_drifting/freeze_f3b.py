"""Freeze a development-selected F3B B0 configuration for confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .diagnostics import write_json
from .f3b_freeze import HERE, frozen_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("compact", "reference_scale"), required=True
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "f3b_freeze.json")
    args = parser.parse_args()
    payload = frozen_payload(
        args.profile, args.steps, args.nfe, args.development.resolve()
    )
    digest = write_json(args.out, payload)
    print("=== F3B B0 CONFIRMATION FREEZE ===")
    print(f"profile={args.profile} steps={args.steps} nfe={args.nfe}")
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
