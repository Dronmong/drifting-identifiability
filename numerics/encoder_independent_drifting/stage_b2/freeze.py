"""Freeze a GO B2 preflight and fresh paired baseline before confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..diagnostics import write_json
from .artifacts import (
    DEFAULT_B0_RESULT,
    DEFAULT_B1_RESULT,
    DEFAULT_BASELINE,
    DEFAULT_FREEZE,
    DEFAULT_PREFLIGHT,
    frozen_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--b0", type=Path, default=DEFAULT_B0_RESULT)
    parser.add_argument("--b1", type=Path, default=DEFAULT_B1_RESULT)
    parser.add_argument("--out", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    payload = frozen_payload(args.preflight, args.baseline, args.b0, args.b1)
    digest = write_json(args.out, payload)
    print(f"B2 confirmation freeze: {digest}")


if __name__ == "__main__":
    main()
