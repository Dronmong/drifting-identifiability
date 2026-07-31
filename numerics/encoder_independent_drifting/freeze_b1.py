"""Freeze B1 only after calibration and the paired B0 baseline return GO."""

from __future__ import annotations

import argparse
from pathlib import Path

from .b1_freeze import (
    DEFAULT_B0_RESULT,
    DEFAULT_BASELINE,
    DEFAULT_CALIBRATION,
    DEFAULT_FREEZE,
    frozen_payload,
)
from .diagnostics import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", type=Path, default=DEFAULT_B0_RESULT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    payload = frozen_payload(args.calibration, args.baseline, args.b0)
    digest = write_json(args.out, payload)
    print(
        f"froze B1 lambda_event={payload['lambda_event']:.6g}; "
        f"effective gradient ratio={payload['effective_gradient_ratio']:.4f}"
    )
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
