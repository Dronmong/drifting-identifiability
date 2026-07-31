"""Create the immutable source/config manifest for the K=200 confirmation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .diagnostics import write_json
from .f1_k200 import (
    HERE,
    frozen_config,
    protocol_sha256,
    source_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=HERE / "f1_k200_freeze.json")
    args = parser.parse_args()
    payload = {
        "status": "f1-k200-frozen-design",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha256(),
        "source_sha256": source_manifest(),
        "config": frozen_config(),
        "note": (
            "This manifest was created before the K=200 preflight artifacts "
            "and confirmation arms. The runner refuses source/config drift."),
    }
    digest = write_json(args.out, payload)
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()

