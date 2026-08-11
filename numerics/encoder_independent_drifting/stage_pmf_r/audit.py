"""Source identity and launch interlock for the developmental S3R package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def source_manifest() -> list[str]:
    paths = [
        *HERE.glob("*.py"),
        ROOT / "numerics" / "EncoderIndependentS3FailureResearch.md",
        ROOT / "numerics" / "encoder_independent_drifting" / "config.py",
        ROOT / "numerics" / "encoder_independent_drifting" / "device.py",
        ROOT / "numerics" / "encoder_independent_drifting" / "diagnostics.py",
        *(ROOT / "numerics" / "encoder_independent_drifting" / "stage_pmf").glob(
            "*.py"
        ),
    ]
    return sorted(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in paths
        if path.is_file()
    )


def source_digest() -> str:
    digest = hashlib.sha256()
    for relative in source_manifest():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def require_developmental_preflight(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError("S3R RUN BLOCKED: preflight artifact is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "s3r-preflight-passed":
        raise RuntimeError("S3R RUN BLOCKED: preflight did not pass")
    if payload.get("profile", {}).get("name") != "developmental":
        raise RuntimeError("S3R RUN BLOCKED: smoke preflight cannot authorize a screen")
    if payload.get("source_sha256") != source_digest():
        raise RuntimeError("S3R RUN BLOCKED: source changed after preflight")
    if payload.get("launch_authorized") is not False:
        raise RuntimeError("S3R RUN BLOCKED: malformed developmental artifact")
    return payload
