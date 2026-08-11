"""Source/config manifests and the explicit full-run authorization gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ..config import config_digest
from .config import INITIAL_UNITS, PMFProfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PROTOCOL = REPO / "numerics" / "EncoderIndependentPMFS3Protocol.md"
AUTHORIZATION = HERE / "s3_launch_authorization.json"

SOURCE_FILES = (
    REPO / "numerics" / "encoder_independent_drifting" / "config.py",
    REPO / "numerics" / "encoder_independent_drifting" / "device.py",
    REPO / "numerics" / "encoder_independent_drifting" / "diagnostics.py",
    REPO / "numerics" / "encoder_independent_drifting" / "fid.py",
    REPO / "numerics" / "encoder_independent_drifting" / "appearance.py",
    REPO / "numerics" / "encoder_independent_drifting" / "diagnose_phase20.py",
    REPO / "numerics" / "encoder_independent_drifting" / "stage_b2" / "metrics.py",
    HERE / "__init__.py",
    HERE / "config.py",
    HERE / "model.py",
    HERE / "objective.py",
    HERE / "data.py",
    HERE / "training.py",
    HERE / "audit.py",
    HERE / "preflight.py",
    HERE / "sanity.py",
    HERE / "run_two_unit.py",
    HERE / "launch_s3.ps1",
    REPO / "numerics" / "encoder_independent_drifting" / "tests" / "test_pmf_s3.py",
    REPO / "numerics" / "encoder_independent_drifting" / "tests" / "run_all.py",
    PROTOCOL,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, str]:
    missing = [str(path) for path in SOURCE_FILES if not path.exists()]
    if missing:
        raise RuntimeError(f"S3 source manifest is incomplete: {missing}")
    return {
        str(path.relative_to(REPO)).replace("\\", "/"): sha256(path)
        for path in SOURCE_FILES
    }


def source_digest() -> str:
    encoded = json.dumps(source_manifest(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def profile_payload(selected: PMFProfile) -> dict:
    return asdict(selected)


def profile_digest(selected: PMFProfile) -> str:
    return config_digest(selected)


def require_launch_authorization(selected: PMFProfile, path: Path) -> dict:
    """Accept only an explicit post-audit token matching exact current sources."""
    if not path.exists():
        raise RuntimeError(
            "FULL S3 RUN BLOCKED: no post-audit authorization file exists. "
            "The requested stopping point is immediately before launch."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "AUTHORIZED-AFTER-S3-AUDIT",
        "profile": selected.name,
        "profile_sha256": profile_digest(selected),
        "source_sha256": source_digest(),
        "units": list(INITIAL_UNITS),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"FULL S3 RUN BLOCKED: authorization field {key!r} does not "
                f"match the current audited build"
            )
    return payload
