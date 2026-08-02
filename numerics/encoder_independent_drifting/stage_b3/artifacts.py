"""Hash-bound prerequisites and executable boundary for Stage B3."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..diagnostics import ROOT
from ..stage_b2.artifacts import (
    DEFAULT_B0_RESULT,
    DEFAULT_B1_RESULT,
    file_sha256,
    load_passing_prerequisites,
    verify_sidecar,
)
from ..stage_b25.artifacts import (
    DEFAULT_DATA,
    DEFAULT_PROVENANCE,
    load_data_provenance,
    load_frozen_inputs,
)
from .core import B3Config, b3_config

PACKAGE = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "numerics" / "EncoderIndependentB3Protocol.md"
DEFAULT_B2_RESULT = PACKAGE / "stage_b2" / "b2_confirmatory.json"
DEFAULT_PHASE30 = PACKAGE / "phase30.json"
DEFAULT_PREFLIGHT = HERE / "b3_preflight.json"
DEFAULT_RESULT = HERE / "b3_matched_reference.json"

_DEPENDENCIES = (
    "appearance.py",
    "cifar.py",
    "config.py",
    "device.py",
    "diagnose_phase25.py",
    "diagnostics.py",
    "f3b.py",
    "f3b_freeze.py",
    "fid.py",
    "fixed_features.py",
    "kernel_gradient.py",
    "kernels.py",
    "models.py",
    "objectives.py",
    "stage_b2/artifacts.py",
    "stage_b2/core.py",
    "stage_b25/artifacts.py",
    "stage_b25/evaluation.py",
)


def config_payload(config: B3Config) -> dict:
    config.validate()
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def source_manifest() -> dict[str, str]:
    paths = [PACKAGE / item for item in _DEPENDENCIES]
    paths.extend(sorted(HERE.rglob("*.py")))
    paths.extend(sorted(HERE.rglob("*.ps1")))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"B3 executable dependency missing: {missing}")
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
        for path in sorted(set(paths))
    }


def _load_b2(path: Path = DEFAULT_B2_RESULT) -> dict:
    digest = verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "b2-confirmatory":
        raise RuntimeError("not a B2 confirmatory artifact")
    if payload.get("verdict", {}).get("decision") != "PASS":
        raise RuntimeError("B3 requires the passing frozen B2 result")
    if set(payload.get("checkpoints", {})) != {"300", "301", "302"}:
        raise RuntimeError("B2 is missing a frozen paired checkpoint")
    for unit, record in payload["checkpoints"].items():
        checkpoint = Path(record["path"])
        if not checkpoint.is_file() or file_sha256(checkpoint) != record["sha256"]:
            raise RuntimeError(f"B2 checkpoint {unit} is missing or changed")
    payload["artifact_sha256"] = digest
    payload["artifact_path"] = str(path.resolve())
    return payload


def load_reference_artifacts(
    b0_path: Path = DEFAULT_B0_RESULT,
    b1_path: Path = DEFAULT_B1_RESULT,
    b2_path: Path = DEFAULT_B2_RESULT,
) -> dict[str, dict]:
    b0, b1 = load_passing_prerequisites(b0_path, b1_path)
    b2 = _load_b2(b2_path)
    if b2.get("b0_artifact_sha256") != b0["artifact_sha256"]:
        raise RuntimeError("B2 and B3 adopted different B0 artifacts")
    if b2.get("b1_artifact_sha256") != b1["artifact_sha256"]:
        raise RuntimeError("B2 and B3 adopted different B1 artifacts")
    if not (b0.get("profile") == b1.get("profile") == b2.get("profile")):
        raise RuntimeError("B0/B1/B2 profiles differ")
    return {"B0": b0, "B1": b1, "B2": b2}


def phase30_source_comparison(path: Path = DEFAULT_PHASE30) -> dict:
    """Record historical source drift; never pretend current code is old code."""
    digest = verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "phase30-capacity-and-batch":
        raise RuntimeError("not the Phase-30 artifact")
    historical = payload.get("provenance", {}).get("source_sha256", {})
    names = (
        "cifar.py",
        "config.py",
        "diagnose_phase25.py",
        "fixed_features.py",
        "kernel_gradient.py",
        "kernels.py",
        "models.py",
        "objectives.py",
        "run_phase30.py",
    )
    rows = {}
    for name in names:
        key = f"numerics/encoder_independent_drifting/{name}"
        current_path = PACKAGE / name
        current = file_sha256(current_path)
        old = historical.get(key)
        rows[name] = {
            "historical_sha256": old,
            "current_sha256": current,
            "identical": bool(old == current),
        }
    return {
        "artifact_path": str(path.resolve()),
        "artifact_sha256": digest,
        "files": rows,
        "all_relevant_sources_identical": all(
            row["identical"] for row in rows.values()
        ),
        "reading": "equation-level compatibility is tested separately; changed source hashes forbid an exact historical-replay claim",
    }


def preflight_header(config: B3Config | None = None) -> dict:
    config = config or b3_config()
    references = load_reference_artifacts()
    frozen = load_frozen_inputs()
    external = load_data_provenance(
        DEFAULT_DATA,
        DEFAULT_PROVENANCE,
        b2_freeze=frozen["b2"],
    )
    return {
        "status": "b3-preflight",
        "protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "b3_config": config_payload(config),
        "reference_artifacts": {
            name: {
                "path": payload["artifact_path"],
                "sha256": payload["artifact_sha256"],
                "status": payload["status"],
            }
            for name, payload in references.items()
        },
        "profile": references["B0"]["profile"],
        "b2_tau": float(references["B2"]["tau"]),
        "external_data": external,
        "phase30_compatibility": phase30_source_comparison(),
    }


def load_preflight(path: Path = DEFAULT_PREFLIGHT) -> dict:
    digest = verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "b3-preflight":
        raise RuntimeError("not a B3 preflight artifact")
    if payload.get("verdict", {}).get("decision") != "GO":
        raise RuntimeError("B3 preflight did not return GO")
    checks = {
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "b3_config": config_payload(b3_config()),
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"B3 changed after preflight at {key}")
    references = load_reference_artifacts()
    for name, prerequisite in references.items():
        if (
            payload["reference_artifacts"][name]["sha256"]
            != prerequisite["artifact_sha256"]
        ):
            raise RuntimeError(f"B3 {name} prerequisite changed after preflight")
    external = payload["external_data"]
    if file_sha256(Path(external["artifact_path"])) != external["artifact_sha256"]:
        raise RuntimeError("B3 shifted source changed after preflight")
    payload["artifact_sha256"] = digest
    return payload


def assert_unused(path: Path) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise RuntimeError(f"refusing to overwrite consumed B3 artifact {path}")
