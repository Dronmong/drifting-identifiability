"""Artifact adoption and source-freeze boundary for paired Stage B1."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .b1 import B1_CONFIRMATION_UNITS, b1_config, config_payload
from .diagnostics import PACKAGE, ROOT
from .f3b_freeze import profile_from_payload

HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "numerics" / "EncoderIndependentB1Protocol.md"
DEFAULT_B0_RESULT = HERE / "f3b_confirmatory.json"
DEFAULT_CALIBRATION = HERE / "b1_calibration.json"
DEFAULT_BASELINE = HERE / "b1_baseline.json"
DEFAULT_FREEZE = HERE / "b1_freeze.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not sidecar.exists():
        raise RuntimeError(f"artifact or SHA sidecar missing: {path}")
    actual = file_sha256(path)
    recorded = sidecar.read_text(encoding="utf-8").split()[0]
    if actual != recorded:
        raise RuntimeError(f"SHA mismatch for {path.name}")
    return actual


def source_manifest() -> dict[str, str]:
    sources = sorted(PACKAGE.glob("*.py")) + sorted((PACKAGE / "tests").glob("*.py"))
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
        for path in sources
    }


def load_adopted_b0(path: Path = DEFAULT_B0_RESULT) -> dict:
    """Verify immutable B0 bytes/checkpoints without re-running its old manifest."""
    artifact_sha = verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "f3b-b0-confirmatory":
        raise RuntimeError("B1 baseline is not the final B0 confirmation artifact")
    if payload.get("verdict", {}).get("decision") != "PASS":
        raise RuntimeError("B1 is licensed only by a passing B0 artifact")
    profile = profile_from_payload(payload["profile"])
    if profile.name != "compact" or profile.model.image_size != 32:
        raise RuntimeError("B1 currently adopts only the frozen 32x32 compact B0")
    expected = {str(unit) for unit in B1_CONFIRMATION_UNITS}
    if set(payload.get("checkpoints", {})) != expected:
        raise RuntimeError("B0 artifact is missing a paired checkpoint")
    for unit, item in payload["checkpoints"].items():
        checkpoint = Path(item["path"])
        if not checkpoint.exists() or file_sha256(checkpoint) != item["sha256"]:
            raise RuntimeError(f"B0 checkpoint {unit} is missing or changed")
    payload["artifact_sha256"] = artifact_sha
    payload["artifact_path"] = str(path)
    return payload


def _load_prerequisite(
    path: Path, status: str, b0_sha: str, require_go: bool = True
) -> dict:
    artifact_sha = verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != status:
        raise RuntimeError(f"{path.name} has the wrong B1 artifact status")
    if payload.get("protocol_sha256") != file_sha256(PROTOCOL):
        raise RuntimeError(f"{path.name} used a different B1 protocol")
    if payload.get("source_sha256") != source_manifest():
        raise RuntimeError(f"{path.name} used different executable sources")
    if payload.get("b0_artifact_sha256") != b0_sha:
        raise RuntimeError(f"{path.name} belongs to another B0 result")
    if require_go and payload.get("verdict", {}).get("decision") != "GO":
        raise RuntimeError(f"{path.name} did not return GO")
    payload["artifact_sha256"] = artifact_sha
    payload["artifact_path"] = str(path)
    return payload


def load_calibration(
    path: Path = DEFAULT_CALIBRATION,
    b0_path: Path = DEFAULT_B0_RESULT,
) -> tuple[dict, dict]:
    """Load the pre-freeze B1 calibration against its adopted B0 artifact."""
    b0 = load_adopted_b0(b0_path)
    calibration = _load_prerequisite(path, "b1-calibration", b0["artifact_sha256"])
    if calibration.get("b1_config") != config_payload(b1_config()):
        raise RuntimeError("B1 calibration used another executable config")
    return b0, calibration


def load_baseline(
    path: Path = DEFAULT_BASELINE,
    calibration_path: Path = DEFAULT_CALIBRATION,
    b0_path: Path = DEFAULT_B0_RESULT,
) -> tuple[dict, dict, dict]:
    """Load the paired B0 baseline after a compatible B1 calibration."""
    b0, calibration = load_calibration(calibration_path, b0_path)
    baseline = _load_prerequisite(path, "b1-b0-paired-baseline", b0["artifact_sha256"])
    if baseline.get("b1_config") != config_payload(b1_config()):
        raise RuntimeError("B1 baseline used another executable config")
    if baseline.get("calibration_sha256") != calibration["artifact_sha256"]:
        raise RuntimeError("B1 baseline used another calibration artifact")
    if baseline.get("allocation_digests") != calibration.get("allocation_digests"):
        raise RuntimeError("B1 baseline used another evaluation allocation")
    return b0, calibration, baseline


def frozen_payload(
    calibration_path: Path = DEFAULT_CALIBRATION,
    baseline_path: Path = DEFAULT_BASELINE,
    b0_path: Path = DEFAULT_B0_RESULT,
) -> dict:
    b0 = load_adopted_b0(b0_path)
    calibration = _load_prerequisite(
        calibration_path, "b1-calibration", b0["artifact_sha256"]
    )
    baseline = _load_prerequisite(
        baseline_path, "b1-b0-paired-baseline", b0["artifact_sha256"]
    )
    if baseline.get("calibration_sha256") != calibration["artifact_sha256"]:
        raise RuntimeError("B1 baseline used another calibration artifact")
    lambda_event = float(calibration["lambda_event"])
    if not math.isfinite(lambda_event) or lambda_event <= 0:
        raise RuntimeError("calibration did not produce a valid event weight")
    if calibration.get("allocation_digests") != baseline.get("allocation_digests"):
        raise RuntimeError("calibration and baseline allocations differ")
    config = b1_config()
    return {
        "status": "b1-confirmation-freeze",
        "protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "b0_artifact": str(b0_path),
        "b0_artifact_sha256": b0["artifact_sha256"],
        "b0_checkpoint_sha256": {
            unit: item["sha256"] for unit, item in b0["checkpoints"].items()
        },
        "calibration_artifact": str(calibration_path),
        "calibration_sha256": calibration["artifact_sha256"],
        "baseline_artifact": str(baseline_path),
        "baseline_sha256": baseline["artifact_sha256"],
        "profile": b0["profile"],
        "b1_config": config_payload(config),
        "lambda_event": lambda_event,
        "effective_gradient_ratio": config.effective_gradient_ratio,
        "scale": float(calibration["scale"]),
        "scale_indices_digest": calibration["scale_indices_digest"],
        "allocation_digests": calibration["allocation_digests"],
        "confirmation_units": list(B1_CONFIRMATION_UNITS),
        "claim_scope": (
            "paired finite-sample test of whether an encoder-free spectral "
            "regularizer reduces held-out source discrepancy while retaining "
            "B0 detectable coverage; not a finite-bank identifiability theorem"
        ),
    }


def load_freeze(path: Path = DEFAULT_FREEZE) -> dict:
    verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "b1-confirmation-freeze":
        raise RuntimeError("not a B1 confirmation freeze")
    if payload.get("protocol_sha256") != file_sha256(PROTOCOL):
        raise RuntimeError("B1 protocol changed after freeze")
    if payload.get("source_sha256") != source_manifest():
        raise RuntimeError("B1 executable sources changed after freeze")
    if payload.get("b1_config") != config_payload(b1_config()):
        raise RuntimeError("B1 executable configuration changed after freeze")
    if payload.get("confirmation_units") != list(B1_CONFIRMATION_UNITS):
        raise RuntimeError("paired B1 unit IDs changed")
    profile_from_payload(payload["profile"])
    return payload


def load_compatible_artifacts(
    freeze_path: Path, freeze: dict
) -> tuple[dict, dict, dict]:
    b0_path = Path(freeze["b0_artifact"])
    calibration_path = Path(freeze["calibration_artifact"])
    baseline_path = Path(freeze["baseline_artifact"])
    b0 = load_adopted_b0(b0_path)
    if b0["artifact_sha256"] != freeze["b0_artifact_sha256"]:
        raise RuntimeError("B0 bytes differ from the B1 freeze")
    calibration = _load_prerequisite(
        calibration_path, "b1-calibration", b0["artifact_sha256"]
    )
    baseline = _load_prerequisite(
        baseline_path, "b1-b0-paired-baseline", b0["artifact_sha256"]
    )
    if calibration["artifact_sha256"] != freeze["calibration_sha256"]:
        raise RuntimeError("B1 calibration differs from freeze")
    if baseline["artifact_sha256"] != freeze["baseline_sha256"]:
        raise RuntimeError("B1 baseline differs from freeze")
    if freeze.get("allocation_digests") != calibration.get("allocation_digests"):
        raise RuntimeError("B1 allocation differs from freeze")
    # The sidecar is checked here so the final artifact can record it exactly.
    verify_sidecar(freeze_path)
    return b0, calibration, baseline
