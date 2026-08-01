"""Immutable prerequisite and source boundaries for Stage B2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ..diagnostics import ROOT
from ..f3b import TimeConditionedUNet, f3b_seed
from ..f3b_freeze import profile_from_payload
from .core import B2_CONFIRMATION_UNITS, b2_config, config_payload

PACKAGE = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "numerics" / "EncoderIndependentB2Protocol.md"
DEFAULT_B0_RESULT = PACKAGE / "f3b_confirmatory.json"
DEFAULT_B1_RESULT = PACKAGE / "b1_confirmatory.json"
DEFAULT_B1_CALIBRATION = PACKAGE / "b1_calibration.json"
DEFAULT_PREFLIGHT = HERE / "b2_preflight.json"
DEFAULT_BASELINE = HERE / "b2_baseline.json"
DEFAULT_FREEZE = HERE / "b2_freeze.json"

_TOP_LEVEL_EXECUTABLES = (
    "appearance.py",
    "cifar.py",
    "config.py",
    "device.py",
    "diagnostics.py",
    "f3b.py",
    "f3b_freeze.py",
    "fid.py",
)


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
    """Hash only B2 and its declared execution dependencies.

    The old B1 manifest globbed every future package file.  B2 deliberately
    uses an explicit dependency set so adding Stage B3 cannot retroactively
    invalidate a frozen B2 result.
    """
    paths = [PACKAGE / name for name in _TOP_LEVEL_EXECUTABLES]
    paths.extend(sorted(HERE.rglob("*.py")))
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"B2 executable source missing: {missing}")
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
        for path in sorted(set(paths))
    }


def _load_passing(path: Path, status: str) -> dict:
    artifact_sha = verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != status:
        raise RuntimeError(f"{path.name} has the wrong prerequisite status")
    if payload.get("verdict", {}).get("decision") != "PASS":
        raise RuntimeError(f"{path.name} is not a passing prerequisite")
    payload["artifact_sha256"] = artifact_sha
    payload["artifact_path"] = str(path)
    return payload


def load_passing_prerequisites(
    b0_path: Path = DEFAULT_B0_RESULT,
    b1_path: Path = DEFAULT_B1_RESULT,
) -> tuple[dict, dict]:
    b0 = _load_passing(b0_path, "f3b-b0-confirmatory")
    b1 = _load_passing(b1_path, "b1-confirmatory")
    if b1.get("b0_artifact_sha256") != b0["artifact_sha256"]:
        raise RuntimeError("B1 and B2's adopted B0 artifact differ")
    for label, payload in (("B0", b0), ("B1", b1)):
        checkpoints = payload.get("checkpoints", {})
        if set(checkpoints) != {"300", "301", "302"}:
            raise RuntimeError(f"{label} is missing paired confirmation checkpoints")
        for unit, item in checkpoints.items():
            path = Path(item["path"])
            if not path.exists() or file_sha256(path) != item["sha256"]:
                raise RuntimeError(f"{label} checkpoint {unit} is missing or changed")
    return b0, b1


def load_b1_calibration(b1: dict, path: Path = DEFAULT_B1_CALIBRATION) -> dict:
    artifact_sha = verify_sidecar(path)
    if artifact_sha != b1.get("calibration_sha256"):
        raise RuntimeError("B1 calibration bytes differ from its confirmation")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "b1-calibration":
        raise RuntimeError("not the B1 calibration artifact")
    if payload.get("verdict", {}).get("decision") != "GO":
        raise RuntimeError("B1 calibration did not return GO")
    payload["artifact_sha256"] = artifact_sha
    return payload


def load_preflight(path: Path = DEFAULT_PREFLIGHT) -> dict:
    artifact_sha = verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "b2-preflight":
        raise RuntimeError("not a B2 preflight artifact")
    if payload.get("verdict", {}).get("decision") != "GO":
        raise RuntimeError("B2 preflight did not return GO")
    if payload.get("protocol_sha256") != file_sha256(PROTOCOL):
        raise RuntimeError("B2 protocol changed after preflight")
    if payload.get("source_sha256") != source_manifest():
        raise RuntimeError("B2 executable sources changed after preflight")
    payload["artifact_sha256"] = artifact_sha
    payload["artifact_path"] = str(path)
    return payload


def load_baseline(
    path: Path = DEFAULT_BASELINE,
    preflight_path: Path = DEFAULT_PREFLIGHT,
) -> tuple[dict, dict, dict, dict]:
    preflight = load_preflight(preflight_path)
    b0, b1 = load_passing_prerequisites()
    artifact_sha = verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "b2-b0-paired-baseline":
        raise RuntimeError("not a B2 paired-baseline artifact")
    if payload.get("verdict", {}).get("decision") != "GO":
        raise RuntimeError("B2 paired baseline did not return GO")
    if payload.get("protocol_sha256") != file_sha256(PROTOCOL):
        raise RuntimeError("B2 protocol changed after baseline measurement")
    if payload.get("source_sha256") != source_manifest():
        raise RuntimeError("B2 sources changed after baseline measurement")
    if payload.get("preflight_sha256") != preflight["artifact_sha256"]:
        raise RuntimeError("B2 baseline used another preflight")
    if payload.get("b0_artifact_sha256") != b0["artifact_sha256"]:
        raise RuntimeError("B2 baseline used another B0")
    if payload.get("b1_artifact_sha256") != b1["artifact_sha256"]:
        raise RuntimeError("B2 baseline used another B1")
    if payload.get("b2_config") != config_payload(b2_config()):
        raise RuntimeError("B2 baseline used another executable configuration")
    payload["artifact_sha256"] = artifact_sha
    payload["artifact_path"] = str(path)
    return b0, b1, preflight, payload


def frozen_payload(
    preflight_path: Path = DEFAULT_PREFLIGHT,
    baseline_path: Path = DEFAULT_BASELINE,
    b0_path: Path = DEFAULT_B0_RESULT,
    b1_path: Path = DEFAULT_B1_RESULT,
) -> dict:
    preflight = load_preflight(preflight_path)
    b0, b1 = load_passing_prerequisites(b0_path, b1_path)
    artifact_sha = verify_sidecar(baseline_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("status") != "b2-b0-paired-baseline":
        raise RuntimeError("not a B2 paired baseline")
    if baseline.get("verdict", {}).get("decision") != "GO":
        raise RuntimeError("B2 paired baseline did not return GO")
    expected = {
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "preflight_sha256": preflight["artifact_sha256"],
        "b0_artifact_sha256": b0["artifact_sha256"],
        "b1_artifact_sha256": b1["artifact_sha256"],
        "b2_config": config_payload(b2_config()),
    }
    for key, value in expected.items():
        if baseline.get(key) != value:
            raise RuntimeError(f"B2 baseline differs at {key}")
    fresh = baseline["fresh_data"]
    if not fresh.get("operator_attested_unused"):
        raise RuntimeError("B2 fresh-data non-reuse was not attested")
    return {
        "status": "b2-confirmation-freeze",
        "protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": expected["protocol_sha256"],
        "source_sha256": expected["source_sha256"],
        "b0_artifact": str(b0_path),
        "b0_artifact_sha256": b0["artifact_sha256"],
        "b1_artifact": str(b1_path),
        "b1_artifact_sha256": b1["artifact_sha256"],
        "preflight_artifact": str(preflight_path),
        "preflight_sha256": preflight["artifact_sha256"],
        "baseline_artifact": str(baseline_path),
        "baseline_sha256": artifact_sha,
        "profile": b0["profile"],
        "b2_config": config_payload(b2_config()),
        "tau": float(preflight["tau"]),
        "lambda_event": float(preflight["lambda_event"]),
        "effective_gradient_ratio": b2_config().effective_gradient_ratio,
        "fresh_data": fresh,
        "allocation_digests": baseline["allocation_digests"],
        "confirmation_units": list(B2_CONFIRMATION_UNITS),
        "claim_scope": (
            "paired finite-sample mechanism test on a genuinely fresh source; "
            "not an empirical proof of the exact zero-drift converse"
        ),
    }


def load_freeze(path: Path = DEFAULT_FREEZE) -> dict:
    verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "b2-confirmation-freeze":
        raise RuntimeError("not a B2 confirmation freeze")
    if payload.get("protocol_sha256") != file_sha256(PROTOCOL):
        raise RuntimeError("B2 protocol changed after freeze")
    if payload.get("source_sha256") != source_manifest():
        raise RuntimeError("B2 executable sources changed after freeze")
    if payload.get("b2_config") != config_payload(b2_config()):
        raise RuntimeError("B2 executable configuration changed after freeze")
    if payload.get("confirmation_units") != list(B2_CONFIRMATION_UNITS):
        raise RuntimeError("paired B2 unit IDs changed")
    profile_from_payload(payload["profile"])
    return payload


def load_compatible_artifacts(
    freeze_path: Path, freeze: dict
) -> tuple[dict, dict, dict, dict]:
    b0, b1 = load_passing_prerequisites(
        Path(freeze["b0_artifact"]), Path(freeze["b1_artifact"])
    )
    preflight = load_preflight(Path(freeze["preflight_artifact"]))
    baseline_path = Path(freeze["baseline_artifact"])
    baseline_sha = verify_sidecar(baseline_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    checks = {
        "b0_artifact_sha256": b0["artifact_sha256"],
        "b1_artifact_sha256": b1["artifact_sha256"],
        "preflight_sha256": preflight["artifact_sha256"],
        "baseline_sha256": baseline_sha,
    }
    for key, actual in checks.items():
        if freeze.get(key) != actual:
            raise RuntimeError(f"B2 freeze differs at {key}")
    if baseline.get("allocation_digests") != freeze.get("allocation_digests"):
        raise RuntimeError("B2 allocation differs from freeze")
    verify_sidecar(freeze_path)
    return b0, b1, preflight, baseline


def load_checkpoint_model(
    checkpoint_record: dict,
    frozen_profile: dict,
    unit: int,
    device,
) -> TimeConditionedUNet:
    """Load a hash-checked B0/B1 EMA checkpoint for report or comparison."""
    path = Path(checkpoint_record["path"])
    if not path.exists() or file_sha256(path) != checkpoint_record["sha256"]:
        raise RuntimeError(f"paired checkpoint {unit} is missing or changed")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if int(payload.get("unit", -1)) != unit:
        raise RuntimeError(f"checkpoint payload has the wrong unit for {unit}")
    if payload.get("profile") != frozen_profile:
        raise RuntimeError(f"checkpoint {unit} used a different frozen profile")
    selected = profile_from_payload(frozen_profile)
    model = TimeConditionedUNet(
        selected.model, f3b_seed("confirmation", unit, "model-init")
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model
