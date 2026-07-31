"""Freeze and compatibility checks for the F3B confirmation boundary."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from .config import config_digest
from .diagnostics import PACKAGE, ROOT
from .f3b import (
    CONFIRMATION_UNITS,
    DEVELOPMENT_UNITS,
    F3BEvalConfig,
    F3BModelConfig,
    F3BProfile,
    F3BTrainConfig,
    confirmation_profile,
    profile,
)

HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "numerics" / "EncoderIndependentF3BProtocol.md"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not sidecar.exists():
        raise RuntimeError(f"artifact or SHA sidecar missing: {path}")
    recorded = sidecar.read_text(encoding="utf-8").split()[0]
    actual = file_sha256(path)
    if recorded != actual:
        raise RuntimeError(f"SHA mismatch for {path.name}: {recorded} != {actual}")
    return actual


def source_manifest() -> dict[str, str]:
    sources = sorted(PACKAGE.glob("*.py")) + sorted((PACKAGE / "tests").glob("*.py"))
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
        for path in sources
    }


def profile_payload(value: F3BProfile) -> dict:
    value.validate()
    return asdict(value)


def profile_from_payload(payload: dict) -> F3BProfile:
    model = dict(payload["model"])
    train = dict(payload["train"])
    evaluation = dict(payload["evaluation"])
    for name in ("channel_multipliers", "attention_resolutions"):
        model[name] = tuple(model[name])
    train["checkpoint_steps"] = tuple(train["checkpoint_steps"])
    evaluation["nfe_ladder"] = tuple(evaluation["nfe_ladder"])
    result = F3BProfile(
        name=payload["name"],
        purpose=payload["purpose"],
        model=F3BModelConfig(**model),
        train=F3BTrainConfig(**train),
        evaluation=F3BEvalConfig(**evaluation),
    )
    result.validate()
    return result


def frozen_payload(
    profile_name: str, steps: int, nfe: int, development_path: Path
) -> dict:
    development_sha = verify_sidecar(development_path)
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if development.get("status") != "f3b-b0-development":
        raise RuntimeError("the selection source is not an F3B development run")
    if development.get("mechanics_only"):
        raise RuntimeError("a mechanics-only development artifact cannot be frozen")
    if development.get("provenance", {}).get("source_sha256") != source_manifest():
        raise RuntimeError(
            "development artifact used sources different from current code"
        )
    developed = profile_from_payload(development["profile"])
    if developed.name != profile_name:
        raise RuntimeError("selected profile differs from development artifact")
    # Registry equality prevents a hand-edited development payload from
    # silently becoming a new architecture under an old profile name.
    if profile_payload(developed) != profile_payload(profile(profile_name)):
        raise RuntimeError("development profile differs from the code registry")
    if development.get("units") != list(DEVELOPMENT_UNITS):
        raise RuntimeError("the freeze requires all three declared development units")
    measured_by_unit = {
        unit: {
            (int(row["step"]), int(row["nfe"]))
            for row in development.get("evaluations", [])
            if int(row["unit"]) == unit
        }
        for unit in DEVELOPMENT_UNITS
    }
    missing = [
        unit
        for unit, measured in measured_by_unit.items()
        if (steps, nfe) not in measured
    ]
    if missing:
        raise RuntimeError(
            "selected (training steps, NFE) was not evaluated in every "
            f"development unit; missing {missing}"
        )
    selected = confirmation_profile(developed, steps, nfe)
    return {
        "status": "f3b-b0-confirmation-freeze",
        "protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "confirmation_units": list(CONFIRMATION_UNITS),
        "profile": profile_payload(selected),
        "profile_digest": config_digest(selected),
        "selected_development_artifact": str(
            development_path.relative_to(ROOT)
        ).replace("\\", "/")
        if development_path.is_relative_to(ROOT)
        else str(development_path),
        "selected_development_sha256": development_sha,
        "selection": {"steps": int(steps), "nfe": int(nfe)},
        "claim_scope": (
            "detectable fresh-sample reachability for this frozen bridge; "
            "not general flow-matching or harness validity"
        ),
    }


def load_freeze(path: Path) -> dict:
    verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "f3b-b0-confirmation-freeze":
        raise RuntimeError("not an F3B confirmation freeze")
    if payload.get("protocol_sha256") != file_sha256(PROTOCOL):
        raise RuntimeError("F3B protocol changed after confirmation freeze")
    if payload.get("source_sha256") != source_manifest():
        raise RuntimeError("F3B executable sources changed after freeze")
    value = profile_from_payload(payload["profile"])
    if payload.get("profile_digest") != config_digest(value):
        raise RuntimeError("frozen F3B profile digest is inconsistent")
    if payload.get("confirmation_units") != list(CONFIRMATION_UNITS):
        raise RuntimeError("confirmation unit IDs changed")
    return payload


def load_compatible_calibration(path: Path, freeze_path: Path, freeze: dict) -> dict:
    calibration_sha = verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "f3b-b0-calibration":
        raise RuntimeError("not an F3B B0 calibration artifact")
    if payload.get("freeze_sha256") != verify_sidecar(freeze_path):
        raise RuntimeError("calibration belongs to a different F3B freeze")
    if payload.get("source_sha256") != freeze["source_sha256"]:
        raise RuntimeError("calibration used sources different from the freeze")
    if payload.get("verdict", {}).get("decision") != "GO":
        raise RuntimeError("F3B calibration did not return GO")
    verdict = payload["verdict"]
    if float(verdict.get("recall_gate", -1.0)) != 0.05:
        raise RuntimeError("F3B calibration used a different recall gate")
    if not verdict.get("full_null_power"):
        raise RuntimeError("F3B null calibration used fewer than 200 replicates")
    if not float(verdict.get("p_null_upper", 1.0)) < float(
        verdict.get("null_tolerance", 0.0)
    ):
        raise RuntimeError("F3B null exceedance bound is not below tolerance")
    required = {
        "effective_rank",
        "one_minus_duplicate_rate",
        "nn_diversity",
        "nearest_train_normalized",
    }
    if set(payload.get("thresholds", {})) != required:
        raise RuntimeError("F3B calibration is missing a required veto")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0
        for value in payload["thresholds"].values()
    ):
        raise RuntimeError("F3B calibration has a non-positive veto threshold")
    normalizer = float(payload.get("normalizer", 0.0))
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise RuntimeError("F3B calibration has an invalid distance normalizer")
    payload["artifact_sha256"] = calibration_sha
    return payload
