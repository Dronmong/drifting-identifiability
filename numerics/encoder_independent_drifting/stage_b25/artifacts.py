"""Prospective artifact boundary for the unconsumed B2.5 development run."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from ..b1_freeze import DEFAULT_FREEZE as DEFAULT_B1_FREEZE
from ..b1_freeze import load_freeze as load_b1_freeze
from ..diagnostics import ROOT
from ..f3b import F3BProfile
from ..f3b_freeze import profile_from_payload
from ..stage_b2.artifacts import (
    DEFAULT_FREEZE as DEFAULT_B2_FREEZE,
)
from ..stage_b2.artifacts import (
    file_sha256,
    verify_sidecar,
)
from ..stage_b2.artifacts import (
    load_freeze as load_b2_freeze,
)
from .core import B25Config, b25_config

PACKAGE = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "numerics" / "EncoderIndependentB25Protocol.md"
DEFAULT_DATA = HERE / "data" / "cinic10_imagenet_only_b25_disjoint.npz"
DEFAULT_PROVENANCE = DEFAULT_DATA.with_suffix(".provenance.json")
DEFAULT_PREFLIGHT = HERE / "b25_preflight.json"
DEFAULT_RESULT = HERE / "b25_development.json"

_DEPENDENCIES = (
    "appearance.py",
    "b1.py",
    "b1_freeze.py",
    "cifar.py",
    "config.py",
    "device.py",
    "diagnostics.py",
    "f3b.py",
    "f3b_freeze.py",
    "fid.py",
    "spectral_anchor.py",
    "stage_b2/__init__.py",
    "stage_b2/artifacts.py",
    "stage_b2/core.py",
    "stage_b2/fresh_data.py",
    "stage_b2/metrics.py",
    "stage_b2/source_cinic10_pool.py",
)


def config_payload(config: B25Config) -> dict:
    config.validate()
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def source_manifest() -> dict[str, str]:
    paths = [PACKAGE / name for name in _DEPENDENCIES]
    paths.extend(sorted(HERE.rglob("*.py")))
    if any(not path.is_file() for path in paths):
        raise RuntimeError("a declared B2.5 source dependency is missing")
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
        for path in sorted(set(paths))
    }


def development_profile(b2_freeze: dict, config: B25Config) -> F3BProfile:
    profile = profile_from_payload(b2_freeze["profile"])
    result = replace(
        profile,
        purpose="B2.5 paired development factorial",
        train=replace(
            profile.train,
            steps=config.final_step,
            checkpoint_steps=config.checkpoint_steps,
        ),
    )
    result.validate()
    return result


def load_frozen_inputs(
    b1_path: Path = DEFAULT_B1_FREEZE,
    b2_path: Path = DEFAULT_B2_FREEZE,
) -> dict:
    b1_sha = verify_sidecar(b1_path)
    b2_sha = verify_sidecar(b2_path)
    b1 = load_b1_freeze(b1_path)
    b2 = load_b2_freeze(b2_path)
    if b1.get("profile") != b2.get("profile"):
        raise RuntimeError("B1 and B2 were not calibrated on the same bridge profile")
    values = {
        "b1_scale": float(b1["scale"]),
        "lambda_b1": float(b1["lambda_event"]),
        "tau_b2": float(b2["tau"]),
        "lambda_b2": float(b2["lambda_event"]),
    }
    if any(value <= 0 for value in values.values()):
        raise RuntimeError("a frozen B1/B2 intervention constant is non-positive")
    return {
        "b1": b1,
        "b2": b2,
        "b1_freeze_path": str(b1_path.resolve()),
        "b2_freeze_path": str(b2_path.resolve()),
        "b1_freeze_sha256": b1_sha,
        "b2_freeze_sha256": b2_sha,
        **values,
    }


def load_data_provenance(
    data_path: Path = DEFAULT_DATA,
    provenance_path: Path = DEFAULT_PROVENANCE,
    b2_freeze: dict | None = None,
) -> dict:
    if not data_path.is_file() or not provenance_path.is_file():
        raise RuntimeError("B2.5 disjoint data/provenance has not been built")
    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    if record.get("schema") != "drifting-identifiability-b25-external-pool-v1":
        raise RuntimeError("not a B2.5 disjoint-pool provenance record")
    digest = file_sha256(data_path)
    if record.get("output_sha256") != digest:
        raise RuntimeError("B2.5 data bytes differ from provenance")
    guarantees = record.get("selection_guarantees", {})
    required_zero = (
        "decoded_pixel_overlap_with_complete_cifar10",
        "source_path_overlap_with_excluded_pools",
        "decoded_pixel_overlap_with_excluded_pools",
        "decoded_pixel_duplicates_within_pool",
    )
    if any(int(guarantees.get(name, -1)) != 0 for name in required_zero):
        raise RuntimeError(
            "B2.5 data provenance lacks a required zero-overlap guarantee"
        )
    if b2_freeze is not None:
        consumed = Path(b2_freeze["fresh_data"]["path"])
        consumed_sha = file_sha256(consumed)
        excluded = {item["sha256"] for item in record.get("excluded_pools", [])}
        if consumed_sha not in excluded:
            raise RuntimeError("B2.5 data did not exclude the consumed B2 pool")
    return {
        **record,
        "artifact_path": str(data_path.resolve()),
        "artifact_sha256": digest,
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": file_sha256(provenance_path),
    }


def load_preflight(path: Path = DEFAULT_PREFLIGHT) -> dict:
    digest = verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "b25-preflight":
        raise RuntimeError("not a B2.5 preflight artifact")
    if payload.get("verdict", {}).get("decision") != "GO":
        raise RuntimeError("B2.5 preflight did not return GO")
    if payload.get("protocol_sha256") != file_sha256(PROTOCOL):
        raise RuntimeError("B2.5 protocol changed after preflight")
    if payload.get("source_sha256") != source_manifest():
        raise RuntimeError("B2.5 executable sources changed after preflight")
    if payload.get("b25_config") != config_payload(b25_config()):
        raise RuntimeError("B2.5 configuration changed after preflight")
    for prefix in ("b1", "b2"):
        frozen_path = Path(payload[f"{prefix}_freeze_path"])
        if verify_sidecar(frozen_path) != payload[f"{prefix}_freeze_sha256"]:
            raise RuntimeError(f"B2.5 {prefix.upper()} freeze changed after preflight")
    external = payload.get("external_data", {})
    external_path = Path(external["artifact_path"])
    provenance_path = Path(external["provenance_path"])
    if file_sha256(external_path) != external.get("artifact_sha256"):
        raise RuntimeError("B2.5 external data changed after preflight")
    if file_sha256(provenance_path) != external.get("provenance_sha256"):
        raise RuntimeError("B2.5 external provenance changed after preflight")
    payload["artifact_sha256"] = digest
    return payload


def preflight_header(config: B25Config | None = None) -> dict:
    config = config or b25_config()
    frozen = load_frozen_inputs()
    data = load_data_provenance(b2_freeze=frozen["b2"])
    profile = development_profile(frozen["b2"], config)
    return {
        "status": "b25-preflight",
        "protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "b25_config": config_payload(config),
        "profile": asdict(profile),
        "b1_freeze_path": frozen["b1_freeze_path"],
        "b1_freeze_sha256": frozen["b1_freeze_sha256"],
        "b2_freeze_path": frozen["b2_freeze_path"],
        "b2_freeze_sha256": frozen["b2_freeze_sha256"],
        "frozen_constants": {
            name: frozen[name]
            for name in ("b1_scale", "lambda_b1", "tau_b2", "lambda_b2")
        },
        "external_data": data,
    }


def assert_result_path_unused(path: Path = DEFAULT_RESULT) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise RuntimeError("refusing to overwrite a consumed B2.5 development result")
