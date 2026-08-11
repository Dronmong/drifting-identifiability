"""Corrected v2 artifact boundary for the matched S1 continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ..diagnostics import write_json
from ..f3b import CONFIRMATION_UNITS
from ..f3b_freeze import file_sha256, verify_sidecar
from .continuation import CONTINUATION_PHASE, ContinuationConfig
from .freeze import DEFAULT_FREEZE as DEFAULT_S0_FREEZE
from .freeze import load_freeze as load_s0_freeze
from .s1_evaluation import S1AuditConfig

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = PACKAGE.parents[1]
PROTOCOL = ROOT / "numerics" / "EncoderIndependentSinkhornS1Protocol.md"
DEFAULT_B0 = PACKAGE / "f3b_confirmatory.json"
DEFAULT_B2_FREEZE = PACKAGE / "stage_b2" / "b2_freeze.json"
DEFAULT_B2_BASELINE = PACKAGE / "stage_b2" / "b2_baseline.json"
DEFAULT_B2_RESULT = PACKAGE / "stage_b2" / "b2_confirmatory.json"
DEFAULT_PREFLIGHT = HERE / "s1_v2_preflight_v2.json"
DEFAULT_S1_FREEZE = HERE / "s1_v2_freeze.json"

_SOURCE_FILES = (
    PACKAGE / "appearance.py",
    PACKAGE / "b1.py",
    PACKAGE / "cifar.py",
    PACKAGE / "device.py",
    PACKAGE / "diagnostics.py",
    PACKAGE / "f3b.py",
    PACKAGE / "f3b_freeze.py",
    PACKAGE / "fid.py",
    PACKAGE / "stage_b2" / "artifacts.py",
    PACKAGE / "stage_b2" / "core.py",
    PACKAGE / "stage_b2" / "evaluation.py",
    PACKAGE / "stage_b2" / "fresh_data.py",
    PACKAGE / "stage_b2" / "metrics.py",
    HERE / "continuation.py",
    HERE / "core.py",
    HERE / "freeze.py",
    HERE / "s1.py",
    HERE / "s1_evaluation.py",
    HERE / "s1_freeze.py",
    HERE / "s1_preflight.py",
    HERE / "training.py",
)


def source_manifest() -> dict[str, str]:
    missing = [path for path in _SOURCE_FILES if not path.exists()]
    if missing:
        raise RuntimeError(f"corrected S1 executable source missing: {missing}")
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
        for path in _SOURCE_FILES
    }


def deterministic_unit_order(s0_freeze_sha: str) -> list[int]:
    protocol_sha = file_sha256(PROTOCOL)

    def score(unit: int) -> str:
        material = (
            f"{protocol_sha}:{s0_freeze_sha}:{CONTINUATION_PHASE}:{unit}"
        ).encode()
        return hashlib.sha256(material).hexdigest()

    return sorted(CONFIRMATION_UNITS, key=score)


def _passing(path: Path, status: str) -> tuple[dict, str]:
    digest = verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != status:
        raise RuntimeError(f"{path.name} has the wrong status")
    if payload.get("verdict", {}).get("decision") != "PASS":
        raise RuntimeError(f"{path.name} is not a passing prerequisite")
    return payload, digest


def frozen_payload(preflight_path: Path = DEFAULT_PREFLIGHT) -> dict:
    s0 = load_s0_freeze(DEFAULT_S0_FREEZE)
    s0_sha = verify_sidecar(DEFAULT_S0_FREEZE)
    b0, b0_sha = _passing(DEFAULT_B0, "f3b-b0-confirmatory")
    b2, b2_sha = _passing(DEFAULT_B2_RESULT, "b2-confirmatory")
    b2_freeze_sha = verify_sidecar(DEFAULT_B2_FREEZE)
    b2_baseline_sha = verify_sidecar(DEFAULT_B2_BASELINE)
    b2_freeze = json.loads(DEFAULT_B2_FREEZE.read_text(encoding="utf-8"))
    baseline = json.loads(DEFAULT_B2_BASELINE.read_text(encoding="utf-8"))
    preflight_sha = verify_sidecar(preflight_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "sinkhorn-s1-v2-preflight":
        raise RuntimeError("not the corrected S1 preflight")
    if preflight.get("verdict", {}).get("decision") != "GO":
        raise RuntimeError("corrected S1 preflight did not return GO")
    if preflight.get("protocol_sha256") != file_sha256(PROTOCOL):
        raise RuntimeError("corrected S1 protocol changed after preflight")
    if preflight.get("source_sha256") != source_manifest():
        raise RuntimeError("corrected S1 sources changed after preflight")
    if b2.get("b0_artifact_sha256") != b0_sha:
        raise RuntimeError("B2 and corrected S1 do not share B0")
    if b2.get("freeze_sha256") != b2_freeze_sha:
        raise RuntimeError("B2 result and freeze differ")
    if b2.get("baseline_sha256") != b2_baseline_sha:
        raise RuntimeError("B2 result and baseline differ")
    if baseline.get("allocation_digests") != b2.get("allocation_digests"):
        raise RuntimeError("B2 allocation artifacts differ")
    order = deterministic_unit_order(s0_sha)
    if preflight.get("unit_order") != order:
        raise RuntimeError("corrected S1 preflight used another unit order")
    continuation = ContinuationConfig()
    continuation.validate()
    audit = S1AuditConfig(nfe=int(b0["profile"]["evaluation"]["nfe_ladder"][0]))
    audit.validate()
    return {
        "status": "sinkhorn-s1-v2-freeze",
        "decision": "GO-INITIAL-TWO",
        "protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "preflight": str(preflight_path),
        "preflight_sha256": preflight_sha,
        "s0_freeze": str(DEFAULT_S0_FREEZE),
        "s0_freeze_sha256": s0_sha,
        "b0_artifact": str(DEFAULT_B0),
        "b0_sha256": b0_sha,
        "b2_freeze": str(DEFAULT_B2_FREEZE),
        "b2_freeze_sha256": b2_freeze_sha,
        "b2_baseline": str(DEFAULT_B2_BASELINE),
        "b2_baseline_sha256": b2_baseline_sha,
        "b2_result": str(DEFAULT_B2_RESULT),
        "b2_result_sha256": b2_sha,
        "profile": b0["profile"],
        "continuation_config": asdict(continuation),
        "sinkhorn": {
            "config": s0["config"],
            "cost_scale": s0["target_cost_scale"],
            "lambda_event": s0["lambda_event"],
        },
        "laplace": {
            "config": b2_freeze["b2_config"],
            "tau": b2_freeze["tau"],
            "lambda_event": b2_freeze["lambda_event"],
        },
        "audit_config": asdict(audit),
        "unit_order": order,
        "initial_units": order[:2],
        "tiebreaker_unit": order[2],
        "unit_selection": (
            "SHA256 sort of protocol hash, S0-freeze hash, fixed continuation "
            "label, and unit ID; independent of all B2/Sinkhorn outcomes"
        ),
        "fresh_data": b2_freeze["fresh_data"],
        "allocation_digests": b2_freeze["allocation_digests"],
        "staging_rule": {
            "two_pass": "PROVISIONAL-GO",
            "zero_pass": "STOP",
            "one_pass": f"RUN-TIEBREAKER-{order[2]}",
        },
        "optimizer_scope": (
            "historical optimizer unavailable; identical fresh AdamW state "
            "for all continuations"
        ),
        "claim_scope": (
            "two-unit developmental matched continuation on a reused B2 "
            "evaluation source; not a confirmation"
        ),
    }


def load_s1_freeze(path: Path = DEFAULT_S1_FREEZE) -> dict:
    verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "sinkhorn-s1-v2-freeze":
        raise RuntimeError("not a corrected S1 freeze")
    if payload.get("decision") != "GO-INITIAL-TWO":
        raise RuntimeError("corrected S1 freeze does not authorize training")
    if payload.get("protocol_sha256") != file_sha256(PROTOCOL):
        raise RuntimeError("corrected S1 protocol changed after freeze")
    if payload.get("source_sha256") != source_manifest():
        raise RuntimeError("corrected S1 executable sources changed after freeze")
    artifact_checks = {
        "preflight_sha256": Path(payload["preflight"]),
        "s0_freeze_sha256": DEFAULT_S0_FREEZE,
        "b0_sha256": DEFAULT_B0,
        "b2_freeze_sha256": DEFAULT_B2_FREEZE,
        "b2_baseline_sha256": DEFAULT_B2_BASELINE,
        "b2_result_sha256": DEFAULT_B2_RESULT,
    }
    for key, artifact in artifact_checks.items():
        if payload.get(key) != verify_sidecar(artifact):
            raise RuntimeError(f"corrected S1 prerequisite changed at {key}")
    if payload.get("unit_order") != deterministic_unit_order(
        payload["s0_freeze_sha256"]
    ):
        raise RuntimeError("corrected S1 unit selection changed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--out", type=Path, default=DEFAULT_S1_FREEZE)
    args = parser.parse_args()
    payload = frozen_payload(args.preflight)
    digest = write_json(args.out, payload)
    print(
        f"Corrected S1 freeze: units={payload['initial_units']}; "
        f"reserve={payload['tiebreaker_unit']}; sha256={digest}"
    )


if __name__ == "__main__":
    main()
