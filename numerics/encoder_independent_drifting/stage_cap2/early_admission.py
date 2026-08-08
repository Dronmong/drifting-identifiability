"""Immutable 50k raw-state numerical gate for CAP-EMF-2 continuation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .artifacts import (
    assert_unused,
    load_checkpoint,
    load_preflight,
    profile_payload,
    verify_json,
    write_json_atomic,
)
from .config import apply_calibrated_gate, screen_profile
from .numerical_admission import admission_matrix_complete

STATUS = "cap-emf2-50k-raw-admission"
STEP = 50_000


def _reference(path: Path, anchor: Path) -> str:
    return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()


def _resolve(reference: object, anchor: Path) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise RuntimeError("early admission contains an empty artifact reference")
    path = Path(reference)
    return path if path.is_absolute() else (anchor / path).resolve()


def _same_hardware(left: object, right: object) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    fields = (
        "actual_gpu_name",
        "compute_capability",
        "torch_version",
        "cuda_runtime",
        "cudnn_version",
        "cublas_workspace_config",
    )
    return (
        left.get("matches") is True
        and right.get("matches") is True
        and all(left.get(field) == right.get(field) for field in fields)
    )


def _inputs(
    *,
    preflight_path: Path,
    result_path: Path,
    raw_checkpoint_path: Path,
    readmission_path: Path,
    arm: str | None = None,
    candidate: str | None = None,
) -> dict:
    preflight = load_preflight(preflight_path)
    result = verify_json(result_path, "cap-emf2-screen-unit")
    readmission = verify_json(readmission_path, "cap-emf2-numerical-admission")
    selected_arm = arm or result.get("arm")
    selected_candidate = candidate or result.get("numerical_candidate")
    calibration = preflight.get("inputs", {}).get("gate_calibration", {})
    expected_declared = apply_calibrated_gate(
        screen_profile(selected_arm, selected_candidate, updates=STEP), calibration
    )
    declared = result.get("declared_profile")
    realized = result.get("realized_profile")
    raw_record = result.get("checkpoints", {}).get(str(STEP), {}).get("raw", {})
    checkpoint = load_checkpoint(
        raw_checkpoint_path,
        expected_sha=raw_record.get("sha256"),
        step=STEP,
        kind="raw",
        arm=selected_arm,
        declared_profile=declared,
        realized_profile=realized,
        preflight_sha256=preflight["artifact_sha256"],
        run_identity_sha256=result.get("run_identity_sha256"),
        unit_seed=int(result.get("unit_seed", -1)),
    )
    initial = preflight.get("inputs", {}).get("numerical_admission", {})
    benchmark = preflight.get("inputs", {}).get("benchmark", {})
    realized_train = realized.get("train", {}) if isinstance(realized, dict) else {}
    checks = {
        "result_completed_50k": (
            int(result.get("training", {}).get("optimizer_updates", -1)) == STEP
            and int(result.get("training", {}).get("nonfinite_updates", -1)) == 0
        ),
        "result_development_only": result.get("development_only") is True,
        "result_identity": (
            result.get("arm") == selected_arm
            and result.get("numerical_candidate") == selected_candidate
            and result.get("preflight_sha256") == preflight["artifact_sha256"]
        ),
        "result_declared_profile": declared == profile_payload(expected_declared),
        "result_realized_batch": (
            isinstance(realized, dict)
            and int(realized_train.get("updates", -1)) == STEP
            and int(realized_train.get("micro_batch", -1))
            == int(benchmark.get("micro_batch", -2))
            and int(realized_train.get("accumulation_steps", -1))
            == int(benchmark.get("accumulation_steps", -2))
        ),
        "result_checkpoint_ladder": (
            set(result.get("checkpoints", {})) == {str(STEP)}
            and set(result.get("checkpoints", {}).get(str(STEP), {}))
            == {"raw", "ema"}
            and raw_record.get("sha256") == checkpoint["artifact_sha256"]
        ),
        "readmission_complete": admission_matrix_complete(readmission),
        "readmission_checkpoint": (
            readmission.get("checkpoint_sha256") == checkpoint["artifact_sha256"]
            and int(readmission.get("checkpoint_step", -1)) == STEP
            and readmission.get("checkpoint_identity", {}).get("valid") is True
            and readmission.get("checkpoint_identity", {}).get("stage")
            == "cap-emf-2-screen"
            and readmission.get("checkpoint_identity", {}).get("kind") == "raw"
            and readmission.get("checkpoint_identity", {}).get("arm") == selected_arm
        ),
        "readmission_candidate": (
            readmission.get("candidate", {}).get("name") == selected_candidate
        ),
        "readmission_sources": (
            readmission.get("source_sha256") == preflight.get("source_sha256")
        ),
        "readmission_environment": (
            _same_hardware(readmission.get("hardware"), initial.get("hardware"))
            and _same_hardware(readmission.get("hardware"), benchmark.get("hardware"))
            and _same_hardware(readmission.get("hardware"), result.get("hardware"))
            and readmission.get("production_numerical_mode")
            == initial.get("production_numerical_mode")
        ),
    }
    return {
        "preflight": preflight,
        "result": result,
        "checkpoint": checkpoint,
        "readmission": readmission,
        "arm": selected_arm,
        "candidate": selected_candidate,
        "checks": checks,
    }


def build_early_admission(
    *,
    preflight_path: Path,
    result_path: Path,
    raw_checkpoint_path: Path,
    readmission_path: Path,
    out: Path,
) -> dict:
    assert_unused(out)
    inputs = _inputs(
        preflight_path=preflight_path,
        result_path=result_path,
        raw_checkpoint_path=raw_checkpoint_path,
        readmission_path=readmission_path,
    )
    failed = sorted(name for name, valid in inputs["checks"].items() if not valid)
    payload = {
        "status": STATUS,
        "decision": "GO" if not failed else "NO_GO",
        "step": STEP,
        "arm": inputs["arm"],
        "candidate": inputs["candidate"],
        "preflight_sha256": inputs["preflight"]["artifact_sha256"],
        "result_sha256": inputs["result"]["artifact_sha256"],
        "raw_checkpoint_sha256": inputs["checkpoint"]["artifact_sha256"],
        "readmission_sha256": inputs["readmission"]["artifact_sha256"],
        "checks": inputs["checks"],
        "failed": failed,
        "references": {
            "preflight": _reference(preflight_path, out.parent),
            "result_50k": _reference(result_path, out.parent),
            "checkpoint_50k_raw": _reference(raw_checkpoint_path, out.parent),
            "readmission_50k_raw": _reference(readmission_path, out.parent),
        },
        "limits": [
            "This authorizes only continuation of the exact bound arm from 50k to at most 150k.",
            "It is a numerical/mechanical gate, not an image-quality result.",
        ],
    }
    payload["artifact_sha256"] = write_json_atomic(out, payload)
    return payload


def revalidate_early_admission(path: Path) -> dict:
    record = verify_json(path, STATUS)
    references = record.get("references")
    required = {
        "preflight",
        "result_50k",
        "checkpoint_50k_raw",
        "readmission_50k_raw",
    }
    if not isinstance(references, dict) or set(references) != required:
        raise RuntimeError("early admission has an incomplete reference ledger")
    inputs = _inputs(
        preflight_path=_resolve(references["preflight"], path.parent),
        result_path=_resolve(references["result_50k"], path.parent),
        raw_checkpoint_path=_resolve(references["checkpoint_50k_raw"], path.parent),
        readmission_path=_resolve(references["readmission_50k_raw"], path.parent),
        arm=record.get("arm"),
        candidate=record.get("candidate"),
    )
    failed = sorted(name for name, valid in inputs["checks"].items() if not valid)
    decision = "GO" if not failed else "NO_GO"
    bindings = {
        "decision": record.get("decision") == decision,
        "failed": record.get("failed") == failed,
        "step": record.get("step") == STEP,
        "preflight": record.get("preflight_sha256")
        == inputs["preflight"]["artifact_sha256"],
        "result": record.get("result_sha256")
        == inputs["result"]["artifact_sha256"],
        "checkpoint": record.get("raw_checkpoint_sha256")
        == inputs["checkpoint"]["artifact_sha256"],
        "readmission": record.get("readmission_sha256")
        == inputs["readmission"]["artifact_sha256"],
        "checks": record.get("checks") == inputs["checks"],
    }
    invalid = sorted(name for name, valid in bindings.items() if not valid)
    if invalid:
        raise RuntimeError(f"early admission revalidation failed: {invalid}")
    record["revalidated"] = True
    return record


def load_early_admission(path: Path, *, arm: str, candidate: str) -> dict:
    record = revalidate_early_admission(path)
    if record.get("decision") != "GO":
        raise RuntimeError("CAP2 50k raw-state admission did not return GO")
    if record.get("arm") != arm or record.get("candidate") != candidate:
        raise RuntimeError("CAP2 early admission belongs to another arm/candidate")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--result-50k", type=Path, required=True)
    parser.add_argument("--checkpoint-50k-raw", type=Path, required=True)
    parser.add_argument("--readmission-50k-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build_early_admission(
        preflight_path=args.preflight,
        result_path=args.result_50k,
        raw_checkpoint_path=args.checkpoint_50k_raw,
        readmission_path=args.readmission_50k_raw,
        out=args.out,
    )
    print(json.dumps({"decision": result["decision"], "failed": result["failed"]}, indent=2))
    print(f"wrote {args.out} sha256={result['artifact_sha256']}")
    return 0 if result["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
