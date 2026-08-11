"""Freeze the outcome-blind S0 mechanics before any S1 model is trained."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..diagnostics import write_json
from ..f3b_freeze import file_sha256, verify_sidecar
from .preflight import DEFAULT_B0, HERE, PROTOCOL, source_manifest

DEFAULT_PREFLIGHT = HERE / "s0_preflight_v3.json"
DEFAULT_FREEZE = HERE / "s0_freeze.json"


def _selection_key(candidate: dict) -> tuple[float, int, float, float]:
    """Prefer diffuse, fast-converging mechanics without using quality data."""
    summary = candidate["summary"]
    return (
        float(summary["conditional_max_weight_maximum"]),
        int(summary["maximum_iterations"]),
        float(summary["mean_event_seconds"]),
        float(candidate["epsilon"]),
    )


def select_candidate(candidates: list[dict]) -> dict:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.get("summary", {}).get("all_mechanical_gates_pass") is True
    ]
    if not eligible:
        raise RuntimeError("no mechanically valid Sinkhorn candidate to freeze")
    return min(eligible, key=_selection_key)


def frozen_payload(preflight_path: Path = DEFAULT_PREFLIGHT) -> dict:
    preflight_sha = verify_sidecar(preflight_path)
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    if payload.get("status") != "sinkhorn-s0-preflight":
        raise RuntimeError("not a Sinkhorn S0 preflight")
    if payload.get("verdict", {}).get("decision") != "GO-TO-REVIEW":
        raise RuntimeError("Sinkhorn preflight did not reach review")
    if payload.get("protocol_sha256") != file_sha256(PROTOCOL):
        raise RuntimeError("Sinkhorn protocol changed after preflight")
    if payload.get("source_sha256") != source_manifest():
        raise RuntimeError("Sinkhorn executable sources changed after preflight")
    if payload.get("b0_sha256") != file_sha256(DEFAULT_B0):
        raise RuntimeError("B0 artifact changed after Sinkhorn preflight")
    selected = select_candidate(payload["candidates"])
    return {
        "status": "sinkhorn-s0-freeze",
        "decision": "GO-S1",
        "protocol": payload["protocol"],
        "protocol_sha256": payload["protocol_sha256"],
        "source_sha256": payload["source_sha256"],
        "preflight_artifact": str(preflight_path),
        "preflight_sha256": preflight_sha,
        "b0_artifact": payload["b0_path"],
        "b0_sha256": payload["b0_sha256"],
        "target_cost_scale": float(payload["target_cost_scale"]),
        "cost_scale_samples": int(payload["cost_scale_samples"]),
        "cost_scale_indices_sha256": payload["cost_scale_indices_sha256"],
        "epsilon": float(selected["epsilon"]),
        "lambda_event": float(selected["lambda_event"]),
        "config": selected["config"],
        "mechanical_summary": selected["summary"],
        "selection_rule": (
            "Among candidates passing every mechanical gate, minimize maximum "
            "conditional coupling weight; then maximum iterations, event time, "
            "and epsilon. No model-quality metric or sample image is read."
        ),
        "selection_reading": (
            "The selected coupling is the less concentrated and faster of the "
            "eligible candidates; this is a numerical freeze, not evidence of "
            "better generated samples."
        ),
        "claim_scope": (
            "Identity-pixel balanced cross-minus-independent-self correction "
            "for the matched S1 component screen; not yet a model result."
        ),
    }


def load_freeze(path: Path = DEFAULT_FREEZE) -> dict:
    verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "sinkhorn-s0-freeze":
        raise RuntimeError("not a Sinkhorn S0 freeze")
    if payload.get("decision") != "GO-S1":
        raise RuntimeError("Sinkhorn S0 did not authorize S1")
    if payload.get("protocol_sha256") != file_sha256(PROTOCOL):
        raise RuntimeError("Sinkhorn protocol changed after freeze")
    if payload.get("source_sha256") != source_manifest():
        raise RuntimeError("Sinkhorn S0 sources changed after freeze")
    if payload.get("preflight_sha256") != verify_sidecar(
        Path(payload["preflight_artifact"])
    ):
        raise RuntimeError("Sinkhorn preflight changed after freeze")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--out", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    payload = frozen_payload(args.preflight)
    digest = write_json(args.out, payload)
    print(
        f"Sinkhorn S0 freeze: epsilon={payload['epsilon']}; "
        f"lambda={payload['lambda_event']:.8g}; sha256={digest}"
    )


if __name__ == "__main__":
    main()
