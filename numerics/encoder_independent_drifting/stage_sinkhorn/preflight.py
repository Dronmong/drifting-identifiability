"""Outcome-blind S0.3 preflight on immutable B0 EMA foundations.

This script measures numerical transport health, gradient scale, wall time,
and memory for epsilon candidates. It does not train a candidate, inspect an
image metric, or select a research outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .. import cifar
from ..b1 import load_b0_checkpoint_model
from ..device import configure, resolve_device
from ..diagnostics import write_json
from ..f3b import CONFIRMATION_UNITS, bridge_streams, independent_bridge_batch
from ..f3b_freeze import profile_from_payload, verify_sidecar
from .core import SinkhornConfig, target_cost_scale
from .training import (
    calibrated_event_lambda,
    minimax_log_event_lambda,
    parameter_gradient_geometry,
    sinkhorn_correction_term,
    sinkhorn_seed,
    sinkhorn_streams,
)

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = PACKAGE.parents[1]
DEFAULT_B0 = PACKAGE / "f3b_confirmatory.json"
DEFAULT_OUTPUT = HERE / "s0_preflight.json"
PROTOCOL = ROOT / "numerics" / "EncoderIndependentSinkhornProtocol.md"
EPSILON_CANDIDATES = (0.05, 0.10)
_SOURCE_FILES = (
    PACKAGE / "b1.py",
    PACKAGE / "cifar.py",
    PACKAGE / "device.py",
    PACKAGE / "diagnostics.py",
    PACKAGE / "f3b.py",
    PACKAGE / "f3b_freeze.py",
    HERE / "core.py",
    HERE / "preflight.py",
    HERE / "training.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, str]:
    missing = [path for path in _SOURCE_FILES if not path.exists()]
    if missing:
        raise RuntimeError(f"Sinkhorn preflight source missing: {missing}")
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
        for path in _SOURCE_FILES
    }


def _calibration_indices(pool_size: int, count: int) -> np.ndarray:
    if count < 2 or pool_size < count:
        raise ValueError("invalid target-only cost-scale calibration count")
    generator = np.random.default_rng(
        sinkhorn_seed("s0-preflight", 0, "cost-scale-indices")
    )
    return np.sort(generator.choice(pool_size, size=count, replace=False))


def _candidate_config(epsilon: float, real_batch: int) -> SinkhornConfig:
    result = SinkhornConfig(epsilon=epsilon, real_batch=real_batch)
    result.validate()
    return result


def _candidate_row(
    b0: dict,
    selected,
    train: torch.Tensor,
    device: torch.device,
    unit: int,
    cost_scale: float,
    config: SinkhornConfig,
) -> dict:
    model = load_b0_checkpoint_model(
        b0["checkpoints"][str(unit)],
        b0["profile"],
        selected.model,
        unit,
        device,
    )
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    flow_streams = bridge_streams("sinkhorn-s0-preflight", unit)
    mixed, target, _, _, time_value = independent_bridge_batch(
        train,
        selected.train.batch,
        flow_streams,
        device,
        selected.train.horizontal_flip,
    )
    prediction = model(mixed, time_value)
    flow = F.mse_loss(prediction, target)
    correction, health = sinkhorn_correction_term(
        model,
        train,
        selected.model,
        sinkhorn_streams("sinkhorn-s0-preflight", unit),
        device,
        cost_scale,
        config,
        horizontal_flip=selected.train.horizontal_flip,
    )
    gradient_geometry = parameter_gradient_geometry(flow, correction, model)
    flow_norm = gradient_geometry["first_norm"]
    correction_norm = gradient_geometry["second_norm"]
    event_weight = calibrated_event_lambda(flow_norm, correction_norm, config)
    combined = flow + event_weight * correction.to(flow.dtype)
    model.zero_grad(set_to_none=True)
    combined.backward()
    combined_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), selected.train.gradient_clip)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = peak_reserved = 0
    elapsed = time.perf_counter() - started
    row = {
        "unit": unit,
        "epsilon": config.epsilon,
        "flow_loss": float(flow.detach()),
        "correction_loss": float(correction.detach()),
        "flow_gradient_norm": flow_norm,
        "correction_gradient_norm": correction_norm,
        "flow_correction_gradient_cosine": gradient_geometry["cosine"],
        "unit_lambda_event": event_weight,
        "unit_weighted_event_gradient_ratio": (
            event_weight * correction_norm / flow_norm
        ),
        "combined_gradient_norm_before_clip": combined_norm,
        "elapsed_seconds": elapsed,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "transport_health": health,
    }
    scalar_fields = (
        "flow_loss",
        "correction_loss",
        "flow_gradient_norm",
        "correction_gradient_norm",
        "flow_correction_gradient_cosine",
        "unit_lambda_event",
        "combined_gradient_norm_before_clip",
        "elapsed_seconds",
    )
    positive_fields = tuple(
        name for name in scalar_fields if name != "flow_correction_gradient_cosine"
    )
    if (
        not all(
            math.isfinite(float(row[name])) and float(row[name]) > 0
            for name in positive_fields
        )
        or not math.isfinite(row["flow_correction_gradient_cosine"])
        or not -1.000001 <= row["flow_correction_gradient_cosine"] <= 1.000001
    ):
        raise FloatingPointError(f"invalid Sinkhorn preflight row: {row}")
    del combined, correction, flow, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _summarize_candidate(
    rows: list[dict],
    config: SinkhornConfig,
    total_memory: int | None,
) -> dict:
    chosen_lambda = minimax_log_event_lambda([row["unit_lambda_event"] for row in rows])
    for row in rows:
        row["frozen_lambda_weighted_event_gradient_ratio"] = (
            chosen_lambda * row["correction_gradient_norm"] / row["flow_gradient_norm"]
        )
    plans = [
        row["transport_health"][role] for row in rows for role in ("cross", "self")
    ]
    peak_allocated = max(row["peak_memory_allocated_bytes"] for row in rows)
    peak_reserved = max(row["peak_memory_reserved_bytes"] for row in rows)
    memory_fraction = 0.0 if total_memory is None else peak_allocated / total_memory
    ratio_lower = config.event_gradient_ratio / 2
    ratio_upper = config.event_gradient_ratio * 2
    gradient_ratio_passes = all(
        ratio_lower <= row["frozen_lambda_weighted_event_gradient_ratio"] <= ratio_upper
        for row in rows
    )
    solver_passes = all(
        plan["converged"]
        and plan["maximum_relative_error"] <= config.relative_tolerance
        for plan in plans
    )
    memory_passes = memory_fraction <= 0.95
    return {
        "epsilon": config.epsilon,
        "config": asdict(config),
        "lambda_event": chosen_lambda,
        "rows": rows,
        "summary": {
            "solver_passes": solver_passes,
            "gradient_ratio_passes": gradient_ratio_passes,
            "memory_passes": memory_passes,
            "all_mechanical_gates_pass": bool(
                solver_passes and gradient_ratio_passes and memory_passes
            ),
            "maximum_relative_marginal_error": max(
                plan["maximum_relative_error"] for plan in plans
            ),
            "maximum_iterations": max(plan["iterations"] for plan in plans),
            "cap_hits": sum(int(plan["iteration_cap_hit"]) for plan in plans),
            "cross_entropy_mean": float(
                np.mean(
                    [
                        row["transport_health"]["cross"]["conditional_entropy_mean"]
                        for row in rows
                    ]
                )
            ),
            "self_entropy_mean": float(
                np.mean(
                    [
                        row["transport_health"]["self"]["conditional_entropy_mean"]
                        for row in rows
                    ]
                )
            ),
            "conditional_max_weight_maximum": max(
                row["transport_health"][role]["conditional_max_weight_maximum"]
                for row in rows
                for role in ("cross", "self")
            ),
            "update_to_sample_rms_mean": float(
                np.mean(
                    [row["transport_health"]["update_to_sample_rms"] for row in rows]
                )
            ),
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "peak_allocated_memory_fraction": memory_fraction,
            "mean_event_seconds": float(
                np.mean([row["elapsed_seconds"] for row in rows])
            ),
            "frozen_lambda_ratio_range": [
                min(row["frozen_lambda_weighted_event_gradient_ratio"] for row in rows),
                max(row["frozen_lambda_weighted_event_gradient_ratio"] for row in rows),
            ],
            "flow_correction_gradient_cosine_range": [
                min(row["flow_correction_gradient_cosine"] for row in rows),
                max(row["flow_correction_gradient_cosine"] for row in rows),
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", type=Path, default=DEFAULT_B0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--real-batch", type=int, default=128)
    parser.add_argument("--cost-scale-samples", type=int, default=256)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    verify_sidecar(args.b0)
    b0 = json.loads(args.b0.read_text(encoding="utf-8"))
    if (
        b0.get("status") != "f3b-b0-confirmatory"
        or b0.get("verdict", {}).get("decision") != "PASS"
    ):
        raise RuntimeError("S0.3 requires the immutable passing B0 artifact")
    selected = profile_from_payload(b0["profile"])
    train = cifar.cifar_pool(selected.model.image_size, "train", args.data_root)
    indices = _calibration_indices(len(train), args.cost_scale_samples)
    calibration = train[torch.as_tensor(indices)]
    cost_scale = target_cost_scale(calibration)
    total_memory = (
        int(torch.cuda.get_device_properties(device).total_memory)
        if device.type == "cuda"
        else None
    )

    candidates = []
    for epsilon in EPSILON_CANDIDATES:
        config = _candidate_config(epsilon, args.real_batch)
        rows = [
            _candidate_row(
                b0,
                selected,
                train,
                device,
                unit,
                cost_scale,
                config,
            )
            for unit in CONFIRMATION_UNITS
        ]
        candidates.append(_summarize_candidate(rows, config, total_memory))

    mechanical_go = any(
        candidate["summary"]["all_mechanical_gates_pass"] for candidate in candidates
    )
    payload = {
        "status": "sinkhorn-s0-preflight",
        "scope": (
            "outcome-blind B0-foundation mechanics; no candidate training, image "
            "metric, checkpoint selection, or epsilon selection"
        ),
        "protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": _sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "b0_path": str(args.b0.resolve()),
        "b0_sha256": _sha256(args.b0),
        "b0_sidecar_sha256": _sha256(args.b0.with_suffix(args.b0.suffix + ".sha256")),
        "device": settings,
        "target_cost_scale": cost_scale,
        "cost_scale_samples": args.cost_scale_samples,
        "cost_scale_indices_sha256": hashlib.sha256(
            indices.astype(np.int64).tobytes()
        ).hexdigest(),
        "real_batch": args.real_batch,
        "candidates": candidates,
        "verdict": {
            "decision": "GO-TO-REVIEW" if mechanical_go else "NO-GO",
            "mechanically_valid_candidate_exists": mechanical_go,
            "epsilon_selected": None,
            "reading": (
                "At least one epsilon is mechanically viable; review numerical "
                "diagnostics and freeze before any S1 training."
                if mechanical_go
                else "No epsilon candidate passed every mechanical gate."
            ),
        },
    }
    write_json(args.out, payload)
    print(
        f"Sinkhorn S0 preflight: {payload['verdict']['decision']}; "
        f"cost_scale={cost_scale:.6g}; output={args.out}"
    )


if __name__ == "__main__":
    main()
