"""Outcome-blind gradient, kernel, and resource preflight for Stage B2."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .. import cifar
from ..device import configure, resolve_device
from ..diagnostics import provenance, write_json
from ..f3b import (
    TimeConditionedUNet,
    bridge_streams,
    independent_bridge_batch,
)
from ..f3b_freeze import profile_from_payload
from .artifacts import (
    DEFAULT_B0_RESULT,
    DEFAULT_B1_RESULT,
    DEFAULT_PREFLIGHT,
    PROTOCOL,
    file_sha256,
    load_passing_prerequisites,
    source_manifest,
)
from .core import (
    B2_CALIBRATION_UNITS,
    b2_config,
    b2_seed,
    b2_streams,
    calibrate_laplace_bandwidth,
    calibrated_event_lambda,
    config_payload,
    correction_term,
    laplace_drift_energy,
    parameter_gradient_norm,
)


def calibration_indices(pool_size: int, count: int) -> np.ndarray:
    if pool_size < count:
        raise ValueError("training pool is too small for B2 calibration")
    return np.sort(
        np.random.default_rng(b2_seed("calibration", 0, "bandwidth-indices")).choice(
            pool_size, size=count, replace=False
        )
    )


def _gradient_calibration(
    train: torch.Tensor,
    selected,
    tau: float,
    device: torch.device,
    config,
) -> tuple[list[dict], float, dict]:
    rows = []
    maximum_peak = 0
    started_all = time.perf_counter()
    for unit in B2_CALIBRATION_UNITS:
        model = TimeConditionedUNet(
            selected.model, b2_seed("calibration", unit, "model-init")
        ).to(device)
        flow_streams = bridge_streams("b2-calibration", unit)
        mixed, target, _, _, time_value = independent_bridge_batch(
            train,
            selected.train.batch,
            flow_streams,
            device,
            selected.train.horizontal_flip,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = model(mixed, time_value)
        flow = F.mse_loss(prediction, target)
        flow_norm = parameter_gradient_norm(flow, model)
        correction, health = correction_term(
            model,
            train,
            selected.model,
            b2_streams("calibration", unit),
            device,
            tau,
            config,
            horizontal_flip=selected.train.horizontal_flip,
        )
        correction_norm = parameter_gradient_norm(correction, model)
        event_weight = calibrated_event_lambda(flow_norm, correction_norm, config)
        combined = flow + event_weight * correction
        model.zero_grad(set_to_none=True)
        combined.backward()
        combined_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), selected.train.gradient_clip
            )
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
        else:
            peak = 0
        maximum_peak = max(maximum_peak, peak)
        row = {
            "unit": unit,
            "flow_loss": float(flow.detach()),
            "correction_loss": float(correction.detach()),
            "flow_gradient_norm": flow_norm,
            "correction_gradient_norm": correction_norm,
            "lambda_event": event_weight,
            "weighted_event_gradient_ratio": (
                event_weight * correction_norm / flow_norm
            ),
            "combined_gradient_norm_before_clip": combined_norm,
            "kernel_health": health,
            "peak_memory_bytes": peak,
            "elapsed_seconds": time.perf_counter() - started,
        }
        required = (
            "flow_loss",
            "correction_loss",
            "flow_gradient_norm",
            "correction_gradient_norm",
            "lambda_event",
            "combined_gradient_norm_before_clip",
        )
        if not all(
            math.isfinite(float(row[name])) and float(row[name]) > 0
            for name in required
        ):
            raise FloatingPointError(f"invalid B2 gradient calibration row: {row}")
        rows.append(row)
        del combined, correction, flow, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    chosen = float(np.median([row["lambda_event"] for row in rows]))
    for row in rows:
        row["frozen_lambda_weighted_event_gradient_ratio"] = (
            chosen
            * float(row["correction_gradient_norm"])
            / float(row["flow_gradient_norm"])
        )
    return (
        rows,
        chosen,
        {
            "maximum_peak_memory_bytes": maximum_peak,
            "gradient_calibration_seconds": time.perf_counter() - started_all,
            "mean_seconds_per_unit": float(
                np.mean([row["elapsed_seconds"] for row in rows])
            ),
        },
    )


def _safety_checks(target: torch.Tensor, tau: float, device: torch.device) -> dict:
    count = min(16, len(target))
    positive = target[:count].to(device)
    negative = target[count : 2 * count].to(device)
    if len(negative) != count:
        raise ValueError("B2 safety checks need at least 32 target samples")
    probes = 0.5 * (positive + negative)
    identical, identical_health = laplace_drift_energy(probes, positive, positive, tau)
    near, _ = laplace_drift_energy(probes, positive, negative, tau)
    remote, remote_health = laplace_drift_energy(
        probes, positive + 100.0, negative, tau
    )
    result = {
        "identical_empirical_energy": float(identical),
        "near_energy": float(near),
        "remote_energy": float(remote),
        "remote_to_near_ratio": float(remote / near.clamp_min(1e-30)),
        "identical_health": identical_health,
        "remote_health": remote_health,
    }
    result["passes"] = bool(
        result["identical_empirical_energy"] <= 1e-12
        and math.isfinite(result["remote_energy"])
        and result["remote_energy"] > result["near_energy"]
        and remote_health["positive"]["row_sum_error_maximum"] < 1e-5
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", type=Path, default=DEFAULT_B0_RESULT)
    parser.add_argument("--b1", type=Path, default=DEFAULT_B1_RESULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_PREFLIGHT)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    b0, b1 = load_passing_prerequisites(args.b0, args.b1)
    selected = profile_from_payload(b0["profile"])
    config = b2_config()
    train = cifar.cifar_pool(selected.model.image_size, "train", args.data_root)
    indices = calibration_indices(len(train), config.ess_samples)
    calibration_target = train[torch.as_tensor(indices)]
    tau, bandwidth = calibrate_laplace_bandwidth(calibration_target, config)
    rows, event_weight, resource = _gradient_calibration(
        train, selected, tau, device, config
    )
    if device.type == "cuda":
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        memory_fraction = resource["maximum_peak_memory_bytes"] / total_memory
    else:
        total_memory = None
        memory_fraction = 0.0
    memory_passes = bool(memory_fraction <= config.maximum_preflight_memory_fraction)
    resource.update(
        {
            "total_device_memory_bytes": total_memory,
            "peak_device_memory_fraction": memory_fraction,
            "maximum_allowed_memory_fraction": (
                config.maximum_preflight_memory_fraction
            ),
            "memory_passes": memory_passes,
        }
    )
    tolerance = config.calibration_ratio_tolerance_factor
    lower = config.event_gradient_ratio / tolerance
    upper = config.event_gradient_ratio * tolerance
    ratios_pass = all(
        lower <= row["frozen_lambda_weighted_event_gradient_ratio"] <= upper
        for row in rows
    )
    safety = _safety_checks(calibration_target, tau, device)
    decision = "GO" if ratios_pass and safety["passes"] and memory_passes else "NO-GO"
    payload = {
        "status": "b2-preflight",
        "protocol": str(PROTOCOL.relative_to(PROTOCOL.parents[1])).replace("\\", "/"),
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "b0_artifact_sha256": b0["artifact_sha256"],
        "b1_artifact_sha256": b1["artifact_sha256"],
        "profile": b0["profile"],
        "b2_config": config_payload(config),
        "tau": tau,
        "bandwidth_calibration": bandwidth,
        "calibration_indices_sha256": hashlib.sha256(
            indices.astype(np.int64).tobytes()
        ).hexdigest(),
        "lambda_event": event_weight,
        "effective_gradient_ratio": config.effective_gradient_ratio,
        "gradient_calibration": rows,
        "resource": resource,
        "safety_checks": safety,
        "verdict": {
            "decision": decision,
            "gradient_ratio_passes": ratios_pass,
            "safety_passes": safety["passes"],
            "memory_passes": memory_passes,
            "reading": (
                "GO: the theory-aligned loss is differentiable, calibrated, "
                "numerically stable, and preserves B0/B1 prerequisite integrity"
                if decision == "GO"
                else "NO-GO: repair B2 before any candidate training"
            ),
        },
        "claim_scope": (
            "outcome-blind mechanics only; no B2 candidate and no confirmation "
            "metric is inspected"
        ),
        "provenance": provenance(),
        "device": settings,
    }
    write_json(args.out, payload)
    print(f"B2 preflight: {decision}; tau={tau:.6g}; lambda={event_weight:.6g}")


if __name__ == "__main__":
    main()
