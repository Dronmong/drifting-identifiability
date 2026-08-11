"""Outcome-blind real-GPU preflight for corrected S1 continuations."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .. import cifar
from ..b1 import load_b0_checkpoint_model
from ..device import configure, resolve_device
from ..diagnostics import provenance, write_json
from ..f3b import bridge_streams, independent_bridge_batch
from ..f3b_freeze import file_sha256, profile_from_payload, verify_sidecar
from ..stage_b2.artifacts import load_compatible_artifacts
from ..stage_b2.artifacts import load_freeze as load_b2_freeze
from ..stage_b2.core import (
    b2_config,
    b2_streams,
)
from ..stage_b2.core import (
    correction_term as laplace_correction_term,
)
from .continuation import (
    CONTINUATION_ARMS,
    CONTINUATION_PHASE,
    ContinuationConfig,
    batch_sha256,
    state_sha256,
    train_continuation_arm,
)
from .core import SinkhornConfig
from .freeze import DEFAULT_FREEZE as DEFAULT_S0_FREEZE
from .freeze import load_freeze as load_s0_freeze
from .s1_freeze import (
    DEFAULT_B0,
    DEFAULT_B2_FREEZE,
    DEFAULT_PREFLIGHT,
    PROTOCOL,
    deterministic_unit_order,
    source_manifest,
)
from .training import (
    parameter_gradient_geometry,
    sinkhorn_correction_term,
    sinkhorn_streams,
)


def _gradient_row(
    *,
    arm: str,
    b0: dict,
    selected,
    train: torch.Tensor,
    unit: int,
    device: torch.device,
    sinkhorn_config: SinkhornConfig,
    sinkhorn_cost_scale: float,
    sinkhorn_lambda: float,
    laplace_config,
    laplace_tau: float,
    laplace_lambda: float,
) -> dict:
    model = load_b0_checkpoint_model(
        b0["checkpoints"][str(unit)],
        b0["profile"],
        selected.model,
        unit,
        device,
    )
    model.train()
    streams = bridge_streams(CONTINUATION_PHASE, unit)
    mixed, target, _, _, time_value = independent_bridge_batch(
        train,
        selected.train.batch,
        streams,
        device,
        selected.train.horizontal_flip,
    )
    flow = F.mse_loss(model(mixed, time_value), target)
    if arm == "sinkhorn":
        correction, health = sinkhorn_correction_term(
            model,
            train,
            selected.model,
            sinkhorn_streams(CONTINUATION_PHASE, unit),
            device,
            sinkhorn_cost_scale,
            sinkhorn_config,
            horizontal_flip=selected.train.horizontal_flip,
        )
        event_weight = sinkhorn_lambda
    elif arm == "laplace":
        correction, health = laplace_correction_term(
            model,
            train,
            selected.model,
            b2_streams(CONTINUATION_PHASE, unit),
            device,
            laplace_tau,
            laplace_config,
            horizontal_flip=selected.train.horizontal_flip,
        )
        event_weight = laplace_lambda
    else:
        raise ValueError("gradient preflight supports correction arms only")
    geometry = parameter_gradient_geometry(flow, correction, model)
    weighted_ratio = event_weight * geometry["second_norm"] / geometry["first_norm"]
    result = {
        "arm": arm,
        "unit": unit,
        "start_state_sha256": state_sha256(model),
        "flow_batch_sha256": batch_sha256(mixed, target, time_value),
        "flow_loss": float(flow.detach()),
        "correction_loss": float(correction.detach()),
        "flow_gradient_norm": geometry["first_norm"],
        "correction_gradient_norm": geometry["second_norm"],
        "flow_correction_gradient_cosine": geometry["cosine"],
        "event_weight": event_weight,
        "weighted_event_gradient_ratio": weighted_ratio,
        "correction_health": health,
    }
    del model, flow, correction
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--short-steps", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_PREFLIGHT)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(args.out.suffix + ".sha256").exists():
        raise RuntimeError("corrected S1 preflight output already exists")
    if args.short_steps < 20:
        raise ValueError("corrected S1 preflight requires at least 20 steps")

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    s0 = load_s0_freeze(DEFAULT_S0_FREEZE)
    s0_sha = verify_sidecar(DEFAULT_S0_FREEZE)
    b2_freeze = load_b2_freeze(DEFAULT_B2_FREEZE)
    b0, _, _, _ = load_compatible_artifacts(DEFAULT_B2_FREEZE, b2_freeze)
    if file_sha256(DEFAULT_B0) != b0["artifact_sha256"]:
        raise RuntimeError("corrected S1 preflight loaded another B0")
    selected = profile_from_payload(b0["profile"])
    sinkhorn_config = SinkhornConfig(**s0["config"])
    laplace_config = b2_config()
    sinkhorn_config.validate()
    laplace_config.validate()
    train = cifar.cifar_pool(selected.model.image_size, "train", args.data_root)
    order = deterministic_unit_order(s0_sha)
    unit = int(order[0])
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    gradient_rows = [
        _gradient_row(
            arm=arm,
            b0=b0,
            selected=selected,
            train=train,
            unit=unit,
            device=device,
            sinkhorn_config=sinkhorn_config,
            sinkhorn_cost_scale=float(s0["target_cost_scale"]),
            sinkhorn_lambda=float(s0["lambda_event"]),
            laplace_config=laplace_config,
            laplace_tau=float(b2_freeze["tau"]),
            laplace_lambda=float(b2_freeze["lambda_event"]),
        )
        for arm in ("laplace", "sinkhorn")
    ]
    short = ContinuationConfig(steps=args.short_steps, log_every=10)
    outcomes = []
    for arm in CONTINUATION_ARMS:
        outcome = train_continuation_arm(
            arm=arm,
            pool=train,
            checkpoint_record=b0["checkpoints"][str(unit)],
            frozen_profile=b0["profile"],
            model_config=selected.model,
            base_train_config=selected.train,
            continuation_config=short,
            unit=unit,
            device=device,
            sinkhorn_cost_scale=float(s0["target_cost_scale"]),
            sinkhorn_lambda=float(s0["lambda_event"]),
            sinkhorn_config=sinkhorn_config,
            laplace_tau=float(b2_freeze["tau"]),
            laplace_lambda=float(b2_freeze["lambda_event"]),
            laplace_config=laplace_config,
        )
        outcomes.append(
            {
                "arm": arm,
                "start_state_sha256": outcome.start_state_sha256,
                "first_flow_batch_sha256": outcome.first_flow_batch_sha256,
                "optimizer_updates": outcome.optimizer_updates,
                "correction_events": outcome.correction_events,
                "wall_seconds": outcome.wall_seconds,
                "peak_memory_bytes": outcome.peak_memory_bytes,
                "correction_summary": outcome.correction_summary,
                "history": outcome.history,
            }
        )
        del outcome
        if device.type == "cuda":
            torch.cuda.empty_cache()

    starting_states_match = len({row["start_state_sha256"] for row in outcomes}) == 1
    flow_batches_match = len({row["first_flow_batch_sha256"] for row in outcomes}) == 1
    correction_counts_pass = all(
        row["correction_events"] == (0 if row["arm"] == "control" else 2)
        for row in outcomes
    )
    sinkhorn_short = next(row for row in outcomes if row["arm"] == "sinkhorn")
    laplace_short = next(row for row in outcomes if row["arm"] == "laplace")
    sinkhorn_solver_pass = bool(
        sinkhorn_short["correction_summary"]["cap_hits"] == 0
        and sinkhorn_short["correction_summary"]["maximum_relative_error"] <= 1e-3
    )
    laplace_health_pass = bool(
        laplace_short["correction_summary"]["positive_row_sum_error_maximum"] <= 1e-5
        and laplace_short["correction_summary"]["negative_row_sum_error_maximum"]
        <= 1e-5
    )
    gradients_pass = all(
        math.isfinite(float(row["weighted_event_gradient_ratio"]))
        and 0.01 <= float(row["weighted_event_gradient_ratio"]) <= 1.0
        and math.isfinite(float(row["flow_correction_gradient_cosine"]))
        and -1.000001 <= float(row["flow_correction_gradient_cosine"]) <= 1.000001
        for row in gradient_rows
    )
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        memory_fraction = peak_memory / total_memory
    else:
        peak_memory = total_memory = 0
        memory_fraction = 0.0
    memory_pass = memory_fraction <= 0.95
    checks = {
        "starting_states_match": starting_states_match,
        "flow_batches_match": flow_batches_match,
        "correction_counts_pass": correction_counts_pass,
        "sinkhorn_solver_pass": sinkhorn_solver_pass,
        "laplace_health_pass": laplace_health_pass,
        "gradients_pass": gradients_pass,
        "memory_pass": memory_pass,
    }
    decision = "GO" if all(checks.values()) else "NO-GO"
    payload = {
        "status": "sinkhorn-s1-v2-preflight",
        "verdict": {
            "decision": decision,
            "checks": checks,
            "reading": (
                "Corrected matched continuations are mechanically ready."
                if decision == "GO"
                else "Corrected S1 must be repaired before training."
            ),
        },
        "scope": (
            "outcome-blind mechanics only; no generated image, metric, or "
            "candidate checkpoint inspected"
        ),
        "protocol": str(PROTOCOL.relative_to(PROTOCOL.parents[1])).replace("\\", "/"),
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "s0_freeze_sha256": s0_sha,
        "b0_sha256": b0["artifact_sha256"],
        "b2_freeze_sha256": verify_sidecar(DEFAULT_B2_FREEZE),
        "unit_order": order,
        "preflight_unit": unit,
        "short_steps": args.short_steps,
        "gradient_rows": gradient_rows,
        "continuation_rows": outcomes,
        "peak_memory_bytes": peak_memory,
        "total_memory_bytes": total_memory,
        "peak_memory_fraction": memory_fraction,
        "elapsed_seconds": time.time() - started,
        "device": settings,
        "provenance": provenance(),
    }
    write_json(args.out, payload)
    print(
        f"Corrected S1 preflight: {decision}; unit_order={order}; "
        f"memory={memory_fraction:.3f}"
    )


if __name__ == "__main__":
    main()
