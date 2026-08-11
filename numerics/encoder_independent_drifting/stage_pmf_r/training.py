"""Matched train-only S3R mechanism screen with collapse-aware logging."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from ..config import MASTER_SEED, derive_seed
from ..stage_pmf.objective import sample_time_triangle
from ..stage_pmf.training import EMAState, PMFStreams, pmf_streams, training_batch
from .config import S3R_ARMS, S3RProfile
from .diagnostics import (
    developmental_gate,
    objective_health,
    pmf_gradient_conflict,
    raw_and_ema_health,
)
from .model import RepairedPixelMeanFlowTransformer
from .objectives import alpha_flow_loss, alpha_schedule, emf_x1_loss, pmf_loss

S3R_SEED_OFFSET = 179_000


def s3r_seed(phase: str, unit: int | str, role: str) -> int:
    return derive_seed(MASTER_SEED + S3R_SEED_OFFSET, "pmf-s3r", phase, unit, role)


def _model_config(selected: S3RProfile, arm: str):
    # Continuous pMF preserves the interval-only network derivative that was
    # audited in S3.  Discrete AlphaFlow and EMF can safely use absolute time.
    return replace(selected.model, condition_on_absolute_time=(arm != "pmf"))


def _fixed_endpoint_data(
    pool: torch.Tensor,
    selected: S3RProfile,
    phase: str,
    unit: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = selected.train.health_samples
    index_generator = torch.Generator().manual_seed(
        s3r_seed(phase, unit, "health-targets") % (2**63 - 1)
    )
    noise_generator = torch.Generator().manual_seed(
        s3r_seed(phase, unit, "health-noise") % (2**63 - 1)
    )
    indices = torch.randint(len(pool), (count,), generator=index_generator)
    target = pool[indices].clone().to(device)
    noise = torch.randn(target.shape, generator=noise_generator, dtype=target.dtype)
    return noise.to(device), target


@dataclass
class S3RTrainResult:
    arm: str
    model: RepairedPixelMeanFlowTransformer
    ema: EMAState
    optimizer: torch.optim.Optimizer
    history: list[dict]
    endpoint_history: list[dict]
    optimizer_updates: int
    examples_seen: int
    clipping_fraction: float
    wall_seconds: float
    peak_memory_bytes: int | None


def _stream_state(streams: PMFStreams) -> dict:
    return {
        "data": streams.data.bit_generator.state,
        "noise": streams.noise.get_state(),
        "time_values": streams.time_values.get_state(),
        "diagonal_mask": streams.diagonal_mask.get_state(),
        "augmentation": streams.augmentation.get_state(),
    }


def checkpoint_payload(
    *,
    update: int,
    arm: str,
    unit: int,
    model: RepairedPixelMeanFlowTransformer,
    ema: EMAState,
    optimizer: torch.optim.Optimizer,
    streams: PMFStreams,
    history: list[dict],
    endpoint_history: list[dict],
    clipped_updates: int,
    examples_seen: int,
    elapsed_seconds: float,
    source_sha256: str,
) -> dict:
    return {
        "status": "s3r-developmental-checkpoint",
        "update": int(update),
        "arm": arm,
        "unit": int(unit),
        "source_sha256": source_sha256,
        "model": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "ema": {name: value.detach().cpu() for name, value in ema.shadow.items()},
        "optimizer": optimizer.state_dict(),
        "streams": _stream_state(streams),
        "history": list(history),
        "endpoint_history": list(endpoint_history),
        "clipped_updates": int(clipped_updates),
        "examples_seen": int(examples_seen),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _restore(
    payload: dict,
    *,
    arm: str,
    unit: int,
    model: RepairedPixelMeanFlowTransformer,
    ema: EMAState,
    optimizer: torch.optim.Optimizer,
    streams: PMFStreams,
    device: torch.device,
) -> tuple[int, list[dict], list[dict], int, int, float]:
    if payload.get("arm") != arm or int(payload.get("unit", -1)) != unit:
        raise ValueError("resume checkpoint arm/unit mismatch")
    model.load_state_dict(payload["model"], strict=True)
    ema.shadow = {
        name: value.detach().to(device).clone()
        for name, value in payload["ema"].items()
    }
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    streams.data.bit_generator.state = payload["streams"]["data"]
    streams.noise.set_state(payload["streams"]["noise"].cpu())
    streams.time_values.set_state(payload["streams"]["time_values"].cpu())
    streams.diagonal_mask.set_state(payload["streams"]["diagonal_mask"].cpu())
    streams.augmentation.set_state(payload["streams"]["augmentation"].cpu())
    return (
        int(payload["update"]) + 1,
        list(payload.get("history", [])),
        list(payload.get("endpoint_history", [])),
        int(payload.get("clipped_updates", 0)),
        int(payload.get("examples_seen", 0)),
        float(payload.get("elapsed_seconds", 0.0)),
    )


CheckpointCallback = Callable[[dict], None]


def train_arm(
    pool: torch.Tensor,
    selected: S3RProfile,
    arm: str,
    phase: str,
    unit: int,
    device: torch.device | str,
    checkpoint: CheckpointCallback | None = None,
    resume_payload: dict | None = None,
    source_sha256: str = "developmental-unsealed",
    stop_after_update: int | None = None,
) -> S3RTrainResult:
    """Train one developmental arm without touching the official test split."""
    selected.validate()
    if arm not in S3R_ARMS:
        raise ValueError(f"unknown S3R arm {arm!r}")
    device = torch.device(device)
    model = RepairedPixelMeanFlowTransformer(
        _model_config(selected, arm), s3r_seed(phase, unit, "model-init")
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=selected.train.learning_rate,
        betas=(selected.train.beta1, selected.train.beta2),
        weight_decay=selected.train.weight_decay,
    )
    ema = EMAState(model, selected.train.ema_decay)
    # Reuse the audited independent stream implementation, but give S3R a new
    # phase label so it cannot replay a frozen S3 training stream by accident.
    streams = pmf_streams(f"s3r-{phase}", unit)
    fixed_noise, fixed_target = _fixed_endpoint_data(
        pool, selected, phase, unit, device
    )
    history: list[dict] = []
    endpoint_history: list[dict] = []
    clipped_updates = 0
    examples_seen = 0
    first_update = 1
    elapsed_before = 0.0
    if resume_payload is not None:
        (
            first_update,
            history,
            endpoint_history,
            clipped_updates,
            examples_seen,
            elapsed_before,
        ) = _restore(
            resume_payload,
            arm=arm,
            unit=unit,
            model=model,
            ema=ema,
            optimizer=optimizer,
            streams=streams,
            device=device,
        )
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    final_update = selected.train.updates
    if stop_after_update is not None:
        if not first_update <= stop_after_update <= selected.train.updates:
            raise ValueError(
                "stop_after_update must include at least one pending update"
            )
        final_update = stop_after_update
    last_completed = first_update - 1
    for update in range(first_update, final_update + 1):
        last_completed = update
        model.train()
        optimizer.zero_grad(set_to_none=True)
        health_parts: list[dict] = []
        last_batch = None
        for _ in range(selected.train.accumulation_steps):
            clean, noise = training_batch(
                pool,
                selected.train.micro_batch,
                streams,
                device,
                selected.train.horizontal_flip,
            )
            triangle = sample_time_triangle(
                selected.train.micro_batch,
                selected.objective,
                streams.time_values,
                streams.diagonal_mask,
                dtype=clean.dtype,
            )
            if arm == "pmf":
                outcome = pmf_loss(model, clean, noise, triangle, selected.objective)
            elif arm == "alpha":
                alpha = alpha_schedule(
                    update, selected.train.updates, selected.objective
                )
                outcome = alpha_flow_loss(
                    model, clean, noise, triangle, selected.objective, alpha
                )
            else:
                outcome = emf_x1_loss(model, clean, noise, triangle, selected.objective)
            if not torch.isfinite(outcome.loss):
                raise FloatingPointError(f"non-finite {arm} loss at update {update}")
            (outcome.loss / selected.train.accumulation_steps).backward()
            health_parts.append(objective_health(outcome))
            last_batch = (clean.detach(), noise.detach(), triangle)
            examples_seen += len(clean)

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), selected.train.gradient_clip
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError(f"non-finite {arm} gradient at update {update}")
        was_clipped = float(gradient_norm) > selected.train.gradient_clip
        clipped_updates += int(was_clipped)
        optimizer.step()
        ema.update(model)

        if update == 1 or update % selected.train.log_every == 0:
            jvp_values = [part["jvp_rms"] for part in health_parts]
            row = {
                "update": update,
                "arm": arm,
                "raw_mse": sum(part["raw_mse"] for part in health_parts)
                / len(health_parts),
                "diagonal_raw_mse": sum(
                    part["diagonal_raw_mse"] for part in health_parts
                )
                / len(health_parts),
                "interior_raw_mse": sum(
                    part["interior_raw_mse"] for part in health_parts
                )
                / len(health_parts),
                "auxiliary_raw_mse": sum(
                    part["auxiliary_raw_mse"] for part in health_parts
                )
                / len(health_parts),
                "jvp_rms_p50": sum(part["p50"] for part in jvp_values)
                / len(jvp_values),
                "jvp_rms_p90": max(part["p90"] for part in jvp_values),
                "jvp_rms_max": max(part["max"] for part in jvp_values),
                "alpha": health_parts[-1]["alpha"],
                "gradient_norm_preclip": float(gradient_norm),
                "was_clipped": was_clipped,
                "clipping_fraction": clipped_updates / update,
                "examples_seen": examples_seen,
                "wall_seconds": elapsed_before + time.time() - started,
            }
            history.append(row)
            print(
                f"S3R {arm} u{unit} update={update}/{selected.train.updates} "
                f"raw={row['raw_mse']:.5f} diag={row['diagonal_raw_mse']:.5f} "
                f"interior={row['interior_raw_mse']:.5f} "
                f"clip_fraction={row['clipping_fraction']:.3f}",
                flush=True,
            )

        if (
            update % selected.train.health_every == 0
            or update == selected.train.updates
        ):
            endpoint = raw_and_ema_health(model, ema.shadow, fixed_noise, fixed_target)
            endpoint["update"] = update
            endpoint["gate"] = developmental_gate(endpoint, clipped_updates / update)
            endpoint_history.append(endpoint)

        if (
            arm == "pmf"
            and update % selected.train.gradient_cosine_every == 0
            and last_batch is not None
        ):
            clean, noise, triangle = last_batch
            # Four samples are enough to expose a sign conflict and bound this
            # relatively expensive train-only diagnostic.
            small = slice(0, min(4, len(clean)))
            small_triangle = type(triangle)(
                t=triangle.t[small],
                r=triangle.r[small],
                diagonal=triangle.diagonal[small],
            )
            conflict = pmf_gradient_conflict(
                model,
                clean[small],
                noise[small],
                small_triangle,
                selected.objective,
            )
            history[-1]["tfm_tc_gradient"] = conflict

        if checkpoint is not None and update in selected.train.checkpoint_updates:
            checkpoint(
                checkpoint_payload(
                    update=update,
                    arm=arm,
                    unit=unit,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    streams=streams,
                    history=history,
                    endpoint_history=endpoint_history,
                    clipped_updates=clipped_updates,
                    examples_seen=examples_seen,
                    elapsed_seconds=elapsed_before + time.time() - started,
                    source_sha256=source_sha256,
                )
            )

    return S3RTrainResult(
        arm=arm,
        model=model,
        ema=ema,
        optimizer=optimizer,
        history=history,
        endpoint_history=endpoint_history,
        optimizer_updates=last_completed,
        examples_seen=examples_seen,
        clipping_fraction=clipped_updates / max(last_completed, 1),
        wall_seconds=elapsed_before + time.time() - started,
        peak_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    )
