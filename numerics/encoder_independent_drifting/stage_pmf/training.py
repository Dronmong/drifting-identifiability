"""Reproducible local training mechanics for the Stage-S3 pMF foundation."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..config import MASTER_SEED, derive_seed
from .config import PMFProfile
from .model import PixelMeanFlowTransformer
from .objective import meanflow_loss, sample_time_triangle

PMF_SEED_OFFSET = 113_000


def pmf_seed(phase: str, unit: int | str, role: str) -> int:
    return derive_seed(MASTER_SEED + PMF_SEED_OFFSET, "pmf-s3", phase, unit, role)


def pmf_evaluation_seed(phase: str, role: str) -> int:
    """A sealed evaluation seed shared across units for paired comparisons."""
    return pmf_seed(phase, "shared-evaluation", role)


@dataclass
class PMFStreams:
    data: np.random.Generator
    noise: torch.Generator
    time_values: torch.Generator
    diagonal_mask: torch.Generator
    augmentation: torch.Generator


def pmf_streams(phase: str, unit: int) -> PMFStreams:
    def generator(role: str) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(
            pmf_seed(phase, unit, role) % (2**63 - 1)
        )

    return PMFStreams(
        data=np.random.default_rng(pmf_seed(phase, unit, "data-order")),
        noise=generator("path-noise"),
        time_values=generator("triangle-values"),
        diagonal_mask=generator("triangle-diagonal-mask"),
        augmentation=generator("horizontal-flip"),
    )


def training_batch(
    pool: torch.Tensor,
    batch: int,
    streams: PMFStreams,
    device: torch.device | str,
    horizontal_flip: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pool.ndim != 4 or batch <= 0:
        raise ValueError("pool must be an image tensor and batch must be positive")
    indices = streams.data.integers(0, len(pool), batch)
    clean = pool[torch.as_tensor(indices)].clone()
    if horizontal_flip:
        flip = torch.rand(batch, generator=streams.augmentation) < 0.5
        clean[flip] = torch.flip(clean[flip], dims=(-1,))
    noise = torch.randn(clean.shape, generator=streams.noise, dtype=clean.dtype)
    return clean.to(device), noise.to(device)


class EMAState:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[name].lerp_(value.detach(), 1 - self.decay)
            else:
                self.shadow[name].copy_(value)

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        original = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(original, strict=True)

    def cpu_state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.shadow.items()}


@dataclass
class PMFTrainResult:
    model: PixelMeanFlowTransformer
    ema: EMAState
    optimizer: torch.optim.Optimizer
    streams: PMFStreams
    history: list[dict]
    wall_seconds: float
    peak_memory_bytes: int | None
    optimizer_updates: int
    examples_seen: int


CheckpointCallback = Callable[
    [
        int,
        PixelMeanFlowTransformer,
        EMAState,
        torch.optim.Optimizer,
        PMFStreams,
        dict,
        list[dict],
    ],
    None,
]


def _stream_state(streams: PMFStreams) -> dict:
    return {
        "data": streams.data.bit_generator.state,
        "noise": streams.noise.get_state(),
        "time_values": streams.time_values.get_state(),
        "diagonal_mask": streams.diagonal_mask.get_state(),
        "augmentation": streams.augmentation.get_state(),
    }


def checkpoint_payload(
    update: int,
    model: PixelMeanFlowTransformer,
    ema: EMAState,
    optimizer: torch.optim.Optimizer,
    streams: PMFStreams,
    profile_payload: dict,
    unit: int,
    history: list[dict],
    source_sha256: str,
    last_row: dict | None = None,
    peak_memory_bytes: int | None = None,
) -> dict:
    return {
        "status": "pmf-s3-checkpoint",
        "unit": int(unit),
        "update": int(update),
        "profile": profile_payload,
        "source_sha256": source_sha256,
        "model": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "ema": ema.cpu_state(),
        "optimizer": optimizer.state_dict(),
        "streams": _stream_state(streams),
        "history": list(history),
        "last_row": dict(last_row or {}),
        "peak_memory_bytes_so_far": peak_memory_bytes,
    }


def _restore_state(
    payload: dict,
    model: PixelMeanFlowTransformer,
    ema: EMAState,
    optimizer: torch.optim.Optimizer,
    streams: PMFStreams,
    device: torch.device,
) -> tuple[int, list[dict]]:
    """Restore every state that can influence subsequent optimization."""
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
    return int(payload["update"]) + 1, list(payload.get("history", []))


def train_pmf(
    pool: torch.Tensor,
    selected: PMFProfile,
    phase: str,
    unit: int,
    device: torch.device | str,
    checkpoint: CheckpointCallback | None = None,
    resume_payload: dict | None = None,
) -> PMFTrainResult:
    """Train one pMF unit from scratch using only the declared train pool."""
    selected.validate()
    device = torch.device(device)
    model = PixelMeanFlowTransformer(
        selected.model, pmf_seed(phase, unit, "model-init")
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=selected.train.learning_rate,
        betas=(selected.train.beta1, selected.train.beta2),
        weight_decay=selected.train.weight_decay,
    )
    ema = EMAState(model, selected.train.ema_decay)
    streams = pmf_streams(phase, unit)
    history: list[dict] = []
    first_update = 1
    if resume_payload is not None:
        first_update, history = _restore_state(
            resume_payload, model, ema, optimizer, streams, device
        )
    elapsed_before = float(history[-1].get("wall_seconds", 0.0)) if history else 0.0
    started = time.time()
    examples_seen = (first_update - 1) * selected.train.effective_batch
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for update in range(first_update, selected.train.updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw_mse = 0.0
        jvp_rms = 0.0
        auxiliary_raw_mse = 0.0
        diagonal_fraction = 0.0
        adaptive_loss = 0.0
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
            outcome = meanflow_loss(model, clean, noise, triangle, selected.objective)
            if not torch.isfinite(outcome.loss):
                raise FloatingPointError(f"non-finite pMF loss at update {update}")
            (outcome.loss / selected.train.accumulation_steps).backward()
            adaptive_loss += float(outcome.loss.detach())
            raw_mse += float(outcome.raw_mse.detach())
            jvp_rms += float(
                outcome.directional_derivative.detach().square().mean().sqrt()
            )
            auxiliary_raw_mse += float(outcome.auxiliary_raw_mse.detach())
            diagonal_fraction += float(outcome.triangle.diagonal.float().mean())
            examples_seen += len(clean)

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), selected.train.gradient_clip
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError(f"non-finite pMF gradient at update {update}")
        optimizer.step()
        ema.update(model)
        divisor = selected.train.accumulation_steps
        row = {
            "update": update,
            "adaptive_loss": adaptive_loss / divisor,
            "raw_velocity_mse": raw_mse / divisor,
            "jvp_rms": jvp_rms / divisor,
            "auxiliary_raw_velocity_mse": auxiliary_raw_mse / divisor,
            "diagonal_fraction": diagonal_fraction / divisor,
            "gradient_norm_preclip": float(gradient_norm),
            "examples_seen": examples_seen,
            "wall_seconds": elapsed_before + time.time() - started,
        }
        if update == 1 or update % selected.train.log_every == 0:
            history.append(row)
            print(
                f"pMF u{unit} update={update}/{selected.train.updates} "
                f"loss={row['adaptive_loss']:.6f} "
                f"raw_mse={row['raw_velocity_mse']:.6f} "
                f"aux_mse={row['auxiliary_raw_velocity_mse']:.6f} "
                f"jvp={row['jvp_rms']:.5f}",
                flush=True,
            )
        if checkpoint is not None and update in selected.train.checkpoint_updates:
            checkpoint(update, model, ema, optimizer, streams, row, history)

    return PMFTrainResult(
        model=model,
        ema=ema,
        optimizer=optimizer,
        streams=streams,
        history=history,
        wall_seconds=elapsed_before + time.time() - started,
        peak_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        optimizer_updates=selected.train.updates,
        examples_seen=examples_seen,
    )
