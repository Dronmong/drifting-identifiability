"""CAP-EMF-1 training loop: EMA, checkpoints, health, and real recovery.

**This loop can resume mid-unit.**  B2.5 could not: it saved EMA weights at
three declared steps and nothing else, so an interrupted unit lost every
completed hour *and* the surviving checkpoints then blocked its own restart.
A 160,000-update rented run must not repeat that, so optimizer, model, EMA and
every RNG stream are written atomically on a fixed cadence and a restart
continues from the last verified point.

No correction of any kind appears here.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from ..config import MASTER_SEED, derive_seed
from . import CAP_PHASE, CAP_UNIT
from .config import CAPProfile
from .data import flip_batch
from .diagnostics import endpoint_health
from .model import CAPPixelTransformer, one_step_sample
from .objective import emf_loss, sample_time_triangle


def cap_seed(role: str, index: int = 0) -> int:
    return derive_seed(MASTER_SEED + 141_000, CAP_PHASE, CAP_UNIT, role, index)


@dataclass
class TrainOutcome:
    history: list[dict] = field(default_factory=list)
    health: list[dict] = field(default_factory=list)
    checkpoints: dict[str, dict] = field(default_factory=dict)
    snapshots: list[int] = field(default_factory=list)
    wall_seconds: float = 0.0
    peak_memory_bytes: int = 0
    peak_memory_reserved_bytes: int = 0
    optimizer_updates: int = 0
    examples_seen: int = 0
    model_forwards: int = 0
    clipped_updates: int = 0
    clipped_updates_final_window: int = 0
    final_window_updates: int = 0
    nonfinite_updates: int = 0
    best_rank_ratio: float = 0.0
    parameter_count: int = 0


class EMAState:
    """Exponential moving average with an explicit maturity accounting."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must lie in [0,1)")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: value.detach().clone().float()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }
        self.buffers = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if not value.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    value.detach().float(), alpha=1.0 - self.decay
                )
            else:
                self.buffers[name] = value.detach().clone()
        self.updates += 1

    def state_dict(self) -> dict:
        merged = {name: value.clone() for name, value in self.shadow.items()}
        merged.update({name: value.clone() for name, value in self.buffers.items()})
        return merged

    def initialization_weight(self) -> float:
        """Share of the average still contributed by the starting network."""
        return self.decay**self.updates

    def recovery_state(self) -> dict:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": self.shadow,
            "buffers": self.buffers,
        }

    def load_recovery_state(self, payload: dict) -> None:
        self.decay = float(payload["decay"])
        self.updates = int(payload["updates"])
        self.shadow = {k: v.clone() for k, v in payload["shadow"].items()}
        self.buffers = {k: v.clone() for k, v in payload["buffers"].items()}


#: Endpoint-weighted buckets.  The one-call sampler evaluates at t=1, and
#: logit-normal(0.8, 0.8) puts only ~4% of rows above 0.9, so the top bucket is
#: the one that decides whether the sampled configuration is trained at all.
T_BUCKETS: tuple[float, ...] = (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0)


def bucket_by_time(
    t: torch.Tensor, per_sample: torch.Tensor
) -> dict[str, dict[str, float]]:
    """Mean error and share of rows per time bucket."""
    result: dict[str, dict[str, float]] = {}
    total = max(len(t), 1)
    for low, high in zip(T_BUCKETS, T_BUCKETS[1:]):
        mask = (t >= low) & (t < high) if high < 1.0 else (t >= low) & (t <= high)
        count = int(mask.sum())
        result[f"{low:g}-{high:g}"] = {
            "share": count / total,
            "mean_raw_mse": float(per_sample[mask].mean()) if count else float("nan"),
            "count": count,
        }
    return result


def learning_rate_at(update: int, train) -> float:
    """Linear warmup, then constant.  Standard for CIFAR-10 at this scale."""
    if train.warmup_updates <= 0 or update >= train.warmup_updates:
        return train.learning_rate
    return train.learning_rate * (update + 1) / train.warmup_updates


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def _health_batch(
    model: nn.Module, noise: torch.Tensor, batch: int
) -> torch.Tensor:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(noise), batch):
            chunks.append(one_step_sample(model, noise[start : start + batch]).cpu())
    return torch.cat(chunks)


def train_cap_unit(
    pool: torch.Tensor,
    profile: CAPProfile,
    device: torch.device | str,
    *,
    recovery_path: Path | None = None,
    checkpoint: Callable[[int, dict, dict], None] | None = None,
    snapshot: Callable[[int, dict], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> TrainOutcome:
    profile.validate()
    train = profile.train
    device = torch.device(device)
    announce = progress or (lambda message: None)

    model = CAPPixelTransformer(profile.model, cap_seed("model-init")).to(device)
    outcome = TrainOutcome(parameter_count=model.parameter_count())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train.learning_rate,
        betas=(train.beta1, train.beta2),
        weight_decay=train.weight_decay,
    )
    ema = EMAState(model, train.ema_decay)

    data_generator = torch.Generator().manual_seed(cap_seed("data-order"))
    noise_generator = torch.Generator().manual_seed(cap_seed("endpoint-noise"))
    time_generator = torch.Generator().manual_seed(cap_seed("time-triangle"))
    flip_generator = torch.Generator().manual_seed(cap_seed("horizontal-flip"))

    # Sealed train-only health noise: the same latents at every checkpoint, so
    # health movement is the model moving, not the sample moving.
    health_generator = torch.Generator().manual_seed(cap_seed("health-noise"))
    shape = (train.audit_samples, profile.model.channels) + (
        profile.model.image_size,
    ) * 2
    health_noise = torch.randn(shape, generator=health_generator).to(device)
    health_target = pool[
        torch.randperm(len(pool), generator=health_generator)[: train.audit_samples]
    ]

    start_update = 0
    if recovery_path is not None and recovery_path.exists():
        payload = torch.load(recovery_path, map_location="cpu", weights_only=False)
        if payload.get("stage") != "cap-emf-1-recovery":
            raise RuntimeError("not a CAP-EMF-1 recovery file")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        ema.load_recovery_state(payload["ema"])
        data_generator.set_state(payload["generators"]["data"])
        noise_generator.set_state(payload["generators"]["noise"])
        time_generator.set_state(payload["generators"]["time"])
        flip_generator.set_state(payload["generators"]["flip"])
        outcome.history = payload["history"]
        outcome.health = payload["health"]
        outcome.checkpoints = payload["checkpoints"]
        outcome.wall_seconds = float(payload["wall_seconds"])
        outcome.optimizer_updates = int(payload["optimizer_updates"])
        outcome.examples_seen = int(payload["examples_seen"])
        outcome.model_forwards = int(payload["model_forwards"])
        outcome.clipped_updates = int(payload["clipped_updates"])
        outcome.nonfinite_updates = int(payload["nonfinite_updates"])
        outcome.best_rank_ratio = float(payload["best_rank_ratio"])
        start_update = int(payload["completed_updates"])
        announce(f"resumed CAP-EMF-1 from update {start_update}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    window_start = max(0, train.updates - profile.gate.clip_window_updates)

    for update in range(start_update, train.updates):
        current_lr = learning_rate_at(update, train)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        losses = []
        times: list[torch.Tensor] = []
        errors: list[torch.Tensor] = []
        for _ in range(train.accumulation_steps):
            order = torch.randint(
                0, len(pool), (train.micro_batch,), generator=data_generator
            )
            clean = pool[order].to(device)
            if train.horizontal_flip:
                flips = (
                    torch.rand(train.micro_batch, generator=flip_generator) < 0.5
                )
                clean = flip_batch(clean, flips)
            noise = torch.randn(
                clean.shape, generator=noise_generator, dtype=clean.dtype
            ).to(device)
            triangle = sample_time_triangle(
                train.micro_batch, profile.objective, time_generator, device
            )
            result = emf_loss(model, clean, noise, triangle, profile.objective)
            (result.loss / train.accumulation_steps).backward()
            # Not a constant: the two stopped evaluations run on active rows
            # only, so this is batch + 2*active rather than 3*batch.
            outcome.model_forwards += result.model_evaluations
            times.append(result.t.cpu())
            errors.append(result.per_sample_raw_mse.cpu())
            losses.append(
                (
                    float(result.loss.detach()),
                    float(result.raw_mse.detach()),
                    float(result.diagonal_raw_mse.detach()),
                    float(result.interior_raw_mse.detach()),
                )
            )
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train.gradient_clip)
        pre_clip = float(norm)
        if not torch.isfinite(norm):
            outcome.nonfinite_updates += 1
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
            ema.update(model)
        if pre_clip > train.gradient_clip:
            outcome.clipped_updates += 1
            if update >= window_start:
                outcome.clipped_updates_final_window += 1
        if update >= window_start:
            outcome.final_window_updates += 1
        outcome.optimizer_updates += 1
        outcome.examples_seen += train.effective_batch
        step = update + 1

        if step % train.log_every == 0 or step == train.updates:
            mean = [sum(values) / len(values) for values in zip(*losses)]
            outcome.history.append(
                {
                    "step": step,
                    "loss": mean[0],
                    "raw_mse": mean[1],
                    "diagonal_raw_mse": mean[2],
                    "interior_raw_mse": mean[3],
                    "gradient_norm_before_clip": pre_clip,
                    "learning_rate": current_lr,
                    "examples_seen": outcome.examples_seen,
                    "ema_updates": ema.updates,
                    "time_buckets": bucket_by_time(
                        torch.cat(times), torch.cat(errors)
                    ),
                    "wall_seconds": time.time() - started + outcome.wall_seconds,
                }
            )

        if step % train.health_every == 0 or step in train.checkpoint_updates:
            samples = train.audit_samples if step in train.checkpoint_updates else (
                train.health_samples
            )
            model.eval()
            generated = _health_batch(
                model, health_noise[:samples], train.micro_batch * 2
            )
            model.train()
            record = endpoint_health(generated, health_target[:samples])
            record["step"] = step
            record["ema_updates"] = ema.updates
            record["ema_initialization_weight"] = ema.initialization_weight()
            record["ema_mature"] = step >= train.ema_mature_at()
            outcome.health.append(record)
            outcome.best_rank_ratio = max(
                outcome.best_rank_ratio, record["effective_rank_ratio"]
            )
            announce(
                f"step {step}: moment {record['second_moment_ratio']:.3f} "
                f"var {record['centered_variance_ratio']:.3f} "
                f"rank {record['effective_rank_ratio']:.3f} "
                f"HH {record['haar_HH_ratio']:.3f}"
            )

        if step in train.checkpoint_updates and checkpoint is not None:
            checkpoint(step, model.state_dict(), ema.state_dict())

        if snapshot is not None and step % train.snapshot_every == 0:
            # Raw weights for post-hoc EMA synthesis. Secondary by
            # construction: the declared 0.9999 EMA remains the primary result,
            # so this cannot become checkpoint selection on a metric.
            snapshot(step, model.state_dict())
            outcome.snapshots.append(step)

        if recovery_path is not None and (
            step % train.recovery_every == 0 or step == train.updates
        ):
            _atomic_save(
                {
                    "stage": "cap-emf-1-recovery",
                    "completed_updates": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema": ema.recovery_state(),
                    "generators": {
                        "data": data_generator.get_state(),
                        "noise": noise_generator.get_state(),
                        "time": time_generator.get_state(),
                        "flip": flip_generator.get_state(),
                    },
                    "history": outcome.history,
                    "health": outcome.health,
                    "checkpoints": outcome.checkpoints,
                    "wall_seconds": time.time() - started + outcome.wall_seconds,
                    "optimizer_updates": outcome.optimizer_updates,
                    "examples_seen": outcome.examples_seen,
                    "model_forwards": outcome.model_forwards,
                    "clipped_updates": outcome.clipped_updates,
                    "nonfinite_updates": outcome.nonfinite_updates,
                    "best_rank_ratio": outcome.best_rank_ratio,
                },
                recovery_path,
            )

    outcome.wall_seconds += time.time() - started
    if device.type == "cuda":
        outcome.peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        outcome.peak_memory_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    return outcome


def clip_fraction(outcome: TrainOutcome) -> float:
    if not outcome.final_window_updates:
        return 0.0
    return outcome.clipped_updates_final_window / outcome.final_window_updates


def history_to_json(outcome: TrainOutcome) -> str:
    return json.dumps({"history": outcome.history, "health": outcome.health})
