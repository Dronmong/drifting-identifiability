"""Exact-memory Stage-B3 training for the one-step drifting proxy.

The R11 teacher is detached.  We therefore construct it from the complete
field cloud under ``no_grad`` and re-evaluate the same latent cloud in chunks
for backpropagation.  Chunk losses are sums divided by the *full* cloud size;
their accumulated gradient is the full-cloud mean-loss gradient.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .. import kernel_gradient as KG
from ..config import MASTER_SEED, GeometryConfig, TrainConfig, derive_seed
from ..diagnose_phase25 import rectangular_ess
from ..fixed_features import build_family
from ..kernels import BlockKernel, calibrate_block_kernel
from ..models import OneStepGenerator, sample_latent
from ..objectives import corrected_teacher

B3_PHASE = "b3-matched-reference"
B3_SEED_OFFSET = 131_000
B3_UNITS = (600, 601, 602)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# This must be present before the first cuBLAS operation when deterministic
# algorithms are enabled.  Merely calling ``torch.use_deterministic_algorithms``
# without it makes a fresh overnight process fail at the first calibration
# matrix multiplication.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)


@dataclass(frozen=True)
class B3ArmSpec:
    name: str
    width: int
    field_cloud: int = 256
    backward_microbatch: int = 256

    def validate(self) -> None:
        if (
            not self.name
            or min(self.width, self.field_cloud, self.backward_microbatch) <= 0
        ):
            raise ValueError("invalid B3 arm")
        if self.field_cloud % self.backward_microbatch:
            raise ValueError("B3 cloud must divide exactly into backward microbatches")


B3_ARMS = (
    B3ArmSpec("B3-native", 64),
    B3ArmSpec("B3-capacity", 368),
)


@dataclass(frozen=True)
class B3Config:
    units: tuple[int, ...] = B3_UNITS
    arms: tuple[B3ArmSpec, ...] = B3_ARMS
    steps: int = 30_000
    checkpoint_steps: tuple[int, ...] = (10_000, 20_000, 30_000)
    positives: int = 64
    calibration_samples: int = 256
    target_ess_fraction: float = 0.05
    eta: float = 0.5
    learning_rate: float = 2e-3
    latent_dim: int = 32
    image_size: int = 32
    channels: int = 3
    log_every: int = 500
    recovery_every: int = 1_000
    bootstrap_replicates: int = 500
    density_coverage_neighbors: int = 5
    bridge_parameter_count: int = 3_893_443
    capacity_parameter_tolerance_fraction: float = 0.01
    maximum_preflight_memory_fraction: float = 0.95

    def validate(self) -> None:
        if self.units != B3_UNITS or len(set(self.units)) != len(self.units):
            raise ValueError("B3 uses the frozen units 600, 601, 602")
        if self.arms != B3_ARMS:
            raise ValueError("B3 arm definitions changed")
        for arm in self.arms:
            arm.validate()
        if self.steps <= 0 or self.checkpoint_steps != tuple(
            sorted(set(self.checkpoint_steps))
        ):
            raise ValueError("invalid B3 training/checkpoint budget")
        if not self.checkpoint_steps or self.checkpoint_steps[-1] != self.steps:
            raise ValueError("the final B3 checkpoint must equal the training budget")
        if any(step <= 0 or step > self.steps for step in self.checkpoint_steps):
            raise ValueError("B3 checkpoint outside the budget")
        if min(self.positives, self.calibration_samples, self.latent_dim) <= 0:
            raise ValueError("invalid B3 sample dimensions")
        if not 0 < self.target_ess_fraction < 1 or self.eta <= 0:
            raise ValueError("invalid B3 kernel/teacher scale")
        if self.learning_rate <= 0 or min(self.log_every, self.recovery_every) <= 0:
            raise ValueError("invalid B3 optimization scale")
        if self.steps % self.recovery_every:
            raise ValueError("B3 recovery cadence must divide the training budget")
        if self.bootstrap_replicates < 100 or self.density_coverage_neighbors < 1:
            raise ValueError("invalid B3 evaluation budget")
        if not 0 < self.capacity_parameter_tolerance_fraction < 1:
            raise ValueError("invalid B3 capacity tolerance")
        if not 0 < self.maximum_preflight_memory_fraction < 1:
            raise ValueError("invalid B3 memory threshold")


def b3_config() -> B3Config:
    result = B3Config()
    result.validate()
    return result


def configure_deterministic_execution(device: torch.device | str) -> dict:
    """Enable the deterministic CUDA contract and make it artifact-visible."""
    device = torch.device(device)
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if device.type == "cuda" and configured != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "B3 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA starts"
        )
    torch.use_deterministic_algorithms(True)
    return {
        "deterministic_algorithms": True,
        "cublas_workspace_config": configured,
    }


def b3_seed(unit: int, role: str, event: int | None = None) -> int:
    labels: tuple[object, ...] = ("b3", B3_PHASE, unit, role)
    if event is not None:
        labels += (event,)
    return derive_seed(MASTER_SEED + B3_SEED_OFFSET, *labels)


def seed_manifest(unit: int) -> dict:
    return {
        "unit": int(unit),
        "master_plus_offset": MASTER_SEED + B3_SEED_OFFSET,
        "stage": B3_PHASE,
        "shared_streams": {
            role: b3_seed(unit, role)
            for role in (
                "model-init",
                "calibration-indices",
                "target-training",
                "evaluation-latent",
                "audit-latent",
                "bootstrap",
            )
        },
        "event_seed_rule": "b3_seed(unit, role, zero_based_event)",
    }


def build_generator(
    arm: B3ArmSpec, config: B3Config, unit: int, device
) -> OneStepGenerator:
    arm.validate()
    return OneStepGenerator(
        config.latent_dim,
        config.channels,
        config.image_size,
        arm.width,
        b3_seed(unit, "model-init"),
    ).to(device)


def assert_samplewise_generator(model: nn.Module) -> None:
    """Reject layers that would make chunked and full forwards inequivalent."""
    forbidden = (
        nn.modules.batchnorm._BatchNorm,
        nn.modules.dropout._DropoutNd,
        nn.Dropout,
        nn.AlphaDropout,
    )
    offenders = [
        type(module).__name__
        for module in model.modules()
        if isinstance(module, forbidden)
    ]
    if offenders:
        raise TypeError(
            f"B3 microbatching requires samplewise deterministic layers: {offenders}"
        )


def calibration_indices(pool_size: int, unit: int, config: B3Config) -> np.ndarray:
    if pool_size <= 0:
        raise ValueError("empty B3 target pool")
    rng = np.random.default_rng(b3_seed(unit, "calibration-indices"))
    return rng.integers(0, pool_size, config.calibration_samples, dtype=np.int64)


def calibrate_operator(pool: torch.Tensor, unit: int, config: B3Config, device):
    """One target-only calibration shared exactly by both B3 arms."""
    config.validate()
    indices = calibration_indices(len(pool), unit, config)
    target = pool[torch.as_tensor(indices)].to(device)
    branch = build_family(
        GeometryConfig(
            family="raw",
            base_kernel="smooth_laplace",
            target_ess_fraction=config.target_ess_fraction,
        ),
        config.channels,
    ).branches[0]
    kernel = calibrate_block_kernel(
        branch,
        target,
        "smooth_laplace",
        0.5,
        1.0,
        1e-3,
        combine="sum",
        target_ess_fraction=config.target_ess_fraction,
    )
    return branch, kernel, indices


def kernel_payload(kernel: BlockKernel) -> dict:
    return {
        "base": kernel.base,
        "taus": [float(value) for value in kernel.taus],
        "weights": [float(value) for value in kernel.weights],
        "eps": float(kernel.eps),
        "combine": kernel.combine,
        "legacy_target_ess_fraction": 0.05,
    }


@torch.no_grad()
def construct_full_teacher(
    model: nn.Module,
    latent: torch.Tensor,
    positive: torch.Tensor,
    branch,
    kernel: BlockKernel,
    eta: float,
    *,
    diagnostics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    cloud = model(latent).detach()
    drift, health = KG.field(
        cloud,
        positive,
        cloud,
        branch,
        kernel,
        direction_mode="paper",
        normalization="rms",
        diagnostics=diagnostics,
    )
    teacher = corrected_teacher(cloud + eta * drift, positive, mode="scalar").detach()
    if teacher.shape != cloud.shape or not bool(torch.isfinite(teacher).all()):
        raise FloatingPointError("invalid B3 full-cloud teacher")
    return teacher, cloud, health


def regression_backward(
    model: nn.Module,
    latent: torch.Tensor,
    teacher: torch.Tensor,
    microbatch: int,
) -> float:
    """Backpropagate the exact full-cloud mean with bounded activation memory."""
    if latent.shape[0] != teacher.shape[0] or len(latent) < 1:
        raise ValueError("B3 latent and teacher clouds must match and be nonempty")
    if microbatch <= 0:
        raise ValueError("B3 microbatch must be positive")
    total = torch.zeros((), device=latent.device, dtype=torch.float64)
    full_count = len(latent)
    for start in range(0, full_count, microbatch):
        stop = min(start + microbatch, full_count)
        output = model(latent[start:stop])
        per_sample = (output - teacher[start:stop]).square().flatten(1).sum(1)
        chunk_loss = per_sample.sum() / full_count
        if not bool(torch.isfinite(chunk_loss)):
            raise FloatingPointError("non-finite B3 regression loss")
        chunk_loss.backward()
        total += chunk_loss.detach().double()
    return float(total)


@dataclass
class B3TrainResult:
    model: OneStepGenerator
    history: list[dict]
    wall_seconds: float
    peak_memory_bytes: int | None
    peak_memory_reserved_bytes: int | None
    optimizer_updates: int
    generated_examples: int
    target_examples: int
    model_batch_forward_calls: int
    sample_forward_equivalents: int
    resumed_from_step: int


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


B3ProgressCallback = Callable[
    [
        int,
        OneStepGenerator,
        torch.optim.Optimizer,
        np.random.Generator,
        dict,
        list[dict],
        float,
    ],
    None,
]


def train_b3_arm(
    pool: torch.Tensor,
    unit: int,
    arm: B3ArmSpec,
    config: B3Config,
    branch,
    kernel: BlockKernel,
    device: torch.device | str,
    *,
    checkpoint: B3ProgressCallback | None = None,
    recovery: B3ProgressCallback | None = None,
    resume: dict | None = None,
) -> B3TrainResult:
    """Train one arm while replaying shared latent and target streams."""
    config.validate()
    arm.validate()
    if unit not in config.units:
        raise ValueError(f"B3 unit must lie in {config.units}")
    device = torch.device(device)
    model = build_generator(arm, config, unit, device)
    assert_samplewise_generator(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    target_rng = np.random.default_rng(b3_seed(unit, "target-training"))
    history: list[dict] = []
    start_step = 0
    elapsed_before = 0.0
    prior_peak_memory = 0
    prior_peak_reserved = 0
    if resume is not None:
        start_step = int(resume.get("step", -1))
        if not 0 <= start_step <= config.steps:
            raise ValueError("B3 recovery step lies outside the training budget")
        model.load_state_dict(resume["state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        target_rng.bit_generator.state = resume["target_rng_state"]
        history = list(resume.get("history", []))
        elapsed_before = float(resume.get("wall_seconds", 0.0))
        prior_peak_memory = int(resume.get("peak_memory_bytes") or 0)
        prior_peak_reserved = int(resume.get("peak_memory_reserved_bytes") or 0)
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(start_step, config.steps):
        learning_rate = (
            config.learning_rate
            * 0.5
            * (1.0 + math.cos(math.pi * step / max(config.steps, 1)))
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        latent = sample_latent(
            arm.field_cloud,
            config.latent_dim,
            b3_seed(unit, "training-latent", step),
            device,
        )
        indices = target_rng.integers(0, len(pool), config.positives)
        positive = pool[torch.as_tensor(indices)].to(device)
        recorded_step = step + 1
        diagnose = recorded_step == 1 or recorded_step in config.checkpoint_steps
        teacher, cloud, field_health = construct_full_teacher(
            model,
            latent,
            positive,
            branch,
            kernel,
            config.eta,
            diagnostics=diagnose,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = regression_backward(model, latent, teacher, arm.backward_microbatch)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise FloatingPointError("non-finite B3 gradient")
        optimizer.step()

        if diagnose:
            realized_ess, dead_rows = rectangular_ess(kernel, branch, cloud, positive)
        else:
            realized_ess = dead_rows = None
        record = {
            "step": recorded_step,
            "loss": loss,
            "learning_rate": learning_rate,
            "gradient_norm": float(gradient_norm),
            "realized_ess": realized_ess,
            "dead_row_fraction": dead_rows,
            "field_health": field_health if diagnose else None,
            "wall_seconds": elapsed_before + time.time() - started,
        }
        if diagnose or recorded_step % config.log_every == 0:
            history.append(record)
        if checkpoint is not None and recorded_step in config.checkpoint_steps:
            checkpoint(
                recorded_step,
                model,
                optimizer,
                target_rng,
                record,
                history,
                elapsed_before + time.time() - started,
            )
        if recovery is not None and recorded_step % config.recovery_every == 0:
            recovery(
                recorded_step,
                model,
                optimizer,
                target_rng,
                record,
                history,
                elapsed_before + time.time() - started,
            )

    chunk_calls = math.ceil(arm.field_cloud / arm.backward_microbatch)
    return B3TrainResult(
        model=model,
        history=history,
        wall_seconds=elapsed_before + time.time() - started,
        peak_memory_bytes=(
            max(prior_peak_memory, torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        peak_memory_reserved_bytes=(
            max(prior_peak_reserved, torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        ),
        optimizer_updates=config.steps,
        generated_examples=2 * config.steps * arm.field_cloud,
        target_examples=config.steps * config.positives,
        model_batch_forward_calls=config.steps * (1 + chunk_calls),
        sample_forward_equivalents=2 * config.steps * arm.field_cloud,
        resumed_from_step=start_step,
    )


def phase30_train_config(config: B3Config, arm: B3ArmSpec) -> TrainConfig:
    """Explicitly expose the historical optimizer/model fields B3 inherits."""
    return TrainConfig(
        steps=config.steps,
        batch=config.positives,
        field_cloud=arm.field_cloud,
        learning_rate=config.learning_rate,
        latent_dim=config.latent_dim,
        width=arm.width,
        image_size=config.image_size,
        channels=config.channels,
    )
