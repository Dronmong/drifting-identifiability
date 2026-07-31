"""Reform R1: the skyline arm and target admissibility.

The Phase-1 screen compared nine arms of which the best reached calibrated
precision 0.199 against a target-null level of 0.98, and four of nine targets
had zero coverage for *every* arm.  A comparison in which nothing succeeds
measures which method degrades most gracefully, not which one supplies an
image prior.  The deep diagnosis confirmed the point: at the frozen budget a
sliced-Wasserstein objective on the same generator scores *worse* than the
raw-pixel baseline, and two of three probed targets stay unsolvable even at
four times the budget.

This module makes that failure impossible to repeat:

* :func:`train_skyline` trains the SAME generator with a well-posed
  distribution-matching objective that shares none of the suspected defects
  -- no kernel, no bandwidth, no feature map, no field normalization;
* :func:`admissible_targets` refuses any target the skyline cannot solve;
* :func:`sufficient_budget` derives the training budget from the skyline
  rather than from a baseline arm.

The skyline is **not an arm**.  It says nothing about encoder-free drifting
and never enters a gate as a candidate; it calibrates the testbed and bounds
what any arm could achieve there.

The Phase-1 budget was frozen from the baseline's *coverage*, which saturates
at 1.0 while precision is still 0.199.  Admissibility here is stated in terms
of precision, which does not saturate early.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import metrics as M
from .config import TrainConfig, derive_seed
from .datasets import ImageTarget
from .models import OneStepGenerator, sample_latent

# A target is admissible if the skyline reaches this fraction of the
# target-vs-target null precision.  Frozen here, before any successor screen.
ADMISSIBLE_PRECISION_FRACTION = 0.5
SKYLINE_PROJECTIONS = 64


def sliced_w2_loss(generated: torch.Tensor, target: torch.Tensor,
                   directions: torch.Tensor) -> torch.Tensor:
    """Differentiable sliced Wasserstein-2 between two equal-size batches."""
    if len(generated) != len(target):
        raise ValueError("sliced Wasserstein needs equal batch sizes")
    left = torch.sort(generated.reshape(len(generated), -1) @ directions.T,
                      dim=0).values
    right = torch.sort(target.reshape(len(target), -1) @ directions.T,
                       dim=0).values
    return ((left - right) ** 2).mean()


@dataclass
class SkylineResult:
    model: OneStepGenerator
    steps: int
    metrics: dict
    score: dict
    precision: float
    coverage: float


def train_skyline(target: ImageTarget, config: TrainConfig, seed: int,
                  pools: dict, null: dict) -> SkylineResult:
    """Train the shared generator with the sliced-Wasserstein objective."""
    model = OneStepGenerator(config.latent_dim, config.channels,
                             config.image_size, config.width,
                             derive_seed(seed, "generator"))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(derive_seed(seed, "skyline", target.name))
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(
        derive_seed(seed, "skyline-latent", target.name) % (2 ** 31))
    dim = config.channels * config.image_size ** 2
    for _ in range(config.steps):
        batch = target.sample(config.batch, rng)
        latent = torch.randn(config.batch, config.latent_dim,
                             generator=torch_generator)
        directions = torch.randn(SKYLINE_PROJECTIONS, dim,
                                 generator=torch_generator)
        directions = directions / directions.norm(dim=1, keepdim=True)
        loss = sliced_w2_loss(model(latent), batch, directions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    latent = sample_latent(config.eval_samples, config.latent_dim,
                           derive_seed(seed, "eval-latent"))
    with torch.no_grad():
        generated = model(latent)
    measured = M.raw_metrics(
        generated, pools["eval"], pools["cal_a"], pools["cal_b"],
        np.random.default_rng(derive_seed(seed, "skyline-metrics")),
        target, target_null=pools["null"])
    return SkylineResult(
        model=model, steps=config.steps, metrics=measured,
        score=M.normalized_geometry_score_v2(measured, null),
        precision=float(measured["precision"]),
        coverage=float(measured["coverage"]))


def free_particle_reference(target: ImageTarget, config: TrainConfig,
                            seed: int, branch, kernel, direction_mode: str,
                            pools: dict, null: dict,
                            step_size: float = 0.2) -> dict:
    """Reform R13: what the SAME field achieves without a generator.

    The Phase-2 entry gate asked whether a field's zero could be reached by
    *some* cloud and answered yes by moving free particles.  The generator's
    reachable set is a low-dimensional manifold, and the whole Phase-2
    failure lived in that difference.

    The natural-looking version of this measurement -- comparing the field
    *residual* at the generator's output against the free cloud's -- is
    confounded and must not be used: a collapsed generator sits in a dense
    region where the field is weak, so it records a *lower* residual than a
    good one (measured: 0.965 for a collapsed arm against 1.402 for a healthy
    one).  Field residual does not rank generators.

    What is well posed is the same *quality* score under both
    parametrizations.  This returns the free-particle cloud's score, so a
    runner can report ``generator_score / free_particle_score`` -- the
    parametric quality gap, which is exactly the quantity the Phase-2
    diagnosis found to be ~3.5 and predicted R11 would close.
    """
    from . import kernel_gradient as KG
    from . import metrics as M

    rng = np.random.default_rng(derive_seed(seed, "r13", target.name))
    cloud = torch.tensor(
        rng.normal(scale=0.5,
                   size=(config.eval_samples, config.channels,
                         config.image_size, config.image_size)),
        dtype=torch.float32)
    steps = config.steps
    for index in range(steps):
        fresh = target.sample(config.batch, rng)
        drift, _ = KG.field(cloud, fresh, cloud, branch, kernel,
                            direction_mode=direction_mode,
                            normalization="rms", diagnostics=False)
        cloud = cloud + step_size * (1.0 - index / steps) * drift
    measured = M.raw_metrics(
        cloud, pools["eval"], pools["cal_a"], pools["cal_b"],
        np.random.default_rng(derive_seed(seed, "r13-metrics")),
        None, target_null=pools["null"])
    score = M.normalized_geometry_score_v2(measured, null)
    return {
        "free_particle_score": score["geometry_score"],
        "free_particle_ed2": measured["ed2"],
        "free_particle_coverage": measured["coverage"],
        "free_particle_effective_dimension_ratio": measured.get(
            "effective_dimension_ratio"),
    }


def admissibility(target: ImageTarget, config: TrainConfig, seed: int,
                  pools: dict, null: dict) -> dict:
    """Is this target solvable at this budget?  Verdict plus evidence."""
    result = train_skyline(target, config, seed, pools, null)
    required = ADMISSIBLE_PRECISION_FRACTION * float(null["precision"])
    return {
        "target": target.name,
        "steps": config.steps,
        "skyline_precision": result.precision,
        "skyline_coverage": result.coverage,
        "null_precision": float(null["precision"]),
        "required_precision": required,
        "skyline_geometry_score": result.score["geometry_score"],
        "admissible": bool(result.precision >= required),
    }


def admissible_targets(rows: list[dict]) -> list[str]:
    return [row["target"] for row in rows if row["admissible"]]


def sufficient_budget(target: ImageTarget, seed: int, pools_for,
                      null_for, base: TrainConfig,
                      candidates=(300, 600, 1200, 2400)) -> dict:
    """Smallest budget in ``candidates`` at which the skyline clears the bar.

    Reform R1: the training budget is derived from the skyline, not from a
    baseline arm.  Returns the first admissible budget, or the full sweep
    with ``admissible: False`` if the target never clears -- in which case the
    target must be dropped rather than the bar lowered.
    """
    sweep = []
    for steps in candidates:
        config = TrainConfig(**{**base.__dict__, "steps": steps})
        pools = pools_for(target, config, seed)
        null = null_for(target, pools, seed)
        row = admissibility(target, config, seed, pools, null)
        sweep.append(row)
        if row["admissible"]:
            return {"target": target.name, "budget": steps,
                    "admissible": True, "sweep": sweep}
    return {"target": target.name, "budget": None, "admissible": False,
            "sweep": sweep}
