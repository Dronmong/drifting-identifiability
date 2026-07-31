"""Phase-4 failure investigation (post-hoc; feeds no gate).

Phase 4 left three things unexplained:

  * R11 fails at the paper's declared temperatures, and at tau = .05 it
    actively hurts while not restoring effective dimension at all;
  * the collapse is dimension-dependent and the self-mask mitigates it;
  * two mechanism hypotheses were refuted, so the cause is unknown.

The refuted P4C tests share a defect worth naming: both probed the teacher
map **at the target law**, where the field is near zero by construction and
nothing can happen.  Training spends almost all of its time far from there.
These experiments probe along the trajectory instead, and decompose the
generator's contribution into capacity, self-reference and dynamics.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from . import metrics as M
from .config import (
    GeometryConfig, MASTER_SEED, TrainConfig, derive_seed,
)
from .diagnostics import provenance, write_json
from .evaluate import evaluation_pools, null_reference
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import variance_matched_teacher

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 4000


def _kernel(train, resolution: int, rng, tau: float | None):
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace",
                              bandwidth_tau=tau)
    branch = build_family(geometry, 3).branches[0]
    calibration = train.sample(256, rng)
    if tau is not None:
        kernel = calibrate_block_kernel(
            branch, calibration, "smooth_laplace",
            geometry.bandwidth_quantile, tau, geometry.kernel_eps,
            combine=geometry.combine, target_ess_fraction=None)
    else:
        kernel = calibrate_block_kernel(
            branch, calibration, "smooth_laplace",
            geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
            geometry.kernel_eps, combine=geometry.combine,
            target_ess_fraction=geometry.target_ess_fraction)
    return branch, kernel


# ---------------------------------------------------------------------------
# H4: why does R11 fail at the paper's temperatures?
# ---------------------------------------------------------------------------


def h4_temperature_health(resolution: int, root: str | None) -> dict:
    """Is the field at the paper's tau carrying any information at all?

    The paper's temperature grid is stated in normalized units where the mean
    pairwise feature distance is 1.  At 768 ambient dimensions a very sharp
    kernel may see a single neighbour per probe, in which case the field is
    noise and no correction to the teacher's second moment can help.  That
    would explain P4A's temperature failures without any new mechanism.
    """
    train = cifar.cifar_target(resolution, "train", root)
    rows = []
    for tau in (None, 0.02, 0.05, 0.2, 1.0):
        rng = np.random.default_rng(derive_seed(MASTER_SEED, "h4", str(tau)))
        branch, kernel = _kernel(train, resolution, rng, tau)
        positive = train.sample(64, rng)
        # Two clouds: an untrained generator (where training starts) and a
        # real sample (where it should end).
        model = OneStepGenerator(32, 3, resolution, 64,
                                 derive_seed(MASTER_SEED, "generator"))
        with torch.no_grad():
            init_cloud = model(sample_latent(64, 32,
                                             derive_seed(MASTER_SEED, "l")))
        for name, cloud in (("init", init_cloud),
                            ("real", train.sample(64, rng))):
            _, stats = KG.field(cloud, positive, cloud, branch, kernel,
                                direction_mode="paper", normalization="none")
            rows.append({
                "tau": tau, "cloud": name,
                "bandwidth": float(kernel.taus.median()),
                "ess_fraction": stats["ess_fraction"],
                "affinity_median": stats["affinity_median"],
                "collapsed_row_fraction": stats["collapsed_row_fraction"],
                "denominator_floor_fraction": stats[
                    "denominator_floor_fraction"],
                "drift_rms_raw": stats["drift_rms_raw"],
                "distance_median": stats["distance_median"],
            })
            print(f"    H4 tau={str(tau):6} {name:5} "
                  f"bw={rows[-1]['bandwidth']:9.4f} "
                  f"ESS={rows[-1]['ess_fraction']:7.4f} "
                  f"aff={rows[-1]['affinity_median']:10.3e} "
                  f"collapsed={rows[-1]['collapsed_row_fraction']:5.3f}",
                  flush=True)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# H1 / H3: decompose the generator's contribution
# ---------------------------------------------------------------------------


def _fit_to_cloud(target_cloud: torch.Tensor, config: TrainConfig,
                  seed: int, steps: int) -> torch.Tensor:
    """Least-squares fit of the generator to a FIXED cloud."""
    model = OneStepGenerator(config.latent_dim, 3, config.image_size,
                             config.width, derive_seed(seed, "generator"))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    latent = sample_latent(len(target_cloud), config.latent_dim,
                           derive_seed(seed, "fit-latent"))
    for _ in range(steps):
        loss = ((model(latent) - target_cloud) ** 2).flatten(1).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(latent)


def h1h3_decompose(resolution: int, seeds: int, steps: int,
                   root: str | None) -> dict:
    """Capacity, self-reference, or dynamics?

    Four regimes sharing one field and one generator architecture:

      `free`            particles moved directly by the field (no generator);
      `fit_to_free`     the generator fitted by least squares to the
                        converged free cloud -- pure capacity;
      `frozen_teacher`  trained by stop-gradient regression, but with the
                        teacher computed against a FROZEN reference cloud
                        instead of the generator's own output -- removes the
                        self-reference while keeping the regression;
      `self_teacher`    the ordinary recipe.

    If `fit_to_free` collapses, it is capacity.  If only `self_teacher`
    collapses, it is the self-referential fixed-point iteration.
    """
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                         audit_batch=32, eval_samples=512,
                         image_size=resolution)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        rng = np.random.default_rng(derive_seed(seed, "h1"))
        branch, kernel = _kernel(train, resolution, rng, None)
        reference_dimension = M.effective_dimension(pools["eval"])

        def record(name: str, cloud: torch.Tensor) -> None:
            measured = M.raw_metrics(
                cloud, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "h1m")), None,
                target_null=pools["null"])
            rows.append({
                "regime": name, "seed": seed,
                "geometry_score_v2": M.normalized_geometry_score_v2(
                    measured, null)["geometry_score"],
                "ed2": measured["ed2"],
                "effective_dimension_ratio": measured[
                    "effective_dimension_ratio"],
                "reference_effective_dimension": reference_dimension})
            print(f"    H1 {name:16} score="
                  f"{rows[-1]['geometry_score_v2']:7.3f} "
                  f"ed2={rows[-1]['ed2']:7.4f} "
                  f"eff_dim={rows[-1]['effective_dimension_ratio']:5.3f}",
                  flush=True)

        # free particles
        cloud = torch.tensor(
            rng.normal(scale=0.5, size=(config.eval_samples, 3, resolution,
                                        resolution)), dtype=torch.float32)
        for step in range(steps):
            drift, _ = KG.field(cloud, train.sample(64, rng), cloud, branch,
                                kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            cloud = cloud + 0.2 * (1.0 - step / steps) * drift
        free_cloud = cloud
        record("free", free_cloud)

        # capacity: fit the generator to that converged cloud
        record("fit_to_free", _fit_to_cloud(free_cloud, config, seed, 1500))

        # frozen teacher vs self teacher
        for regime in ("frozen_teacher", "self_teacher"):
            model = OneStepGenerator(config.latent_dim, 3, resolution,
                                     config.width,
                                     derive_seed(seed, "generator"))
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config.learning_rate)
            generator = torch.Generator().manual_seed(
                derive_seed(seed, "latent") % (2 ** 31))
            frozen = free_cloud[:config.batch].clone()
            for _ in range(steps):
                positives = train.sample(config.batch, rng)
                latent = torch.randn(config.batch, config.latent_dim,
                                     generator=generator)
                output = model(latent)
                with torch.no_grad():
                    negatives = (frozen if regime == "frozen_teacher"
                                 else output.detach())
                    drift, _ = KG.field(output.detach(), positives,
                                        negatives, branch, kernel,
                                        direction_mode="paper",
                                        normalization="rms",
                                        diagnostics=False)
                    teacher = output.detach() + 0.5 * drift
                loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            latent = sample_latent(config.eval_samples, config.latent_dim,
                                   derive_seed(seed, "eval-latent"))
            with torch.no_grad():
                record(regime, model(latent))
    return {"rows": rows}


# ---------------------------------------------------------------------------
# H2: is the teacher contractive AWAY from the target law?
# ---------------------------------------------------------------------------


def h2_trajectory_contraction(resolution: int, root: str | None) -> dict:
    """The refuted P4C test probed the teacher at the target law only.

    Repeat it along an interpolation from an untrained generator's output to
    a real sample, which is roughly the path training takes.  A map that is
    benign at the fixed point can still be strongly contractive on the way
    there.
    """
    train = cifar.cifar_target(resolution, "train", root)
    rng = np.random.default_rng(derive_steps := derive_seed(MASTER_SEED,
                                                            "h2"))
    del derive_steps
    branch, kernel = _kernel(train, resolution, rng, None)
    model = OneStepGenerator(32, 3, resolution, 64,
                             derive_seed(MASTER_SEED, "generator"))
    with torch.no_grad():
        start = model(sample_latent(512, 32, derive_seed(MASTER_SEED, "l")))
    end = train.sample(512, rng)
    rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        cloud = (1 - alpha) * start + alpha * end
        positives = train.sample(64, rng)
        drift, _ = KG.field(cloud, positives, cloud, branch, kernel,
                            direction_mode="paper", normalization="rms",
                            diagnostics=False)
        teacher = cloud + 0.5 * drift
        before = M.effective_dimension(cloud)
        after = M.effective_dimension(teacher)
        matched = variance_matched_teacher(teacher, positives)
        rows.append({
            "alpha": alpha,
            "eff_dim_cloud": before, "eff_dim_teacher": after,
            "eff_dim_ratio": after / max(before, 1e-12),
            "eff_dim_teacher_matched": M.effective_dimension(matched),
            "variance_ratio": float(
                teacher.reshape(len(teacher), -1).var(0).mean()
                / cloud.reshape(len(cloud), -1).var(0).mean()),
        })
        print(f"    H2 alpha={alpha:4.2f} eff_dim {before:6.2f} -> "
              f"{after:6.2f} (ratio {rows[-1]['eff_dim_ratio']:6.4f}) "
              f"var_ratio={rows[-1]['variance_ratio']:6.4f}", flush=True)
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--only", type=str, default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase5_diagnosis.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if not cifar.available(args.root):
        raise SystemExit("CIFAR-10 is not present locally.")

    started = time.time()
    stages = {
        "H4_temperature_health": lambda: h4_temperature_health(
            args.resolution, args.root),
        "H2_trajectory": lambda: h2_trajectory_contraction(
            args.resolution, args.root),
        "H1H3_decompose": lambda: h1h3_decompose(
            args.resolution, args.seeds, args.steps, args.root),
    }
    wanted = set(stages) if args.only == "all" else set(args.only.split(","))
    results = {}
    for name, function in stages.items():
        if name in wanted:
            print(f"--- {name} ---", flush=True)
            results[name] = function()

    payload = {
        "status": "phase4-failure-investigation-feeds-no-gate",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    digest = write_json(args.out, payload)
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
