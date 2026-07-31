"""Phase-2 failure investigation (post-hoc; feeds no gate).

Phase 2 found fixed compositional geometry 37% worse than raw pixels on
CIFAR-16, and *every* drifting arm roughly 4x worse than a plain
sliced-Wasserstein skyline.  The instrumentation localized the geometry
failure to density-seeking -- better `nearest_real`, worse coverage -- but
left the skyline gap unexplained.

This module chases that gap, starting from a discrepancy found by re-reading
the plan against the repository: the field every phase has used is the
row-normalized SNIS mean shift, which `lowdim_drift` labels "DIAGNOSTIC
ONLY", not the paper's Algorithm-2 bi-softmax estimator.  The omitted
column reweighting is measurably an anti-density-seeking mechanism, which is
exactly the failure mode observed.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import oracle as O
from .config import (
    ArmConfig, FieldConfig, GeometryConfig, MASTER_SEED, MixtureConfig,
    ObjectiveConfig, TrainConfig, derive_seed,
)
from . import kernel_gradient as KG
from . import metrics as M
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .diagnostics import paired_log_ratio, provenance, write_json
from .evaluate import evaluate_arm, evaluation_pools, null_reference
from .train import train_arm

HERE = Path(__file__).resolve().parent


def _arm(arm_id: str, family: str, direction: str, anchor: bool = False,
         **geometry) -> ArmConfig:
    objective = (ObjectiveConfig(lambda_anchor=1.0, lambda_geometry=1.0)
                 if anchor else
                 ObjectiveConfig(lambda_anchor=0.0, lambda_geometry=1.0))
    return ArmConfig(
        arm_id, anchor,
        GeometryConfig(family=family, base_kernel="smooth_laplace",
                       **geometry),
        FieldConfig(direction_mode=direction), MixtureConfig(adaptive=False),
        objective, note=f"{family} / {direction}")


def _cell(train_target, eval_target, seed: int, config: TrainConfig,
          arms, with_skyline: bool = True) -> list[dict]:
    pools = evaluation_pools(eval_target, config, seed)
    null = null_reference(eval_target, pools, seed)
    rows = []
    for arm in arms:
        outcome = train_arm(arm, train_target, config, seed)
        row = {"arm": arm.arm_id, "seed": seed, "note": arm.note}
        row.update(evaluate_arm(outcome, eval_target, config, pools, null,
                                seed))
        rows.append(row)
        print(f"      {arm.arm_id:22} score_v2="
              f"{row.get('geometry_score_v2', float('nan')):8.3f} "
              f"ed2={row.get('ed2', float('nan')):7.4f} "
              f"cover={row.get('coverage', float('nan')):5.3f} "
              f"near={row.get('nearest_real', float('nan')):5.3f} "
              f"{outcome.wall_seconds:5.1f}s", flush=True)
    if with_skyline:
        skyline = O.train_skyline(train_target, config, seed, pools, null)
        rows.append({
            "arm": "SKY", "seed": seed, "note": "skyline",
            "geometry_score_v2": skyline.score["geometry_score"],
            "geometry_ratios_v2": skyline.score["geometry_ratios"],
            "ed2": skyline.metrics["ed2"],
            "coverage": skyline.coverage,
            "precision": skyline.precision,
            "nearest_real": skyline.metrics.get("nearest_real"),
        })
        print(f"      {'SKY':22} score_v2="
              f"{skyline.score['geometry_score']:8.3f} "
              f"ed2={skyline.metrics['ed2']:7.4f} "
              f"cover={skyline.coverage:5.3f}", flush=True)
    return rows


def f1_paper_field(resolution: int, seeds: int, steps: int,
                   root: str | None) -> dict:
    """Does the paper's real Algorithm-2 estimator close the skyline gap?

    Compares the SNIS field every phase has used against the paper's
    bi-softmax estimator, for the raw and the wavelet kernel, with the
    skyline as the reference.
    """
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                         audit_batch=32, eval_samples=512,
                         image_size=resolution)
    arms = [
        _arm("raw_snis", "raw", "standard"),
        _arm("raw_paper", "raw", "paper"),
        _arm("wavelet_snis", "wavelet", "standard"),
        _arm("wavelet_paper", "wavelet", "paper"),
        _arm("wavelet_paper_anchor", "wavelet", "paper", anchor=True),
    ]
    rows = []
    for seed in range(seeds):
        print(f"    F1 seed {seed}", flush=True)
        rows.extend(_cell(train, evaluation, MASTER_SEED + seed, config,
                          arms))
    return {"rows": rows, "steps": steps}


def f2_batch_sweep(resolution: int, steps: int, root: str | None) -> dict:
    """Is the drifting field simply starved of samples at batch 64?

    The paper operates at a declared per-class batch of 64, but a kernel
    field's quality depends on how many neighbours each probe sees.  The
    skyline sorts a whole batch globally and may simply be using its samples
    better.
    """
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for batch in (32, 64, 128, 256):
        config = TrainConfig(steps=steps, batch=batch,
                             controller_batch=32, audit_batch=32,
                             eval_samples=512, image_size=resolution)
        print(f"    F2 batch={batch}", flush=True)
        for row in _cell(train, evaluation, MASTER_SEED, config,
                         [_arm("raw_paper", "raw", "paper"),
                          _arm("raw_snis", "raw", "standard")]):
            row["batch"] = batch
            rows.append(row)
    return {"rows": rows}


def f3_budget_sweep(resolution: int, root: str | None) -> dict:
    """Does drifting catch the skyline with more steps, or plateau?"""
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for steps in (300, 600, 1200, 2400):
        config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                             audit_batch=32, eval_samples=512,
                             image_size=resolution)
        print(f"    F3 steps={steps}", flush=True)
        for row in _cell(train, evaluation, MASTER_SEED, config,
                         [_arm("raw_paper", "raw", "paper")]):
            row["steps"] = steps
            rows.append(row)
    return {"rows": rows}


def f4_free_particles(resolution: int, seeds: int, steps: int,
                      root: str | None) -> dict:
    """Is the gap in the FIELD, or in the generator that has to realize it?

    G0.5 certified that the raw field's zero-set is reachable -- but it
    tested a *free particle cloud*, which can take any configuration.  The
    generator maps a 32-dimensional latent through a smooth network, so its
    reachable set is a low-dimensional manifold.  A field whose zero-set is
    reachable by particles may have no zero inside that manifold.

    This runs the identical field on both and scores them the same way.  If
    free particles reach skyline-level quality, the field is sound and the
    failure is in the coupling between the objective and the parametrization
    -- a different defect, needing a different reform.
    """
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                         audit_batch=32, eval_samples=512,
                         image_size=resolution)
    rows = []
    for seed in range(seeds):
        pools = evaluation_pools(evaluation, config, MASTER_SEED + seed)
        null = null_reference(evaluation, pools, MASTER_SEED + seed)
        rng = np.random.default_rng(derive_seed(MASTER_SEED + seed, "f4"))
        calibration = train.sample(256, rng)
        for family in ("raw", "wavelet"):
            geometry = GeometryConfig(family=family,
                                      base_kernel="smooth_laplace")
            branch = build_family(geometry, 3).branches[0]
            kernel = calibrate_block_kernel(
                branch, calibration, "smooth_laplace",
                geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
                geometry.kernel_eps, combine=geometry.combine,
                target_ess_fraction=geometry.target_ess_fraction)
            for mode in ("standard", "paper"):
                cloud = torch.tensor(
                    rng.normal(scale=0.5, size=(config.eval_samples, 3,
                                                resolution, resolution)),
                    dtype=torch.float32)
                for index in range(steps):
                    fresh = train.sample(config.batch, rng)
                    drift, _ = KG.field(
                        cloud, fresh, cloud, branch, kernel,
                        direction_mode=mode, normalization="rms",
                        diagnostics=False)
                    cloud = cloud + 0.2 * (1.0 - index / steps) * drift
                measured = M.raw_metrics(
                    cloud, pools["eval"], pools["cal_a"], pools["cal_b"],
                    np.random.default_rng(derive_seed(seed, "f4m")),
                    None, target_null=pools["null"])
                score = M.normalized_geometry_score_v2(measured, null)
                rows.append({
                    "arm": f"free_{family}_{mode}", "seed": seed,
                    "family": family, "direction_mode": mode,
                    "geometry_score_v2": score["geometry_score"],
                    "geometry_ratios_v2": score["geometry_ratios"],
                    "ed2": measured["ed2"],
                    "coverage": measured["coverage"],
                    "precision": measured["precision"],
                    "nearest_real": measured.get("nearest_real"),
                })
                print(f"      free_{family}_{mode:9} score_v2="
                      f"{score['geometry_score']:8.3f} "
                      f"ed2={measured['ed2']:7.4f} "
                      f"cover={measured['coverage']:5.3f}", flush=True)
    return {"rows": rows, "steps": steps}


def f5_rollout_teacher(resolution: int, seeds: int, steps: int,
                       root: str | None) -> dict:
    """The implied repair: regress to a ROLLED-OUT cloud, not a single step.

    F4 shows the field is sound -- free particles reach skyline quality --
    and the generator loses a further 3.5x.  The suspect is the transfer:
    the paper-style stop-gradient regression asks the generator to match
    ``output + eta * V``, a single Euler step, which is a very weak teacher
    when the generator must also stay a smooth function of its latent.

    A rollout teacher runs the particle dynamics ``K`` steps from the current
    output and regresses onto where they land.  ``K = 1`` recovers the
    current behaviour.  The repository has an independently confirmed
    precedent for this in the transport program
    (`AdaptiveRolloutConfirmationResults.md`).
    """
    from .models import OneStepGenerator, sample_latent
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                         audit_batch=32, eval_samples=512,
                         image_size=resolution)
    rows = []
    for seed in range(seeds):
        pools = evaluation_pools(evaluation, config, MASTER_SEED + seed)
        null = null_reference(evaluation, pools, MASTER_SEED + seed)
        rng = np.random.default_rng(derive_seed(MASTER_SEED + seed, "f5"))
        calibration = train.sample(256, rng)
        geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace")
        branch = build_family(geometry, 3).branches[0]
        kernel = calibrate_block_kernel(
            branch, calibration, "smooth_laplace",
            geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
            geometry.kernel_eps, combine=geometry.combine,
            target_ess_fraction=geometry.target_ess_fraction)
        for rollout in (1, 4, 16):
            model = OneStepGenerator(
                config.latent_dim, 3, resolution, config.width,
                derive_seed(MASTER_SEED + seed, "generator"))
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config.learning_rate)
            generator = torch.Generator().manual_seed(
                derive_seed(MASTER_SEED + seed, "latent") % (2 ** 31))
            for _ in range(steps):
                positives = train.sample(config.batch, rng)
                latent = torch.randn(config.batch, config.latent_dim,
                                     generator=generator)
                output = model(latent)
                with torch.no_grad():
                    cloud = output.detach().clone()
                    for _ in range(rollout):
                        drift, _ = KG.field(
                            cloud, positives, cloud, branch, kernel,
                            direction_mode="paper", normalization="rms",
                            diagnostics=False)
                        cloud = cloud + 0.5 * drift
                loss = ((output - cloud) ** 2).flatten(1).sum(1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            latent = sample_latent(config.eval_samples, config.latent_dim,
                                   derive_seed(MASTER_SEED + seed,
                                               "eval-latent"))
            with torch.no_grad():
                generated = model(latent)
            measured = M.raw_metrics(
                generated, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "f5m")),
                None, target_null=pools["null"])
            score = M.normalized_geometry_score_v2(measured, null)
            rows.append({
                "arm": f"rollout{rollout}", "seed": seed, "rollout": rollout,
                "geometry_score_v2": score["geometry_score"],
                "geometry_ratios_v2": score["geometry_ratios"],
                "ed2": measured["ed2"], "coverage": measured["coverage"],
                "precision": measured["precision"],
                "nearest_real": measured.get("nearest_real"),
            })
            print(f"      rollout K={rollout:3} score_v2="
                  f"{score['geometry_score']:8.3f} "
                  f"ed2={measured['ed2']:7.4f} "
                  f"cover={measured['coverage']:5.3f}", flush=True)
    return {"rows": rows, "steps": steps}


def summarize(rows: list[dict], key: str = "geometry_score_v2") -> dict:
    out: dict[str, float] = {}
    for arm in sorted({r["arm"] for r in rows}):
        values = [r[key] for r in rows if r["arm"] == arm
                  and isinstance(r.get(key), (int, float))
                  and np.isfinite(r[key])]
        if values:
            out[arm] = float(np.median(values))
    return out


def paired(rows: list[dict], candidate: str, baseline: str) -> dict:
    scores: dict[str, dict[int, float]] = {}
    for row in rows:
        value = row.get("geometry_score_v2")
        if value is not None and np.isfinite(value):
            scores.setdefault(row["arm"], {})[row["seed"]] = value
    if candidate not in scores or baseline not in scores:
        return {"ratio": float("nan"), "pairs": 0}
    keys = sorted(set(scores[candidate]) & set(scores[baseline]))
    return paired_log_ratio([scores[candidate][k] for k in keys],
                            [scores[baseline][k] for k in keys])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--only", type=str, default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase2_diagnosis.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if not cifar.available(args.root):
        raise SystemExit("CIFAR-10 is not present locally.")

    started = time.time()
    stages = {
        "F1_paper_field": lambda: f1_paper_field(
            args.resolution, args.seeds, args.steps, args.root),
        "F2_batch_sweep": lambda: f2_batch_sweep(
            args.resolution, args.steps, args.root),
        "F3_budget_sweep": lambda: f3_budget_sweep(
            args.resolution, args.root),
        "F4_free_particles": lambda: f4_free_particles(
            args.resolution, args.seeds, args.steps, args.root),
        "F5_rollout_teacher": lambda: f5_rollout_teacher(
            args.resolution, args.seeds, args.steps, args.root),
    }
    wanted = set(stages) if args.only == "all" else set(args.only.split(","))
    results = {}
    for name, function in stages.items():
        if name in wanted:
            print(f"--- {name} ---", flush=True)
            results[name] = function()

    if "F1_paper_field" in results:
        rows = results["F1_paper_field"]["rows"]
        results["F1_paper_field"]["median_score"] = summarize(rows)
        results["F1_paper_field"]["comparisons"] = {
            "raw_paper/raw_snis": paired(rows, "raw_paper", "raw_snis"),
            "wavelet_paper/wavelet_snis": paired(
                rows, "wavelet_paper", "wavelet_snis"),
            "wavelet_paper/raw_paper": paired(
                rows, "wavelet_paper", "raw_paper"),
            "raw_paper/SKY": paired(rows, "raw_paper", "SKY"),
            "raw_snis/SKY": paired(rows, "raw_snis", "SKY"),
            "wavelet_paper_anchor/wavelet_paper": paired(
                rows, "wavelet_paper_anchor", "wavelet_paper"),
        }

    payload = {
        "status": "phase2-post-hoc-diagnosis-feeds-no-gate",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    digest = write_json(args.out, payload)
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
