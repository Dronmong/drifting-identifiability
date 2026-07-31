"""Deep diagnosis of the Phase-1 failure (post-hoc; feeds no gate).

Each function isolates one candidate cause with a measurement rather than an
argument.  Nothing here selects an arm or tunes a hyperparameter for a
reported result; the outputs are written to `phase1_diagnosis.json` and
interpreted in `EncoderIndependentPhase1Diagnosis.md`.

    uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import datasets as D
from . import kernel_gradient as KG
from . import metrics as M
from .config import (
    FieldConfig, GeometryConfig, MASTER_SEED, ObjectiveConfig, TrainConfig,
    derive_seed,
)
from .diagnostics import provenance, write_json
from .evaluate import evaluate_arm, evaluation_pools, null_reference
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import geometry_loss
from .train import ArmConfig, arm_by_id, train_arm

HERE = Path(__file__).resolve().parent
DIAG_TARGETS = ("checkerboard", "texture_blocks", "rings_islands")


def _cfg(steps: int = 300) -> TrainConfig:
    return TrainConfig(steps=steps, batch=64, controller_batch=32,
                       audit_batch=32, eval_samples=512)


# ---------------------------------------------------------------------------
# D1: is the geometry loss a constant?
# ---------------------------------------------------------------------------


def d1_loss_constancy() -> dict:
    """The stop-gradient loss under RMS normalization equals eta^2 exactly.

    ``L = mean_i |f_i - sg(f_i + eta V_i)|^2 = eta^2 * mean_i |V_i|^2``, and
    RMS normalization sets ``mean_i |V_i|^2 = 1`` by construction.  So the
    loss cannot move, and cannot ever reach zero.
    """
    g = torch.Generator().manual_seed(1)
    out = []
    for eta in (0.25, 0.5, 1.0):
        for trial in range(3):
            drift = torch.randn(64, 3, 16, 16, generator=g)
            rms = float(torch.sqrt(
                (drift.reshape(64, -1) ** 2).sum(dim=1).mean()))
            normalized = drift / rms
            output = torch.randn(64, 3, 16, 16, generator=g)
            out.append({
                "eta": eta, "trial": trial,
                "loss_rms_normalized": float(
                    geometry_loss(output, normalized, eta)),
                "eta_squared": eta ** 2,
                "loss_unnormalized": float(
                    geometry_loss(output, drift, eta)),
            })
    pinned = all(abs(r["loss_rms_normalized"] - r["eta_squared"]) < 1e-4
                 for r in out)
    return {"rows": out, "loss_is_pinned_to_eta_squared": pinned}


# ---------------------------------------------------------------------------
# D2: does the field vanish when the laws match?
# ---------------------------------------------------------------------------


def d2_field_floor(target_names=DIAG_TARGETS) -> dict:
    """Raw field magnitude at q = p (the finite-sample floor) vs at the
    trained A4 output.

    If a trained arm sits at the floor it has converged as far as the
    estimator allows.  If it sits far above, it is still being driven.
    """
    rows = []
    for name in target_names:
        target = D.named(name)
        seed = derive_seed(MASTER_SEED, "d2", name)
        rng = np.random.default_rng(seed)
        calibration = target.sample(256, np.random.default_rng(
            derive_seed(MASTER_SEED, "calibration")))
        positive = target.sample(64, rng)
        matched = target.sample(64, rng)          # q = p, fresh draw
        init_cloud = torch.randn(
            64, 3, 16, 16,
            generator=torch.Generator().manual_seed(seed % (2 ** 31))) * 0.5

        trained = {}
        for arm_id in ("A1", "A3", "A4"):
            outcome = train_arm(arm_by_id(arm_id), target, _cfg(),
                                MASTER_SEED, log_every=10_000)
            latent = sample_latent(64, _cfg().latent_dim,
                                   derive_seed(seed, "latent"))
            with torch.no_grad():
                trained[arm_id] = outcome.model(latent)

        for family_name, config in (
                ("raw", GeometryConfig(family="raw")),
                ("wavelet", GeometryConfig(family="wavelet"))):
            family = build_family(config, 3)
            for branch in family.branches:
                kernel = calibrate_block_kernel(
                    branch, calibration, "smooth_laplace",
                    config.bandwidth_quantile, config.bandwidth_multiplier,
                    config.kernel_eps, combine=config.combine,
                    target_ess_fraction=config.target_ess_fraction)
                clouds = {"matched_q_eq_p": matched, "init_noise": init_cloud}
                clouds.update({f"trained_{k}": v for k, v in trained.items()})
                for cloud_name, cloud in clouds.items():
                    _, stats = KG.field(
                        cloud, positive, cloud, branch, kernel,
                        direction_mode="kernel_gradient",
                        normalization="none")
                    rows.append({
                        "target": name, "family": family_name,
                        "branch": branch.name, "cloud": cloud_name,
                        "raw_drift_rms": stats["drift_rms_raw"],
                        "ess_fraction": stats["ess_fraction"],
                        "affinity_median": stats["affinity_median"],
                        "distance_median": stats["distance_median"],
                        "bandwidth_median": stats["bandwidth_median"],
                    })
    return {"rows": rows}


# ---------------------------------------------------------------------------
# D3: was the Phase-0 kernel-health probe representative?
# ---------------------------------------------------------------------------


def d3_probe_representativeness(target_names=DIAG_TARGETS) -> dict:
    """Phase-0 G0.3 measured ESS on `target*0.4 + noise`.

    Compare that probe with the clouds a real run actually visits.  If the
    probe is much closer to the target than the real generator output, the
    Phase-0 gate certified a bandwidth that goes flat in training.
    """
    rows = []
    for name in target_names:
        target = D.named(name)
        seed = derive_seed(MASTER_SEED, "d3", name)
        rng = np.random.default_rng(seed)
        calibration = target.sample(256, np.random.default_rng(
            derive_seed(MASTER_SEED, "calibration")))
        positive = target.sample(64, rng)
        phase0_probe = target.sample(64, rng) * 0.4 + torch.tensor(
            rng.normal(scale=0.4, size=(64, 3, 16, 16)), dtype=torch.float32)

        outcome = train_arm(arm_by_id("A4"), target, _cfg(), MASTER_SEED,
                            log_every=10_000)
        latent = sample_latent(64, _cfg().latent_dim,
                               derive_seed(seed, "latent"))
        with torch.no_grad():
            trained = outcome.model(latent)
        untrained = train_arm(arm_by_id("A4"), target, _cfg(steps=1),
                              MASTER_SEED, log_every=10_000)
        with torch.no_grad():
            at_init = untrained.model(latent)

        config = GeometryConfig(family="wavelet")
        family = build_family(config, 3)
        for branch in family.branches:
            kernel = calibrate_block_kernel(
                branch, calibration, "smooth_laplace",
                config.bandwidth_quantile, config.bandwidth_multiplier,
                config.kernel_eps, combine=config.combine,
                target_ess_fraction=config.target_ess_fraction)
            for cloud_name, cloud in (
                    ("target_vs_target", target.sample(64, rng)),
                    ("phase0_probe", phase0_probe),
                    ("A4_at_init", at_init),
                    ("A4_trained", trained)):
                _, stats = KG.field(
                    cloud, positive, cloud, branch, kernel,
                    direction_mode="kernel_gradient", normalization="none")
                rows.append({
                    "target": name, "branch": branch.name,
                    "cloud": cloud_name,
                    "ess_fraction": stats["ess_fraction"],
                    "distance_median": stats["distance_median"],
                    "bandwidth_median": stats["bandwidth_median"],
                    "distance_over_bandwidth": (
                        stats["distance_median"]
                        / max(stats["bandwidth_median"], 1e-12)),
                    "affinity_median": stats["affinity_median"],
                })
    return {"rows": rows}


# ---------------------------------------------------------------------------
# D4: where does each direction rule put its output?
# ---------------------------------------------------------------------------


def d4_output_locality(target_names=DIAG_TARGETS) -> dict:
    """Standard displacement moves toward a convex combination of TARGET
    samples; the kernel gradient has no such constraint.

    Measures how far each arm's output sits from the nearest real target
    sample, in units of the target's own nearest-neighbour scale.
    """
    rows = []
    for name in target_names:
        target = D.named(name)
        seed = derive_seed(MASTER_SEED, "d4", name)
        pools = evaluation_pools(target, _cfg(), MASTER_SEED)
        reference = pools["eval"].reshape(len(pools["eval"]), -1)
        null = pools["null"].reshape(len(pools["null"]), -1)
        target_nn = float(torch.cdist(null, reference).min(dim=1).values
                          .median())
        for arm_id in ("A0", "A1", "A3", "A4", "A5"):
            outcome = train_arm(arm_by_id(arm_id), target, _cfg(),
                                MASTER_SEED, log_every=10_000)
            latent = sample_latent(256, _cfg().latent_dim,
                                   derive_seed(seed, "latent"))
            with torch.no_grad():
                generated = outcome.model(latent).reshape(256, -1)
            nearest = torch.cdist(generated, reference).min(dim=1).values
            rows.append({
                "target": name, "arm": arm_id,
                "target_nn_scale": target_nn,
                "median_nearest_real": float(nearest.median()),
                "nearest_in_target_nn_units": float(
                    nearest.median() / max(target_nn, 1e-12)),
                "output_rms": float(generated.pow(2).mean().sqrt()),
                "target_rms": float(reference.pow(2).mean().sqrt()),
            })
    return {"rows": rows}


# ---------------------------------------------------------------------------
# D5 / D6: candidate repairs
# ---------------------------------------------------------------------------


def _variant(arm_id: str, **geometry_overrides) -> ArmConfig:
    base = arm_by_id(arm_id)
    geometry = base.geometry
    if geometry is not None and geometry_overrides:
        geometry = GeometryConfig(**{
            **{f.name: getattr(geometry, f.name)
               for f in geometry.__dataclass_fields__.values()},
            **geometry_overrides})
    return ArmConfig(base.arm_id, base.use_anchor, geometry, base.field,
                     base.mixture, base.objective, base.anchor, base.note)


def d5_bandwidth_repair(target_names=DIAG_TARGETS) -> dict:
    """Is the failure a bandwidth calibrated on the wrong distance scale?

    The declared calibration solves for median ESS = 0.5 on TARGET-vs-TARGET
    data.  Training spends its time much further out.  This sweeps the
    declared selectivity and a plain bandwidth multiplier.
    """
    rows = []
    for name in target_names:
        target = D.named(name)
        pools = evaluation_pools(target, _cfg(), MASTER_SEED)
        null = null_reference(target, pools, MASTER_SEED)
        for label, overrides in (
                ("declared_ess0.5", {}),
                ("ess0.8", {"target_ess_fraction": 0.8}),
                ("ess0.95", {"target_ess_fraction": 0.95}),
                ("median_heuristic", {"target_ess_fraction": None}),
                ("median_x4", {"target_ess_fraction": None,
                               "bandwidth_multiplier": 4.0}),
                ("median_x16", {"target_ess_fraction": None,
                                "bandwidth_multiplier": 16.0}),
                ("median_x64", {"target_ess_fraction": None,
                                "bandwidth_multiplier": 64.0})):
            arm = _variant("A4", **overrides)
            outcome = train_arm(arm, target, _cfg(), MASTER_SEED,
                                log_every=10_000)
            row = evaluate_arm(outcome, target, _cfg(), pools, null,
                               MASTER_SEED)
            rows.append({
                "target": name, "variant": label,
                "geometry_score": row.get("geometry_score"),
                "ed2": row.get("ed2"), "coverage": row.get("coverage"),
                "precision": row.get("precision"),
                "median_ess": row.get(
                    "median_branch_wavelet_s0_ess_fraction"),
                "median_raw_drift": row.get(
                    "median_branch_wavelet_s0_drift_rms_raw"),
            })
            print(f"    D5 {name:15} {label:18} "
                  f"score={row.get('geometry_score', float('nan')):8.3f} "
                  f"cover={row.get('coverage', float('nan')):5.3f}",
                  flush=True)
    return {"rows": rows}


def d6_normalization_repair(target_names=DIAG_TARGETS) -> dict:
    """Does an unnormalized field -- one that can actually decay -- help?

    With ``normalization="none"`` the loss is a genuine function of the
    field, so the objective can descend and the exact-zero argument stops
    being vacuous.  Run for the structured and the raw kernel alike, since
    the raw arms share the defect.
    """
    rows = []
    for name in target_names:
        target = D.named(name)
        pools = evaluation_pools(target, _cfg(), MASTER_SEED)
        null = null_reference(target, pools, MASTER_SEED)
        for arm_id in ("A1", "A3", "A4"):
            base = arm_by_id(arm_id)
            for norm in ("rms", "none"):
                for eta in (0.5, 2.0):
                    if norm == "rms" and eta != 0.5:
                        continue
                    arm = ArmConfig(
                        base.arm_id, base.use_anchor, base.geometry,
                        FieldConfig(direction_mode=base.field.direction_mode,
                                    normalization=norm),
                        base.mixture,
                        ObjectiveConfig(
                            lambda_anchor=base.objective.lambda_anchor,
                            lambda_geometry=base.objective.lambda_geometry,
                            step_eta=eta),
                        base.anchor, base.note)
                    outcome = train_arm(arm, target, _cfg(), MASTER_SEED,
                                        log_every=10_000)
                    row = evaluate_arm(outcome, target, _cfg(), pools, null,
                                       MASTER_SEED)
                    rows.append({
                        "target": name, "arm": arm_id,
                        "normalization": norm, "eta": eta,
                        "geometry_score": row.get("geometry_score"),
                        "ed2": row.get("ed2"),
                        "coverage": row.get("coverage"),
                        "precision": row.get("precision"),
                        "final_loss": row.get("final_loss_loss_total"),
                        "median_loss": row.get("median_loss_loss_total"),
                    })
                    print(f"    D6 {name:15} {arm_id} {norm:5} eta={eta:4} "
                          f"score={row.get('geometry_score', float('nan')):8.3f}"
                          f" cover={row.get('coverage', float('nan')):5.3f}",
                          flush=True)
    return {"rows": rows}


def _sliced_w2_loss(generated: torch.Tensor, target: torch.Tensor,
                    directions: torch.Tensor) -> torch.Tensor:
    """Differentiable sliced Wasserstein-2 between two equal-size batches."""
    gp = torch.sort(generated.reshape(len(generated), -1) @ directions.T,
                    dim=0).values
    tp = torch.sort(target.reshape(len(target), -1) @ directions.T,
                    dim=0).values
    return ((gp - tp) ** 2).mean()


def d7_capacity_oracle(target_names=DIAG_TARGETS) -> dict:
    """Is the testbed solvable at all, by anything, at this budget?

    The Phase-1 screen compared nine arms none of which reached the target's
    own support level (best median precision 0.199 against a null of 0.98).
    That makes it a comparison between degrees of failure, which is only
    interpretable if *some* objective can succeed with this generator, this
    budget and these targets.

    Sliced Wasserstein is the control precisely because it shares none of
    the suspected defects: no kernel, no bandwidth, no feature map, no field
    normalization.  It is not an arm and says nothing about encoder-free
    drifting; it calibrates the testbed.
    """
    rows = []
    for name in target_names:
        target = D.named(name)
        for steps in (300, 1200):
            config = _cfg(steps)
            pools = evaluation_pools(target, config, MASTER_SEED)
            null = null_reference(target, pools, MASTER_SEED)
            model = OneStepGenerator(
                config.latent_dim, config.channels, config.image_size,
                config.width, derive_seed(MASTER_SEED, "generator"))
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config.learning_rate)
            rng = np.random.default_rng(derive_seed(MASTER_SEED, "d7", name))
            gen = torch.Generator().manual_seed(
                derive_seed(MASTER_SEED, "d7-latent", name) % (2 ** 31))
            dim = config.channels * config.image_size ** 2
            for _ in range(steps):
                batch = target.sample(config.batch, rng)
                latent = torch.randn(config.batch, config.latent_dim,
                                     generator=gen)
                directions = torch.randn(64, dim, generator=gen)
                directions = directions / directions.norm(dim=1, keepdim=True)
                loss = _sliced_w2_loss(model(latent), batch, directions)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            latent = sample_latent(config.eval_samples, config.latent_dim,
                                   derive_seed(MASTER_SEED, "eval-latent"))
            with torch.no_grad():
                generated = model(latent)
            metrics = M.raw_metrics(
                generated, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(MASTER_SEED, "d7-metrics")),
                target)
            score = M.normalized_geometry_score(metrics, null)
            rows.append({
                "target": name, "objective": "sliced_wasserstein",
                "steps": steps,
                "geometry_score": score["geometry_score"],
                "ed2": metrics["ed2"], "coverage": metrics["coverage"],
                "precision": metrics["precision"],
                "output_rms": float(generated.pow(2).mean().sqrt()),
                "target_rms": float(pools["eval"].pow(2).mean().sqrt()),
            })
            print(f"    D7 {name:15} SW steps={steps:5} "
                  f"score={score['geometry_score']:8.3f} "
                  f"cover={metrics['coverage']:5.3f} "
                  f"prec={metrics['precision']:5.3f}", flush=True)
    return {"rows": rows}


def d8_output_character(target_names=DIAG_TARGETS) -> dict:
    """Shrinkage and diversity of each arm's output against the target.

    Separates three failures that all read as "bad score": collapse (low
    diversity), shrinkage toward the mean (low RMS), and off-manifold spray
    (high diversity, low precision).
    """
    rows = []
    config = _cfg()
    for name in target_names:
        target = D.named(name)
        pools = evaluation_pools(target, config, MASTER_SEED)
        reference = pools["eval"].reshape(len(pools["eval"]), -1)
        ref_pair = float(torch.cdist(reference[:128], reference[128:256])
                         .median())
        ref_rms = float(reference.pow(2).mean().sqrt())
        for arm_id in ("A0", "A1", "A3", "A4", "A5", "A7"):
            outcome = train_arm(arm_by_id(arm_id), target, config,
                                MASTER_SEED, log_every=10_000)
            latent = sample_latent(256, config.latent_dim,
                                   derive_seed(MASTER_SEED, "eval-latent"))
            with torch.no_grad():
                generated = outcome.model(latent).reshape(256, -1)
            pair = float(torch.cdist(generated[:128], generated[128:])
                         .median())
            rms = float(generated.pow(2).mean().sqrt())
            rows.append({
                "target": name, "arm": arm_id,
                "output_rms": rms, "target_rms": ref_rms,
                "rms_ratio": rms / max(ref_rms, 1e-12),
                "generated_pair_distance": pair,
                "target_pair_distance": ref_pair,
                "diversity_ratio": pair / max(ref_pair, 1e-12),
            })
            print(f"    D8 {name:15} {arm_id} rms_ratio="
                  f"{rows[-1]['rms_ratio']:5.3f} diversity_ratio="
                  f"{rows[-1]['diversity_ratio']:5.3f}", flush=True)
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase1_diagnosis.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()

    stages = {
        "D1_loss_constancy": d1_loss_constancy,
        "D2_field_floor": d2_field_floor,
        "D3_probe_representativeness": d3_probe_representativeness,
        "D4_output_locality": d4_output_locality,
        "D5_bandwidth_repair": d5_bandwidth_repair,
        "D6_normalization_repair": d6_normalization_repair,
        "D7_capacity_oracle": d7_capacity_oracle,
        "D8_output_character": d8_output_character,
    }
    wanted = (set(stages) if args.only == "all"
              else set(args.only.split(",")))
    results = {}
    for name, function in stages.items():
        if name not in wanted:
            continue
        print(f"--- {name} ---", flush=True)
        results[name] = function()

    payload = {
        "status": "phase1-post-hoc-diagnosis-feeds-no-gate",
        "provenance": provenance(),
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    digest = write_json(args.out, payload)
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
