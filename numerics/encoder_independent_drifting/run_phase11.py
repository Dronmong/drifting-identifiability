"""Phase 11 (protocol `EncoderIndependentPhase11Protocol.md`).

Whiten the regression metric.

The tail-destruction pass showed the target is fine and the metric is the
problem: the field carries 16x more tail than the generator's cloud has, the
generator STARTS with a healthy tail (0.221, against real data's 0.164), and
training destroys it ~29x within the first 100 steps.  ``|f - T|^2`` weights
each direction by its own variance, so trailing directions barely enter the
loss and an imperfect fit drops them first.

Every intervention this program has tried acts on the TARGET.  This one acts
on the metric, and it is the first derived from a measured cause.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase11
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
    GeometryConfig, MASTER_SEED, TrainConfig, config_digest, derive_seed,
)
from .diagnostics import provenance, write_json
from .evaluate import evaluation_pools, null_reference
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher
from .train import optimizer_report

HERE = Path(__file__).resolve().parent

# Frozen (protocol section 2): disjoint from every earlier phase.
SEED_OFFSET = 20000
MOMENT_BAND = (0.7, 1.3)
SUPERSEDE_TOLERANCE = 1.25
GOOD_ESS = 0.9
FIELD_CLOUD = 256
TAIL_KEEP = 32
RIDGE_FLOOR = 1e-3          # declared; keeps the unspanned directions bounded

# (label, gamma, R11)
ARMS = (("W0_gamma=0", 0.0, False),
        ("W1_gamma=0.5", 0.5, False),
        ("W2_gamma=0.9", 0.9, False),
        ("W3_gamma=0.99", 0.99, False),
        ("W2R_gamma=0.9+R11", 0.9, True),
        ("E1_r11", 0.0, True))


def _setup(resolution: int, seed: int, root: str | None):
    train = cifar.cifar_target(resolution, "train", root)
    rng = np.random.default_rng(derive_seed(seed, "p11-setup"))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace",
                              target_ess_fraction=GOOD_ESS)
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace",
        geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
        geometry.kernel_eps, combine=geometry.combine,
        target_ess_fraction=GOOD_ESS)
    return train, branch, kernel, rng


def _tail(sample: torch.Tensor) -> float:
    flat = sample.reshape(len(sample), -1)
    power = torch.linalg.svdvals(flat - flat.mean(dim=0, keepdim=True)) ** 2
    return float(power[TAIL_KEEP:].sum() / power.sum())


def whitened_loss(output: torch.Tensor, teacher: torch.Tensor,
                  gamma: float, report: dict | None = None) -> torch.Tensor:
    """``|Sigma_gamma^(-1/2) (f - T)|^2`` with a DETACHED metric.

    Applied through the SVD of the centred batch, so no ``d x d`` matrix is
    formed.  The covariance has rank <= n-1 in d >> n dimensions, so the
    ~512 directions the batch does not span are handled by the declared
    ridge floor rather than being amplified without bound.

    ``gamma = 0`` reproduces the ordinary loss up to a constant factor, which
    Adam is invariant to (R24) -- so W0 is the current recipe exactly.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    n = len(output)
    residual = (output - teacher).reshape(n, -1)
    with torch.no_grad():
        flat = output.detach().reshape(n, -1)
        centred = flat - flat.mean(dim=0, keepdim=True)
        _, singular, right = torch.linalg.svd(centred, full_matrices=False)
        variance = singular ** 2 / n
        mean_variance = float(centred.pow(2).sum() / (n * flat.shape[1]))
        ridge = max((1.0 - gamma) * mean_variance,
                    RIDGE_FLOOR * mean_variance)
        inside = (gamma * variance + ridge).clamp_min(1e-30) ** -0.5
        outside = max(ridge, 1e-30) ** -0.5
        if report is not None:
            report["metric_ridge"] = ridge
            report["metric_scale_min"] = float(inside.min())
            report["metric_scale_max"] = max(float(inside.max()), outside)
    coefficients = residual @ right.T
    parallel = (coefficients * inside) @ right
    perpendicular = residual - coefficients @ right
    whitened = parallel + outside * perpendicular
    return whitened.pow(2).sum(dim=1).mean()


def run_arm(train, branch, kernel, rng, config, seed, *, gamma: float,
            correction: bool, steps: int, eta: float = 0.5,
            trace_every: int = 50) -> dict:
    model = OneStepGenerator(config.latent_dim, 3, config.image_size,
                             config.width, derive_seed(seed, "generator"))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    torch_rng = torch.Generator().manual_seed(
        derive_seed(seed, "p11-latent") % (2 ** 31))
    trace, report = [], {}
    for step in range(steps):
        latent = torch.randn(FIELD_CLOUD, config.latent_dim,
                             generator=torch_rng)
        output = model(latent)
        if step % trace_every == 0:
            trace.append({"step": step, "tail": _tail(output.detach())})
        with torch.no_grad():
            positives = train.sample(config.batch, rng)
            drift, _ = KG.field(output.detach(), positives, output.detach(),
                                branch, kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            teacher = output.detach() + eta * drift
            if correction:
                teacher = corrected_teacher(teacher, positives,
                                            mode="scalar")
        loss = whitened_loss(output, teacher, gamma, report)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {"model": model, "trace": trace, "metric": report}


def stage_11(resolution: int, seeds: int, steps: int,
             root: str | None) -> dict:
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        config = TrainConfig(steps=steps, batch=64, field_cloud=FIELD_CLOUD,
                             eval_samples=512, image_size=resolution)
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())
        reference_tail = _tail(pools["eval"])
        probe = sample_latent(config.eval_samples, config.latent_dim,
                              derive_seed(seed, "p11-probe"))
        for label, gamma, correction in ARMS:
            train, branch, kernel, rng = _setup(resolution, seed, root)
            result = run_arm(train, branch, kernel, rng, config, seed,
                             gamma=gamma, correction=correction, steps=steps)
            with torch.no_grad():
                generated = result["model"](probe)
            measured = M.raw_metrics(
                generated, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p11-m")), None,
                target_null=pools["null"])
            trace = result["trace"]
            row = {"arm": label, "gamma": gamma, "r11": correction,
                   "seed": seed,
                   "second_moment_ratio": float(
                       generated.flatten(1).var(0).mean()) / reference_moment,
                   "ed2": measured["ed2"],
                   "geometry_score_v2": M.normalized_geometry_score_v2(
                       measured, null)["geometry_score"],
                   "tail_final": _tail(generated),
                   "tail_at_init": trace[0]["tail"],
                   "tail_at_100": next((t["tail"] for t in trace
                                        if t["step"] >= 100), float("nan")),
                   "reference_tail": reference_tail,
                   "trace": trace, **result["metric"]}
            rows.append(row)
            print(f"    {label:20} seed{index} "
                  f"tail {row['tail_at_init']:.4f} -> "
                  f"{row['tail_at_100']:.4f} -> {row['tail_final']:.4f}  "
                  f"2nd={row['second_moment_ratio']:6.3f} "
                  f"ed2={row['ed2']:8.4f}", flush=True)
    summary = {}
    for label, gamma, correction in ARMS:
        group = [r for r in rows if r["arm"] == label]
        moment = float(np.median([r["second_moment_ratio"] for r in group]))
        summary[label] = {
            "gamma": gamma, "r11": correction,
            "median_second_moment_ratio": moment,
            "median_ed2": float(np.median([r["ed2"] for r in group])),
            "median_score_v2": float(np.median(
                [r["geometry_score_v2"] for r in group])),
            "median_tail_at_init": float(np.median(
                [r["tail_at_init"] for r in group])),
            "median_tail_at_100": float(np.median(
                [r["tail_at_100"] for r in group])),
            "median_tail_final": float(np.median(
                [r["tail_final"] for r in group])),
            "median_metric_scale_ratio": float(np.median(
                [r["metric_scale_max"] / max(r["metric_scale_min"], 1e-30)
                 for r in group])),
            "moment_in_band": bool(MOMENT_BAND[0] <= moment
                                   <= MOMENT_BAND[1])}
    return {"rows": [{k: v for k, v in r.items() if k != "trace"}
                     for r in rows],
            "traces": {r["arm"]: r["trace"] for r in rows},
            "summary": summary,
            "reference_tail": rows[0]["reference_tail"]}


def gate_11(summary: dict) -> dict:
    """Can the metric fix supersede R11?  Declared before the run."""
    r11 = [v for k, v in summary.items() if v["r11"]]
    whitened = {k: v for k, v in summary.items()
                if not v["r11"] and v["gamma"] > 0}
    if not r11 or not whitened:
        return {"passed": False, "reason": "missing arms"}
    best_r11 = min(r11, key=lambda v: v["median_ed2"])
    ceiling = best_r11["median_ed2"] * SUPERSEDE_TOLERANCE
    qualifying = [k for k, v in whitened.items()
                  if v["moment_in_band"] and v["median_ed2"] <= ceiling]
    baseline = summary.get("W0_gamma=0", {})
    tail_moved = {
        k: v["median_tail_at_100"] / max(
            baseline.get("median_tail_at_100", 1e-12), 1e-12)
        for k, v in whitened.items()}
    return {
        "passed": bool(qualifying),
        "meaning": ("the metric fix supersedes R11" if qualifying else
                    "no whitened arm reproduces R11"),
        "ed2_ceiling": ceiling,
        "qualifying_arms": qualifying,
        "tail_at_100_vs_baseline": tail_moved,
        "prediction_1_tail_did_not_collapse": bool(
            max(tail_moved.values(), default=0.0) > 2.0),
        "note": "if the tail rose but the second moment did not follow, the "
                "Phase-10 law's causal reading is refuted, not the metric",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase11.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    print("=== Phase 11: whiten the regression metric ===", flush=True)
    stage = stage_11(args.resolution, args.seeds, args.steps, args.data_root)
    gate = gate_11(stage["summary"])

    payload = {
        "status": "phase11-frozen-protocol",
        "protocol": "numerics/EncoderIndependentPhase11Protocol.md",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "config_digest": config_digest(
            TrainConfig(steps=args.steps, image_size=args.resolution)),
        "optimizer": optimizer_report(TrainConfig()),
        "frozen_thresholds": {
            "moment_band": list(MOMENT_BAND),
            "supersede_tolerance": SUPERSEDE_TOLERANCE,
            "gammas": [a[1] for a in ARMS], "ridge_floor": RIDGE_FLOOR,
            "target_ess": GOOD_ESS, "field_cloud": FIELD_CLOUD},
        "elapsed_seconds": time.time() - started,
        "summary": stage["summary"], "gate": gate,
        "reference_tail": stage["reference_tail"],
        "rows": stage["rows"], "traces": stage["traces"],
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 11 ===")
    print(f"{'arm':22}{'tail@init':>11}{'tail@100':>10}{'tail_end':>10}"
          f"{'2nd_mom':>9}{'ed2':>9}{'score':>8}  band")
    for key, entry in payload["summary"].items():
        print(f"{key:22}{entry['median_tail_at_init']:11.4f}"
              f"{entry['median_tail_at_100']:10.4f}"
              f"{entry['median_tail_final']:10.4f}"
              f"{entry['median_second_moment_ratio']:9.3f}"
              f"{entry['median_ed2']:9.4f}{entry['median_score_v2']:8.3f}"
              f"  {'in ' if entry['moment_in_band'] else 'out'}")
    print(f"    (real data tail = {payload['reference_tail']:.4f})")
    print("\n  tail at step 100, relative to the unwhitened baseline:")
    for arm, ratio in gate["tail_at_100_vs_baseline"].items():
        print(f"    {arm:22} x{ratio:7.2f}")
    print(f"\n  prediction 1 (tail did not collapse): "
          f"{gate['prediction_1_tail_did_not_collapse']}")
    print(f"  gate passed={gate['passed']}  -- {gate['meaning']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
