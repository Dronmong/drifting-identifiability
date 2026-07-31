"""Investigating Phase 12: is it the external target, or the fixed latents?

Phase 12 found that A2 -- an external particle target with index pairing --
reaches tail 0.3243 (x79 over the self-referential baseline), second moment
0.922 (in band) and ED2 0.2091 (6.2x better), while both amortizing
assignments failed.  The result was attributed to removing self-reference.

**A2 differs from A0 in two ways, not one**: the target is external AND the
latents are fixed.  The attribution is only sound if fixed latents alone do
not produce the effect, and that control was never run.

  I1  the confound.  Cross {self-referential, external} x {fresh, fixed}
      latents.  If fixed latents alone carry the effect, Phase 12's reading
      is wrong and the whole line changes.
  I2  the amortization gap.  A2's numbers are on fresh latents; how do they
      compare with its own training latents?  That difference is what an
      assignment would have to close.
  I3  diagnose the two failures directly rather than by inference:
      collision counts for the greedy match, and the *target's* own tail and
      second moment for the barycentric projection.
  I4  the proposed fix, prototyped: an exact balanced hard assignment
      (Hungarian on the batch), which is both non-colliding and
      non-averaging -- the two properties the failed schemes each lacked.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase13
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

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

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 24000
GOOD_ESS = 0.9
PARTICLES = 512
GENERATOR_BATCH = 256
POSITIVES = 64
TAIL_KEEP = 32
PARTICLE_ETA = 0.2
TEACHER_ETA = 0.5
SINKHORN_EPSILON = 0.05
SINKHORN_ITERS = 30


def _setup(resolution: int, seed: int, root: str | None):
    train = cifar.cifar_target(resolution, "train", root)
    rng = np.random.default_rng(derive_seed(seed, "p13-setup"))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace",
                              target_ess_fraction=GOOD_ESS)
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace",
        geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
        geometry.kernel_eps, combine=geometry.combine,
        target_ess_fraction=GOOD_ESS)
    return train, branch, kernel, rng


def _tail(x: torch.Tensor) -> float:
    flat = x.reshape(len(x), -1)
    power = torch.linalg.svdvals(flat - flat.mean(dim=0, keepdim=True)) ** 2
    return float(power[TAIL_KEEP:].sum() / power.sum())


def _sinkhorn_plan(output: torch.Tensor, particles: torch.Tensor):
    flat = output.reshape(len(output), -1)
    flat_particles = particles.reshape(len(particles), -1)
    cost = torch.cdist(flat, flat_particles) ** 2
    scale = cost.median().clamp_min(1e-12)
    log_kernel = -cost / (SINKHORN_EPSILON * scale)
    n, m = cost.shape
    log_u, log_v = torch.zeros(n), torch.zeros(m)
    log_a = torch.full((n,), -float(np.log(n)))
    log_b = torch.full((m,), -float(np.log(m)))
    for _ in range(SINKHORN_ITERS):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    return torch.exp(log_kernel + log_u[:, None] + log_v[None, :]), cost


def make_target(mode: str, output: torch.Tensor, particles: torch.Tensor,
                report: dict | None = None) -> torch.Tensor:
    """The target for one arm, plus diagnostics on the assignment itself."""
    flat_particles = particles.reshape(len(particles), -1)
    if mode == "index":
        index = torch.arange(len(output)) % len(particles)
    elif mode == "nearest":
        flat = output.reshape(len(output), -1)
        index = torch.cdist(flat, flat_particles).argmin(dim=1)
    elif mode == "hungarian":
        # Exact balanced hard assignment: every generated sample gets a
        # DISTINCT particle, and no averaging is performed.  These are the
        # two properties the greedy match and the barycentric projection
        # each lacked.
        flat = output.reshape(len(output), -1)
        cost = torch.cdist(flat, flat_particles).numpy()
        rows, columns = linear_sum_assignment(cost)
        index = torch.zeros(len(output), dtype=torch.long)
        index[torch.as_tensor(rows)] = torch.as_tensor(columns)
    elif mode == "sinkhorn":
        plan, _ = _sinkhorn_plan(output, particles)
        weight = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-30)
        target = (weight @ flat_particles).reshape(
            (len(output),) + particles.shape[1:])
        if report is not None:
            report["target_tail"] = _tail(target)
            report["target_second_moment"] = float(
                target.flatten(1).var(0).mean()
                / particles.flatten(1).var(0).mean())
            report["distinct_particles"] = float("nan")
        return target
    else:
        raise ValueError(f"unknown assignment {mode!r}")
    if report is not None:
        report["distinct_particles"] = float(len(torch.unique(index)))
        chosen = particles[index]
        report["target_tail"] = _tail(chosen)
        report["target_second_moment"] = float(
            chosen.flatten(1).var(0).mean()
            / particles.flatten(1).var(0).mean())
    return particles[index]


# (label, target_mode, fixed_latents)
ARMS = (("B0_self_fresh", "self", False),
        ("B1_self_FIXED", "self", True),          # the missing control
        ("B2_index_FIXED", "index", True),        # Phase 12's A2
        ("B3_index_fresh", "index", False),
        ("B4_hungarian_fresh", "hungarian", False),
        ("B5_hungarian_FIXED", "hungarian", True),
        ("B6_nearest_fresh", "nearest", False),
        ("B7_sinkhorn_fresh", "sinkhorn", False))


def investigate(resolution: int, seeds: int, steps: int,
                root: str | None) -> dict:
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        train, branch, kernel, rng = _setup(resolution, seed, root)
        config = TrainConfig(steps=steps, batch=POSITIVES,
                             eval_samples=PARTICLES, image_size=resolution)
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())

        particles = torch.tensor(
            rng.normal(scale=0.5,
                       size=(PARTICLES, 3, resolution, resolution)),
            dtype=torch.float32)
        for _ in range(steps):
            drift, _ = KG.field(particles, train.sample(POSITIVES, rng),
                                particles, branch, kernel,
                                direction_mode="paper", normalization="rms",
                                diagnostics=False)
            particles = particles + PARTICLE_ETA * drift
        particle_tail = _tail(particles)
        particle_ed2 = M.raw_metrics(
            particles, pools["eval"], pools["cal_a"], pools["cal_b"],
            np.random.default_rng(derive_seed(seed, "p13-p")), None,
            target_null=pools["null"])["ed2"]

        fixed_latent = sample_latent(GENERATOR_BATCH, config.latent_dim,
                                     derive_seed(seed, "p13-fixed"))
        probe = sample_latent(PARTICLES, config.latent_dim,
                              derive_seed(seed, "p13-probe"))

        for label, mode, fixed in ARMS:
            model = OneStepGenerator(config.latent_dim, 3, resolution,
                                     config.width,
                                     derive_seed(seed, "generator"))
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config.learning_rate)
            torch_rng = torch.Generator().manual_seed(
                derive_seed(seed, "p13-latent") % (2 ** 31))
            report: dict = {}
            for _ in range(steps):
                latent = (fixed_latent if fixed
                          else torch.randn(GENERATOR_BATCH,
                                           config.latent_dim,
                                           generator=torch_rng))
                output = model(latent)
                with torch.no_grad():
                    if mode == "self":
                        own, _ = KG.field(
                            output.detach(), train.sample(POSITIVES, rng),
                            output.detach(), branch, kernel,
                            direction_mode="paper", normalization="rms",
                            diagnostics=False)
                        target = output.detach() + TEACHER_ETA * own
                    else:
                        target = make_target(mode, output.detach(),
                                             particles, report)
                loss = ((output - target) ** 2).flatten(1).sum(1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                fresh = model(probe)
                trained = model(fixed_latent)
            fresh_metrics = M.raw_metrics(
                fresh, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p13-m")), None,
                target_null=pools["null"])
            trained_metrics = M.raw_metrics(
                trained, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p13-m")), None,
                target_null=pools["null"])
            row = {"arm": label, "mode": mode, "fixed_latents": fixed,
                   "seed": seed,
                   "tail_fresh": _tail(fresh),
                   "second_moment_fresh": float(
                       fresh.flatten(1).var(0).mean()) / reference_moment,
                   "ed2_fresh": fresh_metrics["ed2"],
                   "score_fresh": M.normalized_geometry_score_v2(
                       fresh_metrics, null)["geometry_score"],
                   "ed2_on_training_latents": trained_metrics["ed2"],
                   "particle_tail": particle_tail,
                   "particle_ed2": particle_ed2,
                   **report}
            row["amortization_gap"] = (row["ed2_fresh"]
                                       / max(row["ed2_on_training_latents"],
                                             1e-12))
            rows.append(row)
            print(f"    {label:22} seed{index} "
                  f"tail={row['tail_fresh']:7.4f} "
                  f"2nd={row['second_moment_fresh']:6.3f} "
                  f"ed2={row['ed2_fresh']:8.4f} "
                  f"(train {row['ed2_on_training_latents']:7.4f}) "
                  f"distinct={report.get('distinct_particles', float('nan')):6.1f}",
                  flush=True)
        print(f"    (seed{index} particle cloud: tail={particle_tail:.4f} "
              f"ed2={particle_ed2:.4f})", flush=True)

    summary = {}
    for label, mode, fixed in ARMS:
        group = [r for r in rows if r["arm"] == label]
        summary[label] = {
            "mode": mode, "fixed_latents": fixed,
            **{k: float(np.median([r[k] for r in group]))
               for k in ("tail_fresh", "second_moment_fresh", "ed2_fresh",
                         "score_fresh", "ed2_on_training_latents",
                         "amortization_gap")},
            "median_distinct_particles": float(np.median(
                [r.get("distinct_particles", np.nan) for r in group])),
            "median_target_tail": float(np.median(
                [r.get("target_tail", np.nan) for r in group])),
            "median_target_second_moment": float(np.median(
                [r.get("target_second_moment", np.nan) for r in group])),
        }
    return {"rows": rows, "summary": summary,
            "particle_tail": float(np.median([r["particle_tail"]
                                              for r in rows])),
            "particle_ed2": float(np.median([r["particle_ed2"]
                                             for r in rows]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase13_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    print("=== Investigating Phase 12 ===", flush=True)
    stage = investigate(args.resolution, args.seeds, args.steps,
                        args.data_root)
    payload = {
        "status": "phase12-investigation-feeds-no-gate",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "question": "is Phase 12's A2 effect due to the external target or "
                    "to the fixed latents, and does an exact balanced hard "
                    "assignment fix the amortizing arms?",
        "elapsed_seconds": time.time() - started,
        **stage,
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE-12 INVESTIGATION ===")
    print(f"{'arm':24}{'latents':>9}{'tail':>9}{'2nd':>8}{'ed2 fresh':>11}"
          f"{'ed2 train':>11}{'gap':>7}{'distinct':>10}")
    for key, entry in payload["summary"].items():
        print(f"{key:24}{'fixed' if entry['fixed_latents'] else 'fresh':>9}"
              f"{entry['tail_fresh']:9.4f}{entry['second_moment_fresh']:8.3f}"
              f"{entry['ed2_fresh']:11.4f}"
              f"{entry['ed2_on_training_latents']:11.4f}"
              f"{entry['amortization_gap']:7.2f}"
              f"{entry['median_distinct_particles']:10.1f}")
    print(f"\n  the particle cloud itself: tail={payload['particle_tail']:.4f}"
          f"  ed2={payload['particle_ed2']:.4f}")
    print("\n  assignment diagnostics (target's own shape):")
    for key, entry in payload["summary"].items():
        if np.isfinite(entry["median_target_tail"]):
            print(f"    {key:24} target tail="
                  f"{entry['median_target_tail']:7.4f}  target 2nd="
                  f"{entry['median_target_second_moment']:6.3f}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
