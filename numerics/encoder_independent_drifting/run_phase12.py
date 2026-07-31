"""Phase 12 (protocol `EncoderIndependentPhase12Protocol.md`).

Replace the self-referential teacher with an external one.

The audit established that the deficit is caused by self-reference, not by
the size or coherence of the teacher's demand: rollout K=16 (large AND
committed) leaves the tail at 0.0035, 16x averaging leaves it at 0.0035 and
hurts quality, and every failing arm has the form ``T = f + Delta``.  The two
arms whose target never references ``f`` reach tail 0.095-0.159 against
0.0035-0.0049, measured on fresh latents.

Here the generator chases an **evolving particle cloud** with a transport
assignment.  All arms in a seed share one particle trajectory -- advanced
once per step, not pre-converged -- so they differ only in what the generator
is asked to match, and the particle cost is paid once and reported.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase12
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
SEED_OFFSET = 23000
MOMENT_BAND = (0.7, 1.3)
SUPERSEDE_TOLERANCE = 1.25
GOOD_ESS = 0.9
PARTICLES = 512
GENERATOR_BATCH = 256
POSITIVES = 64
TAIL_KEEP = 32
PARTICLE_ETA = 0.2
TEACHER_ETA = 0.5

# Declared, not swept (protocol section 5).
SINKHORN_EPSILON = 0.05      # relative to the median squared cost
SINKHORN_ITERS = 30

# (label, mode, r11, fixed_latents, pre_converged)
#
# The two `*P` arms were added after a 60-step smoke test, for the reason
# recorded in the protocol's amendment: with the cloud evolving concurrently
# the generator starts far from the particles, where nearest-neighbour
# collapses onto a few of them and Sinkhorn's barycentric projection averages
# hard enough to contract the target (smoke second moment 0.005).  The
# particle cost is identical either way -- the same field evaluations, in a
# different order -- so the ledger is unaffected.
ARMS = (("A0_self_reference", "self", False, False, False),
        ("A1_self+R11", "self", True, False, False),
        ("A2_particles_index", "index", False, True, False),
        ("A3_particles_nearest", "nearest", False, False, False),
        ("A4_particles_sinkhorn", "sinkhorn", False, False, False),
        ("A4R_sinkhorn+R11", "sinkhorn", True, False, False),
        ("A3P_pre_nearest", "nearest", False, False, True),
        ("A4P_pre_sinkhorn", "sinkhorn", False, False, True))


def _setup(resolution: int, seed: int, root: str | None):
    train = cifar.cifar_target(resolution, "train", root)
    rng = np.random.default_rng(derive_seed(seed, "p12-setup"))
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


def nearest_target(output: torch.Tensor,
                   particles: torch.Tensor) -> torch.Tensor:
    """Each generated sample takes its nearest particle.

    Collisions are allowed -- several samples may claim one particle -- so
    this is a greedy match rather than a balanced assignment.  That is the
    honest cheap baseline; A4 is the balanced version.
    """
    flat = output.reshape(len(output), -1)
    flat_particles = particles.reshape(len(particles), -1)
    index = torch.cdist(flat, flat_particles).argmin(dim=1)
    return particles[index]


def sinkhorn_target(output: torch.Tensor, particles: torch.Tensor,
                    epsilon: float = SINKHORN_EPSILON,
                    iterations: int = SINKHORN_ITERS) -> torch.Tensor:
    """Entropic-OT barycentric projection of the batch onto the particles.

    A balanced plan, so every particle carries its share of the mass and no
    region of the cloud is over-claimed -- which a greedy nearest match
    cannot promise.  Run in the log domain for stability.
    """
    flat = output.reshape(len(output), -1)
    flat_particles = particles.reshape(len(particles), -1)
    cost = torch.cdist(flat, flat_particles) ** 2
    scale = cost.median().clamp_min(1e-12)
    log_kernel = -cost / (epsilon * scale)
    n, m = cost.shape
    log_u = torch.zeros(n)
    log_v = torch.zeros(m)
    log_a = torch.full((n,), -np.log(n))
    log_b = torch.full((m,), -np.log(m))
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    plan = torch.exp(log_kernel + log_u[:, None] + log_v[None, :])
    weight = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-30)
    return (weight @ flat_particles).reshape(
        (len(output),) + particles.shape[1:])


def stage_12(resolution: int, seeds: int, steps: int, root: str | None,
             trace_every: int = 50) -> dict:
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
        reference_tail = _tail(pools["eval"])

        # One particle trajectory, shared by every arm in this seed.
        particles = torch.tensor(
            rng.normal(scale=0.5,
                       size=(PARTICLES, 3, resolution, resolution)),
            dtype=torch.float32)

        # A second cloud, advanced to convergence BEFORE generator training
        # and then held fixed.  Identical number of field steps as the
        # concurrent cloud -- the same cost, differently ordered -- so the
        # ledger is unaffected.  See the protocol's amendment for why the
        # `*P` arms exist.
        pre_particles = particles.clone()
        for _ in range(steps):
            pre_drift, _ = KG.field(
                pre_particles, train.sample(POSITIVES, rng), pre_particles,
                branch, kernel, direction_mode="paper", normalization="rms",
                diagnostics=False)
            pre_particles = pre_particles + PARTICLE_ETA * pre_drift

        models, optimizers, generators, traces = {}, {}, {}, {}
        for label, _, _, _, _ in ARMS:
            models[label] = OneStepGenerator(
                config.latent_dim, 3, resolution, config.width,
                derive_seed(seed, "generator"))
            optimizers[label] = torch.optim.Adam(
                models[label].parameters(), lr=config.learning_rate)
            generators[label] = torch.Generator().manual_seed(
                derive_seed(seed, "p12-latent") % (2 ** 31))
            traces[label] = []
        fixed_latent = sample_latent(GENERATOR_BATCH, config.latent_dim,
                                     derive_seed(seed, "p12-fixed"))
        ledger = {label: {"field_evaluations": 0, "kernel_pairs": 0}
                  for label, _, _, _, _ in ARMS}
        particle_ledger = {"field_evaluations": 0, "kernel_pairs": 0}
        started = time.time()

        for step in range(steps):
            # Advance the shared particle cloud one field step.
            positives = train.sample(POSITIVES, rng)
            drift, _ = KG.field(particles, positives, particles, branch,
                                kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            particles = particles + PARTICLE_ETA * drift
            particle_ledger["field_evaluations"] += 1
            particle_ledger["kernel_pairs"] += PARTICLES * (POSITIVES
                                                            + PARTICLES)

            for label, mode, r11, fixed, pre in ARMS:
                model = models[label]
                cloud_for_arm = pre_particles if pre else particles
                latent = (fixed_latent if fixed
                          else torch.randn(GENERATOR_BATCH, config.latent_dim,
                                           generator=generators[label]))
                output = model(latent)
                if step % trace_every == 0:
                    traces[label].append({"step": step,
                                          "tail": _tail(output.detach())})
                with torch.no_grad():
                    if mode == "self":
                        own, _ = KG.field(output.detach(), positives,
                                          output.detach(), branch, kernel,
                                          direction_mode="paper",
                                          normalization="rms",
                                          diagnostics=False)
                        target = output.detach() + TEACHER_ETA * own
                        ledger[label]["field_evaluations"] += 1
                        ledger[label]["kernel_pairs"] += (
                            GENERATOR_BATCH * (POSITIVES + GENERATOR_BATCH))
                    elif mode == "index":
                        target = cloud_for_arm[:GENERATOR_BATCH]
                    elif mode == "nearest":
                        target = nearest_target(output.detach(), cloud_for_arm)
                    else:
                        target = sinkhorn_target(output.detach(), cloud_for_arm)
                    if r11:
                        target = corrected_teacher(target, positives,
                                                   mode="scalar")
                loss = ((output - target) ** 2).flatten(1).sum(1).mean()
                optimizers[label].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[label].step()

        wall = time.time() - started
        probe = sample_latent(PARTICLES, config.latent_dim,
                              derive_seed(seed, "p12-probe"))
        for label, mode, r11, fixed, pre in ARMS:
            with torch.no_grad():
                generated = models[label](probe)
            measured = M.raw_metrics(
                generated, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p12-m")), None,
                target_null=pools["null"])
            uses_particles = mode != "self"
            row = {"arm": label, "mode": mode, "r11": r11,
                   "fixed_latents": fixed, "seed": seed,
                   "second_moment_ratio": float(
                       generated.flatten(1).var(0).mean()) / reference_moment,
                   "ed2": measured["ed2"],
                   "geometry_score_v2": M.normalized_geometry_score_v2(
                       measured, null)["geometry_score"],
                   "tail_final": _tail(generated),
                   "tail_at_init": traces[label][0]["tail"],
                   "reference_tail": reference_tail,
                   "trace": traces[label],
                   "field_evaluations": (
                       ledger[label]["field_evaluations"]
                       + (particle_ledger["field_evaluations"]
                          if uses_particles else 0)),
                   "kernel_pairs": (
                       ledger[label]["kernel_pairs"]
                       + (particle_ledger["kernel_pairs"]
                          if uses_particles else 0))}
            rows.append(row)
            print(f"    {label:24} seed{index} "
                  f"tail {row['tail_at_init']:.4f} -> {row['tail_final']:.4f}"
                  f"  2nd={row['second_moment_ratio']:6.3f} "
                  f"ed2={row['ed2']:8.4f} "
                  f"score={row['geometry_score_v2']:7.3f}", flush=True)
        print(f"    (seed{index} wall {wall:.0f}s for all arms; particle "
              f"cloud cost {particle_ledger['kernel_pairs']:,} kernel pairs)",
              flush=True)

    summary = {}
    for label, mode, r11, fixed, pre in ARMS:
        group = [r for r in rows if r["arm"] == label]
        moment = float(np.median([r["second_moment_ratio"] for r in group]))
        summary[label] = {
            "mode": mode, "r11": r11, "fixed_latents": fixed,
            "median_second_moment_ratio": moment,
            "median_ed2": float(np.median([r["ed2"] for r in group])),
            "median_score_v2": float(np.median(
                [r["geometry_score_v2"] for r in group])),
            "median_tail_final": float(np.median(
                [r["tail_final"] for r in group])),
            "median_kernel_pairs": float(np.median(
                [r["kernel_pairs"] for r in group])),
            "moment_in_band": bool(MOMENT_BAND[0] <= moment
                                   <= MOMENT_BAND[1])}
    return {"rows": [{k: v for k, v in r.items() if k != "trace"}
                     for r in rows],
            "traces": {r["arm"]: r["trace"] for r in rows},
            "summary": summary, "reference_tail": rows[0]["reference_tail"]}


def gate_12(summary: dict) -> dict:
    """Can an external target supersede R11?  Declared before the run."""
    r11 = [v for v in summary.values() if v["r11"]]
    external = {k: v for k, v in summary.items()
                if v["mode"] != "self" and not v["r11"]}
    if not r11 or not external:
        return {"passed": False, "reason": "missing arms"}
    best_r11 = min(r11, key=lambda v: v["median_ed2"])
    ceiling = best_r11["median_ed2"] * SUPERSEDE_TOLERANCE
    qualifying = [k for k, v in external.items()
                  if v["moment_in_band"] and v["median_ed2"] <= ceiling]
    baseline = summary.get("A0_self_reference", {})
    return {
        "passed": bool(qualifying),
        "meaning": ("an external target supersedes R11" if qualifying else
                    "no external target reproduces R11"),
        "ed2_ceiling": ceiling,
        "qualifying_arms": qualifying,
        "amortizing_arms_only": [k for k in qualifying
                                 if not summary[k]["fixed_latents"]],
        "tail_vs_baseline": {
            k: v["median_tail_final"]
            / max(baseline.get("median_tail_final", 1e-12), 1e-12)
            for k, v in external.items()},
        "cost_vs_baseline": {
            k: v["median_kernel_pairs"]
            / max(baseline.get("median_kernel_pairs", 1e-12), 1e-12)
            for k, v in external.items()},
        "note": "a qualifying arm with fixed latents memorizes and does not "
                "amortize; only the fresh-latent arms are the proposal",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase12.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    print("=== Phase 12: external targets by transport ===", flush=True)
    stage = stage_12(args.resolution, args.seeds, args.steps, args.data_root)
    gate = gate_12(stage["summary"])

    payload = {
        "status": "phase12-frozen-protocol",
        "protocol": "numerics/EncoderIndependentPhase12Protocol.md",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "config_digest": config_digest(
            TrainConfig(steps=args.steps, image_size=args.resolution)),
        "optimizer": optimizer_report(TrainConfig()),
        "frozen_thresholds": {
            "moment_band": list(MOMENT_BAND),
            "supersede_tolerance": SUPERSEDE_TOLERANCE,
            "particles": PARTICLES, "generator_batch": GENERATOR_BATCH,
            "positives": POSITIVES, "target_ess": GOOD_ESS,
            "sinkhorn_epsilon": SINKHORN_EPSILON,
            "sinkhorn_iterations": SINKHORN_ITERS},
        "elapsed_seconds": time.time() - started,
        "summary": stage["summary"], "gate": gate,
        "reference_tail": stage["reference_tail"],
        "rows": stage["rows"], "traces": stage["traces"],
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 12 ===")
    print(f"{'arm':26}{'latents':>9}{'tail':>9}{'2nd_mom':>9}{'ed2':>9}"
          f"{'score':>8}{'cost x':>9}  band")
    for key, entry in payload["summary"].items():
        cost = gate["cost_vs_baseline"].get(key)
        print(f"{key:26}{'fixed' if entry['fixed_latents'] else 'fresh':>9}"
              f"{entry['median_tail_final']:9.4f}"
              f"{entry['median_second_moment_ratio']:9.3f}"
              f"{entry['median_ed2']:9.4f}{entry['median_score_v2']:8.3f}"
              f"{(f'{cost:.2f}' if cost else '1.00'):>9}"
              f"  {'in ' if entry['moment_in_band'] else 'out'}")
    print(f"    (real data tail = {payload['reference_tail']:.4f})")
    print(f"\n  gate passed={gate['passed']}  -- {gate['meaning']}")
    if gate.get("qualifying_arms"):
        print(f"  qualifying: {gate['qualifying_arms']}")
        print(f"  of which genuinely amortizing (fresh latents): "
              f"{gate['amortizing_arms_only']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
