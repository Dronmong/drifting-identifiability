"""Phase 13 (protocol `EncoderIndependentPhase13Protocol.md`).

Does the amortizer close the gap to the particle cloud as pairs grow?

The investigation isolated **correspondence stability** as the requirement:
an exact Hungarian assignment preserves the target (tail 0.327 against the
cloud's 0.406) and still leaves the generator at tail 0.0009 on fresh latents
against 0.3430 on fixed ones.  With a stable correspondence the generator
reaches ED2 0.1640 from 256 pairs against the particle cloud's 0.0752, and
the bank size has never been varied.

One 2048-particle cloud per seed; the banks are nested prefixes of it, so
only the NUMBER of pairs changes.  Evaluation is always on fresh latents.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase13
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
SEED_OFFSET = 25000
MOMENT_BAND = (0.7, 1.3)
SUPERSEDE_TOLERANCE = 1.25
TREND_TOLERANCE = 1.25       # "approaches the particle cloud" means within 25%
GOOD_ESS = 0.9
MAX_PARTICLES = 2048
FIELD_NEGATIVES = 512        # declared cap; a no-op at 512 particles
GENERATOR_BATCH = 256
POSITIVES = 64
TAIL_KEEP = 32
PARTICLE_ETA = 0.2
TEACHER_ETA = 0.5
JITTER = 0.1
BANKS = (256, 512, 1024, 2048)

# (label, mode, bank, r11, jitter)
ARMS = (("C0_self", "self", 0, False, 0.0),
        ("C1_self+R11", "self", 0, True, 0.0),
        ("C2_bank256", "bank", 256, False, 0.0),
        ("C3_bank512", "bank", 512, False, 0.0),
        ("C4_bank1024", "bank", 1024, False, 0.0),
        ("C5_bank2048", "bank", 2048, False, 0.0),
        ("C6_bank2048_jitter", "bank", 2048, False, JITTER))


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


def evolve_particles(train, branch, kernel, rng, steps: int, count: int,
                     resolution: int) -> tuple[torch.Tensor, int]:
    """One cloud, with the negative side capped at a declared size.

    The cap keeps the cloud's cost linear in its particle count instead of
    quadratic.  At 512 particles it is a no-op, which anchors this sweep to
    the configuration every earlier phase measured.
    """
    cloud = torch.tensor(
        rng.normal(scale=0.5, size=(count, 3, resolution, resolution)),
        dtype=torch.float32)
    pairs = 0
    for _ in range(steps):
        if count > FIELD_NEGATIVES:
            index = torch.as_tensor(
                rng.choice(count, size=FIELD_NEGATIVES, replace=False))
            negatives = cloud[index]
        else:
            negatives = cloud
        drift, _ = KG.field(cloud, train.sample(POSITIVES, rng), negatives,
                            branch, kernel, direction_mode="paper",
                            normalization="rms", diagnostics=False)
        cloud = cloud + PARTICLE_ETA * drift
        pairs += count * (POSITIVES + len(negatives))
    return cloud, pairs


def stage_13(resolution: int, seeds: int, steps: int,
             root: str | None) -> dict:
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        train, branch, kernel, rng = _setup(resolution, seed, root)
        config = TrainConfig(steps=steps, batch=POSITIVES,
                             eval_samples=MAX_PARTICLES,
                             image_size=resolution)
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())

        particles, particle_pairs = evolve_particles(
            train, branch, kernel, rng, steps, MAX_PARTICLES, resolution)
        # The reference every arm is scored against.
        particle_metrics = M.raw_metrics(
            particles, pools["eval"], pools["cal_a"], pools["cal_b"],
            np.random.default_rng(derive_seed(seed, "p13-p")), None,
            target_null=pools["null"])
        particle_ed2 = particle_metrics["ed2"]
        particle_tail = _tail(particles)
        print(f"    [seed{index}] particle cloud ({MAX_PARTICLES}): "
              f"ed2={particle_ed2:.4f} tail={particle_tail:.4f} "
              f"({particle_pairs:,} kernel pairs)", flush=True)

        probe = sample_latent(MAX_PARTICLES, config.latent_dim,
                              derive_seed(seed, "p13-probe"))
        for label, mode, bank, r11, jitter in ARMS:
            model = OneStepGenerator(config.latent_dim, 3, resolution,
                                     config.width,
                                     derive_seed(seed, "generator"))
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config.learning_rate)
            torch_rng = torch.Generator().manual_seed(
                derive_seed(seed, "p13-latent") % (2 ** 31))
            bank_latent = (sample_latent(bank, config.latent_dim,
                                         derive_seed(seed, "p13-bank"))
                           if bank else None)
            generator_pairs = 0
            for _ in range(steps):
                if mode == "self":
                    latent = torch.randn(GENERATOR_BATCH, config.latent_dim,
                                         generator=torch_rng)
                    output = model(latent)
                    with torch.no_grad():
                        positives = train.sample(POSITIVES, rng)
                        own, _ = KG.field(
                            output.detach(), positives, output.detach(),
                            branch, kernel, direction_mode="paper",
                            normalization="rms", diagnostics=False)
                        target = output.detach() + TEACHER_ETA * own
                        if r11:
                            target = corrected_teacher(target, positives,
                                                       mode="scalar")
                    generator_pairs += GENERATOR_BATCH * (POSITIVES
                                                          + GENERATOR_BATCH)
                else:
                    pick = torch.as_tensor(
                        rng.choice(bank,
                                   size=min(GENERATOR_BATCH, bank),
                                   replace=False))
                    latent = bank_latent[pick]
                    if jitter:
                        latent = latent + jitter * torch.randn(
                            latent.shape, generator=torch_rng)
                    output = model(latent)
                    target = particles[pick]
                loss = ((output - target) ** 2).flatten(1).sum(1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                generated = model(probe)
            measured = M.raw_metrics(
                generated, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p13-m")), None,
                target_null=pools["null"])
            row = {"arm": label, "mode": mode, "bank": bank, "r11": r11,
                   "jitter": jitter, "seed": seed,
                   "ed2": measured["ed2"],
                   "geometry_score_v2": M.normalized_geometry_score_v2(
                       measured, null)["geometry_score"],
                   "second_moment_ratio": float(
                       generated.flatten(1).var(0).mean()) / reference_moment,
                   "tail": _tail(generated),
                   "particle_ed2": particle_ed2,
                   "particle_tail": particle_tail,
                   "ed2_over_particles": measured["ed2"] / max(particle_ed2,
                                                               1e-12),
                   "visits_per_pair": (steps * min(GENERATOR_BATCH, bank)
                                       / bank if bank else float("nan")),
                   "kernel_pairs": generator_pairs + (particle_pairs
                                                      if mode == "bank"
                                                      else 0)}
            rows.append(row)
            print(f"    {label:22} seed{index} "
                  f"ed2={row['ed2']:8.4f} "
                  f"(x{row['ed2_over_particles']:5.2f} particles) "
                  f"2nd={row['second_moment_ratio']:6.3f} "
                  f"tail={row['tail']:7.4f}", flush=True)

    summary = {}
    for label, mode, bank, r11, jitter in ARMS:
        group = [r for r in rows if r["arm"] == label]
        moment = float(np.median([r["second_moment_ratio"] for r in group]))
        summary[label] = {
            "mode": mode, "bank": bank, "r11": r11, "jitter": jitter,
            "median_ed2": float(np.median([r["ed2"] for r in group])),
            "median_score_v2": float(np.median(
                [r["geometry_score_v2"] for r in group])),
            "median_second_moment_ratio": moment,
            "median_tail": float(np.median([r["tail"] for r in group])),
            "median_ed2_over_particles": float(np.median(
                [r["ed2_over_particles"] for r in group])),
            "median_kernel_pairs": float(np.median(
                [r["kernel_pairs"] for r in group])),
            # At a fixed step budget a larger bank means FEWER visits per
            # pair, so the sweep confounds "more pairs" with "less training
            # per pair".  Reported rather than hidden.
            "visits_per_pair": (
                float(np.median([r["visits_per_pair"] for r in group]))
                if bank else float("nan")),
            "moment_in_band": bool(MOMENT_BAND[0] <= moment
                                   <= MOMENT_BAND[1])}
    return {"rows": rows, "summary": summary,
            "particle_ed2": float(np.median([r["particle_ed2"]
                                             for r in rows])),
            "particle_tail": float(np.median([r["particle_tail"]
                                              for r in rows]))}


def verdict_13(summary: dict, particle_ed2: float) -> dict:
    """The gate, and the separately declared trend criterion."""
    r11 = [v for v in summary.values() if v["r11"]]
    banks = {k: v for k, v in summary.items() if v["mode"] == "bank"
             and not v["jitter"]}
    if not r11 or not banks:
        return {"passed": False, "reason": "missing arms"}
    best_r11 = min(r11, key=lambda v: v["median_ed2"])
    ceiling = best_r11["median_ed2"] * SUPERSEDE_TOLERANCE
    qualifying = [k for k, v in banks.items()
                  if v["moment_in_band"] and v["median_ed2"] <= ceiling]

    ordered = sorted(banks.items(), key=lambda kv: kv[1]["bank"])
    series = [(v["bank"], v["median_ed2"]) for _, v in ordered]
    monotone = all(b[1] <= a[1] + 1e-9 for a, b in zip(series, series[1:]))
    best_bank = min(banks.values(), key=lambda v: v["median_ed2"])
    approaches = bool(best_bank["median_ed2"]
                      <= particle_ed2 * TREND_TOLERANCE)
    return {
        "passed": bool(qualifying),
        "meaning": ("a stable-correspondence external target supersedes R11"
                    if qualifying else "no bank reproduces R11"),
        "ed2_ceiling": ceiling,
        "qualifying_arms": qualifying,
        "bank_series_ed2": series,
        "monotone_in_bank_size": monotone,
        "best_bank_ed2": best_bank["median_ed2"],
        "particle_ed2": particle_ed2,
        "approaches_particle_cloud": approaches,
        "trend_verdict": (
            "the amortizer approaches the particle cloud" if approaches
            else "ED2 plateaus above the particle cloud -- the map does not "
                 "reach the particle law at these bank sizes"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase13.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    print("=== Phase 13: the pair-bank sweep ===", flush=True)
    stage = stage_13(args.resolution, args.seeds, args.steps, args.data_root)
    verdict = verdict_13(stage["summary"], stage["particle_ed2"])

    payload = {
        "status": "phase13-frozen-protocol",
        "protocol": "numerics/EncoderIndependentPhase13Protocol.md",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "config_digest": config_digest(
            TrainConfig(steps=args.steps, image_size=args.resolution)),
        "optimizer": optimizer_report(TrainConfig()),
        "frozen_thresholds": {
            "moment_band": list(MOMENT_BAND),
            "supersede_tolerance": SUPERSEDE_TOLERANCE,
            "trend_tolerance": TREND_TOLERANCE, "banks": list(BANKS),
            "max_particles": MAX_PARTICLES,
            "field_negatives": FIELD_NEGATIVES, "jitter": JITTER,
            "target_ess": GOOD_ESS},
        "elapsed_seconds": time.time() - started,
        "summary": stage["summary"], "verdict": verdict,
        "particle_ed2": stage["particle_ed2"],
        "particle_tail": stage["particle_tail"],
        "rows": stage["rows"],
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 13 ===")
    print(f"{'arm':24}{'bank':>7}{'visits':>8}{'ed2':>10}{'x part':>9}"
          f"{'2nd_mom':>9}{'tail':>9}{'score':>8}  band")
    for key, entry in payload["summary"].items():
        visits = entry["visits_per_pair"]
        print(f"{key:24}{entry['bank'] or '-':>7}"
              f"{(f'{visits:.0f}' if np.isfinite(visits) else '-'):>8}"
              f"{entry['median_ed2']:10.4f}"
              f"{entry['median_ed2_over_particles']:9.2f}"
              f"{entry['median_second_moment_ratio']:9.3f}"
              f"{entry['median_tail']:9.4f}{entry['median_score_v2']:8.3f}"
              f"  {'in ' if entry['moment_in_band'] else 'out'}")
    print(f"    (particle cloud: ed2={payload['particle_ed2']:.4f} "
          f"tail={payload['particle_tail']:.4f})")
    print(f"\n  bank series (size, ed2): {verdict['bank_series_ed2']}")
    print(f"  monotone in bank size: {verdict['monotone_in_bank_size']}")
    print(f"  gate passed={verdict['passed']}  -- {verdict['meaning']}")
    print(f"  trend: {verdict['trend_verdict']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
