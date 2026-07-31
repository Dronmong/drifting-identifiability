"""Formal audit of the load-bearing claim, plus the replication owed.

Phase 11 localized tail destruction to the self-referential teacher: with
everything else identical, a *fixed* high-tail target grows the tail to 0.096
while the drifting teacher collapses it to 0.0037.  The proposed reading is
that ``T = f + eta V`` never accumulates a persistent demand for SHAPE
because ``eta V`` is small (~6% of the output norm) and recomputed from a
fresh batch every step.

That reading has a load-bearing and unmeasured part: **is the field's tail
component actually incoherent across batches?**  If V's tail is reproducible
from batch to batch then the demand does persist, averaging preserves it, and
the reading is wrong -- the smallness of ``eta V`` alone would have to carry
the explanation.

  S1  coherence.  At a fixed cloud, compute V from two disjoint positive
      batches and correlate them, resolved into the data's leading
      directions (bulk) and its trailing ones (tail).  A bulk correlation
      near 1 with a tail correlation near 0 is the signature.
  S2  the replication owed.  Phase 11's discriminating measurement -- moving
      teacher against fixed targets -- was one seed.  Repeat at three, with
      the second moment logged alongside the tail, which the original probe
      did not record.
  S3  scale versus shape.  Is it the SMALLNESS of eta*V or its
      INCOHERENCE?  Rollout K applies the field K times, which grows the
      displacement AND commits it to a direction; averaging M batches at
      K=1 grows coherence WITHOUT growing displacement.  Running both
      separates the two, which no experiment in this program has done.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase12
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

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 22000
GOOD_ESS = 0.9
TAIL_KEEP = 32


def _setup(resolution: int, seed: int, root: str | None):
    train = cifar.cifar_target(resolution, "train", root)
    rng = np.random.default_rng(derive_seed(seed, "p12"))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace",
                              target_ess_fraction=GOOD_ESS)
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace",
        geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
        geometry.kernel_eps, combine=geometry.combine,
        target_ess_fraction=GOOD_ESS)
    return train, branch, kernel, rng


def _tail(x: torch.Tensor, keep: int = TAIL_KEEP) -> float:
    flat = x.reshape(len(x), -1)
    power = torch.linalg.svdvals(flat - flat.mean(dim=0, keepdim=True)) ** 2
    return float(power[keep:].sum() / power.sum())


def _basis(sample: torch.Tensor) -> torch.Tensor:
    flat = sample.reshape(len(sample), -1)
    _, _, right = torch.linalg.svd(flat - flat.mean(dim=0, keepdim=True),
                                   full_matrices=False)
    return right


def _split(v: torch.Tensor, basis: torch.Tensor):
    """Split a displacement field into bulk and tail parts of the data basis."""
    flat = v.reshape(len(v), -1)
    coefficients = flat @ basis.T
    bulk = coefficients[:, :TAIL_KEEP] @ basis[:TAIL_KEEP]
    tail = flat - bulk
    return bulk, tail


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = float(a.norm()) * float(b.norm())
    return float((a * b).sum()) / denominator if denominator > 0 else 0.0


# ---------------------------------------------------------------------------
# S1: is the field's tail demand coherent across batches?
# ---------------------------------------------------------------------------


def s1_coherence(resolution: int, seeds: int, steps: int, root: str | None,
                 cloud_size: int = 256, repeats: int = 8) -> dict:
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        train, branch, kernel, rng = _setup(resolution, seed, root)
        basis = _basis(train.sample(512, rng))
        config = TrainConfig(steps=steps, batch=64, image_size=resolution)

        # A trained generator, and free particles: two clouds that behave
        # very differently and should be checked separately.
        model = OneStepGenerator(config.latent_dim, 3, resolution,
                                 config.width, derive_seed(seed, "generator"))
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
        torch_rng = torch.Generator().manual_seed(
            derive_seed(seed, "p12-latent") % (2 ** 31))
        for _ in range(steps):
            latent = torch.randn(cloud_size, config.latent_dim,
                                 generator=torch_rng)
            output = model(latent)
            with torch.no_grad():
                drift, _ = KG.field(output.detach(), train.sample(64, rng),
                                    output.detach(), branch, kernel,
                                    direction_mode="paper",
                                    normalization="rms", diagnostics=False)
                teacher = output.detach() + 0.5 * drift
            loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            generated = model(torch.randn(cloud_size, config.latent_dim,
                                          generator=torch_rng))

        particles = torch.tensor(
            rng.normal(scale=0.5,
                       size=(cloud_size, 3, resolution, resolution)),
            dtype=torch.float32)
        for _ in range(steps):
            drift, _ = KG.field(particles, train.sample(64, rng), particles,
                                branch, kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            particles = particles + 0.2 * drift

        for name, cloud in (("generator", generated),
                            ("particles", particles),
                            ("real_data", train.sample(cloud_size, rng))):
            bulk_cos, tail_cos, tail_share = [], [], []
            for _ in range(repeats):
                fields = []
                for _ in range(2):
                    drift, _ = KG.field(cloud, train.sample(64, rng), cloud,
                                        branch, kernel,
                                        direction_mode="paper",
                                        normalization="none",
                                        diagnostics=False)
                    fields.append(drift)
                bulk_a, tail_a = _split(fields[0], basis)
                bulk_b, tail_b = _split(fields[1], basis)
                bulk_cos.append(_cosine(bulk_a, bulk_b))
                tail_cos.append(_cosine(tail_a, tail_b))
                tail_share.append(float(tail_a.pow(2).sum()
                                        / (bulk_a.pow(2).sum()
                                           + tail_a.pow(2).sum())))
            rows.append({"cloud": name, "seed": seed,
                         "bulk_coherence": float(np.mean(bulk_cos)),
                         "tail_coherence": float(np.mean(tail_cos)),
                         "field_tail_share": float(np.mean(tail_share))})
            print(f"    S1 {name:11} seed{index} "
                  f"bulk_coh={rows[-1]['bulk_coherence']:+7.4f} "
                  f"tail_coh={rows[-1]['tail_coherence']:+7.4f} "
                  f"tail_share={rows[-1]['field_tail_share']:6.4f}",
                  flush=True)
    summary = {}
    for name in ("generator", "particles", "real_data"):
        group = [r for r in rows if r["cloud"] == name]
        summary[name] = {k: float(np.median([r[k] for r in group]))
                         for k in ("bulk_coherence", "tail_coherence",
                                   "field_tail_share")}
    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------------------
# S2 / S3: replication, and separating smallness from incoherence
# ---------------------------------------------------------------------------


def s2s3_targets(resolution: int, seeds: int, steps: int, root: str | None,
                 cloud_size: int = 256, eta: float = 0.5) -> dict:
    """Moving teacher vs fixed targets, and rollout vs batch-averaging.

    ``rollout=K`` applies the field K times, growing the displacement *and*
    committing it to a direction.  ``average=M`` averages M independent
    single-step fields, growing coherence at *fixed* displacement.  Running
    both separates "eta V is too small" from "eta V is incoherent".
    """
    evaluation = cifar.cifar_target(resolution, "eval", root)
    arms = [("moving_K1", "moving", 1, 1), ("moving_K4", "moving", 4, 1),
            ("moving_K16", "moving", 16, 1),
            ("moving_avg4", "moving", 1, 4),
            ("moving_avg16", "moving", 1, 16),
            ("fixed_particles", "fixed_particles", 0, 0),
            ("fixed_data", "fixed_data", 0, 0)]
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        train, branch, kernel, rng = _setup(resolution, seed, root)
        config = TrainConfig(steps=steps, batch=64, eval_samples=cloud_size,
                             image_size=resolution)
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())

        particle_target = torch.tensor(
            rng.normal(scale=0.5,
                       size=(cloud_size, 3, resolution, resolution)),
            dtype=torch.float32)
        for _ in range(steps):
            drift, _ = KG.field(particle_target, train.sample(64, rng),
                                particle_target, branch, kernel,
                                direction_mode="paper", normalization="rms",
                                diagnostics=False)
            particle_target = particle_target + 0.2 * drift
        data_target = train.sample(cloud_size, rng)
        fixed_latent = sample_latent(cloud_size, config.latent_dim,
                                     derive_seed(seed, "p12-fixed"))

        for label, mode, rollout, average in arms:
            model = OneStepGenerator(config.latent_dim, 3, resolution,
                                     config.width,
                                     derive_seed(seed, "generator"))
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config.learning_rate)
            torch_rng = torch.Generator().manual_seed(
                derive_seed(seed, "p12-latent") % (2 ** 31))
            trace = []
            for step in range(steps):
                if mode == "moving":
                    latent = torch.randn(cloud_size, config.latent_dim,
                                         generator=torch_rng)
                else:
                    latent = fixed_latent
                output = model(latent)
                if step % 100 == 0:
                    trace.append({"step": step,
                                  "tail": _tail(output.detach())})
                if mode == "moving":
                    with torch.no_grad():
                        cloud = output.detach().clone()
                        for _ in range(rollout):
                            total = torch.zeros_like(cloud)
                            for _ in range(average):
                                drift, _ = KG.field(
                                    cloud, train.sample(config.batch, rng),
                                    cloud, branch, kernel,
                                    direction_mode="paper",
                                    normalization="rms", diagnostics=False)
                                total = total + drift
                            cloud = cloud + eta * total / average
                        target = cloud
                else:
                    target = (particle_target if mode == "fixed_particles"
                              else data_target)
                loss = ((output - target) ** 2).flatten(1).sum(1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            probe = sample_latent(cloud_size, config.latent_dim,
                                  derive_seed(seed, "p12-probe"))
            with torch.no_grad():
                generated = model(probe)
            measured = M.raw_metrics(
                generated, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p12-m")), None,
                target_null=pools["null"])
            rows.append({
                "arm": label, "seed": seed, "rollout": rollout,
                "average": average,
                "tail_final": _tail(generated),
                "tail_at_init": trace[0]["tail"],
                "second_moment_ratio": float(
                    generated.flatten(1).var(0).mean()) / reference_moment,
                "ed2": measured["ed2"],
                "geometry_score_v2": M.normalized_geometry_score_v2(
                    measured, null)["geometry_score"]})
            print(f"    S2 {label:18} seed{index} "
                  f"tail {rows[-1]['tail_at_init']:.4f} -> "
                  f"{rows[-1]['tail_final']:.4f}  "
                  f"2nd={rows[-1]['second_moment_ratio']:6.3f} "
                  f"ed2={rows[-1]['ed2']:8.4f}", flush=True)
    summary = {}
    for label, _, _, _ in arms:
        group = [r for r in rows if r["arm"] == label]
        summary[label] = {k: float(np.median([r[k] for r in group]))
                          for k in ("tail_final", "second_moment_ratio",
                                    "ed2", "geometry_score_v2")}
    return {"rows": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=("all", "s1", "s2"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase12_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    payload: dict = {
        "status": "phase11-followup-audit-feeds-no-gate",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "question": "is the field's tail demand incoherent across batches, "
                    "and does making it persistent (rollout) or coherent "
                    "(averaging) restore the tail?",
    }

    if args.stage in ("all", "s1"):
        print("=== S1: is the field's tail demand coherent? ===", flush=True)
        payload["s1_coherence"] = s1_coherence(
            args.resolution, args.seeds, args.steps, args.data_root)
    if args.stage in ("all", "s2"):
        print("\n=== S2/S3: targets, rollout, and averaging ===", flush=True)
        payload["s2s3_targets"] = s2s3_targets(
            args.resolution, args.seeds, args.steps, args.data_root)

    payload["elapsed_seconds"] = time.time() - started
    digest = write_json(args.out, payload)

    print("\n=== PHASE-11 FOLLOW-UP AUDIT ===")
    if "s1_coherence" in payload:
        print(f"\nS1  {'cloud':12}{'bulk coherence':>17}{'tail coherence':>17}"
              f"{'field tail share':>19}")
        for name, entry in payload["s1_coherence"]["summary"].items():
            print(f"    {name:12}{entry['bulk_coherence']:17.4f}"
                  f"{entry['tail_coherence']:17.4f}"
                  f"{entry['field_tail_share']:19.4f}")
    if "s2s3_targets" in payload:
        print(f"\nS2/S3  {'arm':20}{'tail end':>11}{'2nd moment':>13}"
              f"{'ed2':>10}{'score':>9}")
        for name, entry in payload["s2s3_targets"]["summary"].items():
            print(f"       {name:20}{entry['tail_final']:11.4f}"
                  f"{entry['second_moment_ratio']:13.3f}"
                  f"{entry['ed2']:10.4f}{entry['geometry_score_v2']:9.3f}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
