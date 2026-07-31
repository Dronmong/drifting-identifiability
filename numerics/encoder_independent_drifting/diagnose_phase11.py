"""Why is the generator's spectral tail so hard to raise?

Phase 10 established the law -- the spectral tail sets the field's
equilibrium radius, predicting six of seven arms to within 0.11 and free
particles out-of-sample to 0.015 -- and left one puzzle.  The generator's
tail is stuck near 0.005 against real data's 0.138.  Latent dimension does
not raise it (N4: 0.0015 -> 0.0058 across 8 -> 512), capacity does not (8A:
flat across 36x), and an explicit tail penalty reached only 0.0113.  Yet N3
showed a generator *can* reproduce a high-tail cloud faithfully when fitted
to one directly, so the model class is not the barrier.

That leaves the teacher.

**H13: the field is tail-blind.**  The drift ``V`` is built from a kernel on
raw pixels, which is dominated by large-scale structure, so ``V`` may carry
almost no energy in the data's trailing principal directions.  If so the
teacher never asks the generator to populate them, no gradient ever arrives
there, and the cloud keeps whatever tail its initialization gave it.

That would also explain the particles, which look like a paradox otherwise:
they *start* from isotropic noise, whose tail beyond 32 directions is ~0.96
by construction, and end at 0.415.  A tail-blind field would neither add tail
to a generator that lacks it nor remove tail from particles that have it.

  R1  decompose V's energy in the data's principal basis, at several clouds.
      Is V's tail fraction far below the cloud's own?
  R2  trace the generator's output tail and its teacher's tail through
      training.  Does the tail decay, or was it never there?
  R3  **the decisive test, run in both directions.**  Initialize particles
      from a LOW-tail cloud: if the field is tail-blind they stay low and
      land at the low second moment the Phase-10 law predicts, rather than
      recovering to 0.995.  Refuted if they recover.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase11
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
from .models import OneStepGenerator

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 19000
GOOD_ESS = 0.9
TAIL_KEEP = 32


def _setup(resolution: int, seed: int, root: str | None):
    train = cifar.cifar_target(resolution, "train", root)
    rng = np.random.default_rng(derive_seed(seed, "p11"))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace",
                              target_ess_fraction=GOOD_ESS)
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace",
        geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
        geometry.kernel_eps, combine=geometry.combine,
        target_ess_fraction=GOOD_ESS)
    return train, branch, kernel, rng


def _basis(sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The data's principal directions and centre."""
    flat = sample.reshape(len(sample), -1)
    centre = flat.mean(dim=0, keepdim=True)
    _, _, right = torch.linalg.svd(flat - centre, full_matrices=False)
    return right, centre


def _tail_in_basis(x: torch.Tensor, basis: torch.Tensor,
                   centre: torch.Tensor | None = None) -> float:
    """Fraction of energy beyond the top ``TAIL_KEEP`` DATA directions."""
    flat = x.reshape(len(x), -1)
    if centre is not None:
        flat = flat - centre
    coefficients = flat @ basis.T
    power = coefficients.pow(2).sum(dim=0)
    total = float(power.sum())
    return float(power[TAIL_KEEP:].sum()) / total if total > 0 else float("nan")


def _own_tail(x: torch.Tensor) -> float:
    flat = x.reshape(len(x), -1)
    power = torch.linalg.svdvals(flat - flat.mean(dim=0, keepdim=True)) ** 2
    return float(power[TAIL_KEEP:].sum() / power.sum())


# ---------------------------------------------------------------------------
# R1: is the field tail-blind?
# ---------------------------------------------------------------------------


def r1_field_spectrum(resolution: int, seeds: int, steps: int,
                      root: str | None, cloud_size: int = 256) -> dict:
    """Decompose V's energy in the data's principal basis, at several clouds."""
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        train, branch, kernel, rng = _setup(resolution, seed, root)
        reference = train.sample(512, rng)
        basis, centre = _basis(reference)

        # A generator trained by the real recipe, and free particles.
        config = TrainConfig(steps=steps, batch=64, image_size=resolution)
        model = OneStepGenerator(config.latent_dim, 3, resolution,
                                 config.width, derive_seed(seed, "generator"))
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
        torch_rng = torch.Generator().manual_seed(
            derive_seed(seed, "p11-latent") % (2 ** 31))
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

        clouds = {"generator": generated, "particles": particles,
                  "real_data": train.sample(cloud_size, rng),
                  "white_noise": torch.tensor(
                      rng.normal(scale=0.5,
                                 size=(cloud_size, 3, resolution,
                                       resolution)), dtype=torch.float32)}
        for name, cloud in clouds.items():
            drift, _ = KG.field(cloud, train.sample(64, rng), cloud, branch,
                                kernel, direction_mode="paper",
                                normalization="none", diagnostics=False)
            rows.append({
                "cloud": name, "seed": seed,
                "cloud_tail_in_data_basis": _tail_in_basis(cloud, basis,
                                                           centre),
                "cloud_own_tail": _own_tail(cloud),
                # V is a displacement, so it is decomposed about zero.
                "field_tail_in_data_basis": _tail_in_basis(drift, basis),
            })
            print(f"    R1 {name:12} seed{index} "
                  f"cloud_tail={rows[-1]['cloud_tail_in_data_basis']:7.4f} "
                  f"FIELD_tail={rows[-1]['field_tail_in_data_basis']:7.4f}",
                  flush=True)
    summary = {}
    for name in ("generator", "particles", "real_data", "white_noise"):
        group = [r for r in rows if r["cloud"] == name]
        summary[name] = {
            "median_cloud_tail": float(np.median(
                [r["cloud_tail_in_data_basis"] for r in group])),
            "median_field_tail": float(np.median(
                [r["field_tail_in_data_basis"] for r in group])),
        }
    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------------------
# R2: does the tail decay, or was it never there?
# ---------------------------------------------------------------------------


def r2_tail_trace(resolution: int, seeds: int, steps: int, root: str | None,
                  cloud_size: int = 256, every: int = 50) -> dict:
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        train, branch, kernel, rng = _setup(resolution, seed, root)
        config = TrainConfig(steps=steps, batch=64, image_size=resolution)
        model = OneStepGenerator(config.latent_dim, 3, resolution,
                                 config.width, derive_seed(seed, "generator"))
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
        torch_rng = torch.Generator().manual_seed(
            derive_seed(seed, "p11-latent") % (2 ** 31))
        trace = []
        for step in range(steps + 1):
            latent = torch.randn(cloud_size, config.latent_dim,
                                 generator=torch_rng)
            output = model(latent)
            with torch.no_grad():
                drift, _ = KG.field(output.detach(), train.sample(64, rng),
                                    output.detach(), branch, kernel,
                                    direction_mode="paper",
                                    normalization="rms", diagnostics=False)
                teacher = output.detach() + 0.5 * drift
            if step % every == 0:
                with torch.no_grad():
                    trace.append({
                        "step": step,
                        "output_tail": _own_tail(output.detach()),
                        "teacher_tail": _own_tail(teacher)})
            if step == steps:
                break
            loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        rows.append({"seed": seed, "trace": trace,
                     "tail_at_init": trace[0]["output_tail"],
                     "tail_at_end": trace[-1]["output_tail"]})
        marks = " ".join(f"{t['step']}:{t['output_tail']:.4f}"
                         for t in trace[::max(len(trace) // 6, 1)])
        print(f"    R2 seed{index} output tail {marks}", flush=True)
        print(f"       teacher tail at end "
              f"{trace[-1]['teacher_tail']:.4f}", flush=True)
    return {"rows": rows,
            "median_tail_at_init": float(np.median(
                [r["tail_at_init"] for r in rows])),
            "median_tail_at_end": float(np.median(
                [r["tail_at_end"] for r in rows]))}


# ---------------------------------------------------------------------------
# R3: the decisive test -- particles from a low-tail start
# ---------------------------------------------------------------------------


def r3_particle_init(resolution: int, seeds: int, steps: int,
                     root: str | None, cloud_size: int = 512,
                     eta: float = 0.2) -> dict:
    """If the field is tail-blind it cannot RESTORE a tail either.

    Particles normally start from isotropic noise, whose tail is ~0.96 by
    construction, and end at 0.995 of the data's variance.  Start them from a
    cloud with the generator's tail instead and, under H13, they should stay
    there -- landing at the low second moment the Phase-10 law predicts
    rather than recovering.  If they recover, the field can build tail and
    H13 is refuted.
    """
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        train, branch, kernel, rng = _setup(resolution, seed, root)
        config = TrainConfig(steps=steps, batch=64, eval_samples=cloud_size,
                             image_size=resolution)
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())
        basis, centre = _basis(pools["eval"])

        starts = {}
        noise = torch.tensor(
            rng.normal(scale=0.5,
                       size=(cloud_size, 3, resolution, resolution)),
            dtype=torch.float32)
        starts["isotropic_noise"] = noise
        # Low-tail start: the same noise projected onto the data's top-32
        # directions, so it differs from the control in tail and (almost)
        # nothing else.
        flat = noise.flatten(1)
        top = basis[:TAIL_KEEP]
        starts["low_tail_noise"] = ((flat @ top.T) @ top).reshape(noise.shape)
        # And a low-tail start at the data's own scale.
        data = train.sample(cloud_size, rng).flatten(1)
        projected = centre + ((data - centre) @ top.T) @ top
        starts["low_tail_data"] = projected.reshape(noise.shape)

        for name, start in starts.items():
            cloud = start.clone()
            for _ in range(steps):
                drift, _ = KG.field(cloud, train.sample(64, rng), cloud,
                                    branch, kernel, direction_mode="paper",
                                    normalization="rms", diagnostics=False)
                cloud = cloud + eta * drift
            measured = M.raw_metrics(
                cloud, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p11-m")), None,
                target_null=pools["null"])
            rows.append({
                "start": name, "seed": seed,
                "start_tail": _own_tail(start),
                "end_tail": _own_tail(cloud),
                "second_moment_ratio": float(
                    cloud.flatten(1).var(0).mean()) / reference_moment,
                "ed2": measured["ed2"],
                "geometry_score_v2": M.normalized_geometry_score_v2(
                    measured, null)["geometry_score"]})
            print(f"    R3 {name:16} seed{index} "
                  f"tail {rows[-1]['start_tail']:.4f} -> "
                  f"{rows[-1]['end_tail']:.4f}  "
                  f"2nd={rows[-1]['second_moment_ratio']:6.3f} "
                  f"ed2={rows[-1]['ed2']:8.4f}", flush=True)
    summary = {}
    for name in ("isotropic_noise", "low_tail_noise", "low_tail_data"):
        group = [r for r in rows if r["start"] == name]
        if group:
            summary[name] = {
                "median_start_tail": float(np.median(
                    [r["start_tail"] for r in group])),
                "median_end_tail": float(np.median(
                    [r["end_tail"] for r in group])),
                "median_second_moment_ratio": float(np.median(
                    [r["second_moment_ratio"] for r in group])),
                "median_ed2": float(np.median([r["ed2"] for r in group]))}
    return {"rows": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=("all", "r1", "r2", "r3"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase11_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    payload: dict = {
        "status": "phase10-followup-probe-feeds-no-gate",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "hypothesis": "H13: the field is tail-blind -- V carries almost no "
                      "energy in the data's trailing principal directions, "
                      "so the teacher never asks for tail and the cloud "
                      "keeps whatever its initialization gave it",
    }

    if args.stage in ("all", "r1"):
        print("=== R1: is the field tail-blind? ===", flush=True)
        payload["r1_field_spectrum"] = r1_field_spectrum(
            args.resolution, args.seeds, args.steps, args.data_root)
    if args.stage in ("all", "r2"):
        print("\n=== R2: does the tail decay, or was it never there? ===",
              flush=True)
        payload["r2_tail_trace"] = r2_tail_trace(
            args.resolution, args.seeds, args.steps, args.data_root)
    if args.stage in ("all", "r3"):
        print("\n=== R3: particles from a low-tail start ===", flush=True)
        payload["r3_particle_init"] = r3_particle_init(
            args.resolution, args.seeds, args.steps, args.data_root)

    payload["elapsed_seconds"] = time.time() - started
    digest = write_json(args.out, payload)

    print("\n=== PHASE-10 FOLLOW-UP PROBE ===")
    if "r1_field_spectrum" in payload:
        print(f"\nR1  {'cloud':14}{'cloud tail':>13}{'FIELD tail':>13}"
              f"{'ratio':>9}")
        for name, entry in payload["r1_field_spectrum"]["summary"].items():
            cloud_tail = entry["median_cloud_tail"]
            field_tail = entry["median_field_tail"]
            print(f"    {name:14}{cloud_tail:13.4f}{field_tail:13.4f}"
                  f"{field_tail / max(cloud_tail, 1e-12):9.3f}")
    if "r2_tail_trace" in payload:
        stage = payload["r2_tail_trace"]
        print(f"\nR2  output tail at init {stage['median_tail_at_init']:.4f} "
              f"-> at end {stage['median_tail_at_end']:.4f}")
    if "r3_particle_init" in payload:
        print(f"\nR3  {'start':18}{'tail start':>12}{'tail end':>11}"
              f"{'2nd moment':>13}{'ed2':>10}")
        for name, entry in payload["r3_particle_init"]["summary"].items():
            print(f"    {name:18}{entry['median_start_tail']:12.4f}"
                  f"{entry['median_end_tail']:11.4f}"
                  f"{entry['median_second_moment_ratio']:13.3f}"
                  f"{entry['median_ed2']:10.4f}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
