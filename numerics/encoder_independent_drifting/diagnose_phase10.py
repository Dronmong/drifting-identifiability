"""Where is the field's radial zero, along clouds shaped like the data?

Phase 9 sharpened the puzzle rather than resolving it.  In the
`gaussian_decay` cell the data is Gaussian and the linear generator (latent
64 = dim 64) can represent it **exactly**, so ``q = p`` is reachable and the
population field vanishes there -- ``q = p`` IS a fixed point.  The generator
converges to 0.648 anyway.  Free particles under the identical field reach
0.998.

So either the dynamics select a different fixed point, or the field does not
actually vanish at ``q = p``.  An incidental measurement during Phase 9
suggests the latter: with the cloud drawn as *real CIFAR data* the field had
magnitude 0.32 rather than 0.

  Q1  Scale a cloud of the DATA's own shape by ``alpha`` about its mean and
      measure the raw radial component of the field.  Where does it cross
      zero?  If the zero sits near ``alpha ~ 0.5`` then the generator is
      simply sitting on the field's own radial equilibrium and the question
      becomes a statement about the estimator -- fully tractable.  If it
      sits at ``alpha = 1`` the generator's deficit is something else.
  Q2  The same, decomposed: population bias (huge positive batch) against
      finite-batch bias, masked against unmasked.  Separates "the bi-softmax
      is biased" from "the estimator is biased at 64 positives".
  Q3  Do free particles actually sit at a data-shaped configuration?  If the
      converged particle cloud is NOT data-shaped, that is how it escapes a
      radial equilibrium the generator cannot.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from .config import GeometryConfig, MASTER_SEED, derive_seed
from .diagnostics import provenance, write_json
from .fixed_features import build_family
from .kernels import calibrate_block_kernel

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 17000
GOOD_ESS = 0.9


def _setup(resolution: int, seed: int, root: str | None, ess: float):
    train = cifar.cifar_target(resolution, "train", root)
    rng = np.random.default_rng(derive_seed(seed, "p10"))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace",
                              target_ess_fraction=ess)
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace",
        geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
        geometry.kernel_eps, combine=geometry.combine,
        target_ess_fraction=ess)
    return train, branch, kernel, rng


def _radial(drift: torch.Tensor, cloud: torch.Tensor) -> tuple[float, float]:
    """Raw mean radial component about the cloud's own centre, and |V|."""
    flat = cloud.reshape(len(cloud), -1)
    v = drift.reshape(len(drift), -1)
    radius = flat - flat.mean(dim=0, keepdim=True)
    unit = radius / radius.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return float((v * unit).sum(dim=1).mean()), float(v.norm(dim=1).mean())


def scaled_cloud(sample: torch.Tensor, alpha: float) -> torch.Tensor:
    """The data's own shape, rescaled about its mean to variance ratio alpha."""
    flat = sample.reshape(len(sample), -1)
    centre = flat.mean(dim=0, keepdim=True)
    scaled = centre + (flat - centre) * alpha ** 0.5
    return scaled.reshape(sample.shape)


def q1q2_radial_zero(resolution: int, seeds: int, root: str | None,
                     alphas=(0.15, 0.25, 0.4, 0.55, 0.7, 1.0, 1.4),
                     cloud_size: int = 256, repeats: int = 6) -> dict:
    """Where does the field's radial component vanish, along data-shaped clouds?"""
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        train, branch, kernel, rng = _setup(resolution, seed, root, GOOD_ESS)
        for alpha in alphas:
            for positives_n, label in ((64, "batch=64"),
                                       (2048, "batch=2048")):
                for mask in (False, True):
                    radial, magnitude = [], []
                    for _ in range(repeats):
                        base = train.sample(cloud_size, rng)
                        cloud = scaled_cloud(base, alpha)
                        drift, _ = KG.field(
                            cloud, train.sample(positives_n, rng), cloud,
                            branch, kernel, direction_mode="paper",
                            normalization="none", diagnostics=False,
                            self_mask=mask)
                        r, m = _radial(drift, cloud)
                        radial.append(r)
                        magnitude.append(m)
                    rows.append({
                        "alpha": alpha, "positives": positives_n,
                        "self_mask": mask, "seed": seed, "regime": label,
                        "radial": float(np.mean(radial)),
                        "magnitude": float(np.mean(magnitude)),
                        "radial_relative": float(np.mean(radial))
                        / max(float(np.mean(magnitude)), 1e-30)})
                    print(f"    Q1 alpha={alpha:5.2f} {label:11} "
                          f"mask={str(mask):5} seed{index} "
                          f"radial={rows[-1]['radial']:+10.5f} "
                          f"|V|={rows[-1]['magnitude']:9.5f}", flush=True)
    summary = {}
    for alpha in alphas:
        for positives_n in (64, 2048):
            for mask in (False, True):
                group = [r for r in rows if r["alpha"] == alpha
                         and r["positives"] == positives_n
                         and r["self_mask"] is mask]
                if group:
                    summary[f"alpha={alpha}_pos={positives_n}_mask={mask}"] = {
                        "median_radial": float(np.median(
                            [r["radial"] for r in group])),
                        "median_magnitude": float(np.median(
                            [r["magnitude"] for r in group])),
                    }
    return {"rows": rows, "summary": summary,
            "zeros": _crossings(rows, alphas)}


def _crossings(rows: list[dict], alphas) -> dict:
    """Linear interpolation of where the radial component crosses zero."""
    out = {}
    for positives_n in (64, 2048):
        for mask in (False, True):
            series = []
            for alpha in alphas:
                group = [r["radial"] for r in rows if r["alpha"] == alpha
                         and r["positives"] == positives_n
                         and r["self_mask"] is mask]
                if group:
                    series.append((alpha, float(np.median(group))))
            crossing = None
            for (a0, v0), (a1, v1) in zip(series, series[1:]):
                if v0 == 0 or (v0 > 0) != (v1 > 0):
                    crossing = a0 + (a1 - a0) * v0 / (v0 - v1)
                    break
            out[f"pos={positives_n}_mask={mask}"] = crossing
    return out


def q3_particle_shape(resolution: int, seeds: int, steps: int,
                      root: str | None, cloud_size: int = 512,
                      eta: float = 0.2) -> dict:
    """Is the deficit a SHAPE phenomenon?

    Q1 puts the field's radial zero at alpha ~ 0.88 for clouds shaped like
    the data, while the generator equilibrates at ~0.47 with a radial
    component of +0.0004 (measured in the contraction pass, N2).  Both cannot
    be true of the same shape, so the generator's cloud must not be
    data-shaped.

    This compares three clouds **at matched second moment**, so any
    difference is shape and not scale:

      `particles`     the converged free-particle cloud;
      `generator`     the converged uncorrected generator's cloud;
      `data_shaped`   real data rescaled to the same second moment.

    Prediction if the deficit is shape: at the generator's own scale its
    radial component is ~0 while a data-shaped cloud at that scale is pushed
    outward.
    """
    from .models import OneStepGenerator, sample_latent   # noqa: PLC0415
    from .config import TrainConfig                       # noqa: PLC0415
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        train, branch, kernel, rng = _setup(resolution, seed, root, GOOD_ESS)
        reference = train.sample(cloud_size, rng)
        cloud = torch.tensor(
            rng.normal(scale=0.5,
                       size=(cloud_size, 3, resolution, resolution)),
            dtype=torch.float32)
        for _ in range(steps):
            drift, _ = KG.field(cloud, train.sample(64, rng), cloud, branch,
                                kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            cloud = cloud + eta * drift

        # The uncorrected generator, trained under the identical field.
        config = TrainConfig(steps=steps, batch=64, image_size=resolution)
        model = OneStepGenerator(config.latent_dim, 3, resolution,
                                 config.width, derive_seed(seed, "generator"))
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
        torch_rng = torch.Generator().manual_seed(
            derive_seed(seed, "p10-latent") % (2 ** 31))
        for _ in range(steps):
            latent = torch.randn(256, config.latent_dim, generator=torch_rng)
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
            generated = model(sample_latent(cloud_size, config.latent_dim,
                                            derive_seed(seed, "p10-probe")))

        flat = cloud.flatten(1)
        ratio = float(flat.var(0).mean()
                      / reference.flatten(1).var(0).mean())
        generator_ratio = float(generated.flatten(1).var(0).mean()
                                / reference.flatten(1).var(0).mean())
        # Each cloud gets a data-shaped comparator at ITS own second moment.
        matched = scaled_cloud(reference, ratio)
        matched_generator = scaled_cloud(reference, generator_ratio)
        out = {"seed": seed, "second_moment_ratio": ratio,
               "generator_second_moment_ratio": generator_ratio}
        for name, sample in (("particles", cloud),
                             ("data_shaped", matched),
                             ("generator", generated),
                             ("data_shaped_at_generator_scale",
                              matched_generator)):
            radial, magnitude = [], []
            for _ in range(6):
                drift, _ = KG.field(sample, train.sample(64, rng), sample,
                                    branch, kernel, direction_mode="paper",
                                    normalization="none", diagnostics=False)
                r, m = _radial(drift, sample)
                radial.append(r)
                magnitude.append(m)
            out[f"{name}_radial"] = float(np.mean(radial))
            out[f"{name}_magnitude"] = float(np.mean(magnitude))
        # Shape statistics: nearest-neighbour spacing regularity, and the
        # spectral tail that has separated the generator from data all along.
        for name, sample in (("particles", cloud), ("data_shaped", matched),
                             ("generator", generated),
                             ("data_shaped_at_generator_scale",
                              matched_generator)):
            f = sample.flatten(1)
            distance = torch.cdist(f, f)
            distance.fill_diagonal_(float("inf"))
            nearest = distance.min(dim=1).values
            out[f"{name}_nn_mean"] = float(nearest.mean())
            out[f"{name}_nn_cv"] = float(nearest.std() / nearest.mean())
            centred = f - f.mean(dim=0, keepdim=True)
            power = torch.linalg.svdvals(centred) ** 2
            out[f"{name}_tail"] = float(power[32:].sum() / power.sum())
        rows.append(out)
        print(f"    Q3 seed{index} particles 2nd={ratio:5.3f} "
              f"radial={out['particles_radial']:+9.5f} | "
              f"generator 2nd={generator_ratio:5.3f} "
              f"radial={out['generator_radial']:+9.5f} | "
              f"data-shaped at that scale "
              f"radial={out['data_shaped_at_generator_scale_radial']:+9.5f}",
              flush=True)
    summary = {k: float(np.median([r[k] for r in rows]))
               for k in rows[0] if k != "seed"}
    return {"rows": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=("all", "q12", "q3"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase10_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    payload: dict = {
        "status": "phase9-followup-probe-feeds-no-gate",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "question": "where is the bi-softmax field's radial zero along "
                    "clouds shaped like the data, and do particles escape "
                    "it by not being data-shaped?",
    }

    if args.stage in ("all", "q12"):
        print("=== Q1/Q2: the field's radial zero ===", flush=True)
        payload["q1q2_radial_zero"] = q1q2_radial_zero(
            args.resolution, args.seeds, args.data_root)
    if args.stage in ("all", "q3"):
        print("\n=== Q3: is the particle cloud data-shaped? ===", flush=True)
        payload["q3_particle_shape"] = q3_particle_shape(
            args.resolution, args.seeds, args.steps, args.data_root)

    payload["elapsed_seconds"] = time.time() - started
    digest = write_json(args.out, payload)

    print("\n=== PHASE-9 FOLLOW-UP PROBE ===")
    if "q1q2_radial_zero" in payload:
        stage = payload["q1q2_radial_zero"]
        print("\nQ1/Q2  raw radial component vs cloud scale alpha")
        print(f"{'alpha':>7}{'pos=64 unmask':>16}{'pos=64 mask':>14}"
              f"{'pos=2048 unmask':>18}{'pos=2048 mask':>16}")
        alphas = sorted({r["alpha"] for r in stage["rows"]})
        for alpha in alphas:
            cells = []
            for positives_n in (64, 2048):
                for mask in (False, True):
                    key = f"alpha={alpha}_pos={positives_n}_mask={mask}"
                    cells.append(stage["summary"][key]["median_radial"])
            print(f"{alpha:7.2f}{cells[0]:16.5f}{cells[1]:14.5f}"
                  f"{cells[2]:18.5f}{cells[3]:16.5f}")
        print("\n  radial zero crossings (alpha):")
        for key, value in stage["zeros"].items():
            shown = f"{value:.3f}" if value is not None else "none in range"
            print(f"    {key:22} {shown}")
    if "q3_particle_shape" in payload:
        s = payload["q3_particle_shape"]["summary"]
        print("\nQ3  clouds compared at MATCHED second moment "
              "(so differences are shape, not scale)")
        print(f"    particles 2nd moment {s['second_moment_ratio']:.4f}   "
              f"generator 2nd moment {s['generator_second_moment_ratio']:.4f}")
        print(f"\n{'cloud':34}{'radial':>12}{'|V|':>11}{'nn_mean':>10}"
              f"{'nn_cv':>9}{'tail':>9}")
        for name in ("particles", "data_shaped", "generator",
                     "data_shaped_at_generator_scale"):
            print(f"{name:34}{s[name + '_radial']:+12.5f}"
                  f"{s[name + '_magnitude']:11.5f}"
                  f"{s[name + '_nn_mean']:10.4f}{s[name + '_nn_cv']:9.4f}"
                  f"{s[name + '_tail']:9.4f}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
