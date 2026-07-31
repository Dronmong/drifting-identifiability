"""Root-cause investigation after three refuted mechanisms.

Refuted so far: minibatch-noise regression attenuation; anisotropic
contraction of the teacher map; per-application contraction along the
trajectory.  All three asked what the *teacher* does.  The surviving
phenomenology says the teacher does nothing wrong -- it preserves effective
dimension to 0.997 per application -- while the generator still falls from
~13.9 to ~2.2.

So the question was aimed at the wrong object.  The stop-gradient loss
``|f - sg(f + eta V)|^2`` has gradient ``-2 eta V / n``: it is a device for
injecting ``V`` as a gradient, and the generator can only move along
directions its Jacobian can produce.  What actually reaches the cloud is the
**projection of the field onto the model's tangent space**, and that is the
one object nobody has measured.

  J1  time course -- when does the collapse happen?
  J2  is it the optimizer (Adam vs SGD, learning rate)?
  J3  tangent-space realization: of the change the field asks for, how much
      is realized, per eigendirection of the cloud?
  J4  does fitting each teacher to convergence change anything?

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase6
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from . import metrics as M
from .config import GeometryConfig, MASTER_SEED, TrainConfig, derive_seed
from .diagnostics import provenance, write_json
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import variance_matched_teacher

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 6000


def _setup(resolution: int, seed: int, root: str | None):
    train = cifar.cifar_target(resolution, "train", root)
    rng = np.random.default_rng(derive_seed(seed, "j-setup"))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace")
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace",
        geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
        geometry.kernel_eps, combine=geometry.combine,
        target_ess_fraction=geometry.target_ess_fraction)
    return train, branch, kernel, rng


def _teacher(output: torch.Tensor, positives: torch.Tensor, branch, kernel,
             eta: float, variance_match: bool) -> torch.Tensor:
    with torch.no_grad():
        drift, _ = KG.field(output.detach(), positives, output.detach(),
                            branch, kernel, direction_mode="paper",
                            normalization="rms", diagnostics=False)
        teacher = output.detach() + eta * drift
        if variance_match:
            teacher = variance_matched_teacher(teacher, positives)
    return teacher


# ---------------------------------------------------------------------------
# J1: when does the collapse happen?
# ---------------------------------------------------------------------------


def j1_time_course(resolution: int, steps: int, root: str | None,
                   every: int = 10) -> dict:
    """Is the collapse sudden and early, or gradual?

    A basic question nobody has asked.  Sudden collapse points at
    initialization or the first few updates; gradual decay points at a
    compounding process.
    """
    rows = []
    for variance_match in (False, True):
        seed = MASTER_SEED + SEED_OFFSET
        train, branch, kernel, rng = _setup(resolution, seed, root)
        config = TrainConfig(steps=steps, batch=64, image_size=resolution)
        model = OneStepGenerator(config.latent_dim, 3, resolution,
                                 config.width, derive_seed(seed, "generator"))
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
        generator = torch.Generator().manual_seed(
            derive_seed(seed, "latent") % (2 ** 31))
        probe = sample_latent(256, config.latent_dim,
                              derive_seed(seed, "probe"))
        for step in range(steps + 1):
            if step % every == 0:
                with torch.no_grad():
                    rows.append({
                        "r11": variance_match, "step": step,
                        "effective_dimension": M.effective_dimension(
                            model(probe)),
                        "output_rms": float(model(probe).pow(2).mean()
                                            .sqrt()),
                    })
            if step == steps:
                break
            positives = train.sample(config.batch, rng)
            latent = torch.randn(config.batch, config.latent_dim,
                                 generator=generator)
            output = model(latent)
            teacher = _teacher(output, positives, branch, kernel, 0.5,
                               variance_match)
            loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        trace = [r for r in rows if r["r11"] is variance_match]
        print(f"    J1 R11={variance_match!s:5} eff_dim: "
              + " ".join(f"{r['step']}:{r['effective_dimension']:.1f}"
                         for r in trace[::max(len(trace) // 8, 1)]),
              flush=True)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# J2: is it the optimizer?
# ---------------------------------------------------------------------------


def j2_optimizer(resolution: int, steps: int, root: str | None) -> dict:
    """Adam normalizes per-parameter step sizes; SGD does not."""
    rows = []
    for name, make, lr in (("adam", torch.optim.Adam, 2e-3),
                           ("adam_lo", torch.optim.Adam, 5e-4),
                           ("sgd", torch.optim.SGD, 2e-3),
                           ("sgd_hi", torch.optim.SGD, 5e-2)):
        seed = MASTER_SEED + SEED_OFFSET
        train, branch, kernel, rng = _setup(resolution, seed, root)
        config = TrainConfig(steps=steps, batch=64, image_size=resolution)
        model = OneStepGenerator(config.latent_dim, 3, resolution,
                                 config.width, derive_seed(seed, "generator"))
        optimizer = make(model.parameters(), lr=lr)
        generator = torch.Generator().manual_seed(
            derive_seed(seed, "latent") % (2 ** 31))
        for _ in range(steps):
            positives = train.sample(config.batch, rng)
            latent = torch.randn(config.batch, config.latent_dim,
                                 generator=generator)
            output = model(latent)
            teacher = _teacher(output, positives, branch, kernel, 0.5, False)
            loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        probe = sample_latent(512, config.latent_dim,
                              derive_seed(seed, "probe"))
        with torch.no_grad():
            generated = model(probe)
        rows.append({
            "optimizer": name, "lr": lr,
            "effective_dimension": M.effective_dimension(generated),
            "output_rms": float(generated.pow(2).mean().sqrt()),
        })
        print(f"    J2 {name:8} lr={lr:<7} eff_dim="
              f"{rows[-1]['effective_dimension']:6.2f} "
              f"rms={rows[-1]['output_rms']:6.3f}", flush=True)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# J3: what fraction of the requested change is actually realized?
# ---------------------------------------------------------------------------


def j3_tangent_realization(resolution: int, steps: int, root: str | None,
                           probes=(0, 50, 150, 300, 599)) -> dict:
    """The measurement the previous three hypotheses never made.

    At several points along training, compare the change the field *asks*
    for with the change the optimizer actually *delivers*, resolved by
    eigendirection of the current output cloud.  If low-variance directions
    are systematically under-realized, the parametrization -- not the
    teacher -- is what collapses the cloud.
    """
    rows = []
    for variance_match in (False, True):
        seed = MASTER_SEED + SEED_OFFSET
        train, branch, kernel, rng = _setup(resolution, seed, root)
        config = TrainConfig(steps=steps, batch=64, image_size=resolution)
        model = OneStepGenerator(config.latent_dim, 3, resolution,
                                 config.width, derive_seed(seed, "generator"))
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
        generator = torch.Generator().manual_seed(
            derive_seed(seed, "latent") % (2 ** 31))
        for step in range(steps):
            positives = train.sample(config.batch, rng)
            latent = torch.randn(config.batch, config.latent_dim,
                                 generator=generator)
            output = model(latent)
            teacher = _teacher(output, positives, branch, kernel, 0.5,
                               variance_match)

            if step in probes:
                before = output.detach().clone()
                snapshot = copy.deepcopy(model.state_dict())
                optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
                # Trial step on its own graph, then rewind model *and*
                # optimizer so the probe does not perturb the trajectory.
                trial = ((model(latent) - teacher) ** 2).flatten(1).sum(
                    1).mean()
                optimizer.zero_grad(set_to_none=True)
                trial.backward()
                optimizer.step()
                with torch.no_grad():
                    after = model(latent)
                model.load_state_dict(snapshot)
                optimizer.load_state_dict(optimizer_snapshot)

                flat_before = before.reshape(len(before), -1).double()
                desired = (teacher.reshape(len(teacher), -1).double()
                           - flat_before)
                realized = (after.reshape(len(after), -1).double()
                            - flat_before)
                centred = flat_before - flat_before.mean(0, keepdim=True)
                _, singular, right = torch.linalg.svd(
                    centred, full_matrices=False)
                # Project both onto the cloud's own eigendirections.
                desired_k = (desired @ right.T).pow(2).sum(0).sqrt()
                realized_k = (realized @ right.T).pow(2).sum(0).sqrt()
                ratio = (realized_k
                         / desired_k.clamp_min(1e-30)).cpu().numpy()
                rank = min(len(singular), len(ratio))
                quarter = max(rank // 4, 1)
                rows.append({
                    "r11": variance_match, "step": step,
                    "global_realization": float(
                        realized.norm() / desired.norm().clamp_min(1e-30)),
                    "top_quartile_realization": float(
                        np.median(ratio[:quarter])),
                    "bottom_quartile_realization": float(
                        np.median(ratio[rank - quarter:rank])),
                    "effective_dimension": M.effective_dimension(before),
                })
                rows[-1]["realization_anisotropy"] = (
                    rows[-1]["top_quartile_realization"]
                    / max(rows[-1]["bottom_quartile_realization"], 1e-30))
                print(f"    J3 R11={variance_match!s:5} step={step:4} "
                      f"global={rows[-1]['global_realization']:7.5f} "
                      f"top={rows[-1]['top_quartile_realization']:7.5f} "
                      f"bottom={rows[-1]['bottom_quartile_realization']:7.5f} "
                      f"anisotropy={rows[-1]['realization_anisotropy']:7.2f}",
                      flush=True)

            # Recomputed rather than reusing `output`, whose graph the probe
            # branch may already have consumed.
            loss = ((model(latent) - teacher) ** 2).flatten(1).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return {"rows": rows}


# ---------------------------------------------------------------------------
# J4: does fitting each teacher to convergence change anything?
# ---------------------------------------------------------------------------


def j4_inner_loop(resolution: int, steps: int, root: str | None,
                  inner=(1, 8, 32)) -> dict:
    """If the generator fully reached each teacher, would it still collapse?

    Distinguishes "the optimizer is too slow to follow the field" from "the
    fixed point itself is low dimensional".
    """
    rows = []
    for inner_steps in inner:
        seed = MASTER_SEED + SEED_OFFSET
        train, branch, kernel, rng = _setup(resolution, seed, root)
        config = TrainConfig(steps=steps, batch=64, image_size=resolution)
        model = OneStepGenerator(config.latent_dim, 3, resolution,
                                 config.width, derive_seed(seed, "generator"))
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
        generator = torch.Generator().manual_seed(
            derive_seed(seed, "latent") % (2 ** 31))
        outer = max(steps // inner_steps, 1)
        for _ in range(outer):
            positives = train.sample(config.batch, rng)
            latent = torch.randn(config.batch, config.latent_dim,
                                 generator=generator)
            with torch.no_grad():
                frozen = model(latent)
            teacher = _teacher(frozen, positives, branch, kernel, 0.5, False)
            for _ in range(inner_steps):
                loss = ((model(latent) - teacher) ** 2).flatten(1).sum(
                    1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                reached = float(
                    (model(latent) - teacher).norm()
                    / (frozen - teacher).norm().clamp_min(1e-30))
        probe = sample_latent(512, config.latent_dim,
                              derive_seed(seed, "probe"))
        with torch.no_grad():
            generated = model(probe)
        rows.append({
            "inner_steps": inner_steps, "outer_steps": outer,
            "residual_to_teacher": reached,
            "effective_dimension": M.effective_dimension(generated),
        })
        print(f"    J4 inner={inner_steps:3} outer={outer:4} "
              f"residual_to_teacher={reached:6.4f} "
              f"eff_dim={rows[-1]['effective_dimension']:6.2f}", flush=True)
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--only", type=str, default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase6_rootcause.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if not cifar.available(args.root):
        raise SystemExit("CIFAR-10 is not present locally.")

    started = time.time()
    stages = {
        "J1_time_course": lambda: j1_time_course(
            args.resolution, args.steps, args.root),
        "J3_tangent_realization": lambda: j3_tangent_realization(
            args.resolution, args.steps, args.root),
        "J2_optimizer": lambda: j2_optimizer(
            args.resolution, args.steps, args.root),
        "J4_inner_loop": lambda: j4_inner_loop(
            args.resolution, args.steps, args.root),
    }
    wanted = set(stages) if args.only == "all" else set(args.only.split(","))
    results = {}
    for name, function in stages.items():
        if name in wanted:
            print(f"--- {name} ---", flush=True)
            results[name] = function()

    payload = {
        "status": "phase6-root-cause-investigation-feeds-no-gate",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    digest = write_json(args.out, payload)
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
