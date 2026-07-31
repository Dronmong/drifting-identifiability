"""Is the spread maintained by FLUCTUATION?

After Phase 8 the generator's deficit is invariant to the optimizer, the
learning rate, eta, the kernel bandwidth, the field cloud size, the latent
dimension, model capacity (36x), teacher-fitting quality (2.3x) and the
training budget (10x).  It is an equilibrium, not a rate.  Free particles
under the identical field at the identical bandwidth reach 0.998 where the
generator reaches 0.47.

In the mean-field limit the two implement the *same* flow, and a partial
step would only make the generator slower, not land it elsewhere.  So the
difference must be something the mean-field limit throws away.

**H11.** The particle cloud's spread is partly maintained by the
*fluctuation* of the finite-batch field: particles keep their realized
random displacements in their state, while the generator regresses onto a
target and averages those fluctuations out.  Better fitting and more
capacity make the averaging *more* complete, which is why they do not help
and why the width trend is if anything slightly downward.  This is the same
phenomenon the MMD-gradient-flow literature treats with noise injection.

The decisive test reverses the arrow -- instead of adding noise to the
generator, take it away from the particles:

  P1  average the field over K independent positive batches before moving
      the particles.  K = 1 is the ordinary algorithm; large K approaches
      the mean-field flow with the same expected drift and less noise.  If
      H11 holds, the particle second moment must FALL toward the
      generator's as K grows, with no other change.
  P2  the same, but averaging over K independent *negative* resamples of the
      cloud, isolating which side's fluctuation matters.
  P3  the noise budget along the trajectory: is the field noise-dominated
      where the generator equilibrates?  Phase 1 measured a noise fraction
      of ~0.001 and dismissed it, but that was measured far from the fixed
      point, where the signal is large.

Refuted if P1 is flat: then averaging the field changes nothing and
fluctuation is not what holds the cloud open.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase9
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
SEED_OFFSET = 15000
GOOD_ESS = 0.9          # the Phase-7C optimum
MOMENT_BAND = (0.7, 1.3)


def _setup(resolution: int, seed: int, root: str | None):
    train = cifar.cifar_target(resolution, "train", root)
    rng = np.random.default_rng(derive_seed(seed, "p9-setup"))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace",
                              target_ess_fraction=GOOD_ESS)
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace",
        geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
        geometry.kernel_eps, combine=geometry.combine,
        target_ess_fraction=GOOD_ESS)
    return train, branch, kernel, rng


def _in_band(value: float) -> bool:
    low, high = MOMENT_BAND
    return bool(np.isfinite(value) and low <= value <= high)


def _median(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
    return float(np.median(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# P1 / P2: take the fluctuation away from the particles
# ---------------------------------------------------------------------------


def averaged_field(cloud, train, branch, kernel, rng, batch: int,
                   repeats: int, negative_repeats: int = 1):
    """Mean of the drift over ``repeats`` independent positive batches.

    Each repeat is normalized before averaging, exactly as a single step
    would be, so the expected step length is unchanged and only its
    *variance* falls -- the whole point of the probe.  Averaging after
    normalization keeps this a pure variance reduction rather than a
    disguised change of step size.
    """
    total = torch.zeros_like(cloud)
    for _ in range(repeats):
        if negative_repeats > 1:
            # P2: average over independent resamples of the NEGATIVE side by
            # subsampling the cloud, isolating self-term fluctuation.
            inner = torch.zeros_like(cloud)
            for _ in range(negative_repeats):
                index = torch.tensor(
                    rng.choice(len(cloud), size=len(cloud) // 2,
                               replace=False))
                drift, _ = KG.field(cloud, train.sample(batch, rng),
                                    cloud[index], branch, kernel,
                                    direction_mode="paper",
                                    normalization="rms", diagnostics=False)
                inner = inner + drift
            drift = inner / negative_repeats
        else:
            drift, _ = KG.field(cloud, train.sample(batch, rng), cloud,
                                branch, kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
        total = total + drift
    return total / repeats


def p1_average_field(resolution: int, seeds: int, steps: int,
                     root: str | None, repeats=(1, 2, 4, 16, 64),
                     cloud_size: int = 512, batch: int = 64,
                     eta: float = 0.2, negative_repeats: int = 1,
                     label: str = "P1") -> dict:
    """Does removing field fluctuation collapse the particle cloud?"""
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        config = TrainConfig(steps=steps, batch=batch,
                             eval_samples=cloud_size,
                             image_size=resolution)
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())
        for k in repeats:
            train, branch, kernel, rng = _setup(resolution, seed, root)
            cloud = torch.tensor(
                rng.normal(scale=0.5,
                           size=(cloud_size, 3, resolution, resolution)),
                dtype=torch.float32)
            for _ in range(steps):
                drift = averaged_field(cloud, train, branch, kernel, rng,
                                       batch, k, negative_repeats)
                cloud = cloud + eta * drift
            measured = M.raw_metrics(
                cloud, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p9-m")), None,
                target_null=pools["null"])
            rows.append({
                "repeats": k, "seed": seed,
                "second_moment_ratio": float(
                    cloud.flatten(1).var(0).mean()) / reference_moment,
                "ed2": measured["ed2"],
                "geometry_score_v2": M.normalized_geometry_score_v2(
                    measured, null)["geometry_score"]})
            print(f"    {label} K={k:3} seed{index} "
                  f"2nd={rows[-1]['second_moment_ratio']:6.3f} "
                  f"ed2={rows[-1]['ed2']:8.4f}", flush=True)
    summary = {}
    for k in repeats:
        group = [r for r in rows if r["repeats"] == k]
        summary[f"K={k}"] = {
            "median_second_moment_ratio": _median(group,
                                                  "second_moment_ratio"),
            "median_ed2": _median(group, "ed2"),
            "in_band": _in_band(_median(group, "second_moment_ratio"))}
    values = [v["median_second_moment_ratio"] for v in summary.values()]
    return {"rows": rows, "summary": summary,
            "falls_with_averaging": bool(
                len(values) >= 2 and values[-1] < values[0] - 0.05),
            "span": [min(values), max(values)] if values else None}


# ---------------------------------------------------------------------------
# P3: the noise budget along the trajectory
# ---------------------------------------------------------------------------


def p3_noise_budget(resolution: int, seeds: int, steps: int,
                    root: str | None, cloud_size: int = 256,
                    batch: int = 64, probes=(0, 50, 150, 300, 599)) -> dict:
    """Is the field noise-dominated where the GENERATOR equilibrates?

    Phase 1 measured a drift signal-to-noise fraction of ~0.001 and used it
    to refute regression attenuation.  Near a fixed point the signal decays
    while the fluctuation does not, so that ratio is a function of where it
    is measured -- and it was measured far from equilibrium.
    """
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        train, branch, kernel, rng = _setup(resolution, seed, root)
        config = TrainConfig(steps=steps, batch=batch, field_cloud=cloud_size,
                             image_size=resolution)
        model = OneStepGenerator(config.latent_dim, 3, resolution,
                                 config.width, derive_seed(seed, "generator"))
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
        generator = torch.Generator().manual_seed(
            derive_seed(seed, "p9-latent") % (2 ** 31))
        for step in range(steps):
            positives = train.sample(batch, rng)
            latent = torch.randn(cloud_size, config.latent_dim,
                                 generator=generator)
            output = model(latent)
            with torch.no_grad():
                drift, _ = KG.field(output.detach(), positives,
                                    output.detach(), branch, kernel,
                                    direction_mode="paper",
                                    normalization="rms", diagnostics=False)
                teacher = output.detach() + 0.5 * drift
            loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step in probes:
                with torch.no_grad():
                    probe = model(latent).detach()
                    # Signal: the field averaged over many batches.
                    mean_field = averaged_field(probe, train, branch, kernel,
                                                rng, batch, 32)
                    # Noise: spread of single-batch estimates about it.
                    residuals = []
                    for _ in range(8):
                        single, _ = KG.field(
                            probe, train.sample(batch, rng), probe, branch,
                            kernel, direction_mode="paper",
                            normalization="rms", diagnostics=False)
                        residuals.append(float((single - mean_field)
                                               .pow(2).sum()))
                    signal = float(mean_field.pow(2).sum())
                    noise = float(np.mean(residuals))
                rows.append({"seed": seed, "step": step,
                             "signal": signal, "noise": noise,
                             "noise_over_signal": noise / max(signal, 1e-30)})
                print(f"    P3 seed{index} step={step:4} "
                      f"signal={signal:10.4f} noise={noise:10.4f} "
                      f"noise/signal={rows[-1]['noise_over_signal']:8.3f}",
                      flush=True)
    summary = {}
    for step in probes:
        group = [r for r in rows if r["step"] == step]
        if group:
            summary[f"step={step}"] = {
                "median_signal": _median(group, "signal"),
                "median_noise": _median(group, "noise"),
                "median_noise_over_signal": _median(group,
                                                    "noise_over_signal")}
    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------------------


def p4_subspace(resolution: int, seeds: int, steps: int, root: str | None,
                dims=(8, 32, 128, 768), cloud_size: int = 512,
                batch: int = 64, eta: float = 0.2) -> dict:
    """Does SPECTRAL CONCENTRATION set the radial balance?

    The hypothesis N4 did not actually test.  N4 swept the latent dimension
    and found the deficit flat -- but it never produced a cloud with a decent
    spectral tail: latent 512 still reached only 0.0058 of its variance
    beyond the top 32 directions, against real data's 0.1375.  So "more
    latent room" was refuted while "a cloud lacking its tail balances at a
    smaller radius" was never put to the question.

    Here free particles -- which otherwise reach 0.998 -- are confined to a
    fixed ``k``-dimensional subspace spanned by the DATA's leading principal
    directions, reproducing the generator's spectral concentration with
    nothing else changed.  The second moment is scored against the data's
    variance *inside the same subspace*, so a shrunken answer cannot be an
    artifact of comparing a projected cloud with an unprojected target.

    If the relative second moment falls as ``k`` falls, spectral
    concentration is the mechanism and the generator's deficit follows from
    the shape of the cloud it can produce.  A flat result refutes it.
    """
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        config = TrainConfig(steps=steps, batch=batch,
                             eval_samples=cloud_size, image_size=resolution)
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        target = pools["eval"].flatten(1)
        centre = target.mean(dim=0, keepdim=True)
        _, _, right = torch.linalg.svd(target - centre, full_matrices=False)
        for k in dims:
            train, branch, kernel, rng = _setup(resolution, seed, root)
            basis = right[:k]                       # k x d, orthonormal rows
            # The data's own variance inside this subspace is the reference.
            projected_target = (target - centre) @ basis.T
            reference_moment = float(projected_target.var(0).mean() * k
                                     / target.shape[1])

            def confine(flat: torch.Tensor) -> torch.Tensor:
                return centre + ((flat - centre) @ basis.T) @ basis

            cloud = torch.tensor(
                rng.normal(scale=0.5,
                           size=(cloud_size, 3, resolution, resolution)),
                dtype=torch.float32)
            cloud = confine(cloud.flatten(1)).reshape(cloud.shape)
            for _ in range(steps):
                drift, _ = KG.field(cloud, train.sample(batch, rng), cloud,
                                    branch, kernel, direction_mode="paper",
                                    normalization="rms", diagnostics=False)
                moved = (cloud + eta * drift).flatten(1)
                cloud = confine(moved).reshape(cloud.shape)
            measured = M.raw_metrics(
                cloud, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p9-p4")), None,
                target_null=pools["null"])
            flat = cloud.flatten(1)
            rows.append({
                "dims": k, "seed": seed,
                "second_moment_ratio": float(flat.var(0).mean())
                / max(reference_moment, 1e-12),
                "raw_second_moment_ratio": float(flat.var(0).mean())
                / float(target.var(0).mean()),
                "ed2": measured["ed2"],
                "geometry_score_v2": M.normalized_geometry_score_v2(
                    measured, null)["geometry_score"]})
            print(f"    P4 k={k:4} seed{index} "
                  f"2nd_in_subspace={rows[-1]['second_moment_ratio']:6.3f} "
                  f"2nd_raw={rows[-1]['raw_second_moment_ratio']:6.3f} "
                  f"ed2={rows[-1]['ed2']:8.4f}", flush=True)
    summary = {}
    for k in dims:
        group = [r for r in rows if r["dims"] == k]
        summary[f"k={k}"] = {
            "median_second_moment_ratio": _median(group,
                                                  "second_moment_ratio"),
            "median_raw_second_moment_ratio": _median(
                group, "raw_second_moment_ratio"),
            "median_ed2": _median(group, "ed2"),
            "in_band": _in_band(_median(group, "second_moment_ratio"))}
    values = [v["median_second_moment_ratio"] for v in summary.values()]
    return {"rows": rows, "summary": summary,
            "falls_with_confinement": bool(
                len(values) >= 2 and values[0] < values[-1] - 0.1),
            "span": [min(values), max(values)] if values else None}


def p5_generator_averaging(resolution: int, seeds: int, steps: int,
                           root: str | None, repeats=(1, 4, 16),
                           cloud_size: int = 256, batch: int = 64,
                           eta: float = 0.5) -> dict:
    """The differential test P3 implies.

    Same knob as P1 -- average the field over K independent positive batches
    -- but applied to the GENERATOR instead of the particles.  P1 already
    showed it is flat for particles (1.012 -> 1.006 over a 64x variance
    reduction), so this is a clean differential: identical intervention,
    and the two systems should respond differently only if the *regression*
    is where the noise does its damage.

    P3 motivates it.  At the generator's equilibrium the field is
    noise-dominated -- noise/signal 4.459 at step 599, against 0.007 at
    initialization, which is where Phase 1 measured the 0.001 figure it used
    to dismiss regression attenuation.  The noise is not independent of the
    predictor either: V is evaluated *at* the cloud the generator produces,
    which is the errors-in-variables situation that attenuates a fit.

    Prediction: the generator's second moment rises with K.  A flat result
    refutes target noise as the mechanism and leaves the deficit unexplained
    by anything measured so far.
    """
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        config = TrainConfig(steps=steps, batch=batch,
                             field_cloud=cloud_size, eval_samples=512,
                             image_size=resolution)
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())
        probe = torch.randn(512, config.latent_dim,
                            generator=torch.Generator().manual_seed(
                                derive_seed(seed, "p9-probe") % (2 ** 31)))
        for k in repeats:
            train, branch, kernel, rng = _setup(resolution, seed, root)
            model = OneStepGenerator(config.latent_dim, 3, resolution,
                                     config.width,
                                     derive_seed(seed, "generator"))
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config.learning_rate)
            generator = torch.Generator().manual_seed(
                derive_seed(seed, "p9-latent") % (2 ** 31))
            for _ in range(steps):
                latent = torch.randn(cloud_size, config.latent_dim,
                                     generator=generator)
                output = model(latent)
                with torch.no_grad():
                    drift = averaged_field(output.detach(), train, branch,
                                           kernel, rng, batch, k)
                    teacher = output.detach() + eta * drift
                loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                generated = model(probe)
            measured = M.raw_metrics(
                generated, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p9-p5")), None,
                target_null=pools["null"])
            rows.append({
                "repeats": k, "seed": seed,
                "second_moment_ratio": float(
                    generated.flatten(1).var(0).mean()) / reference_moment,
                "ed2": measured["ed2"],
                "geometry_score_v2": M.normalized_geometry_score_v2(
                    measured, null)["geometry_score"]})
            print(f"    P5 K={k:3} seed{index} "
                  f"2nd={rows[-1]['second_moment_ratio']:6.3f} "
                  f"ed2={rows[-1]['ed2']:8.4f}", flush=True)
    summary = {}
    for k in repeats:
        group = [r for r in rows if r["repeats"] == k]
        summary[f"K={k}"] = {
            "median_second_moment_ratio": _median(group,
                                                  "second_moment_ratio"),
            "median_ed2": _median(group, "ed2"),
            "in_band": _in_band(_median(group, "second_moment_ratio"))}
    values = [v["median_second_moment_ratio"] for v in summary.values()]
    return {"rows": rows, "summary": summary,
            "rises_with_averaging": bool(
                len(values) >= 2 and values[-1] > values[0] + 0.05),
            "span": [min(values), max(values)] if values else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=("all", "p1", "p2", "p3", "p4", "p5"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase9_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    payload: dict = {
        "status": "phase8-followup-probe-feeds-no-gate",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "bandwidth": f"target_ess={GOOD_ESS} (Phase 7C optimum)",
        "hypothesis": "H11: the particle cloud's spread is maintained partly "
                      "by finite-batch field fluctuation, which the "
                      "generator's regression averages away",
    }

    if args.stage in ("all", "p1"):
        print("=== P1: average the field, watch the particles ===", flush=True)
        payload["p1_average_field"] = p1_average_field(
            args.resolution, args.seeds, args.steps, args.data_root)
    if args.stage in ("all", "p2"):
        print("\n=== P2: average the NEGATIVE side only ===", flush=True)
        payload["p2_negative_average"] = p1_average_field(
            args.resolution, args.seeds, args.steps, args.data_root,
            repeats=(1, 4, 16), negative_repeats=4, label="P2")
    if args.stage in ("all", "p3"):
        print("\n=== P3: noise budget along the trajectory ===", flush=True)
        payload["p3_noise_budget"] = p3_noise_budget(
            args.resolution, args.seeds, args.steps, args.data_root)
    if args.stage in ("all", "p5"):
        print("\n=== P5: average the field for the GENERATOR ===", flush=True)
        payload["p5_generator_averaging"] = p5_generator_averaging(
            args.resolution, args.seeds, args.steps, args.data_root)
    if args.stage in ("all", "p4"):
        print("\n=== P4: confine the particles to a k-dim subspace ===",
              flush=True)
        payload["p4_subspace"] = p4_subspace(
            args.resolution, args.seeds, args.steps, args.data_root)

    payload["elapsed_seconds"] = time.time() - started
    digest = write_json(args.out, payload)

    print("\n=== PHASE-8 FOLLOW-UP PROBE ===")
    for key, title, flag in (
            ("p1_average_field", "P1  particles, positive-batch averaging",
             "falls_with_averaging"),
            ("p2_negative_average", "P2  particles, negative-side averaging",
             "falls_with_averaging"),
            ("p5_generator_averaging", "P5  GENERATOR, positive-batch "
             "averaging", "rises_with_averaging")):
        if key in payload:
            print(f"\n{title}")
            for name, entry in payload[key]["summary"].items():
                print(f"    {name:8} "
                      f"2nd={entry['median_second_moment_ratio']:6.3f} "
                      f"ed2={entry['median_ed2']:8.4f}  "
                      f"{'in ' if entry['in_band'] else 'out'}")
            print(f"    -> {flag}: {payload[key][flag]}  "
                  f"span={[round(x, 3) for x in payload[key]['span']]}")
    if "p4_subspace" in payload:
        print("\nP4  particles confined to k data directions")
        print(f"    {'k':>6}{'2nd (in subspace)':>20}{'2nd (raw)':>12}"
              f"{'ed2':>10}")
        for name, entry in payload["p4_subspace"]["summary"].items():
            print(f"    {name:>6}"
                  f"{entry['median_second_moment_ratio']:20.3f}"
                  f"{entry['median_raw_second_moment_ratio']:12.3f}"
                  f"{entry['median_ed2']:10.4f}")
        print(f"    -> falls with confinement: "
              f"{payload['p4_subspace']['falls_with_confinement']}  "
              f"span={[round(x, 3) for x in payload['p4_subspace']['span']]}")
    if "p3_noise_budget" in payload:
        print(f"\nP3  {'step':>8}{'signal':>12}{'noise':>12}"
              f"{'noise/signal':>14}")
        for name, entry in payload["p3_noise_budget"]["summary"].items():
            print(f"    {name:>8}{entry['median_signal']:12.4f}"
                  f"{entry['median_noise']:12.4f}"
                  f"{entry['median_noise_over_signal']:14.3f}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
