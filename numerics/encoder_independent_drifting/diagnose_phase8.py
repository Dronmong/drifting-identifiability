"""Why does the GENERATOR contract, when the field does not?

Phase 7 sharpened the question to a point.  The field's deficit is
bandwidth-controlled -- free particles at ess = 0.9 reach a second moment of
0.998 and ED2 0.0749.  The generator's is **bandwidth-independent**: 0.101 to
0.443 across 36 runs spanning realized ESS 0.82-0.99, three cloud sizes and
three seeds.  It is also not the optimizer (6A), not the step size (R24 --
eta is inert), not the cloud size (7A), and five direct hypotheses about the
stop-gradient regression were refuted in Phases 3-5.

One structural fact has never been tested.  Free particles converge where
the drift vanishes, ``V = 0``.  The generator converges where the drift's
projection onto the model's tangent space vanishes, ``P_T(V) = 0``, which is
strictly weaker: at the generator's equilibrium the field may still be
pushing hard, as long as it pushes in directions the model cannot realize.

Crucially, the generator is **capable** of the right answer -- R11 drives it
to a second moment of ~1.0 with the same architecture and no extra capacity,
and the head is a plain convolution with no output nonlinearity, so scale is
unbounded.  So this is not a capacity limit.  It is a question about where
the objective's equilibrium sits.

  N1  at convergence, is the field still asking the cloud to EXPAND?
      Measure the mean radial component of V.  If it is positive at
      equilibrium, the field wants a bigger cloud and the generator is
      declining.
  N2  **the decisive test.**  Add ONE learnable scalar output gain, putting
      pure dilation explicitly into the tangent space at a cost of one
      parameter.  If the deficit is that expansion is unrealizable or gets
      out-competed in the shared parameters, the gain grows and repairs the
      second moment.  If the gain stays ~1 and the cloud stays shrunk, then
      dilation IS realizable and the objective's equilibrium genuinely
      prefers a shrunken cloud -- which would mean R11 overrides the
      objective rather than helping it converge.
  N3  Phase 4's `fit_to_free` probe said point-target regression alone costs
      ~2x, but was underparameterized by 3.6x and said so.  Sweep capacity at
      the GOOD bandwidth and find out whether that cost is a capacity
      artifact or intrinsic.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase8
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
from .objectives import corrected_teacher

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 13000

# Phase 7C located the quality optimum here; every probe uses it, so nothing
# below is confounded by the badly-set bandwidth of Phases 2-6.
GOOD_ESS = 0.9


class GainGenerator(torch.nn.Module):
    """The same generator with one learnable scalar output gain.

    ``f(z) = g * net(z)`` with ``g`` initialized at 1.  This adds pure
    dilation to the model's tangent space at a cost of a single parameter,
    so "the cloud cannot expand" and "the cloud will not expand" become
    distinguishable.
    """

    def __init__(self, base: OneStepGenerator) -> None:
        super().__init__()
        self.base = base
        self.gain = torch.nn.Parameter(torch.ones(()))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.gain * self.base(latent)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _setup(resolution: int, seed: int, root: str | None, ess: float):
    train = cifar.cifar_target(resolution, "train", root)
    rng = np.random.default_rng(derive_seed(seed, "p8-setup"))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace",
                              target_ess_fraction=ess)
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace",
        geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
        geometry.kernel_eps, combine=geometry.combine,
        target_ess_fraction=ess)
    return train, branch, kernel, rng


def _radial_component(drift: torch.Tensor,
                      cloud: torch.Tensor) -> dict:
    """How much of the field points outward from the cloud's own centre?

    Positive means the field is asking the cloud to grow.  Reported both as
    a mean projection and as the fraction of samples pushed outward, so a
    few large outliers cannot manufacture the conclusion.
    """
    flat = cloud.reshape(len(cloud), -1)
    flat_drift = drift.reshape(len(drift), -1)
    radius = flat - flat.mean(dim=0, keepdim=True)
    norm = radius.norm(dim=1, keepdim=True).clamp_min(1e-12)
    unit = radius / norm
    projection = (flat_drift * unit).sum(dim=1)
    return {
        "radial_mean": float(projection.mean()),
        "radial_relative": float(projection.mean()
                                 / flat_drift.norm(dim=1).mean()
                                 .clamp_min(1e-12)),
        "outward_fraction": float((projection > 0).to(torch.float32).mean()),
    }


def _tail_fraction(samples: torch.Tensor, keep: int) -> float:
    """Fraction of variance living beyond the top ``keep`` directions.

    H10 says the generator's cloud is confined to a low-dimensional manifold
    (latent 32 into 768 pixels), so it is missing the data's spectral tail;
    a cloud without a tail reaches the field's density balance at a smaller
    radius.  Phase 4 measured real CIFAR-16 at 13.4% beyond the top 32
    against a corrected generator's 4.7%, which is the observation this
    quantifies systematically.
    """
    flat = samples.reshape(len(samples), -1)
    centred = flat - flat.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centred)
    power = (singular ** 2)
    total = float(power.sum())
    if total <= 0:
        return float("nan")
    return float(power[keep:].sum()) / total


def _train(model, train, branch, kernel, config, seed, *, correction: str,
           steps: int, rng, log_gain: bool = False) -> dict:
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate)
    generator = torch.Generator().manual_seed(
        derive_seed(seed, "p8-latent") % (2 ** 31))
    cloud = config.field_cloud or config.batch
    trace = []
    for step in range(steps):
        positives = train.sample(config.batch, rng)
        latent = torch.randn(cloud, config.latent_dim, generator=generator)
        output = model(latent)
        with torch.no_grad():
            drift, _ = KG.field(output.detach(), positives, output.detach(),
                                branch, kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            teacher = output.detach() + 0.5 * drift
            if correction != "none":
                teacher = corrected_teacher(teacher, positives,
                                            mode=correction)
        loss = ((output - teacher) ** 2).flatten(1).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if log_gain and step % max(steps // 30, 1) == 0:
            trace.append({"step": step,
                          "gain": float(model.gain.detach())})
    return {"trace": trace}


# ---------------------------------------------------------------------------
# N1 / N2
# ---------------------------------------------------------------------------


def n1n2_gain(resolution: int, seeds: int, steps: int, root: str | None,
              cloud: int = 256) -> dict:
    """Does the field still ask for expansion, and does one gain deliver it?"""
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        config = TrainConfig(steps=steps, batch=64, field_cloud=cloud,
                             eval_samples=512, image_size=resolution)
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())
        for variant in ("plain", "gain", "plain_r11"):
            train, branch, kernel, rng = _setup(resolution, seed, root,
                                                GOOD_ESS)
            base = OneStepGenerator(config.latent_dim, 3, resolution,
                                    config.width,
                                    derive_seed(seed, "generator"))
            model = GainGenerator(base) if variant == "gain" else base
            correction = "scalar" if variant == "plain_r11" else "none"
            result = _train(model, train, branch, kernel, config, seed,
                            correction=correction, steps=steps, rng=rng,
                            log_gain=(variant == "gain"))
            probe = sample_latent(config.eval_samples, config.latent_dim,
                                  derive_seed(seed, "p8-probe"))
            with torch.no_grad():
                generated = model(probe)
            # N1: at the converged model, what is the field still asking for?
            with torch.no_grad():
                field_probe = generated[:cloud]
                drift, _ = KG.field(field_probe,
                                    train.sample(64, rng), field_probe,
                                    branch, kernel, direction_mode="paper",
                                    normalization="rms", diagnostics=False)
                radial = _radial_component(drift, field_probe)
            measured = M.raw_metrics(
                generated, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p8-m")), None,
                target_null=pools["null"])
            row = {"variant": variant, "seed": seed,
                   "second_moment_ratio": float(
                       generated.flatten(1).var(0).mean()) / reference_moment,
                   "tail_fraction": _tail_fraction(generated,
                                                   config.latent_dim),
                   "reference_tail_fraction": _tail_fraction(
                       pools["eval"], config.latent_dim),
                   "ed2": measured["ed2"],
                   "geometry_score_v2": M.normalized_geometry_score_v2(
                       measured, null)["geometry_score"],
                   "final_gain": (float(model.gain.detach())
                                  if variant == "gain" else None),
                   "gain_trace": result["trace"]}
            row.update(radial)
            rows.append(row)
            print(f"    N1/N2 {variant:10} seed{index} "
                  f"2nd={row['second_moment_ratio']:6.3f} "
                  f"ed2={row['ed2']:8.4f} "
                  f"radial_rel={row['radial_relative']:+7.4f} "
                  f"outward={row['outward_fraction']:5.3f}"
                  + (f" gain={row['final_gain']:6.3f}"
                     if row["final_gain"] is not None else ""), flush=True)
    summary = {}
    for variant in ("plain", "gain", "plain_r11"):
        group = [r for r in rows if r["variant"] == variant]
        summary[variant] = {
            "median_second_moment_ratio": float(np.median(
                [r["second_moment_ratio"] for r in group])),
            "median_ed2": float(np.median([r["ed2"] for r in group])),
            "median_radial_relative": float(np.median(
                [r["radial_relative"] for r in group])),
            "median_outward_fraction": float(np.median(
                [r["outward_fraction"] for r in group])),
            "median_tail_fraction": float(np.median(
                [r["tail_fraction"] for r in group])),
            "reference_tail_fraction": float(np.median(
                [r["reference_tail_fraction"] for r in group])),
            "median_final_gain": (
                float(np.median([r["final_gain"] for r in group]))
                if group[0]["final_gain"] is not None else None)}
    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------------------
# N3: is the regression cost a capacity artifact?
# ---------------------------------------------------------------------------


def n3_capacity(resolution: int, seeds: int, steps: int, root: str | None,
                fits=((64, 64), (64, 128), (64, 256), (64, 512),
                      (128, 256)),
                cloud: int = 512, fit_steps: int = 800) -> dict:
    """Phase 4 said point-target regression costs ~2x, and was 3.6x short.

    Its probe asked 110k parameters to reproduce 512 x 768 = 393k values.
    The cheap way to cross the parameters/values line is to shrink the
    TARGET SET rather than grow the model -- growing width to 256 at 512
    targets is ~290 GFLOP per step, which is hours of CPU for one fit.

    Each arm is ``(width, targets)``.  A fit is compared against the free
    particles it was fitted to, at the same count, so "regression costs
    something" is never confounded with "a smaller cloud is a worse sample".
    If the cost vanishes once parameters exceed values, Phase 4's figure was
    a memorization artifact; if it survives, point-target regression is
    intrinsically contractive.
    """
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        config = TrainConfig(steps=steps, batch=64, eval_samples=cloud,
                             image_size=resolution)
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())
        train, branch, kernel, rng = _setup(resolution, seed, root, GOOD_ESS)

        particles = torch.tensor(
            rng.normal(scale=0.5, size=(cloud, 3, resolution, resolution)),
            dtype=torch.float32)
        for _ in range(steps):
            drift, _ = KG.field(particles, train.sample(64, rng), particles,
                                branch, kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            particles = particles + 0.2 * drift

        def record(name: str, sample: torch.Tensor, parameters: int,
                   targets: int) -> None:
            measured = M.raw_metrics(
                sample, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p8-n3")), None,
                target_null=pools["null"])
            values = targets * 3 * resolution * resolution
            rows.append({
                "regime": name, "seed": seed, "parameters": parameters,
                "targets": targets, "values": values,
                "params_per_value": (parameters / values if values else
                                     float("nan")),
                "second_moment_ratio": float(
                    sample.flatten(1).var(0).mean()) / reference_moment,
                "ed2": measured["ed2"],
                "geometry_score_v2": M.normalized_geometry_score_v2(
                    measured, null)["geometry_score"]})
            print(f"    N3 {name:22} seed{index} "
                  f"p/v={rows[-1]['params_per_value']:6.2f} "
                  f"2nd={rows[-1]['second_moment_ratio']:6.3f} "
                  f"ed2={rows[-1]['ed2']:8.4f}", flush=True)

        for width, targets in fits:
            # Control: the free particles this fit is asked to reproduce.
            record(f"particles_n={targets}", particles[:targets], 0, targets)
            model = OneStepGenerator(config.latent_dim, 3, resolution, width,
                                     derive_seed(seed, "n3-generator"))
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
            latent = sample_latent(targets, config.latent_dim,
                                   derive_seed(seed, "n3-latent"))
            for _ in range(fit_steps):
                loss = ((model(latent) - particles[:targets]) ** 2
                        ).flatten(1).sum(dim=1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                fitted = model(latent)
            record(f"fit_w={width}_n={targets}", fitted,
                   model.parameter_count(), targets)
    summary = {}
    for name in sorted({r["regime"] for r in rows}):
        group = [r for r in rows if r["regime"] == name]
        summary[name] = {
            "parameters": group[0]["parameters"],
            "targets": group[0]["targets"],
            "params_per_value": group[0]["params_per_value"],
            "median_second_moment_ratio": float(np.median(
                [r["second_moment_ratio"] for r in group])),
            "median_ed2": float(np.median([r["ed2"] for r in group]))}
    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------------------


def n4_latent(resolution: int, seeds: int, steps: int, root: str | None,
              latents=(8, 32, 128, 512), cloud: int = 256) -> dict:
    """H10's direct test: does giving the manifold more room reduce it?

    R17 already swept latent dimension and found no effect -- but at ESS 0.5,
    before the bandwidth optimum was known, and reporting effective dimension
    rather than the second moment.  Both reasons to look again.

    If H10 is right, the deficit should shrink as the latent dimension
    approaches the ambient 768.  A flat result refutes H10 and confirms R17
    at the good bandwidth, which is worth having either way.
    """
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        for latent_dim in latents:
            config = TrainConfig(steps=steps, batch=64, field_cloud=cloud,
                                 eval_samples=512, image_size=resolution,
                                 latent_dim=latent_dim)
            pools = evaluation_pools(evaluation, config, seed)
            null = null_reference(evaluation, pools, seed)
            reference_moment = float(pools["eval"].flatten(1).var(0).mean())
            train, branch, kernel, rng = _setup(resolution, seed, root,
                                                GOOD_ESS)
            model = OneStepGenerator(latent_dim, 3, resolution, config.width,
                                     derive_seed(seed, "generator"))
            _train(model, train, branch, kernel, config, seed,
                   correction="none", steps=steps, rng=rng)
            probe = sample_latent(config.eval_samples, latent_dim,
                                  derive_seed(seed, "p8-probe"))
            with torch.no_grad():
                generated = model(probe)
            measured = M.raw_metrics(
                generated, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p8-n4")), None,
                target_null=pools["null"])
            rows.append({
                "latent_dim": latent_dim, "seed": seed,
                "second_moment_ratio": float(
                    generated.flatten(1).var(0).mean()) / reference_moment,
                "tail_fraction": _tail_fraction(generated, 32),
                "ed2": measured["ed2"],
                "geometry_score_v2": M.normalized_geometry_score_v2(
                    measured, null)["geometry_score"]})
            print(f"    N4 latent={latent_dim:4} seed{index} "
                  f"2nd={rows[-1]['second_moment_ratio']:6.3f} "
                  f"tail={rows[-1]['tail_fraction']:6.4f} "
                  f"ed2={rows[-1]['ed2']:8.4f}", flush=True)
    summary = {}
    for latent_dim in latents:
        group = [r for r in rows if r["latent_dim"] == latent_dim]
        summary[f"latent={latent_dim}"] = {
            "median_second_moment_ratio": float(np.median(
                [r["second_moment_ratio"] for r in group])),
            "median_tail_fraction": float(np.median(
                [r["tail_fraction"] for r in group])),
            "median_ed2": float(np.median([r["ed2"] for r in group]))}
    return {"rows": rows, "summary": summary}


def n6_longrun(resolution: int, seeds: int, steps: int, root: str | None,
               cloud: int = 256, trace_every: int = 100) -> dict:
    """Is the uncorrected generator CONVERGED at 600 steps, or still climbing?

    Everything the program concludes about "the generator's equilibrium"
    rests on this and it has never been logged.  N2's evidence says
    equilibrium -- a free dilation parameter declines to grow, and the
    field's radial demand is +0.0004 -- but no long trace exists.  If the
    second moment is still rising at 600 steps then the deficit is a RATE,
    not an equilibrium, and R11 is an accelerant rather than a repair.
    """
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        config = TrainConfig(steps=steps, batch=64, field_cloud=cloud,
                             eval_samples=512, image_size=resolution)
        pools = evaluation_pools(evaluation, config, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())
        probe = sample_latent(512, config.latent_dim,
                              derive_seed(seed, "p8-probe"))
        for correction in ("none", "scalar"):
            train, branch, kernel, rng = _setup(resolution, seed, root,
                                                GOOD_ESS)
            model = OneStepGenerator(config.latent_dim, 3, resolution,
                                     config.width,
                                     derive_seed(seed, "generator"))
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config.learning_rate)
            generator = torch.Generator().manual_seed(
                derive_seed(seed, "p8-latent") % (2 ** 31))
            trace = []
            for step in range(steps):
                positives = train.sample(config.batch, rng)
                latent = torch.randn(cloud, config.latent_dim,
                                     generator=generator)
                output = model(latent)
                with torch.no_grad():
                    drift, _ = KG.field(output.detach(), positives,
                                        output.detach(), branch, kernel,
                                        direction_mode="paper",
                                        normalization="rms",
                                        diagnostics=False)
                    teacher = output.detach() + 0.5 * drift
                    if correction != "none":
                        teacher = corrected_teacher(teacher, positives,
                                                    mode=correction)
                loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step % trace_every == 0:
                    with torch.no_grad():
                        trace.append({
                            "step": step,
                            "second_moment_ratio": float(
                                model(probe).flatten(1).var(0).mean())
                            / reference_moment})
            values = [t["second_moment_ratio"] for t in trace]
            quarter = max(len(values) // 4, 1)
            late = float(np.median(values[-quarter:]))
            earlier = float(np.median(values[-2 * quarter:-quarter]))
            at600 = next((t["second_moment_ratio"] for t in trace
                          if t["step"] >= 600), float("nan"))
            rows.append({"correction": correction, "seed": seed,
                         "final": values[-1], "at_600": at600,
                         "late_window": late, "earlier_window": earlier,
                         "window_growth": late - earlier, "trace": trace})
            print(f"    N6 {correction:8} seed{index} "
                  f"at600={at600:6.3f} final={values[-1]:6.3f} "
                  f"growth={late - earlier:+7.4f}", flush=True)
    summary = {}
    for correction in ("none", "scalar"):
        group = [r for r in rows if r["correction"] == correction]
        growth = float(np.median([r["window_growth"] for r in group]))
        summary[correction] = {
            "median_at_600": float(np.median([r["at_600"] for r in group])),
            "median_final": float(np.median([r["final"] for r in group])),
            "median_window_growth": growth,
            "verdict": ("converged: the last two windows agree"
                        if abs(growth) < 0.02 else
                        "still moving: the trace has not settled")}
    return {"rows": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=("all", "n12", "n3", "n4", "n6"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase8_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    payload: dict = {
        "status": "phase7-followup-probe-feeds-no-gate",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "bandwidth": f"target_ess={GOOD_ESS} (Phase 7C optimum)",
    }

    if args.stage in ("all", "n12"):
        print("=== N1/N2: radial demand, and the one-parameter gain ===",
              flush=True)
        payload["n1n2_gain"] = n1n2_gain(args.resolution, args.seeds,
                                         args.steps, args.data_root)
    if args.stage in ("all", "n3"):
        print("\n=== N3: is the regression cost a capacity artifact? ===",
              flush=True)
        payload["n3_capacity"] = n3_capacity(args.resolution, args.seeds,
                                             args.steps, args.data_root)
    if args.stage in ("all", "n4"):
        print("\n=== N4: latent dimension at the good bandwidth ===",
              flush=True)
        payload["n4_latent"] = n4_latent(args.resolution, args.seeds,
                                         args.steps, args.data_root)
    if args.stage == "n6":
        print("=== N6: converged at 600 steps, or still climbing? ===",
              flush=True)
        payload["n6_longrun"] = n6_longrun(args.resolution, args.seeds,
                                           args.steps, args.data_root)
        payload["elapsed_seconds"] = time.time() - started
        digest = write_json(args.out, payload)
        print("\n=== N6: LONG-RUN CONVERGENCE ===")
        for key, entry in payload["n6_longrun"]["summary"].items():
            print(f"  {key:8} at600={entry['median_at_600']:6.3f} "
                  f"final={entry['median_final']:6.3f} "
                  f"growth={entry['median_window_growth']:+7.4f}  "
                  f"{entry['verdict']}")
        print(f"\nwrote {args.out} sha256={digest[:16]}...")
        return

    payload["elapsed_seconds"] = time.time() - started
    digest = write_json(args.out, payload)

    print("\n=== PHASE-7 FOLLOW-UP PROBE ===")
    if "n1n2_gain" in payload:
        print(f"{'variant':12}{'2nd_mom':>9}{'ed2':>9}{'radial_rel':>12}"
              f"{'outward':>9}{'tail':>8}{'gain':>8}")
        for key, entry in payload["n1n2_gain"]["summary"].items():
            gain = entry["median_final_gain"]
            print(f"{key:12}{entry['median_second_moment_ratio']:9.3f}"
                  f"{entry['median_ed2']:9.4f}"
                  f"{entry['median_radial_relative']:+12.4f}"
                  f"{entry['median_outward_fraction']:9.3f}"
                  f"{entry['median_tail_fraction']:8.4f}"
                  f"{(f'{gain:.3f}' if gain is not None else '--'):>8}")
        reference = next(iter(payload["n1n2_gain"]["summary"].values()))
        print(f"    (real data tail fraction beyond top-32: "
              f"{reference['reference_tail_fraction']:.4f})")
    if "n4_latent" in payload:
        print(f"\n{'arm':12}{'2nd_mom':>9}{'tail':>9}{'ed2':>9}")
        for key, entry in payload["n4_latent"]["summary"].items():
            print(f"{key:12}{entry['median_second_moment_ratio']:9.3f}"
                  f"{entry['median_tail_fraction']:9.4f}"
                  f"{entry['median_ed2']:9.4f}")
    if "n3_capacity" in payload:
        print(f"\n{'regime':22}{'params':>10}{'p/v':>7}{'2nd_mom':>9}"
              f"{'ed2':>9}")
        for key, entry in payload["n3_capacity"]["summary"].items():
            print(f"{key:22}{entry['parameters']:10}"
                  f"{entry['params_per_value']:7.2f}"
                  f"{entry['median_second_moment_ratio']:9.3f}"
                  f"{entry['median_ed2']:9.4f}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
