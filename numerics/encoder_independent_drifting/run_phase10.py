"""Phase 10 (protocol `EncoderIndependentPhase10Protocol.md`).

Measure the shape law, then act on shape.

The mechanism pass showed the deficit is a shape phenomenon: at matched
second moment the generator's cloud is radially balanced (+0.0006) while a
data-shaped cloud at the same scale is pushed outward 43x harder (+0.0264).
Its cloud is clumped -- nearest-neighbour spacing 3.47 against 6.36 at equal
variance -- because it concentrates energy in few directions (spectral tail
0.0034 against 0.142).

  10A  the law, with the two candidates separated:
       family S varies the spectrum (tail AND spacing move together);
       family P holds the covariance fixed and varies only packing.
  10B  intervene on shape -- repulsion penalty, spectral-tail floor -- with
       the same supersession gate R11 has survived three times.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase10
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

# Frozen (protocol sections 2-3): disjoint from every earlier phase.
SEED_OFFSET = 18000
MOMENT_BAND = (0.7, 1.3)
SUPERSEDE_TOLERANCE = 1.25
GOOD_ESS = 0.9
FIELD_CLOUD = 256
TAIL_KEEP = 32

BETAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
PACKINGS = (0, 5, 20, 80)
WEIGHTS = (0.1, 1.0)
ALPHAS = (0.15, 0.25, 0.4, 0.55, 0.7, 1.0, 1.4)


def _setup(resolution: int, seed: int, root: str | None):
    train = cifar.cifar_target(resolution, "train", root)
    rng = np.random.default_rng(derive_seed(seed, "p10-setup"))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace",
                              target_ess_fraction=GOOD_ESS)
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace",
        geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
        geometry.kernel_eps, combine=geometry.combine,
        target_ess_fraction=GOOD_ESS)
    return train, branch, kernel, rng


def _shape_stats(sample: torch.Tensor) -> dict:
    flat = sample.reshape(len(sample), -1)
    centred = flat - flat.mean(dim=0, keepdim=True)
    power = torch.linalg.svdvals(centred) ** 2
    distance = torch.cdist(flat, flat)
    distance.fill_diagonal_(float("inf"))
    nearest = distance.min(dim=1).values
    return {"tail": float(power[TAIL_KEEP:].sum() / power.sum()),
            "nn_mean": float(nearest.mean()),
            "nn_cv": float(nearest.std() / nearest.mean())}


def _radial(drift: torch.Tensor, cloud: torch.Tensor) -> float:
    flat = cloud.reshape(len(cloud), -1)
    v = drift.reshape(len(drift), -1)
    radius = flat - flat.mean(dim=0, keepdim=True)
    unit = radius / radius.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return float((v * unit).sum(dim=1).mean())


# ---------------------------------------------------------------------------
# 10A: constructing clouds of controlled shape
# ---------------------------------------------------------------------------


def spectral_family(sample: torch.Tensor, beta: float) -> torch.Tensor:
    """Reweight singular values as ``S^beta`` at fixed total variance.

    beta > 1 concentrates the spectrum (less tail), beta < 1 flattens it.
    The total variance is restored afterwards so the family varies shape
    alone.
    """
    flat = sample.reshape(len(sample), -1)
    centre = flat.mean(dim=0, keepdim=True)
    centred = flat - centre
    left, singular, right = torch.linalg.svd(centred, full_matrices=False)
    reweighted = singular.clamp_min(1e-12) ** beta
    rebuilt = (left * reweighted) @ right
    scale = (centred.pow(2).sum() / rebuilt.pow(2).sum().clamp_min(1e-30)
             ) ** 0.5
    return (centre + rebuilt * scale).reshape(sample.shape)


def packing_family(sample: torch.Tensor, steps: int,
                   strength: float = 0.15) -> torch.Tensor:
    """Regularize the point arrangement at FIXED covariance.

    Repulsion is applied in the cloud's own principal coordinates and every
    coordinate is rescaled back to its target variance after each step, so
    the spectrum -- and hence the spectral tail -- is held fixed while the
    nearest-neighbour spacing becomes more regular.  This is what separates
    "packing" from "spectrum", which every cloud measured so far confounded.
    """
    flat = sample.reshape(len(sample), -1)
    centre = flat.mean(dim=0, keepdim=True)
    centred = flat - centre
    _, _, right = torch.linalg.svd(centred, full_matrices=False)
    coords = centred @ right.T
    target = coords.var(dim=0)
    scale = float(coords.pow(2).sum(dim=1).mean().sqrt())
    for _ in range(steps):
        distance = torch.cdist(coords, coords) + torch.eye(
            len(coords)) * 1e9
        weight = torch.exp(-distance / (scale * 0.25))
        difference = coords.unsqueeze(1) - coords.unsqueeze(0)
        push = (weight.unsqueeze(2) * difference).sum(dim=1)
        push = push / push.norm(dim=1, keepdim=True).clamp_min(1e-12)
        coords = coords + strength * scale * push / len(coords) ** 0.5
        coords = coords - coords.mean(dim=0, keepdim=True)
        coords = coords * (target / coords.var(dim=0).clamp_min(1e-30)) ** 0.5
    return (centre + coords @ right).reshape(sample.shape)


def radial_zero(cloud: torch.Tensor, train, branch, kernel, rng,
                repeats: int = 4) -> dict:
    """Scan the scale and interpolate where the radial component vanishes."""
    flat = cloud.reshape(len(cloud), -1)
    centre = flat.mean(dim=0, keepdim=True)
    base = flat - centre
    series = []
    for alpha in ALPHAS:
        scaled = (centre + base * alpha ** 0.5).reshape(cloud.shape)
        values = []
        for _ in range(repeats):
            drift, _ = KG.field(scaled, train.sample(64, rng), scaled,
                                branch, kernel, direction_mode="paper",
                                normalization="none", diagnostics=False)
            values.append(_radial(drift, scaled))
        series.append((alpha, float(np.mean(values))))
    crossing = None
    for (a0, v0), (a1, v1) in zip(series, series[1:]):
        if v0 == 0 or (v0 > 0) != (v1 > 0):
            crossing = a0 + (a1 - a0) * v0 / (v0 - v1)
            break
    return {"series": series, "radial_zero": crossing}


def stage_10a(resolution: int, seeds: int, root: str | None,
              cloud_size: int = 256) -> dict:
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        train, branch, kernel, rng = _setup(resolution, seed, root)
        reference = train.sample(cloud_size, rng)
        for beta in BETAS:
            cloud = spectral_family(reference, beta)
            stats = _shape_stats(cloud)
            result = radial_zero(cloud, train, branch, kernel, rng)
            rows.append({"family": "spectrum", "knob": beta, "seed": seed,
                         **stats, "radial_zero": result["radial_zero"]})
            print(f"    10A S beta={beta:4.1f} seed{index} "
                  f"tail={stats['tail']:7.4f} nn={stats['nn_mean']:8.3f} "
                  f"nn_cv={stats['nn_cv']:6.3f} "
                  f"zero={str(round(result['radial_zero'], 4) if result['radial_zero'] else None):>8}",
                  flush=True)
        for steps in PACKINGS:
            cloud = (reference if steps == 0
                     else packing_family(reference, steps))
            stats = _shape_stats(cloud)
            result = radial_zero(cloud, train, branch, kernel, rng)
            rows.append({"family": "packing", "knob": steps, "seed": seed,
                         **stats, "radial_zero": result["radial_zero"]})
            print(f"    10A P steps={steps:4} seed{index} "
                  f"tail={stats['tail']:7.4f} nn={stats['nn_mean']:8.3f} "
                  f"nn_cv={stats['nn_cv']:6.3f} "
                  f"zero={str(round(result['radial_zero'], 4) if result['radial_zero'] else None):>8}",
                  flush=True)
    summary = {}
    for family, knobs in (("spectrum", BETAS), ("packing", PACKINGS)):
        for knob in knobs:
            group = [r for r in rows if r["family"] == family
                     and r["knob"] == knob]
            zeros = [r["radial_zero"] for r in group
                     if r["radial_zero"] is not None]
            summary[f"{family}_{knob}"] = {
                "median_tail": float(np.median([r["tail"] for r in group])),
                "median_nn_mean": float(np.median(
                    [r["nn_mean"] for r in group])),
                "median_nn_cv": float(np.median(
                    [r["nn_cv"] for r in group])),
                "median_radial_zero": (float(np.median(zeros)) if zeros
                                       else None)}
    return {"rows": rows, "summary": summary,
            "law": _fit_law(summary)}


def _fit_law(summary: dict) -> dict:
    """Does the radial zero track the tail, the packing, or both?"""
    def series(family: str, key: str):
        pairs = [(v[key], v["median_radial_zero"])
                 for k, v in summary.items() if k.startswith(family)
                 and v["median_radial_zero"] is not None]
        return [p for p in pairs if np.isfinite(p[0])]

    def spearman(pairs) -> float:
        if len(pairs) < 3:
            return float("nan")
        x = np.argsort(np.argsort([p[0] for p in pairs]))
        y = np.argsort(np.argsort([p[1] for p in pairs]))
        n = len(pairs)
        return float(1 - 6 * ((x - y) ** 2).sum() / (n * (n ** 2 - 1)))

    spectrum_tail = spearman(series("spectrum", "median_tail"))
    packing_nn = spearman(series("packing", "median_nn_cv"))
    packing_zeros = [v["median_radial_zero"] for k, v in summary.items()
                     if k.startswith("packing")
                     and v["median_radial_zero"] is not None]
    packing_span = (max(packing_zeros) - min(packing_zeros)
                    if packing_zeros else float("nan"))
    spectrum_zeros = [v["median_radial_zero"] for k, v in summary.items()
                      if k.startswith("spectrum")
                      and v["median_radial_zero"] is not None]
    spectrum_span = (max(spectrum_zeros) - min(spectrum_zeros)
                     if spectrum_zeros else float("nan"))
    return {
        "spearman_tail_vs_zero_in_spectrum_family": spectrum_tail,
        "spearman_nncv_vs_zero_in_packing_family": packing_nn,
        "spectrum_family_zero_span": spectrum_span,
        "packing_family_zero_span": packing_span,
        "verdict": (
            "the spectrum is the variable; packing at fixed spectrum does "
            "little" if np.isfinite(packing_span) and np.isfinite(
                spectrum_span) and packing_span < 0.25 * spectrum_span
            else "packing moves the zero at fixed spectrum too -- both "
                 "matter"),
    }


# ---------------------------------------------------------------------------
# 10B: intervene on shape
# ---------------------------------------------------------------------------


def repulsion_penalty(output: torch.Tensor, bandwidth: float
                      ) -> torch.Tensor:
    """Push the batch's points apart, with no reference to any scale."""
    flat = output.reshape(len(output), -1)
    distance = torch.cdist(flat, flat)
    mask = 1.0 - torch.eye(len(flat), dtype=flat.dtype)
    return ((torch.exp(-distance / max(bandwidth, 1e-12)) * mask).sum()
            / mask.sum())


def tail_penalty(output: torch.Tensor) -> torch.Tensor:
    """Negative fraction of variance beyond the top ``TAIL_KEEP`` directions."""
    flat = output.reshape(len(output), -1)
    centred = flat - flat.mean(dim=0, keepdim=True)
    power = torch.linalg.svdvals(centred) ** 2
    return -power[TAIL_KEEP:].sum() / power.sum().clamp_min(1e-30)


def stage_10b(resolution: int, seeds: int, steps: int, root: str | None,
              eta: float = 0.5) -> dict:
    evaluation = cifar.cifar_target(resolution, "eval", root)
    arms = [("E0_none", "none", 0.0, 0.0), ("E1_r11", "scalar", 0.0, 0.0)]
    for weight in WEIGHTS:
        arms.append((f"E2_repulsion_{weight}", "none", weight, 0.0))
        arms.append((f"E3_tail_{weight}", "none", 0.0, weight))
    arms.append(("E23_both_1.0", "none", 1.0, 1.0))

    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        config = TrainConfig(steps=steps, batch=64, field_cloud=FIELD_CLOUD,
                             eval_samples=512, image_size=resolution)
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())
        reference_shape = _shape_stats(pools["eval"])
        for name, correction, repulsion_weight, tail_weight in arms:
            train, branch, kernel, rng = _setup(resolution, seed, root)
            bandwidth = float(np.median(
                torch.cdist(pools["eval"].flatten(1),
                            pools["eval"].flatten(1)).numpy()))
            model = OneStepGenerator(config.latent_dim, 3, resolution,
                                     config.width,
                                     derive_seed(seed, "generator"))
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config.learning_rate)
            torch_rng = torch.Generator().manual_seed(
                derive_seed(seed, "p10b-latent") % (2 ** 31))
            for _ in range(steps):
                latent = torch.randn(FIELD_CLOUD, config.latent_dim,
                                     generator=torch_rng)
                output = model(latent)
                with torch.no_grad():
                    positives = train.sample(config.batch, rng)
                    drift, _ = KG.field(output.detach(), positives,
                                        output.detach(), branch, kernel,
                                        direction_mode="paper",
                                        normalization="rms",
                                        diagnostics=False)
                    teacher = output.detach() + eta * drift
                    if correction != "none":
                        teacher = corrected_teacher(teacher, positives,
                                                    mode=correction)
                loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
                if repulsion_weight:
                    loss = loss + repulsion_weight * repulsion_penalty(
                        output, bandwidth)
                if tail_weight:
                    loss = loss + tail_weight * tail_penalty(output)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            probe = sample_latent(config.eval_samples, config.latent_dim,
                                  derive_seed(seed, "p10b-probe"))
            with torch.no_grad():
                generated = model(probe)
            measured = M.raw_metrics(
                generated, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "p10b-m")), None,
                target_null=pools["null"])
            stats = _shape_stats(generated)
            row = {"arm": name, "seed": seed,
                   "second_moment_ratio": float(
                       generated.flatten(1).var(0).mean()) / reference_moment,
                   "ed2": measured["ed2"],
                   "geometry_score_v2": M.normalized_geometry_score_v2(
                       measured, null)["geometry_score"],
                   **stats,
                   "reference_tail": reference_shape["tail"],
                   "reference_nn_mean": reference_shape["nn_mean"]}
            rows.append(row)
            print(f"    10B {name:20} seed{index} "
                  f"2nd={row['second_moment_ratio']:6.3f} "
                  f"ed2={row['ed2']:8.4f} tail={row['tail']:7.4f} "
                  f"nn={row['nn_mean']:8.3f}", flush=True)
    summary = {}
    for name, _, _, _ in arms:
        group = [r for r in rows if r["arm"] == name]
        moment = float(np.median([r["second_moment_ratio"] for r in group]))
        summary[name] = {
            "median_second_moment_ratio": moment,
            "median_ed2": float(np.median([r["ed2"] for r in group])),
            "median_score_v2": float(np.median(
                [r["geometry_score_v2"] for r in group])),
            "median_tail": float(np.median([r["tail"] for r in group])),
            "median_nn_mean": float(np.median([r["nn_mean"] for r in group])),
            "moment_in_band": bool(MOMENT_BAND[0] <= moment
                                   <= MOMENT_BAND[1])}
    return {"rows": rows, "summary": summary,
            "reference_shape": reference_shape}


def gate_10b(summary: dict) -> dict:
    """Can acting on shape supersede R11?  Declared before the run."""
    r11 = summary.get("E1_r11")
    shape = {k: v for k, v in summary.items()
             if k.startswith(("E2", "E3", "E23"))}
    if not r11 or not shape:
        return {"passed": False, "reason": "missing arms"}
    ceiling = r11["median_ed2"] * SUPERSEDE_TOLERANCE
    qualifying = [k for k, v in shape.items()
                  if v["moment_in_band"] and v["median_ed2"] <= ceiling]
    baseline = summary.get("E0_none", {})
    moved = {k: {"tail_vs_baseline": (v["median_tail"]
                                      / max(baseline.get("median_tail", 1e-12),
                                            1e-12)),
                 "nn_vs_baseline": (v["median_nn_mean"]
                                    / max(baseline.get("median_nn_mean",
                                                       1e-12), 1e-12))}
             for k, v in shape.items()}
    return {
        "passed": bool(qualifying),
        "meaning": ("acting on shape supersedes R11" if qualifying else
                    "no shape intervention reproduces R11"),
        "ed2_ceiling": ceiling,
        "qualifying_arms": qualifying,
        "shape_actually_moved": moved,
        "note": "if the shape moved but the second moment did not, the "
                "mechanism is refuted, not the intervention",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=("all", "10a", "10b"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase10.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    payload: dict = {
        "status": "phase10-frozen-protocol",
        "protocol": "numerics/EncoderIndependentPhase10Protocol.md",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "config_digest": config_digest(
            TrainConfig(steps=args.steps, image_size=args.resolution)),
        "optimizer": optimizer_report(TrainConfig()),
        "frozen_thresholds": {
            "moment_band": list(MOMENT_BAND),
            "supersede_tolerance": SUPERSEDE_TOLERANCE,
            "betas": list(BETAS), "packings": list(PACKINGS),
            "weights": list(WEIGHTS), "target_ess": GOOD_ESS,
        },
    }

    if args.stage in ("all", "10a"):
        print("=== 10A: the shape law, spectrum against packing ===",
              flush=True)
        payload["stage_10a"] = stage_10a(args.resolution, args.seeds,
                                         args.data_root)
        print(f"\n  10A: {payload['stage_10a']['law']['verdict']}", flush=True)
    if args.stage in ("all", "10b"):
        print("\n=== 10B: intervene on shape ===", flush=True)
        stage = stage_10b(args.resolution, args.seeds, args.steps,
                          args.data_root)
        payload["stage_10b"] = stage | {"gate": gate_10b(stage["summary"])}
        print(f"\n  10B gate: {payload['stage_10b']['gate']['meaning']}",
              flush=True)

    payload["elapsed_seconds"] = time.time() - started
    digest = write_json(args.out, payload)

    print("\n=== PHASE 10 ===")
    if "stage_10a" in payload:
        print(f"{'cloud':18}{'tail':>9}{'nn_mean':>10}{'nn_cv':>9}"
              f"{'radial zero':>13}")
        for key, entry in payload["stage_10a"]["summary"].items():
            zero = entry["median_radial_zero"]
            print(f"{key:18}{entry['median_tail']:9.4f}"
                  f"{entry['median_nn_mean']:10.3f}{entry['median_nn_cv']:9.3f}"
                  f"{(f'{zero:.4f}' if zero is not None else 'none'):>13}")
        law = payload["stage_10a"]["law"]
        print(f"\n  spectrum family: tail vs zero spearman "
              f"{law['spearman_tail_vs_zero_in_spectrum_family']:+.3f}, "
              f"zero span {law['spectrum_family_zero_span']:.4f}")
        print(f"  packing family:  nn_cv vs zero spearman "
              f"{law['spearman_nncv_vs_zero_in_packing_family']:+.3f}, "
              f"zero span {law['packing_family_zero_span']:.4f}")
        print(f"  -> {law['verdict']}")
    if "stage_10b" in payload:
        print(f"\n{'arm':22}{'2nd_mom':>9}{'ed2':>10}{'score':>9}"
              f"{'tail':>9}{'nn_mean':>10}  band")
        for key, entry in payload["stage_10b"]["summary"].items():
            print(f"{key:22}{entry['median_second_moment_ratio']:9.3f}"
                  f"{entry['median_ed2']:10.4f}{entry['median_score_v2']:9.3f}"
                  f"{entry['median_tail']:9.4f}{entry['median_nn_mean']:10.3f}"
                  f"  {'in ' if entry['moment_in_band'] else 'out'}")
        reference = payload["stage_10b"]["reference_shape"]
        print(f"    (real data: tail={reference['tail']:.4f} "
              f"nn_mean={reference['nn_mean']:.3f})")
        gate = payload["stage_10b"]["gate"]
        print(f"\n  gate passed={gate['passed']}  -- {gate['meaning']}")
        for arm, moved in gate["shape_actually_moved"].items():
            print(f"    {arm:22} tail x{moved['tail_vs_baseline']:6.2f}  "
                  f"nn x{moved['nn_vs_baseline']:6.2f}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
