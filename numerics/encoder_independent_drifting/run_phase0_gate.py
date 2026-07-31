"""Phase-0 exit gate (plan section 9, plus reform R5).

The gate has five conditions, all evaluated here and all recorded whether
they pass or fail:

G0.1  every mathematical unit test passes;
G0.2  the anchor detects every synthetic collision pair;
G0.3  at least one fixed geometry branch has healthier affinity ESS *and*
      drift SNR than raw pixel drifting;
G0.4  kernel-gradient and standard-displacement modes differ in the
      predicted structured directions;
G0.5  minimizing a candidate field actually reaches the `q = p` finite-sample
      floor (reform R5, added after the Phase-1 failure).

G0.3 is a comparison of kernel *health*, so it is measured on the same
probes, the same target batches and the same seeds for every branch.  ESS is
reported as a fraction of the batch: a value near 1 means the kernel is flat
(every neighbour equally weighted) and near 1/N means it has collapsed onto
a single neighbour.  Neither extreme is healthy, so the criterion is
distance from flatness in the useful direction, stated explicitly below.

G0.5 exists because G0.1-G0.4 all passed and Phase 1 still failed.  Kernel
health says the kernel *can* discriminate; it says nothing about whether the
set where the field vanishes contains the target law.  G0.5 measures that
directly and cheaply, and on the Phase-1 configurations it reproduces the
observed ranking before any generator is trained.

Run:
    uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase0_gate
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import collision_suite as CS
from . import datasets as D
from . import kernel_gradient as KG
from . import spectral_anchor as SA
from .config import AnchorConfig, GeometryConfig, MASTER_SEED, derive_seed
from .diagnostics import provenance, write_json
from .fixed_features import build_family
from .kernels import calibrate_block_kernel, min_eigenvalue, mmd2_unbiased

HERE = Path(__file__).resolve().parent

# A flat kernel weights every neighbour alike and carries no geometry; the
# raw pixel kernel is the thing being improved on, so "healthier" means
# strictly further from flatness than raw, in the direction of selectivity.
FLAT_ESS_FRACTION = 1.0


def _anchor_bank(reference: torch.Tensor, features: int,
                 seed: int) -> SA.SpectralBank:
    config = AnchorConfig(features=features)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_seed(seed, "scale") % (2 ** 63 - 1))
    scale = SA.projected_scale(reference, config, generator)
    return SA.build_bank(config, reference[0].numel(), scale,
                         derive_seed(seed, "bank"))


def gate_unit_tests() -> dict:
    from .tests import (
        test_crossfit_controller, test_fixed_features, test_kernel_gradients,
        test_positive_kernel_mixture, test_reforms, test_reproducibility,
        test_spectral_anchor,
    )
    from .tests.harness import run_module

    modules = {
        "spectral_anchor": test_spectral_anchor,
        "fixed_features": test_fixed_features,
        "kernel_gradients": test_kernel_gradients,
        "positive_kernel_mixture": test_positive_kernel_mixture,
        "crossfit_controller": test_crossfit_controller,
        "reproducibility": test_reproducibility,
        "diagnosis_reforms": test_reforms,
    }
    counts = {name: run_module(name, vars(module))
              for name, module in modules.items()}
    total = sum(counts.values())
    return {"failures_by_module": counts, "total_failures": total,
            "pass": total == 0}


def gate_anchor_collisions(samples: int, permutations: int,
                           features: int) -> dict:
    """G0.2: the anchor must see every declared source collision."""
    rng = np.random.default_rng(derive_seed(MASTER_SEED, "phase0-collision"))
    reference = torch.cat([pair.left(128, rng) for pair in CS.suite()])
    bank = _anchor_bank(reference, features,
                        derive_seed(MASTER_SEED, "phase0-anchor"))

    def discrepancy(left: torch.Tensor, right: torch.Tensor) -> float:
        return float(SA.anchor_loss(bank, left, right, "unbiased"))

    result = CS.run_suite(discrepancy, samples, permutations, rng)
    result["bank"] = bank.summary()
    result["pass"] = result["detected"] == result["total"]
    return result


def gate_geometry_blindness(samples: int, permutations: int) -> dict:
    """Context for G0.2: which collisions each fixed geometry cannot see.

    A branch that misses a collision is not a defect -- it is the reason the
    geometry branch may never be the correctness authority.
    """
    rng = np.random.default_rng(derive_seed(MASTER_SEED, "phase0-blind"))
    reference = torch.cat([pair.left(128, rng) for pair in CS.suite()])
    out = {}
    for family in ("raw", "wavelet", "randconv"):
        config = GeometryConfig(family=family)
        branch = build_family(config, D.CHANNELS).branches[0]
        kernel = calibrate_block_kernel(
            branch, reference, "smooth_laplace", config.bandwidth_quantile,
            config.bandwidth_multiplier, config.kernel_eps,
            combine=config.combine,
            target_ess_fraction=config.target_ess_fraction)

        def discrepancy(left: torch.Tensor, right: torch.Tensor,
                        branch=branch, kernel=kernel) -> float:
            return mmd2_unbiased(kernel, branch, left, right)

        out[family] = CS.run_suite(discrepancy, samples, permutations, rng)
    return out


def gate_kernel_health(seeds: int, batch: int) -> dict:
    """G0.3: fixed geometry vs raw pixel drifting on kernel health."""
    families = {
        "raw": GeometryConfig(family="raw"),
        "haar_control": GeometryConfig(family="haar_control"),
        "pyramid": GeometryConfig(family="pyramid", combine="sum"),
        "wavelet": GeometryConfig(family="wavelet", combine="sum"),
        "wavelet_product": GeometryConfig(family="wavelet",
                                          combine="product"),
        "randconv": GeometryConfig(family="randconv", combine="sum"),
    }
    rows: list[dict] = []
    for target in D.suite():
        for seed in range(seeds):
            rng = np.random.default_rng(
                derive_seed(MASTER_SEED, "health", target.name, seed))
            calibration = target.sample(256, rng)
            positive = target.sample(batch, rng)
            # A deliberately imperfect cloud: the field is only informative
            # away from the target, and a flat kernel is worst there.
            generated = target.sample(batch, rng) * 0.4 + torch.tensor(
                rng.normal(scale=0.4, size=(batch, D.CHANNELS, D.SIZE,
                                            D.SIZE)), dtype=torch.float32)
            for name, config in families.items():
                family = build_family(config, D.CHANNELS)
                for branch in family.branches:
                    kernel = calibrate_block_kernel(
                        branch, calibration, "smooth_laplace",
                        config.bandwidth_quantile,
                        config.bandwidth_multiplier, config.kernel_eps,
                        combine=config.combine,
                        target_ess_fraction=config.target_ess_fraction)
                    _, stats = KG.field_with_snr(
                        generated, positive, generated, branch, kernel,
                        direction_mode="kernel_gradient",
                        normalization="rms")
                    rows.append({
                        "target": target.name, "seed": seed,
                        "family": name, "branch": branch.name,
                        "ess_fraction": stats["ess_fraction"],
                        "entropy_fraction": stats["entropy_fraction"],
                        "drift_snr": stats["drift_snr"],
                        "affinity_median": stats["affinity_median"],
                        "affinity_zero_fraction":
                            stats["affinity_zero_fraction"],
                        "denominator_floor_fraction":
                            stats["denominator_floor_fraction"],
                        "min_eigenvalue": min_eigenvalue(
                            kernel, branch, calibration[:48]),
                    })

    def aggregate(key: str) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for row in rows:
            out.setdefault(f"{row['family']}::{row['branch']}", []).append(
                row[key])
        return {k: float(np.median(v)) for k, v in out.items()}

    ess = aggregate("ess_fraction")
    snr = aggregate("drift_snr")
    raw_key = "raw::raw"
    raw_ess, raw_snr = ess[raw_key], snr[raw_key]
    # Healthier = strictly more selective than raw (lower ESS fraction, i.e.
    # the kernel actually distinguishes neighbours) AND a higher cross-fit
    # drift signal-to-noise ratio.
    healthier = {
        key: {"ess_fraction": ess[key], "drift_snr": snr[key],
              "more_selective": ess[key] < raw_ess,
              "higher_snr": snr[key] > raw_snr,
              "healthier": ess[key] < raw_ess and snr[key] > raw_snr}
        for key in ess if key != raw_key
    }
    winners = [k for k, v in healthier.items() if v["healthier"]]
    return {
        "raw_ess_fraction": raw_ess, "raw_drift_snr": raw_snr,
        "flat_ess_fraction": FLAT_ESS_FRACTION,
        "median_ess_fraction": ess, "median_drift_snr": snr,
        "median_entropy_fraction": aggregate("entropy_fraction"),
        "median_affinity_median": aggregate("affinity_median"),
        "median_min_eigenvalue": aggregate("min_eigenvalue"),
        "branch_verdicts": healthier,
        "healthier_branches": winners,
        "pass": len(winners) > 0,
        "rows": rows,
    }


def gate_direction_modes(seeds: int, batch: int) -> dict:
    """G0.4: the two movement rules must diverge under a structured kernel.

    The raw Gaussian kernel is the control: there the two modes are provably
    proportional, so a cosine of 1 there and a cosine well below 1 for a
    structured kernel is the predicted signature.
    """
    rows = []
    for target in D.suite():
        for seed in range(seeds):
            rng = np.random.default_rng(
                derive_seed(MASTER_SEED, "modes", target.name, seed))
            calibration = target.sample(256, rng)
            positive = target.sample(batch, rng)
            generated = target.sample(batch, rng) * 0.4
            for name, base, config in (
                    # Positive control: for a raw Gaussian kernel the two
                    # modes are provably proportional, so the cosine must be
                    # 1 here.  Anything else means the measurement is wrong.
                    ("raw_gaussian_control", "gaussian",
                     GeometryConfig(family="raw")),
                    ("raw", "smooth_laplace", GeometryConfig(family="raw")),
                    ("wavelet", "smooth_laplace",
                     GeometryConfig(family="wavelet")),
                    ("randconv", "smooth_laplace",
                     GeometryConfig(family="randconv"))):
                branch = build_family(config, D.CHANNELS).branches[0]
                kernel = calibrate_block_kernel(
                    branch, calibration, base,
                    config.bandwidth_quantile, config.bandwidth_multiplier,
                    config.kernel_eps, combine=config.combine,
                    target_ess_fraction=config.target_ess_fraction)
                fields, spectra = {}, {}
                for mode in KG.DIRECTION_MODES:
                    drift, _ = KG.field(
                        generated, positive, generated, branch, kernel,
                        direction_mode=mode, normalization="rms",
                        diagnostics=False)
                    fields[mode] = drift.flatten()
                    spectra[mode] = KG.drift_spectrum(drift)
                cosine = float(torch.nn.functional.cosine_similarity(
                    fields["standard"], fields["kernel_gradient"], dim=0))
                profile = sum(
                    abs(spectra["standard"][k]
                        - spectra["kernel_gradient"][k])
                    for k in spectra["standard"])
                rows.append({"target": target.name, "seed": seed,
                             "family": name, "cosine": cosine,
                             "spectral_profile_l1": profile})

    def median(family: str, key: str) -> float:
        return float(np.median([r[key] for r in rows
                                if r["family"] == family]))

    structured = {f: {"cosine": median(f, "cosine"),
                      "spectral_profile_l1": median(f,
                                                    "spectral_profile_l1")}
                  for f in ("wavelet", "randconv")}
    control_cosine = median("raw_gaussian_control", "cosine")
    raw_cosine = median("raw", "cosine")
    control_ok = control_cosine > 0.999
    passed = control_ok and all(
        v["cosine"] < 0.9 and v["spectral_profile_l1"] > 0.1
        for v in structured.values())
    return {"gaussian_control_cosine": control_cosine,
            "gaussian_control_ok": bool(control_ok),
            "raw_smooth_laplace_cosine": raw_cosine,
            "structured": structured,
            "pass": bool(passed), "rows": rows}


def gate_zero_set(seeds: int, batch: int, steps: int,
                  step_size: float = 0.2) -> dict:
    """G0.5 (reform R5): does minimizing this field reach the `q=p` floor?

    The single cheapest predictor of the Phase-1 failure, and it was missing.
    A geometry whose field can be driven to the finite-sample floor has a
    zero-set that contains the target; one that plateaus far above it does
    not, and no amount of training will fix that.

    Directly optimizes a free particle cloud on the field -- no generator,
    no optimizer state -- so the result is a property of the *field*, not of
    an architecture.  Measured for every direction rule, since the rule is
    part of what defines the reachable set.
    """
    rows: list[dict] = []
    families = {
        "raw": GeometryConfig(family="raw"),
        "wavelet": GeometryConfig(family="wavelet"),
        "randconv": GeometryConfig(family="randconv"),
    }
    for target in D.suite()[:4]:
        for seed in range(seeds):
            rng = np.random.default_rng(
                derive_seed(MASTER_SEED, "zeroset", target.name, seed))
            calibration = target.sample(256, rng)
            # Held-out evaluation batches.  A cloud optimized against one
            # fixed positive batch can memorize those 64 points and drive
            # the field *below* the q=p floor, which measures batch
            # overfitting rather than law matching.  Training therefore draws
            # a fresh batch every step and the residual is read on unseen
            # samples.
            holdout = target.sample(batch, rng)
            matched = target.sample(batch, rng)      # a genuine q = p cloud
            for name, config in families.items():
                family = build_family(config, D.CHANNELS)
                branch = family.branches[0]
                kernel = calibrate_block_kernel(
                    branch, calibration, "smooth_laplace",
                    config.bandwidth_quantile, config.bandwidth_multiplier,
                    config.kernel_eps, combine=config.combine,
                    target_ess_fraction=config.target_ess_fraction)
                for mode in KG.DIRECTION_MODES:
                    _, floor_stats = KG.field(
                        matched, holdout, matched, branch, kernel,
                        direction_mode=mode, normalization="none")
                    floor = floor_stats["drift_rms_raw"]
                    cloud = torch.tensor(
                        rng.normal(scale=0.5,
                                   size=(batch, D.CHANNELS, D.SIZE, D.SIZE)),
                        dtype=torch.float32)
                    start = None
                    for index in range(steps):
                        fresh = target.sample(batch, rng)
                        drift, stats = KG.field(
                            cloud, fresh, cloud, branch, kernel,
                            direction_mode=mode, normalization="rms")
                        if start is None:
                            start = stats["drift_rms_raw"]
                        # A constant step on a unit-RMS field cannot settle
                        # (the Phase-1 defect of diagnosis D1), so the step
                        # decays linearly to zero and the cloud is allowed to
                        # come to rest wherever the field permits.
                        rate = step_size * (1.0 - index / steps)
                        cloud = cloud + rate * drift
                    _, final = KG.field(
                        cloud, holdout, cloud, branch, kernel,
                        direction_mode=mode, normalization="none")
                    residual = final["drift_rms_raw"]
                    rows.append({
                        "target": target.name, "seed": seed,
                        "family": name, "direction_mode": mode,
                        "floor": floor, "start": start,
                        "residual": residual,
                        "residual_over_floor": residual / max(floor, 1e-12),
                        "descent_fraction": (
                            1.0 - residual / max(start, 1e-12)),
                    })

    def median(family: str, mode: str, key: str) -> float:
        values = [r[key] for r in rows
                  if r["family"] == family and r["direction_mode"] == mode]
        return float(np.median(values)) if values else float("nan")

    summary = {
        f"{family}::{mode}": {
            "residual_over_floor": median(family, mode,
                                          "residual_over_floor"),
            "descent_fraction": median(family, mode, "descent_fraction"),
        }
        for family in families for mode in KG.DIRECTION_MODES
    }
    raw_ratio = summary["raw::kernel_gradient"]["residual_over_floor"]
    # The raw kernel reaches its floor; the bar is stated relative to it so
    # the criterion does not depend on the absolute floor scale.
    threshold = max(2.0 * raw_ratio, 2.0)
    reaching = [key for key, value in summary.items()
                if value["residual_over_floor"] <= threshold]
    return {
        "raw_kernel_gradient_residual_over_floor": raw_ratio,
        "threshold": threshold,
        "summary": summary,
        "reaching_the_floor": reaching,
        "pass": bool(any(k.startswith(("wavelet", "randconv"))
                         for k in reaching)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--collision-samples", type=int, default=192)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--anchor-features", type=int, default=512)
    parser.add_argument("--zero-set-steps", type=int, default=300)
    parser.add_argument("--zero-set-step", type=float, default=0.2)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase0_gate.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.time()

    results: dict = {}
    results["G0.1_unit_tests"] = (
        {"skipped": True, "pass": None} if args.skip_tests
        else gate_unit_tests())
    print("\n--- G0.2 anchor collision detection ---", flush=True)
    results["G0.2_anchor_collisions"] = gate_anchor_collisions(
        args.collision_samples, args.permutations, args.anchor_features)
    print("--- G0.2b geometry blindness (context) ---", flush=True)
    results["G0.2b_geometry_blindness"] = gate_geometry_blindness(
        args.collision_samples, args.permutations)
    print("--- G0.3 kernel health ---", flush=True)
    results["G0.3_kernel_health"] = gate_kernel_health(args.seeds, args.batch)
    print("--- G0.4 direction modes ---", flush=True)
    results["G0.4_direction_modes"] = gate_direction_modes(
        args.seeds, args.batch)
    print("--- G0.5 zero-set reachability (reform R5) ---", flush=True)
    results["G0.5_zero_set"] = gate_zero_set(
        args.seeds, args.batch, args.zero_set_steps, args.zero_set_step)

    verdicts = {k: v.get("pass") for k, v in results.items()}
    overall = all(v for v in verdicts.values() if v is not None)
    payload = {
        "status": "phase0-exit-gate",
        "scope": "synthetic 16x16 structured images; no pretrained encoder; "
                 "finite random-feature anchor, not an exact characteristic "
                 "family",
        "config": vars(args) | {"out": str(args.out)},
        "provenance": provenance(),
        "elapsed_seconds": time.time() - started,
        "verdicts": verdicts,
        "gate_pass": bool(overall),
        "results": results,
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 0 EXIT GATE ===")
    for name, verdict in verdicts.items():
        label = "SKIP" if verdict is None else ("PASS" if verdict else "FAIL")
        print(f"  [{label}] {name}")
    health = results["G0.3_kernel_health"]
    print(f"\n  raw pixel: ESS fraction {health['raw_ess_fraction']:.4f}, "
          f"drift SNR {health['raw_drift_snr']:.3f}")
    for key in sorted(health["healthier_branches"])[:8]:
        verdict = health["branch_verdicts"][key]
        print(f"    healthier: {key:32} ESS {verdict['ess_fraction']:.4f} "
              f"SNR {verdict['drift_snr']:.3f}")
    anchor = results["G0.2_anchor_collisions"]
    print(f"\n  anchor collisions detected: {anchor['detected']}"
          f"/{anchor['total']}  missed={anchor['failed']}")
    for family, blind in results["G0.2b_geometry_blindness"].items():
        print(f"  geometry {family:10} detected {blind['detected']}"
              f"/{blind['total']}  blind to {blind['failed']}")
    modes = results["G0.4_direction_modes"]
    print(f"\n  direction cosine: gaussian control "
          f"{modes['gaussian_control_cosine']:.4f} (must be 1), "
          f"raw smooth-Laplace {modes['raw_smooth_laplace_cosine']:.4f}, "
          + ", ".join(f"{k} {v['cosine']:.4f}"
                      for k, v in modes["structured"].items()))
    zero = results["G0.5_zero_set"]
    print(f"\n  zero-set residual / (q=p floor), threshold "
          f"{zero['threshold']:.2f}:")
    for key in sorted(zero["summary"]):
        value = zero["summary"][key]
        mark = "reaches" if key in zero["reaching_the_floor"] else "PLATEAUS"
        print(f"    {key:44} {value['residual_over_floor']:8.2f} "
              f"(descended {value['descent_fraction'] * 100:5.1f}%) {mark}")
    print(f"\n  overall: {'PASS' if overall else 'FAIL'}")
    print(f"  wrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
