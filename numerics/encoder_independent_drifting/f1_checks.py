"""F1 §11 steps 5-7 — non-confirmatory validation. No result here is an F1 result.

Five checks, all of which must pass before the confirmatory arms may run:

  C1  one-step equivalence: `f1.frozen_update` must reproduce the Phase-26
      update bit-for-bit on identical inputs, to <= 1e-6 relative L2.  If the
      explicit `denominator_floor` / `report` arguments changed the arithmetic,
      every historical anchor would be invalid.
  C2  historical regression at K = 40, run through **Phase 26's own code path
      with its own seeds** (§5.1).  This validates that the recorded anchors
      reproduce in this environment; it deliberately does not route through
      `f1.py`, whose disjoint-index calibration differs by construction.
  C3  replay reproducibility: the same unit run twice must agree to <= 1e-4
      relative L2, which is what makes `z -> x_K(z)` a usable endpoint map.
  C4  index disjointness and uniqueness across every F1 role, all three units.
  C5  an all-arm / two-regime smoke at tiny horizon, exercising every metric,
      both regimes and the veto statistics.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.f1_checks
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from .autoencoder import train_autoencoder
from .config import MASTER_SEED, GeometryConfig, derive_seed
from .device import configure, resolve_device
from .diagnose_phase25 import train_cloud
from .diagnostics import provenance, write_json
from .f1 import (
    ETA,
    POSITIVES,
    REPLAY_TOLERANCE,
    allocate,
    bank_statistics,
    build_kernel,
    frozen_update,
    real_nn_scale,
    relative_l2,
    rollout,
    score,
    source_cloud,
    take,
    unit_seeds,
)
from .fid import inception_features
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .objectives import corrected_teacher

HERE = Path(__file__).resolve().parent

# §5.1 tolerances, tabulated in the protocol.
HISTORICAL = {
    "real_data_recall": (0.717, 0.030),
    "real_data_kid": (0.00041, 0.0015),
}
EQUIVALENCE_TOLERANCE = 1e-6


def c1_one_step_equivalence(resolution, root, device) -> dict:
    """Phase 26's update vs `f1.frozen_update` on identical inputs."""
    allocation = allocate(0, resolution, root)
    branch, kernel = build_kernel(allocation, resolution, root, device)
    state = take(resolution, "train", root,
                 allocation.real_data).to(device)
    positives = take(resolution, "train", root,
                     allocation.bank[:POSITIVES]).to(device)
    with torch.no_grad():
        # Phase 26's exact call: no explicit denominator_floor, no report dict.
        drift, _ = KG.field(state, positives, state, branch, kernel,
                            direction_mode="paper", normalization="rms",
                            diagnostics=False)
        historical = corrected_teacher(state + ETA * drift, positives,
                                       mode="scalar")
        current, _ = frozen_update(state, positives, branch, kernel)
    error = relative_l2(historical, current)
    return {"relative_l2": error, "tolerance": EQUIVALENCE_TOLERANCE,
            "passed": bool(error <= EQUIVALENCE_TOLERANCE)}


def c2_historical_regression(resolution, root, device, samples) -> dict:
    """Phase 26's own path and seeds, checked against the recorded anchors."""
    seed = MASTER_SEED + 44000            # Phase 26's declared offset
    rng = np.random.default_rng(derive_seed(seed, "p26"))
    evaluation = cifar.cifar_target(resolution, "eval", root)
    real = evaluation.sample(samples, rng)
    reference = inception_features(real, device).double().numpy()
    train = cifar.cifar_target(resolution, "train", root)
    train.device = device
    branch = build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace",
                       target_ess_fraction=0.05), 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace", 0.5, 1.0, 1e-3,
        combine="sum", target_ess_fraction=0.05)
    state = train.sample(512, rng)
    stream = np.random.default_rng(derive_seed(seed, "it", "real_data"))
    for _ in range(40):
        with torch.no_grad():
            positives = train.sample(POSITIVES, stream)
            drift, _ = KG.field(state, positives, state, branch, kernel,
                                direction_mode="paper", normalization="rms",
                                diagnostics=False)
            state = corrected_teacher(state + ETA * drift, positives,
                                      mode="scalar")
    got = score(state, reference, real, device)
    checks = {}
    for name, (expected, tolerance) in HISTORICAL.items():
        key = "recall" if name.endswith("recall") else "kid"
        checks[name] = {"expected": expected, "got": got[key],
                        "tolerance": tolerance,
                        "passed": bool(abs(got[key] - expected) <= tolerance)}
    return {"measured": {k: got[k] for k in ("recall", "precision", "kid",
                                             "alpha", "second_moment")},
            "checks": checks,
            "passed": all(c["passed"] for c in checks.values())}


def c3_replay_reproducibility(resolution, root, device, steps) -> dict:
    """The same unit, twice, must agree within the declared tolerance."""
    allocation = allocate(0, resolution, root)
    branch, kernel = build_kernel(allocation, resolution, root, device)
    start = source_cloud("random_generator", 0, resolution, root, device)
    runs = [rollout(start, 0, "replay", branch, kernel, resolution, root,
                    device, checkpoints=(0, steps)) for _ in range(2)]
    error = relative_l2(runs[0]["final"], runs[1]["final"])
    return {"relative_l2": error, "tolerance": REPLAY_TOLERANCE,
            "schedule_digest": runs[0]["schedule_digest"],
            "digests_match": runs[0]["schedule_digest"]
                             == runs[1]["schedule_digest"],
            "passed": bool(error <= REPLAY_TOLERANCE)}


def c4_disjointness(resolution, root) -> dict:
    """Uniqueness and disjointness for every role, every unit."""
    rows = []
    for unit in range(3):
        allocation = allocate(unit, resolution, root)
        allocation.assert_disjoint()          # raises on any violation
        rows.append({
            "unit": unit, "bank": len(allocation.bank),
            "bank_unique": len(np.unique(allocation.bank)),
            "real_data": len(allocation.real_data),
            "digests": allocation.digests,
        })
    banks = [allocate(u, resolution, root).bank for u in range(3)]
    overlaps = {f"{a}v{b}": int(np.intersect1d(banks[a], banks[b]).size)
                for a in range(3) for b in range(a + 1, 3)}
    return {"units": rows, "cross_unit_bank_overlap": overlaps,
            "banks_differ": all(v < len(banks[0]) for v in overlaps.values()),
            "passed": all(r["bank"] == r["bank_unique"] for r in rows)}


def c5_smoke(resolution, root, device, samples, steps) -> dict:
    """All six arms, both regimes, every metric and veto statistic."""
    unit = 0
    allocation = allocate(unit, resolution, root)
    branch, kernel = build_kernel(allocation, resolution, root, device)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rng = np.random.default_rng(derive_seed(MASTER_SEED + 51000, "smoke"))
    real = evaluation.sample(samples, rng)
    reference = inception_features(real, device).double().numpy()
    bank = take(resolution, "train", root, allocation.bank).to(device)
    scale = real_nn_scale(resolution, root, allocation.real_data)

    # `train_autoencoder` samples through the target, so the target must carry
    # the device or it feeds CPU batches to a CUDA model.
    ae_target = cifar.cifar_target(resolution, "train", root)
    ae_target.device = device
    fit = train_autoencoder(ae_target, 60, unit_seeds(unit)["source_init"],
                            device, latent_channels=32)
    bad, _ = train_cloud(60, unit_seeds(unit)["source_init"], device,
                         resolution, root)

    rows = []
    for arm in ("real_data", "random_generator", "trained_bad",
                "ae_reconstruction", "basin_interpolation", "ambient_noise"):
        start = source_cloud(arm, unit, resolution, root, device,
                             ae_model=fit["model"], trained_bad=bad)
        for regime in ("replay", "stochastic"):
            out = rollout(start, unit, regime, branch, kernel, resolution,
                          root, device, checkpoints=(0, steps))
            metrics = score(out["final"], reference, real, device,
                            resamples=4, seed=7)
            vetoes = bank_statistics(out["final"].cpu(), bank.cpu(), scale)
            rows.append({
                "arm": arm, "regime": regime,
                "recall": metrics["recall"], "precision": metrics["precision"],
                "kid": metrics["kid"], "effective_rank": metrics["effective_rank"],
                "duplicate_rate": metrics["duplicate_rate"],
                "nn_diversity": metrics["nn_diversity"],
                "recall_ci": metrics["recall_ci"],
                "nearest_bank_normalized": vetoes["nearest_bank_normalized"],
                "distinct_bank": vetoes["distinct_bank"],
                "diagnostics": out["history"][-1],
            })
            print(f"      {arm:20}{regime:12} recall={metrics['recall']:.4f} "
                  f"rank={metrics['effective_rank']:6.2f} "
                  f"distinct_bank={vetoes['distinct_bank']:5} "
                  f"ess={out['history'][-1].get('ess_fraction', float('nan')):.3f}",
                  flush=True)
    required = {"ess_fraction", "collapsed_row_fraction",
                "denominator_floor_fraction", "drift_rms_raw",
                "correction_ratio_median", "interval_steps"}
    complete = all(required <= set(r["diagnostics"]) for r in rows)
    return {"rows": rows, "diagnostics_complete": complete,
            "passed": bool(complete and len(rows) == 12)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--replay-steps", type=int, default=20)
    parser.add_argument("--smoke-steps", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "f1_checks.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)

    started = time.time()
    results = {}
    print("=== C1 one-step equivalence vs Phase 26 ===", flush=True)
    results["c1"] = c1_one_step_equivalence(args.resolution, args.data_root,
                                            device)
    print(f"    relative L2 = {results['c1']['relative_l2']:.3e} "
          f"(tolerance {EQUIVALENCE_TOLERANCE:.0e}) "
          f"{'PASS' if results['c1']['passed'] else 'FAIL'}\n", flush=True)

    print("=== C4 index disjointness and uniqueness ===", flush=True)
    results["c4"] = c4_disjointness(args.resolution, args.data_root)
    print(f"    all roles unique and disjoint across 3 units: "
          f"{'PASS' if results['c4']['passed'] else 'FAIL'}")
    print(f"    cross-unit bank overlap {results['c4']['cross_unit_bank_overlap']}"
          f"  banks differ: {results['c4']['banks_differ']}\n", flush=True)

    print("=== C3 replay reproducibility ===", flush=True)
    results["c3"] = c3_replay_reproducibility(args.resolution, args.data_root,
                                              device, args.replay_steps)
    print(f"    relative L2 = {results['c3']['relative_l2']:.3e} "
          f"(tolerance {REPLAY_TOLERANCE:.0e}) "
          f"{'PASS' if results['c3']['passed'] else 'FAIL'}\n", flush=True)

    print("=== C2 historical K=40 regression (Phase 26 path and seeds) ===",
          flush=True)
    results["c2"] = c2_historical_regression(args.resolution, args.data_root,
                                             device, args.samples)
    for name, entry in results["c2"]["checks"].items():
        print(f"    {name:20} got {entry['got']:+.5f} vs "
              f"{entry['expected']:+.5f} +-{entry['tolerance']} "
              f"{'PASS' if entry['passed'] else 'FAIL'}")
    print(flush=True)

    print("=== C5 all-arm / two-regime smoke ===", flush=True)
    results["c5"] = c5_smoke(args.resolution, args.data_root, device,
                             args.samples, args.smoke_steps)
    print(f"    12 arm/regime cells, diagnostics complete: "
          f"{'PASS' if results['c5']['passed'] else 'FAIL'}\n", flush=True)

    passed = {k: bool(v["passed"]) for k, v in results.items()}
    verdict = {"checks": passed, "all_passed": all(passed.values())}
    verdict["reading"] = (
        "all §11 validation checks pass; the confirmatory F1 arms become "
        "run-ready once §2.1 calibration returns GO"
        if verdict["all_passed"] else
        "at least one validation check FAILED; the confirmatory run stays NO-GO")

    payload = {"status": "f1-validation-checks-not-an-f1-result",
               "protocol": "numerics/EncoderIndependentF1Protocol.md",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "results": results, "verdict": verdict}
    digest = write_json(args.out, payload)

    print("=== F1 VALIDATION (§11 steps 5-7) ===")
    for name, ok in passed.items():
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
