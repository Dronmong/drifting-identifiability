"""Phase-6 follow-up: where does the second-moment deficit come from?

Phase 6 moved the question.  Five of the six refuted mechanism hypotheses
were aimed at the *generator*, but 6C showed the **particle algorithm has
the deficit too** (~0.6 of the data's variance against the generator's
~0.27).  So the deficit can now be studied with no generator, no optimizer
and no parametrization in the way -- which is a far simpler object than
anything the last six phases have had to work with.

Five measurements, none of them gated:

  M1  is 0.6 an attractor, or an unfinished trajectory?  6C logged no trace,
      so it could not tell.  Run long and look at the plateau.
  M2  does the deficit track the kernel bandwidth?  A mean-shift toward a
      kernel-weighted mean is a blur, and blurs contract.
  M3  is it an artifact of the positive/negative batch asymmetry that 6C
      happened to use (512 particles against 64 positives)?
  M4  **the balancing-depth hypothesis.**  Algorithm 2 forms its plan as
      ``A = sqrt(row * col)`` -- ONE symmetrized normalization pass, not a
      converged balancing.  An under-balanced plan over-weights data points
      that many particles claim, so its barycentric image is pulled toward
      dense regions, which shrinks the cloud.  If that is the mechanism, the
      deficit must fall as the plan is balanced further.
  M5  the analytic check: for a Gaussian target and a Gaussian kernel the
      kernel-weighted mean contracts by exactly ``s^2 / (s^2 + tau^2)``.
      Does the measured deficit match that number?

    uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .config import MASTER_SEED, derive_seed
from .diagnostics import provenance, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED_OFFSET = 11000

# The declared band every phase since R20 has reported against.
MOMENT_BAND = (0.7, 1.3)


# ---------------------------------------------------------------------------
# The field, with balancing depth exposed as an axis
# ---------------------------------------------------------------------------


def _logits(q: np.ndarray, data: np.ndarray, tau: float,
            mask: bool) -> np.ndarray:
    dp = np.linalg.norm(q[:, None, :] - data[None, :, :], axis=2)
    dn = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=2)
    logits = np.concatenate([-dp / tau, -dn / tau], axis=1)
    if mask:
        index = np.arange(len(q))
        logits[index, len(data) + index] -= 1e6 / tau
    return logits


def _drift_from_plan(plan: np.ndarray, q: np.ndarray,
                     data: np.ndarray) -> np.ndarray:
    """Algorithm 2's displacement, given an affinity matrix.

    Verbatim from `lowdim_drift.drift_paper` downstream of the plan, so the
    only thing the depth axis changes is how the plan was balanced.  The
    cross-scaling makes the positive and negative sides carry equal total
    mass, so the field is a difference of two weighted means.
    """
    a_positive, a_negative = plan[:, :len(data)], plan[:, len(data):]
    weight_positive = a_positive * a_negative.sum(axis=1, keepdims=True)
    weight_negative = a_negative * a_positive.sum(axis=1, keepdims=True)
    return weight_positive @ data - weight_negative @ q


def drift_balanced(q: np.ndarray, data: np.ndarray, tau: float, mask: bool,
                   depth: int) -> np.ndarray:
    """Algorithm 2 with the plan balanced ``depth`` times.

    ``depth = 0`` is the paper's own rule -- row and column normalization
    computed in PARALLEL and geometric-meaned, ``A = sqrt(row * col)``, which
    is one symmetrized pass and is what `lowdim_drift.drift_paper` does.
    ``depth >= 1`` runs that many alternating Sinkhorn rounds instead, so the
    plan approaches doubly stochastic.  Both feed the identical downstream
    formula.

    Scale is irrelevant downstream because the field is RMS normalized before
    use; what the depth axis changes is the plan's *shape*.
    """
    if depth < 0:
        raise ValueError("balancing depth must be non-negative")
    logits = _logits(q, data, tau, mask)
    if depth == 0:
        # Each softmax must be shifted along the axis it normalizes.  Using
        # one row-wise shift for both is only correct when every row has the
        # same maximum, which is true unmasked (the self-distance is 0 in
        # every row) and false once the self-term is masked out -- a 0.7
        # relative error, caught by the equivalence test against the audited
        # `lowdim_drift.drift_paper`.
        row = np.exp(logits - logits.max(axis=1, keepdims=True))
        row /= row.sum(axis=1, keepdims=True)
        column = np.exp(logits - logits.max(axis=0, keepdims=True))
        column /= column.sum(axis=0, keepdims=True)
        plan = np.sqrt(row * column)
    else:
        # Sinkhorn's scalings absorb any global shift, so one scalar keeps
        # the plan's shape exactly while staying in range.
        kernel = np.exp(logits - logits.max())
        u = np.ones(kernel.shape[0])
        v = np.ones(kernel.shape[1])
        for _ in range(depth):
            u = 1.0 / (kernel @ v + 1e-300)
            v = 1.0 / (kernel.T @ u + 1e-300)
        plan = u[:, None] * kernel * v[None, :]
    return _drift_from_plan(plan, q, data)


def _normalize(drift: np.ndarray) -> np.ndarray:
    rms = np.sqrt((drift ** 2).sum(axis=1).mean())
    return drift / max(rms, 1e-12)


def run_particles(target, dim: int, seed: int, *, steps: int, particles: int,
                  positives: int, tau: float, depth: int, eta: float,
                  schedule: str, mask: bool, trace_every: int = 0,
                  ) -> dict:
    """Free particles under the field -- no generator, no optimizer."""
    rng = np.random.default_rng(derive_seed(seed, "p7", dim, depth, tau))
    reference = target.sample(4096, rng)
    reference_moment = float(reference.var(axis=0).mean())
    cloud = rng.normal(scale=0.5, size=(particles, dim))
    trace = []
    for step in range(steps):
        data = target.sample(positives, rng)
        drift = _normalize(drift_balanced(cloud, data, tau, mask, depth))
        factor = 1.0 - step / steps if schedule == "linear_decay" else 1.0
        cloud = cloud + eta * factor * drift
        if trace_every and step % trace_every == 0:
            trace.append({"step": step,
                          "second_moment_ratio": float(
                              cloud.var(axis=0).mean()) / reference_moment})
    return {
        "second_moment_ratio": float(
            cloud.var(axis=0).mean()) / reference_moment,
        "trace": trace, "cloud": cloud, "reference": reference,
    }


def _in_band(value: float) -> bool:
    low, high = MOMENT_BAND
    return bool(np.isfinite(value) and low <= value <= high)


def _median(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
    return float(np.median(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# M1: attractor, or unfinished trajectory?
# ---------------------------------------------------------------------------


def m1_attractor(LD, dims, seeds: int, steps: int, tau: float) -> dict:
    """The measurement 6C could not make, because it logged no trace.

    A plateau means the deficit is a genuine fixed point of the dynamics and
    the mechanism question is well posed.  A still-rising trace means the
    particles were simply not finished and 6C's number says much less.
    """
    rows = []
    for dim in dims:
        target = LD.gauss_mixture(f"gauss_{dim}d", 4, dim, 0.15)
        for index in range(seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            result = run_particles(
                target, dim, seed, steps=steps, particles=512, positives=64,
                tau=tau, depth=0, eta=0.2, schedule="constant", mask=True,
                trace_every=max(steps // 60, 1))
            trace = result["trace"]
            # Compare the last quarter with the one before it: if the cloud
            # is still growing, the later window must be clearly larger.
            values = [t["second_moment_ratio"] for t in trace]
            quarter = max(len(values) // 4, 1)
            late = float(np.median(values[-quarter:]))
            earlier = float(np.median(values[-2 * quarter:-quarter]))
            rows.append({
                "dim": dim, "seed": seed,
                "second_moment_ratio": result["second_moment_ratio"],
                "late_window": late, "earlier_window": earlier,
                "window_growth": late - earlier,
                "trace": trace})
            print(f"    M1 dim={dim:3} seed{index} "
                  f"final={result['second_moment_ratio']:6.3f} "
                  f"late={late:6.3f} earlier={earlier:6.3f} "
                  f"growth={late - earlier:+7.4f}", flush=True)
    growth = _median(rows, "window_growth")
    return {"rows": rows, "median_window_growth": growth,
            "median_second_moment_ratio": _median(rows,
                                                  "second_moment_ratio"),
            "verdict": ("attractor: the last two windows agree"
                        if abs(growth) < 0.02 else
                        "still moving: the trace has not settled")}


# ---------------------------------------------------------------------------
# M2 / M5: bandwidth dependence, against the analytic blur factor
# ---------------------------------------------------------------------------


def m2_bandwidth(LD, dim: int, seeds: int, steps: int, taus) -> dict:
    """Does the deficit track the kernel bandwidth, as a blur would?

    For an isotropic Gaussian target of variance ``s^2`` and a Gaussian
    kernel of bandwidth ``tau``, the kernel-weighted mean of the data around
    ``x`` is ``x s^2 / (s^2 + tau^2)``: a contraction whose strength grows
    with ``tau``.  Algorithm 2 is not that map -- it subtracts a second
    weighted mean over the cloud, which is exactly the term that should stop
    the contraction -- so this comparison measures how much of the blur
    survives the negative side.
    """
    rows = []
    target = LD.gauss_mixture(f"gauss_{dim}d", 4, dim, 0.15)
    probe = target.sample(4096, np.random.default_rng(0))
    variance = float(probe.var(axis=0).mean())
    for tau in taus:
        for index in range(seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            result = run_particles(
                target, dim, seed, steps=steps, particles=512, positives=64,
                tau=tau, depth=0, eta=0.2, schedule="constant", mask=True)
            # The single-application blur factor contracts VARIANCE by its
            # square; reported as the second-moment ratio it would predict.
            blur = variance / (variance + tau ** 2)
            rows.append({"tau": tau, "seed": seed,
                         "second_moment_ratio": result[
                             "second_moment_ratio"],
                         "blur_factor": blur,
                         "blur_predicted_ratio": blur ** 2})
            print(f"    M2 tau={tau:<6g} seed{index} "
                  f"2nd={result['second_moment_ratio']:6.3f} "
                  f"blur_predicts={blur ** 2:6.3f}", flush=True)
    summary = {}
    for tau in taus:
        group = [r for r in rows if r["tau"] == tau]
        summary[f"tau={tau:g}"] = {
            "median_second_moment_ratio": _median(group,
                                                  "second_moment_ratio"),
            "blur_predicted_ratio": group[0]["blur_predicted_ratio"],
            "in_band": _in_band(_median(group, "second_moment_ratio"))}
    return {"rows": rows, "summary": summary, "target_variance": variance}


# ---------------------------------------------------------------------------
# M3: is it the batch asymmetry 6C happened to use?
# ---------------------------------------------------------------------------


def m3_batch_shape(LD, dim: int, seeds: int, steps: int, shapes) -> dict:
    """6C ran 512 particles against 64 positives.  Does that matter?

    A confound in my own measurement, checked rather than assumed away.
    """
    rows = []
    target = LD.gauss_mixture(f"gauss_{dim}d", 4, dim, 0.15)
    for particles, positives in shapes:
        for index in range(seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            result = run_particles(
                target, dim, seed, steps=steps, particles=particles,
                positives=positives, tau=0.2, depth=0, eta=0.2,
                schedule="constant", mask=True)
            rows.append({"particles": particles, "positives": positives,
                         "seed": seed,
                         "second_moment_ratio": result[
                             "second_moment_ratio"]})
            print(f"    M3 particles={particles:5} positives={positives:5} "
                  f"seed{index} "
                  f"2nd={result['second_moment_ratio']:6.3f}", flush=True)
    summary = {}
    for particles, positives in shapes:
        group = [r for r in rows if r["particles"] == particles
                 and r["positives"] == positives]
        summary[f"{particles}x{positives}"] = {
            "median_second_moment_ratio": _median(group,
                                                  "second_moment_ratio"),
            "in_band": _in_band(_median(group, "second_moment_ratio"))}
    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------------------
# M4: the balancing-depth hypothesis
# ---------------------------------------------------------------------------


def m4_balancing_depth(LD, dims, seeds: int, steps: int, depths,
                       tau: float) -> dict:
    """Is the deficit an under-balancing artifact?

    Algorithm 2 balances its plan once.  An under-balanced plan lets a data
    point that many particles claim keep a large total weight, so the
    barycentric image over-represents dense regions and the cloud contracts.
    A converged balancing is doubly stochastic, which removes exactly that
    freedom.  If this is the mechanism, the deficit falls with depth.
    """
    rows = []
    for dim in dims:
        target = LD.gauss_mixture(f"gauss_{dim}d", 4, dim, 0.15)
        for depth in depths:
            for index in range(seeds):
                seed = MASTER_SEED + SEED_OFFSET + index
                result = run_particles(
                    target, dim, seed, steps=steps, particles=512,
                    positives=64, tau=tau, depth=depth, eta=0.2,
                    schedule="constant", mask=True)
                ed2 = LD.energy_distance2(result["cloud"],
                                          result["reference"])
                rows.append({"dim": dim, "depth": depth, "seed": seed,
                             "second_moment_ratio": result[
                                 "second_moment_ratio"],
                             "ed2": ed2})
                print(f"    M4 dim={dim:3} depth={depth:3} seed{index} "
                      f"2nd={result['second_moment_ratio']:6.3f} "
                      f"ed2={ed2:8.4f}", flush=True)
    summary = {}
    for dim in dims:
        for depth in depths:
            group = [r for r in rows if r["dim"] == dim
                     and r["depth"] == depth]
            summary[f"dim={dim}_depth={depth}"] = {
                "median_second_moment_ratio": _median(
                    group, "second_moment_ratio"),
                "median_ed2": _median(group, "ed2"),
                "in_band": _in_band(_median(group, "second_moment_ratio"))}
    return {"rows": rows, "summary": summary,
            "note": "depth 0 is the paper's sqrt(row*col); depth >= 1 is "
                    "that many alternating Sinkhorn rounds"}


# ---------------------------------------------------------------------------


def m6_cifar_bandwidth(seeds: int, steps: int, resolution: int,
                       root: str | None, trace_every: int = 40) -> dict:
    """The decisive test: is 6C's particle deficit a bandwidth artifact?

    6C measured free particles at 0.594 of the data's variance and called it
    a property of drifting.  The low-dimensional probe says the fixed point's
    scale is set by the bandwidth and runs from 8.9 (tau = .05) to 1.16
    (tau = .5), crossing 1 in between -- so "drifting shrinks" is the wrong
    reading of a single operating point.

    This repeats 6C's exact particle configuration at CIFAR-16 across the
    bandwidth axis, with the trace 6C never logged, and adds the 64-particle
    cloud the *generator* actually trains with (the probe's M3 found the
    particle count matters and the positive count does not).
    """
    import torch                                           # noqa: PLC0415
    from . import cifar                                    # noqa: PLC0415
    from . import kernel_gradient as KG                    # noqa: PLC0415
    from . import metrics as M                             # noqa: PLC0415
    from .config import GeometryConfig, TrainConfig        # noqa: PLC0415
    from .evaluate import evaluation_pools, null_reference  # noqa: PLC0415
    from .fixed_features import build_family               # noqa: PLC0415
    from .kernels import calibrate_block_kernel            # noqa: PLC0415

    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    config = TrainConfig(steps=steps, batch=64, eval_samples=512,
                         image_size=resolution)
    # (label, normalized tau, target ESS fraction).  `None` tau with an ESS
    # fraction is the repository's calibration; a tau pins the paper's rule.
    arms = [("ess=0.10", None, 0.10), ("ess=0.25", None, 0.25),
            ("ess=0.50", None, 0.50), ("ess=0.90", None, 0.90),
            ("tau=0.02", 0.02, None), ("tau=0.05", 0.05, None),
            ("tau=0.20", 0.20, None), ("tau=1.00", 1.00, None)]
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())
        rng = np.random.default_rng(derive_seed(seed, "p7-cifar"))
        for label, tau, ess in arms:
            geometry = GeometryConfig(family="raw",
                                      base_kernel="smooth_laplace")
            branch = build_family(geometry, 3).branches[0]
            kernel = calibrate_block_kernel(
                branch, train.sample(256, rng), "smooth_laplace",
                geometry.bandwidth_quantile,
                tau if tau is not None else geometry.bandwidth_multiplier,
                geometry.kernel_eps, combine=geometry.combine,
                target_ess_fraction=ess)
            for particles in (512, 64):
                cloud = torch.tensor(
                    rng.normal(scale=0.5,
                               size=(particles, 3, resolution, resolution)),
                    dtype=torch.float32)
                trace = []
                for step in range(steps):
                    drift, _ = KG.field(
                        cloud, train.sample(64, rng), cloud, branch, kernel,
                        direction_mode="paper", normalization="rms",
                        diagnostics=False)
                    cloud = cloud + 0.2 * drift
                    if step % trace_every == 0:
                        trace.append({
                            "step": step,
                            "second_moment_ratio": float(
                                cloud.flatten(1).var(0).mean())
                            / reference_moment})
                values = [t["second_moment_ratio"] for t in trace]
                quarter = max(len(values) // 4, 1)
                late = float(np.median(values[-quarter:]))
                earlier = float(np.median(values[-2 * quarter:-quarter]))
                measured = M.raw_metrics(
                    cloud, pools["eval"], pools["cal_a"], pools["cal_b"],
                    np.random.default_rng(derive_seed(seed, "p7-m")), None,
                    target_null=pools["null"])
                row = {"arm": label, "tau": tau, "ess": ess,
                       "particles": particles, "seed": seed,
                       "second_moment_ratio": float(
                           cloud.flatten(1).var(0).mean()) / reference_moment,
                       "late_window": late, "earlier_window": earlier,
                       "window_growth": late - earlier,
                       "ed2": measured["ed2"],
                       "geometry_score_v2": M.normalized_geometry_score_v2(
                           measured, null)["geometry_score"],
                       "trace": trace}
                rows.append(row)
                print(f"    M6 {label:9} N={particles:4} seed{index} "
                      f"2nd={row['second_moment_ratio']:8.3f} "
                      f"growth={row['window_growth']:+7.4f} "
                      f"ed2={row['ed2']:8.4f} "
                      f"score={row['geometry_score_v2']:7.3f}", flush=True)
    summary = {}
    for label, tau, ess in arms:
        for particles in (512, 64):
            group = [r for r in rows if r["arm"] == label
                     and r["particles"] == particles]
            summary[f"{label}_N={particles}"] = {
                "median_second_moment_ratio": _median(
                    group, "second_moment_ratio"),
                "median_window_growth": _median(group, "window_growth"),
                "median_ed2": _median(group, "ed2"),
                "median_score_v2": _median(group, "geometry_score_v2"),
                "in_band": _in_band(_median(group, "second_moment_ratio"))}
    return {"rows": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="lowdim",
                        choices=("lowdim", "cifar"))
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--long-steps", type=int, default=3000)
    parser.add_argument("--dims", type=str, default="8,32")
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "phase7_probe.json")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "numerics"))
    started = time.time()

    if args.stage == "cifar":
        print("=== M6: is 6C's particle deficit a bandwidth artifact? ===",
              flush=True)
        m6 = m6_cifar_bandwidth(args.seeds, args.steps, args.resolution,
                                args.data_root)
        payload = {
            "status": "phase6-followup-cifar-probe-feeds-no-gate",
            "provenance": provenance(),
            "config": vars(args) | {"out": str(args.out)},
            "moment_band": list(MOMENT_BAND),
            "elapsed_seconds": time.time() - started,
            "m6_cifar_bandwidth": m6["summary"],
            "m6_rows": m6["rows"],
        }
        digest = write_json(args.out, payload)
        print("\n=== M6: CIFAR-16 PARTICLE BANDWIDTH SWEEP ===")
        print(f"{'arm':18}{'2nd_mom':>10}{'growth':>10}{'ed2':>10}"
              f"{'score':>9}  band")
        for key, entry in m6["summary"].items():
            print(f"{key:18}{entry['median_second_moment_ratio']:10.3f}"
                  f"{entry['median_window_growth']:+10.4f}"
                  f"{entry['median_ed2']:10.4f}"
                  f"{entry['median_score_v2']:9.3f}"
                  f"  {'in' if entry['in_band'] else 'out'}")
        print(f"\nwrote {args.out} sha256={digest[:16]}...")
        return

    import lowdim_drift as LD                              # noqa: PLC0415

    dims = [int(x) for x in args.dims.split(",")]

    print("=== M1: attractor, or unfinished trajectory? ===", flush=True)
    m1 = m1_attractor(LD, dims, args.seeds, args.long_steps, args.tau)
    print(f"  -> {m1['verdict']}", flush=True)

    print("\n=== M2/M5: bandwidth, against the blur factor ===", flush=True)
    m2 = m2_bandwidth(LD, dims[0], args.seeds, args.steps,
                      (0.05, 0.1, 0.2, 0.5, 1.0))

    print("\n=== M3: batch shape ===", flush=True)
    m3 = m3_batch_shape(LD, dims[0], args.seeds, args.steps,
                        ((512, 64), (512, 512), (64, 64), (64, 512)))

    print("\n=== M4: balancing depth ===", flush=True)
    m4 = m4_balancing_depth(LD, dims, args.seeds, args.steps,
                            (0, 1, 2, 4, 16, 64), args.tau)

    payload = {
        "status": "phase6-followup-probe-feeds-no-gate",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "moment_band": list(MOMENT_BAND),
        "elapsed_seconds": time.time() - started,
        "m1_attractor": {k: v for k, v in m1.items() if k != "rows"},
        "m1_rows": [{k: v for k, v in r.items()} for r in m1["rows"]],
        "m2_bandwidth": m2["summary"],
        "m2_target_variance": m2["target_variance"],
        "m3_batch_shape": m3["summary"],
        "m4_balancing_depth": m4["summary"],
        "m4_note": m4["note"],
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE-6 FOLLOW-UP PROBE ===")
    print(f"M1  {m1['verdict']}  "
          f"(median final {m1['median_second_moment_ratio']:.3f}, "
          f"window growth {m1['median_window_growth']:+.4f})")
    print("\nM2  bandwidth")
    for key, entry in m2["summary"].items():
        print(f"    {key:12} 2nd={entry['median_second_moment_ratio']:6.3f}"
              f"  blur_predicts={entry['blur_predicted_ratio']:6.3f}"
              f"  {'in' if entry['in_band'] else 'out'}")
    print("\nM3  batch shape (particles x positives)")
    for key, entry in m3["summary"].items():
        print(f"    {key:12} 2nd={entry['median_second_moment_ratio']:6.3f}"
              f"  {'in' if entry['in_band'] else 'out'}")
    print("\nM4  balancing depth")
    for key, entry in m4["summary"].items():
        print(f"    {key:18} 2nd={entry['median_second_moment_ratio']:6.3f}"
              f"  ed2={entry['median_ed2']:8.4f}"
              f"  {'in' if entry['in_band'] else 'out'}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
