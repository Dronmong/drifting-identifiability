"""Stage-3A development: can a neural map retain coherent target routes?

This is the first experiment after the corrected balanced-plan audit. It uses
the same 2x64 Tanh generator architecture as the repository's neural PSQT
program and compares route teachers under matched optimizer updates:

* ``est_fixed``: fixed latent cohort and one persistent sparse balanced EST
  assignment;
* ``euclidean_fixed``: fixed cohort and minimum-Euclidean-cost assignment;
* ``random_fixed``: fixed cohort and random bijection control;
* ``est_replan``: fixed cohort, balanced EST route recomputed each macro-step;
* ``est_fresh``: fresh latent cohort and fresh balanced EST route each macro;
* ``independent_sliced``: the old independently sorted projected-quantile loss.

The final evaluation uses fresh latent samples and disjoint target planning,
support-calibration, and evaluation pools. This is development, not
confirmation and not an official-paper comparison.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from est_plan import (  # noqa: E402
    all_rank_matches,
    balanced_est_assignment,
    balanced_hybrid_assignment,
    match_counts,
    target_marginal_l1,
    unit_directions,
)
from metrics import (  # noqa: E402
    calibrated_precision_coverage,
    checkerboard_leakage,
    energy_distance2,
    sliced_w1,
)
from run_neural_pooled_rank_development import Generator  # noqa: E402
import targets as T  # noqa: E402


CORE_ARMS = (
    "est_fixed",
    "euclidean_fixed",
    "random_fixed",
    "est_replan",
    "est_fresh",
    "independent_sliced",
)
HYBRID_WEIGHTS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
HYBRID_ARMS = tuple(f"hybrid_{weight:g}_fixed" for weight in HYBRID_WEIGHTS)
ARMS = CORE_ARMS + HYBRID_ARMS


def euclidean_assignment(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    cost = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    assign = np.empty(len(X), dtype=int)
    assign[rows] = cols
    return assign


def sliced_loss(
    generated: torch.Tensor,
    target_sorted: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    projected = generated @ directions.T
    generated_sorted = torch.sort(projected, dim=0).values
    d = generated.shape[1]
    L = directions.shape[0]
    return (d / L) * torch.mean((generated_sorted - target_sorted) ** 2)


def assignment_for(
    arm: str,
    generated: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if arm.startswith("est_"):
        Pi = all_rank_matches(generated, Y, U)
        return balanced_est_assignment(Y, Pi, method="sparse")[1]
    if arm.startswith("hybrid_") and arm.endswith("_fixed"):
        weight = float(arm[len("hybrid_") : -len("_fixed")])
        Pi = all_rank_matches(generated, Y, U)
        return balanced_hybrid_assignment(
            generated, Y, Pi, agreement_weight=weight,
            method="sparse")[1]
    if arm == "euclidean_fixed":
        return euclidean_assignment(generated, Y)
    if arm == "random_fixed":
        return rng.permutation(len(Y))
    raise ValueError(f"arm {arm} has no discrete assignment")


def train_arm(
    arm: str,
    base_model: Generator,
    *,
    Y: np.ndarray,
    U: np.ndarray,
    fixed_latent: torch.Tensor,
    seed: int,
    macros: int,
    inner_steps: int,
    learning_rate: float,
) -> tuple[Generator, dict]:
    model = copy.deepcopy(base_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(seed)
    directions_t = torch.tensor(U, dtype=torch.float32)
    target_t = torch.tensor(Y, dtype=torch.float32)
    target_sorted = torch.sort(target_t @ directions_t.T, dim=0).values
    assignment: np.ndarray | None = None
    previous: np.ndarray | None = None
    switches: list[float] = []
    assignment_seconds = 0.0
    target_accesses = 0
    final_loss = float("nan")
    route_costs: list[float] = []
    route_agreements: list[float] = []
    marginal_errors: list[float] = []
    started = time.perf_counter()

    for macro in range(macros):
        if arm in ("est_fresh", "independent_sliced"):
            latent = torch.randn(
                fixed_latent.shape, generator=torch_gen, dtype=torch.float32)
        else:
            latent = fixed_latent

        if arm != "independent_sliced":
            refresh = (
                assignment is None
                or arm in ("est_replan", "est_fresh")
            )
            if refresh:
                with torch.no_grad():
                    generated_np = model(latent).cpu().numpy()
                t0 = time.perf_counter()
                assignment = assignment_for(
                    arm, generated_np, Y, U, rng)
                assignment_seconds += time.perf_counter() - t0
                target_accesses += len(Y)
                if previous is not None and arm != "est_fresh":
                    switches.append(float(np.mean(assignment != previous)))
                previous = assignment.copy()
                route_costs.append(float(np.mean(np.sum(
                    (generated_np - Y[assignment]) ** 2, axis=1))))
                route_pi = all_rank_matches(generated_np, Y, U)
                route_counts = match_counts(route_pi)
                route_agreements.append(float(np.mean(
                    route_counts[np.arange(len(Y)), assignment] / len(U))))
                marginal_errors.append(target_marginal_l1(assignment))
            teacher = torch.tensor(Y[assignment], dtype=torch.float32)

        for _ in range(inner_steps):
            optimizer.zero_grad(set_to_none=True)
            generated = model(latent)
            if arm == "independent_sliced":
                loss = sliced_loss(generated, target_sorted, directions_t)
                target_accesses += len(Y)
            else:
                loss = torch.mean((generated - teacher) ** 2)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())

    wall = time.perf_counter() - started
    with torch.no_grad():
        train_out = model(fixed_latent).cpu().numpy()
    return model, {
        "final_training_loss": final_loss,
        "mean_route_switch_fraction": (
            float(np.mean(switches)) if switches else 0.0),
        "max_route_switch_fraction": (
            float(np.max(switches)) if switches else 0.0),
        "assignment_seconds": assignment_seconds,
        "training_seconds": wall,
        "target_accesses": target_accesses,
        "optimizer_updates": macros * inner_steps,
        "train_cohort_ed2_to_plan": energy_distance2(train_out, Y),
        "mean_teacher_route_sq_l2": (
            float(np.mean(route_costs)) if route_costs else float("nan")),
        "mean_teacher_sliced_agreement": (
            float(np.mean(route_agreements))
            if route_agreements else float("nan")),
        "max_teacher_target_marginal_l1": (
            float(np.max(marginal_errors))
            if marginal_errors else float("nan")),
    }


def run_cell(
    tgt: T.GeoTarget,
    seed: int,
    *,
    N: int,
    L: int,
    macros: int,
    inner_steps: int,
    learning_rate: float,
    eval_samples: int,
    arms: tuple[str, ...],
) -> list[dict]:
    Y = tgt.sampler(N, np.random.default_rng(100_000 + 101 * seed))
    cal_a = tgt.sampler(512, np.random.default_rng(110_000 + 101 * seed))
    cal_b = tgt.sampler(512, np.random.default_rng(120_000 + 101 * seed))
    eval_target = tgt.sampler(
        eval_samples, np.random.default_rng(130_000 + 101 * seed))
    U = unit_directions(L, 2, 140_000 + seed)

    base = Generator(2, "broad", 150_000 + seed).to(dtype=torch.float32)
    latent_dimension = base.latent_dimension
    fixed_latent = torch.tensor(
        np.random.default_rng(160_000 + seed).normal(
            size=(N, latent_dimension)),
        dtype=torch.float32,
    )
    eval_latent = torch.tensor(
        np.random.default_rng(170_000 + seed).normal(
            size=(eval_samples, latent_dimension)),
        dtype=torch.float32,
    )

    rows = []
    for arm_index, arm in enumerate(arms):
        model, ledger = train_arm(
            arm,
            base,
            Y=Y,
            U=U,
            fixed_latent=fixed_latent,
            seed=180_000 + 1009 * seed + arm_index,
            macros=macros,
            inner_steps=inner_steps,
            learning_rate=learning_rate,
        )
        with torch.no_grad():
            values = model(eval_latent).cpu().numpy()
        support = calibrated_precision_coverage(
            values, eval_target, cal_a, cal_b)
        row = {
            "target": tgt.name,
            "seed": seed,
            "arm": arm,
            "ed2": energy_distance2(values, eval_target),
            "sw1": sliced_w1(
                values, eval_target, 256,
                np.random.default_rng(190_000 + seed)),
            "precision": support["precision"],
            "coverage": support["coverage"],
            "off_support": support["off_support"],
            "support_radius": support["radius"],
            "output_rms": float(np.sqrt(np.mean(values ** 2))),
            "model_parameters": sum(p.numel() for p in model.parameters()),
            **ledger,
        }
        if tgt.kind == "checkerboard":
            row["checkerboard_leakage"] = checkerboard_leakage(values)
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["target"], row["arm"])].append(row)
    fields = (
        "ed2", "sw1", "precision", "coverage", "off_support",
        "train_cohort_ed2_to_plan", "mean_route_switch_fraction",
        "assignment_seconds", "training_seconds", "target_accesses",
        "final_training_loss", "mean_teacher_route_sq_l2",
        "mean_teacher_sliced_agreement",
        "max_teacher_target_marginal_l1",
    )
    result = []
    for (target, arm), vals in sorted(buckets.items()):
        item = {"target": target, "arm": arm, "seeds": len(vals)}
        for field in fields:
            xs = np.asarray([v[field] for v in vals], dtype=float)
            finite = xs[np.isfinite(xs)]
            item[f"median_{field}"] = (
                float(np.median(finite)) if len(finite) else None)
        result.append(item)
    return result


def write_json(path: Path, payload: dict) -> None:
    def safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--macros", type=int, default=20)
    parser.add_argument("--inner-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--targets", type=str, default="all")
    parser.add_argument(
        "--arms",
        type=str,
        default=",".join(ARMS),
        help="comma-separated registered development arms",
    )
    parser.add_argument(
        "--out", type=Path,
        default=HERE / "route_amortization_development.json")
    args = parser.parse_args()
    if min(args.seeds, args.n, args.directions, args.macros,
           args.inner_steps, args.eval_samples) <= 0:
        raise ValueError("positive experiment sizes required")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    arms = tuple(args.arms.split(","))
    unknown_arms = set(arms) - set(ARMS)
    if unknown_arms or not arms:
        raise ValueError(f"unknown or empty arms: {sorted(unknown_arms)}")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    suite = T.suite()
    if args.targets != "all":
        wanted = set(args.targets.split(","))
        suite = [target for target in suite if target.name in wanted]
        missing = wanted - {target.name for target in suite}
        if missing:
            raise ValueError(f"unknown targets: {sorted(missing)}")

    started = time.time()
    rows = []
    for tgt in suite:
        for seed in range(args.seeds):
            rows.extend(run_cell(
                tgt, seed, N=args.n, L=args.directions,
                macros=args.macros, inner_steps=args.inner_steps,
                learning_rate=args.learning_rate,
                eval_samples=args.eval_samples, arms=arms))
        print(f"completed {tgt.name}", flush=True)
    aggs = aggregate(rows)
    payload = {
        "status": "route-amortization-development-not-confirmation",
        "config": vars(args) | {"out": str(args.out)},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "aggregate": aggs,
    }
    write_json(args.out, payload)

    print(
        f"{'target':18} {'arm':20} {'ED2':>8} {'SW1':>8} "
        f"{'prec':>6} {'cover':>7} {'trainED':>8} {'switch':>7} {'sec':>7}")
    for row in aggs:
        print(
            f"{row['target']:18} {row['arm']:20} "
            f"{row['median_ed2']:8.4f} {row['median_sw1']:8.4f} "
            f"{row['median_precision']:6.3f} {row['median_coverage']:7.3f} "
            f"{row['median_train_cohort_ed2_to_plan']:8.4f} "
            f"{row['median_mean_route_switch_fraction']:7.3f} "
            f"{row['median_training_seconds']:7.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
