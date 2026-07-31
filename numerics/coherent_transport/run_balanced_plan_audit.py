"""Corrected Stage-1 route audit for coherent sliced plans.

This audit deliberately does NOT call a deterministic row-wise modal route a
balanced coupling. It compares:

* modal_route: one target identity per source, target marginal unconstrained;
* balanced_est_dense: exact permutation maximizing sliced agreement;
* balanced_est_sparse: the same objective on the sparse EST support;
* euclidean_assignment: dense minimum-squared-distance permutation control;
* random_bijection: exact-marginal but geometrically uninformed control; and
* est_barycenter: averaged target identities, included as a failure control.

The endpoint of every bijection is the same planning empirical measure, so
endpoint quality alone cannot establish that EST found a better route. The
primary mechanism outcomes are target-marginal error, sliced agreement,
transport cost, sparse graph density, solve time, and one-step geometry.

Target planning, support calibration, and final evaluation pools are disjoint.
This is a development/audit experiment, not a fresh confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from est_plan import (  # noqa: E402
    all_rank_matches,
    balanced_est_assignment,
    est_barycenter,
    match_counts,
    modal_route,
    target_marginal_l1,
    unit_directions,
)
from metrics import (  # noqa: E402
    calibrated_precision_coverage,
    checkerboard_leakage,
    energy_distance2,
    sliced_w1,
)
import targets as T  # noqa: E402


def euclidean_assignment(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    cost = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    assign = np.empty(len(X), dtype=int)
    assign[rows] = cols
    return assign


def _route_row(
    *,
    target_name: str,
    seed: int,
    arm: str,
    X: np.ndarray,
    Y: np.ndarray,
    assign: np.ndarray | None,
    destination: np.ndarray,
    counts: np.ndarray,
    L: int,
    graph_density: float,
    solve_ms: float,
    cal_a: np.ndarray,
    cal_b: np.ndarray,
    eval_target: np.ndarray,
    eta: float,
    checkerboard: bool,
) -> dict:
    if assign is None:
        marginal_l1 = float("nan")
        unique_fraction = float("nan")
        agreement = float("nan")
    else:
        marginal_l1 = target_marginal_l1(assign)
        unique_fraction = float(len(np.unique(assign)) / len(assign))
        agreement = float(
            counts[np.arange(len(assign)), assign].mean() / max(L, 1))

    displacement = destination - X
    q_one = X + eta * displacement
    support = calibrated_precision_coverage(
        q_one, eval_target, cal_a, cal_b)
    endpoint_support = calibrated_precision_coverage(
        destination, eval_target, cal_a, cal_b)

    row = {
        "target": target_name,
        "seed": seed,
        "arm": arm,
        "target_marginal_l1": marginal_l1,
        "unique_target_fraction": unique_fraction,
        "mean_sliced_agreement": agreement,
        "route_mean_l2": float(np.linalg.norm(displacement, axis=1).mean()),
        "route_mean_sq_l2": float((displacement ** 2).sum(axis=1).mean()),
        "graph_density": graph_density,
        "solve_ms": solve_ms,
        "one_step_precision": support["precision"],
        "one_step_coverage": support["coverage"],
        "one_step_ed2": energy_distance2(q_one, eval_target),
        "one_step_sw1": sliced_w1(
            q_one, eval_target, 128,
            np.random.default_rng(8_000_000 + seed)),
        "endpoint_precision": endpoint_support["precision"],
        "endpoint_coverage": endpoint_support["coverage"],
        "endpoint_ed2": energy_distance2(destination, eval_target),
    }
    if checkerboard:
        row["one_step_checkerboard_leakage"] = checkerboard_leakage(q_one)
        row["endpoint_checkerboard_leakage"] = checkerboard_leakage(destination)
    return row


def run_cell(tgt: T.GeoTarget, seed: int, N: int, L: int, eta: float) -> list[dict]:
    # Four disjoint target pools. The calibration pools determine only the
    # target-only support radius; the evaluation pool is never used by a route.
    Y = tgt.sampler(N, np.random.default_rng(10_000 + 101 * seed))
    cal_a = tgt.sampler(512, np.random.default_rng(20_000 + 101 * seed))
    cal_b = tgt.sampler(512, np.random.default_rng(30_000 + 101 * seed))
    eval_target = tgt.sampler(1024, np.random.default_rng(40_000 + 101 * seed))
    X = np.random.default_rng(50_000 + 101 * seed).normal(
        size=(N, 2)) * tgt.scale
    U = unit_directions(L, 2, 60_000 + seed)
    Pi = all_rank_matches(X, Y, U)
    counts = match_counts(Pi)
    graph_density = float((counts > 0).mean())
    rows: list[dict] = []

    t0 = time.perf_counter()
    modal_dest, modal_assign = modal_route(Y, Pi)
    modal_ms = 1000 * (time.perf_counter() - t0)
    rows.append(_route_row(
        target_name=tgt.name, seed=seed, arm="modal_route",
        X=X, Y=Y, assign=modal_assign, destination=modal_dest,
        counts=counts, L=L, graph_density=graph_density, solve_ms=modal_ms,
        cal_a=cal_a, cal_b=cal_b, eval_target=eval_target, eta=eta,
        checkerboard=tgt.kind == "checkerboard"))

    for method in ("dense", "sparse"):
        t0 = time.perf_counter()
        dest, assign = balanced_est_assignment(Y, Pi, method=method)
        solve_ms = 1000 * (time.perf_counter() - t0)
        rows.append(_route_row(
            target_name=tgt.name, seed=seed,
            arm=f"balanced_est_{method}",
            X=X, Y=Y, assign=assign, destination=dest,
            counts=counts, L=L, graph_density=graph_density, solve_ms=solve_ms,
            cal_a=cal_a, cal_b=cal_b, eval_target=eval_target, eta=eta,
            checkerboard=tgt.kind == "checkerboard"))

    t0 = time.perf_counter()
    euclid_assign = euclidean_assignment(X, Y)
    euclid_ms = 1000 * (time.perf_counter() - t0)
    rows.append(_route_row(
        target_name=tgt.name, seed=seed, arm="euclidean_assignment",
        X=X, Y=Y, assign=euclid_assign, destination=Y[euclid_assign],
        counts=counts, L=L, graph_density=1.0, solve_ms=euclid_ms,
        cal_a=cal_a, cal_b=cal_b, eval_target=eval_target, eta=eta,
        checkerboard=tgt.kind == "checkerboard"))

    random_assign = np.random.default_rng(70_000 + seed).permutation(N)
    rows.append(_route_row(
        target_name=tgt.name, seed=seed, arm="random_bijection",
        X=X, Y=Y, assign=random_assign, destination=Y[random_assign],
        counts=counts, L=L, graph_density=1.0, solve_ms=0.0,
        cal_a=cal_a, cal_b=cal_b, eval_target=eval_target, eta=eta,
        checkerboard=tgt.kind == "checkerboard"))

    bary = est_barycenter(Y, Pi)
    rows.append(_route_row(
        target_name=tgt.name, seed=seed, arm="est_barycenter",
        X=X, Y=Y, assign=None, destination=bary,
        counts=counts, L=L, graph_density=graph_density, solve_ms=0.0,
        cal_a=cal_a, cal_b=cal_b, eval_target=eval_target, eta=eta,
        checkerboard=tgt.kind == "checkerboard"))
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["target"], row["arm"])].append(row)
    fields = [
        "target_marginal_l1", "unique_target_fraction",
        "mean_sliced_agreement", "route_mean_l2", "route_mean_sq_l2",
        "graph_density", "solve_ms", "one_step_precision",
        "one_step_coverage", "one_step_ed2", "one_step_sw1",
        "endpoint_precision", "endpoint_coverage", "endpoint_ed2",
    ]
    out = []
    for (target, arm), vals in sorted(buckets.items()):
        item = {"target": target, "arm": arm, "seeds": len(vals)}
        for field in fields:
            xs = np.asarray([v[field] for v in vals], dtype=float)
            finite = xs[np.isfinite(xs)]
            item[f"median_{field}"] = (
                float(np.median(finite)) if len(finite) else None)
        out.append(item)
    return out


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
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "  " + path.name + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "balanced_plan_audit.json",
    )
    args = parser.parse_args()
    if args.seeds <= 0 or args.n <= 1 or args.directions <= 0:
        raise ValueError("seeds, n, and directions must be positive")
    if not 0 < args.eta <= 1:
        raise ValueError("eta must lie in (0, 1]")

    started = time.time()
    rows = []
    for tgt in T.suite():
        for seed in range(args.seeds):
            rows.extend(run_cell(
                tgt, seed, args.n, args.directions, args.eta))
        print(f"completed {tgt.name}", flush=True)

    aggs = aggregate(rows)
    payload = {
        "status": "development-route-audit-not-confirmation",
        "config": {
            "seeds": args.seeds,
            "n": args.n,
            "directions": args.directions,
            "eta": args.eta,
            "target_pools": "planning/calibration_a/calibration_b/evaluation disjoint",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "aggregate": aggs,
    }
    write_json(args.out, payload)

    print(
        f"{'target':18} {'arm':24} {'margL1':>7} {'uniq':>6} "
        f"{'agree':>7} {'cost':>7} {'prec1':>7} {'cov1':>7} {'ms':>7}")
    for row in aggs:
        def fmt(field, width=7):
            value = row[field]
            return f"{value:{width}.3f}" if value is not None else f"{'-':>{width}}"
        print(
            f"{row['target']:18} {row['arm']:24} "
            f"{fmt('median_target_marginal_l1')} "
            f"{fmt('median_unique_target_fraction', 6)} "
            f"{fmt('median_mean_sliced_agreement')} "
            f"{fmt('median_route_mean_l2')} "
            f"{fmt('median_one_step_precision')} "
            f"{fmt('median_one_step_coverage')} "
            f"{fmt('median_solve_ms')}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
