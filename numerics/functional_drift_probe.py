"""Exploratory functional-preconditioning probe for drifting generators.

The particle NCJ field passed its gate but failed after composition with a
shared-parameter generator.  This probe asks whether the missing bridge is the
parameter optimizer: replace Adam's ``J^T V`` update by a damped natural step

    delta_theta = J^T (J J^T + lambda I)^-1 (eta V),

which is the minimum-norm local parameter update whose *output-space* motion
tracks the requested drifting field.  This is an exploratory mechanism test,
not a frozen comparison and not evidence of a scalable method.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NUMERICS = ROOT / "numerics"
if str(NUMERICS) not in sys.path:
    sys.path.insert(0, str(NUMERICS))

from identifiability_drift import compute_field  # noqa: E402
from lowdim_drift import (  # noqa: E402
    controlled_means,
    energy_distance2,
    sliced_w1,
)
from mode_recovery import (  # noqa: E402
    basin_radius,
    coverage,
    sigma_radius,
)
from run_identifiability_generator import TanhMLP  # noqa: E402
from sliced_fission_probe import RingMixture  # noqa: E402


def hard_target(k: int = 32, d: int = 2, sigma: float = 0.02) -> RingMixture:
    means = controlled_means(k, d, 1.0)
    scale = float(np.linalg.norm(means, axis=1).max() + 3 * sigma)
    return RingMixture(f"hard-K{k}-d{d}", d, scale, means, sigma)


def batch_jacobian(model: TanhMLP, cache) -> np.ndarray:
    """Exact Jacobian of flattened batch outputs in ``model.names`` order."""
    z, h1, h2 = cache
    p = model.params
    n, dout = len(z), p["b3"].shape[0]
    rows = []
    for i in range(n):
        for c in range(dout):
            e = np.zeros(dout)
            e[c] = 1.0
            db2 = p["W3"][:, c] * (1.0 - h2[i] * h2[i])
            db1 = (p["W2"] @ db2) * (1.0 - h1[i] * h1[i])
            pieces = {
                "W1": np.outer(z[i], db1),
                "b1": db1,
                "W2": np.outer(h1[i], db2),
                "b2": db2,
                "W3": np.outer(h2[i], e),
                "b3": e,
            }
            rows.append(np.concatenate(
                [pieces[name].reshape(-1) for name in model.names]))
    return np.asarray(rows)


def apply_flat_update(model: TanhMLP, delta: np.ndarray) -> None:
    start = 0
    for name in model.names:
        size = model.params[name].size
        model.params[name] += delta[start:start + size].reshape(
            model.params[name].shape)
        start += size
    if start != len(delta):
        raise ValueError("flat parameter update has the wrong size")


def natural_step(model: TanhMLP, z: np.ndarray, cache, field: np.ndarray,
                 eta: float, damping: float) -> tuple[float, float]:
    """Damped output-natural step with a same-batch linearization diagnostic."""
    x = model.forward(z)
    J = batch_jacobian(model, cache)
    gram = J @ J.T
    scale = max(float(np.trace(gram) / len(gram)), 1e-12)
    rhs = (eta * field).reshape(-1)
    coeff = np.linalg.solve(gram + damping * scale * np.eye(len(gram)), rhs)
    delta = J.T @ coeff
    predicted = J @ delta
    rel_linear_error = float(
        np.linalg.norm(predicted - rhs) / max(np.linalg.norm(rhs), 1e-12))
    apply_flat_update(model, delta)
    moved = model.forward(z) - x
    rel_realized_error = float(
        np.linalg.norm(moved.reshape(-1) - rhs) /
        max(np.linalg.norm(rhs), 1e-12))
    return rel_linear_error, rel_realized_error


def jacobian_check() -> None:
    target = hard_target(4, 2, 0.05)
    model = TanhMLP(target, "broad", 41)
    z = np.random.default_rng(42).normal(size=(3, model.latent_dim))
    x, cache = model.forward(z, want_cache=True)
    J = batch_jacobian(model, cache)
    rng = np.random.default_rng(43)
    direction = rng.normal(size=J.shape[1])
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    apply_flat_update(model, eps * direction)
    x1 = model.forward(z)
    apply_flat_update(model, -eps * direction)
    fd = (x1 - x).reshape(-1) / eps
    rel = np.linalg.norm(fd - J @ direction) / max(np.linalg.norm(fd), 1e-12)
    if rel > 2e-5:
        raise AssertionError(f"batch Jacobian check failed: {rel}")


def train(target: RingMixture, seed: int, arm: str, tau: float,
          updates: int = 400, batch: int = 32, eta: float = 0.08,
          damping: float = 1e-2, warm_fraction: float = 0.4,
          refine_tau: float = 0.2) -> dict:
    model = TanhMLP(target, "far", seed * 10_000 + 1)
    latent_rng = np.random.default_rng(seed * 10_000 + 2)
    reference_rng = np.random.default_rng(seed * 10_000 + 3)
    data_rng = np.random.default_rng(seed * 10_000 + 4)
    linerr = []
    realerr = []
    natural_steps = int(round(warm_fraction * updates))
    for step in range(updates):
        z = latent_rng.normal(size=(batch, model.latent_dim))
        x, cache = model.forward(z, want_cache=True)
        positive = target.sample(batch, data_rng)
        hybrid_refine = arm == "ncf-natural-paper" and step >= natural_steps
        if arm.startswith("paper") or hybrid_refine:
            field = compute_field(
                x, positive, tau=refine_tau if hybrid_refine else tau,
                gain="paper", mask=True,
                on_degenerate="zero").V
        else:
            zr = reference_rng.normal(size=(batch, model.latent_dim))
            negative = model.forward(zr)
            field = compute_field(
                x, positive, negative, tau=tau, gain="constant", mask=False,
                on_degenerate="zero").V
        if arm.endswith("adam") or hybrid_refine:
            model.stopgrad_step(cache, field)
        else:
            le, re = natural_step(model, z, cache, field, eta, damping)
            linerr.append(le)
            realerr.append(re)
        if not model.finite():
            raise FloatingPointError("non-finite model")

    eval_rng = np.random.default_rng(seed * 10_000 + 5)
    q = model.forward(eval_rng.normal(size=(2048, model.latent_dim)))
    p = target.sample(2048, np.random.default_rng(seed * 10_000 + 6))
    weights = np.full(len(target.means), 1.0 / len(target.means))
    reach = coverage(q, target.means, basin_radius(target.means), weights)
    resolve = coverage(
        q, target.means,
        sigma_radius(target.sigma, K=len(target.means)), weights)
    metric_rng = np.random.default_rng(seed * 10_000 + 7)
    iq = metric_rng.choice(len(q), 768, replace=False)
    ip = metric_rng.choice(len(p), 768, replace=False)
    return {
        "arm": arm,
        "tau": tau,
        "seed": seed,
        "reach": reach["unweighted"],
        "resolve": resolve["unweighted"],
        "ed2": max(0.0, energy_distance2(q[iq], p[ip])),
        "sw1": sliced_w1(q, p, 32, metric_rng),
        "linerr": float(np.median(linerr)) if linerr else math.nan,
        "realerr": float(np.median(realerr)) if realerr else math.nan,
    }


def main() -> None:
    jacobian_check()
    target = hard_target(32, 2, 0.02)
    rows = []
    for seed in range(3):
        for tau in (0.2, 0.4):
            for arm in ("paper-adam", "paper-natural",
                        "ncf-adam", "ncf-natural"):
                rows.append(train(target, seed, arm, tau))
    print("exploratory output-natural drifting probe (median over 3 seeds)")
    print("arm            tau reach resolve     ED2     SW1  linerr realerr")
    for arm in ("paper-adam", "paper-natural", "ncf-adam", "ncf-natural"):
        for tau in (0.2, 0.4):
            group = [r for r in rows if r["arm"] == arm and r["tau"] == tau]
            med = {key: float(np.nanmedian([r[key] for r in group]))
                   for key in ("reach", "resolve", "ed2", "sw1",
                               "linerr", "realerr")}
            print(f"{arm:15s} {tau:3.1f} {med['reach']:5.2f} "
                  f"{med['resolve']:6.2f} {med['ed2']:7.4f} "
                  f"{med['sw1']:7.4f} {med['linerr']:7.3f} "
                  f"{med['realerr']:7.3f}")


if __name__ == "__main__":
    main()
