"""D4: learned-generator transfer test (LowDimPerformanceRoadmap.md).

A small fixed MLP generator G_theta : R^2 -> R^d is trained by the drifting
update: samples x = G(z) are regressed toward stopgrad(x + eta * V_hat(x)),
where V_hat is the exact paper bi-softmax drift under a frozen policy (base
or modified).  Base and modified arms share architecture, initialization,
latent stream, target stream, optimizer, and update count.

Usage:
    uv run --with numpy --with scipy python numerics/lowdim_benchmark.py D2   # freeze first
    uv run --with numpy --with scipy python numerics/lowdim_generator.py [--profile smoke|standard]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.linalg import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lowdim_drift import (  # noqa: E402
    MASTER, Run, TargetSpec, WorkCounter, drift_paper, energy_distance2,
    gauss_mixture, moons_target, paired_bootstrap_ratio, skew_target,
    sliced_w1,
)
from lowdim_benchmark import (  # noqa: E402
    FixedPolicy, load_frozen, make_modified_policy,
)


def d4_targets() -> list[TargetSpec]:
    return [
        gauss_mixture("H-1d-K4-eq", 4, 1, 0.15),
        gauss_mixture("H-2d-K6-sep", 6, 2, 0.10),
        moons_target("H-moons"),
        skew_target("H-skew"),
    ]


# ---------------------------------------------------------------------------
# Minimal MLP + Adam (numpy, manual backprop)
# ---------------------------------------------------------------------------


class MLP:
    def __init__(self, d_out: int, rng: np.random.Generator,
                 hidden: int = 64):
        s = 1.0
        self.p = {
            "W1": rng.normal(0, s / np.sqrt(2), (hidden, 2)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0, s / np.sqrt(hidden), (hidden, hidden)),
            "b2": np.zeros(hidden),
            "W3": rng.normal(0, s / np.sqrt(hidden), (d_out, hidden)),
            "b3": np.zeros(d_out),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, z: np.ndarray):
        a1 = z @ self.p["W1"].T + self.p["b1"]
        h1 = np.tanh(a1)
        a2 = h1 @ self.p["W2"].T + self.p["b2"]
        h2 = np.tanh(a2)
        out = h2 @ self.p["W3"].T + self.p["b3"]
        return out, (z, h1, h2)

    def backward(self, gout: np.ndarray, cache) -> dict:
        z, h1, h2 = cache
        g = {}
        g["W3"] = gout.T @ h2
        g["b3"] = gout.sum(axis=0)
        gh2 = gout @ self.p["W3"]
        ga2 = gh2 * (1 - h2 ** 2)
        g["W2"] = ga2.T @ h1
        g["b2"] = ga2.sum(axis=0)
        gh1 = ga2 @ self.p["W2"]
        ga1 = gh1 * (1 - h1 ** 2)
        g["W1"] = ga1.T @ z
        g["b1"] = ga1.sum(axis=0)
        return g

    def adam(self, g: dict, lr: float = 1e-3, b1: float = 0.9,
             b2: float = 0.999, eps: float = 1e-8) -> None:
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)


def gradcheck(log) -> None:
    rng = np.random.default_rng(MASTER)
    net = MLP(2, rng, hidden=5)
    z = rng.normal(size=(3, 2))
    target = rng.normal(size=(3, 2))
    out, cache = net.forward(z)
    gout = (out - target) / len(z)
    g = net.backward(gout, cache)
    h = 1e-6
    for k in ("W1", "b2", "W3"):
        flat = net.p[k].ravel()
        idx = rng.integers(0, flat.size, size=3)
        for i in idx:
            old = flat[i]
            flat[i] = old + h
            lp = 0.5 * np.sum((net.forward(z)[0] - target) ** 2) / len(z)
            flat[i] = old - h
            lm = 0.5 * np.sum((net.forward(z)[0] - target) ** 2) / len(z)
            flat[i] = old
            num = (lp - lm) / (2 * h)
            assert abs(num - g[k].ravel()[i]) < 1e-5, f"gradcheck {k}"
    log("  invariant MLP gradcheck: PASS")
    # matched-batch zero-update: V(x; data=x, no mask) = 0 -> zero gradient
    out, cache = net.forward(z)
    V = drift_paper(out, out.copy(), 0.7, False)
    assert norm(V) < 1e-11
    g0 = net.backward(-V / len(z), cache)
    assert all(norm(v) < 1e-11 for v in g0.values())
    log("  invariant matched-batch zero generator update: PASS")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_generator(target: TargetSpec, policy, seed: int, updates: int,
                    zbatch: int, counter: WorkCounter) -> tuple[float, float]:
    init_rng = np.random.default_rng(MASTER * 13 + seed * 100 + 1)
    latent_rng = np.random.default_rng(MASTER * 13 + seed * 100 + 2)
    data_rng = np.random.default_rng(MASTER * 13 + seed * 100 + 3)
    policy_rng = np.random.default_rng(MASTER * 13 + seed * 100 + 4)
    net = MLP(target.d, init_rng)
    setup = target.sample(256, data_rng)
    policy.setup(setup, zbatch, policy_rng, counter)
    obs: dict = {"last_V": None, "counter": counter}
    for step in range(updates):
        z = latent_rng.normal(size=(zbatch, 2))
        data = target.sample(zbatch, data_rng)
        x, cache = net.forward(z)
        dec = policy.decide(step, updates, obs)
        V = drift_paper(x, data, dec.tau, dec.mask, counter)
        obs["last_V"] = V
        obs["q"] = x
        obs["data"] = data
        # regression toward stopgrad(x + eta V): residual is -eta V
        gout = (-dec.eta * V) / zbatch
        net.adam(net.backward(gout, cache))
    z_eval = np.random.default_rng(MASTER + 55).normal(size=(2048, 2))
    x_eval, _ = net.forward(z_eval)
    ref = target.sample(2048, np.random.default_rng(MASTER + 56))
    return (energy_distance2(x_eval, ref),
            sliced_w1(x_eval, ref, 32, np.random.default_rng(MASTER + 57)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="standard",
                    choices=["smoke", "standard"])
    args = ap.parse_args()
    updates = 300 if args.profile == "smoke" else 2000
    seeds = 2 if args.profile == "smoke" else 10
    base = load_frozen("*D0-baseline*", "baseline_frozen.json")
    pol = load_frozen("*D2-policy*", "policy_frozen.json")
    run = Run("D4-generator", {"profile": args.profile, "baseline": base,
                               "policy": pol, "updates": updates,
                               "seeds": seeds, "zbatch": 64,
                               "arch": "2-64-64-d tanh MLP, Adam 1e-3"})
    run.log(f"D4: learned-generator transfer; base=(tau*={base['tau_star']},"
            f" eta*={base['eta_abs']:.4f}, mask on) vs "
            f"modified={pol['type']}")
    gradcheck(run.log)
    rows = []
    all_base, all_mod = [], []
    for tgt in d4_targets():
        eb_list, em_list = [], []
        for s in range(seeds):
            base_policy = FixedPolicy("base", base["tau_star"],
                                      base["eta_abs"], True)
            mod_policy = make_modified_policy(pol)
            eb, sb = train_generator(tgt, base_policy, s, updates, 64,
                                     run.counter)
            em, sm = train_generator(tgt, mod_policy, s, updates, 64,
                                     run.counter)
            eb_list.append(eb)
            em_list.append(em)
            all_base.append(eb)
            all_mod.append(em)
            rows.append((tgt.name, s, eb, em, sb, sm))
        run.log(f"  {tgt.name:14s}: base {np.median(eb_list):.5f}  "
                f"mod {np.median(em_list):.5f}  ratio "
                f"{np.median(em_list) / max(np.median(eb_list), 1e-12):.3f}")
    point, lo, hi = paired_bootstrap_ratio(np.asarray(all_mod),
                                           np.asarray(all_base))
    run.log(f"  aggregate paired ratio {point:.3f} CI[{lo:.3f},{hi:.3f}]  "
            f"({'modified wins' if hi < 1 else 'no significant transfer'})")
    run.save_csv("d4_results.csv",
                 ["target", "seed", "base_ed2", "mod_ed2", "base_sw1",
                  "mod_sw1"], rows)
    run.save_json("d4_gate.json",
                  {"aggregate_ratio": point, "ci": [lo, hi],
                   "pass": hi < 1.0})
    run.finish()


if __name__ == "__main__":
    main()
