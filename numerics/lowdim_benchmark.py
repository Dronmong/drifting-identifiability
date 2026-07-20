"""D0-D3 runner for the low-dimensional performance program.

Protocol: LowDimPerformanceProtocol.md (frozen before running).

Usage:
    uv run --with numpy --with scipy python numerics/lowdim_benchmark.py \
        [D0|D1|D2|D3] [--profile smoke|standard]

D0 tunes and freezes the exact-estimator baseline on VALIDATION targets.
D1 runs the one-factor ablations on VALIDATION targets.
D2 selects the adaptive annealing trigger and freezes the modified policy.
D3 runs the held-out paired benchmark and evaluates the pre-declared gate.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.linalg import eigvals, norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lowdim_drift import (  # noqa: E402
    FIELD, MASTER, Policy, Run, StepDecision, TargetSpec, WorkCounter,
    drift_paper, energy_distance2, estimate_geometry, field_invariants,
    heldout_targets, init_cloud, km_median, paired_bootstrap_ratio,
    sliced_w1, train, validation_targets,
)

RUNROOT = HERE / "lowdim_runs"


@dataclass(frozen=True)
class Profile:
    name: str
    steps: int
    tune_seeds: int
    d1_seeds: int
    d2_seeds: int
    d3_seeds: int
    ref_final: int
    ref_cross: int
    N: int
    batch: int


PROFILES = {
    "smoke": Profile("smoke", 60, 1, 2, 2, 3, 256, 128, 24, 32),
    "standard": Profile("standard", 400, 3, 6, 6, 20, 1024, 256, 48, 64),
}


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class FixedPolicy(Policy):
    """Frozen (tau, eta, mask): the baseline shape."""

    def __init__(self, name: str, tau: float, eta: float, mask: bool):
        self.name, self.tau, self.eta, self.mask = name, tau, eta, mask

    def decide(self, step, steps, obs):
        return StepDecision(self.tau, self.eta, self.mask)


class GeometryPolicy(Policy):
    """Base class: estimates (K_hat, L_hat, sigma_hat) from the setup sample
    only.  Estimation is observable information, not oracle."""

    def __init__(self, name: str):
        self.name = name
        self.geo: dict = {}

    def setup(self, setup_sample, N, rng, counter):
        self.geo = estimate_geometry(setup_sample, rng)
        self.N = N


class CoarsePolicy(GeometryPolicy):
    def __init__(self, eta: float, mask: bool):
        super().__init__("coarse-only")
        self.eta, self.mask = eta, mask

    def decide(self, step, steps, obs):
        return StepDecision(self.geo["L_hat"], self.eta, self.mask)


class C2FPolicy(GeometryPolicy):
    def __init__(self, eta: float, mask: bool, joint_eta: bool = False,
                 eta_cap: float | None = None):
        super().__init__("c2f-only" if not joint_eta else "combined")
        self.eta, self.mask_cfg, self.joint = eta, mask, joint_eta
        self.eta_cap = eta_cap
        self.ceiling: float | None = None

    def setup(self, setup_sample, N, rng, counter):
        super().setup(setup_sample, N, rng, counter)
        if self.joint and self.eta_cap is not None:
            self.ceiling = surrogate_euler_ceiling(
                setup_sample, self.geo["L_hat"], rng, counter)

    def decide(self, step, steps, obs):
        frac = min(1.0, step / max(1.0, 0.7 * steps))
        tau = self.geo["L_hat"] * \
            (self.geo["sigma_hat"] / self.geo["L_hat"]) ** frac
        if self.joint:
            eta = 0.1 * tau
            if self.ceiling is not None and self.eta_cap is not None:
                eta = min(eta, self.eta_cap * self.ceiling)
            mask = self.N >= 8 * self.geo["K_hat"] if self.mask_cfg == "auto" \
                else bool(self.mask_cfg)
            return StepDecision(tau, eta, mask)
        return StepDecision(tau, self.eta, bool(self.mask_cfg))


class StepOnlyPolicy(GeometryPolicy):
    """tau*, mask on; eta from the coupled-generator safety rule."""

    def __init__(self, tau_star: float, kind: str, value: float):
        super().__init__(f"step-{kind}-{value}")
        self.tau_star, self.kind, self.value = tau_star, kind, value
        self.eta = None

    def setup(self, setup_sample, N, rng, counter):
        super().setup(setup_sample, N, rng, counter)
        if self.kind == "ceil":
            ceiling = surrogate_euler_ceiling(
                setup_sample, self.tau_star, rng, counter)
            self.eta = self.value * ceiling if ceiling is not None \
                else 0.1 * self.tau_star
        else:                                    # fixed multiple of tau
            self.eta = self.value * self.tau_star

    def decide(self, step, steps, obs):
        return StepDecision(self.tau_star, self.eta, True)


class MaskPolicy(GeometryPolicy):
    def __init__(self, tau_star: float, eta: float):
        super().__init__("mask-only")
        self.tau_star, self.eta = tau_star, eta

    def decide(self, step, steps, obs):
        mask = self.N >= 8 * self.geo["K_hat"]
        return StepDecision(self.tau_star, self.eta, mask)


class AdaptivePolicy(GeometryPolicy):
    """D2 candidate: coarse at L_hat until an observable trigger, then
    geometric anneal to sigma_hat over the remaining budget.  Step and mask
    follow the combined rule.  Fallback: fixed70."""

    def __init__(self, trigger: str, eta_cap: float):
        super().__init__(f"adaptive-{trigger}")
        self.trigger = trigger
        self.eta_cap = eta_cap
        self.trigger_step: int | None = None
        self.vhist: list[float] = []
        self.ceiling: float | None = None

    def setup(self, setup_sample, N, rng, counter):
        super().setup(setup_sample, N, rng, counter)
        self.trigger_step = None
        self.vhist = []
        self.ceiling = surrogate_euler_ceiling(
            setup_sample, self.geo["L_hat"], rng, counter)

    def _check_trigger(self, step, steps, obs) -> bool:
        if self.trigger == "fixed70":
            return step >= 0.7 * steps
        if step >= 0.7 * steps:                  # fallback in all cases
            return True
        if obs.get("last_V") is None:
            return False
        if self.trigger == "plateau":
            self.vhist.append(float(np.mean(norm(obs["last_V"], axis=1))))
            if len(self.vhist) >= 40:
                recent = np.mean(self.vhist[-20:])
                older = np.mean(self.vhist[-40:-20])
                if older > 0 and abs(older - recent) / older < 0.15:
                    return True
            return False
        if self.trigger == "agree":
            if step % 10 != 0 or "q" not in obs:
                return False
            counter: WorkCounter = obs["counter"]
            v_fine = drift_paper(obs["q"], obs["data"],
                                 self.geo["sigma_hat"], True, counter)
            v_coarse = obs["last_V"]
            num = np.sum(v_fine * v_coarse, axis=1)
            den = norm(v_fine, axis=1) * norm(v_coarse, axis=1) + 1e-12
            return bool(np.mean(num / den) > 0.8)
        raise ValueError(self.trigger)

    def decide(self, step, steps, obs):
        if self.trigger_step is None and self._check_trigger(step, steps, obs):
            self.trigger_step = step
        if self.trigger_step is None:
            tau = self.geo["L_hat"]
        else:
            rem = max(1, steps - self.trigger_step)
            frac = min(1.0, (step - self.trigger_step) / rem)
            tau = self.geo["L_hat"] * \
                (self.geo["sigma_hat"] / self.geo["L_hat"]) ** frac
        eta = 0.1 * tau
        if self.ceiling is not None:
            eta = min(eta, self.eta_cap * self.ceiling)
        mask = self.N >= 8 * self.geo["K_hat"]
        return StepDecision(tau, eta, mask)


def surrogate_euler_ceiling(setup_sample: np.ndarray, tau: float,
                            rng: np.random.Generator,
                            counter: WorkCounter) -> float | None:
    """One-time full central-difference Jacobian of the UNMASKED paper field
    on a 16-point surrogate at its own empirical truth (an exact equilibrium
    by matched-batch cancellation).  Cost is charged to the counter."""
    m = min(16, len(setup_sample))
    idx = rng.choice(len(setup_sample), size=m, replace=False)
    q = setup_sample[idx]
    data = q.copy()
    base = q.ravel()
    h = 1e-5 * tau
    J = np.empty((base.size, base.size))

    def f(v):
        return drift_paper(v.reshape(q.shape), data, tau, False,
                           counter).ravel()

    for i in range(base.size):
        e = np.zeros_like(base)
        e[i] = h
        J[:, i] = (f(base + e) - f(base - e)) / (2 * h)
    eigs = eigvals(J)
    if np.max(eigs.real) >= -1e-8:
        return None
    return float(np.min(-2 * eigs.real / np.abs(eigs) ** 2))


# ---------------------------------------------------------------------------
# Shared evaluation
# ---------------------------------------------------------------------------


def run_one(target: TargetSpec, policy: Policy, seed: int, prof: Profile,
            counter: WorkCounter, init_kind: str = "missing",
            steps: int | None = None):
    steps = steps or prof.steps
    init_rng = np.random.default_rng(MASTER * 7 + seed * 1000 + 1)
    q0 = init_cloud(init_kind, target, prof.N, init_rng)
    data_rng = np.random.default_rng(MASTER * 7 + seed * 1000 + 2)
    policy_rng = np.random.default_rng(MASTER * 7 + seed * 1000 + 3)
    ref_final = target.sample(prof.ref_final,
                              np.random.default_rng(MASTER + 11))
    ref_cross = target.sample(prof.ref_cross,
                              np.random.default_rng(MASTER + 12))
    ed_tol = 0.05 * target.scale
    return train(q0, target, policy, steps, prof.batch, data_rng, policy_rng,
                 ref_final, ref_cross, ed_tol, counter)


def geo_mean(values) -> float:
    v = np.maximum(np.asarray(values, dtype=float), 1e-12)
    return float(np.exp(np.mean(np.log(v))))


# ---------------------------------------------------------------------------
# D0: baseline tuning and freezing
# ---------------------------------------------------------------------------


def D0(prof: Profile) -> None:
    config = {
        "profile": prof.name,
        "tau_grid": [0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.75],
        "eta_frac_grid": [0.05, 0.15, 0.4],
        "paper_temps": [0.02, 0.05, 0.2],
    }
    run = Run("D0-baseline", config)
    run.log("D0: tune + freeze the exact-estimator baseline (mask ON)")
    field_invariants(run.log)
    targets = validation_targets()
    rows = []
    table = {}
    for tau in config["tau_grid"]:
        for frac in config["eta_frac_grid"]:
            per_target = []
            for tgt in targets:
                eds = []
                for s in range(prof.tune_seeds):
                    res = run_one(tgt, FixedPolicy("tune", tau, frac * tau,
                                                   True),
                                  10_000 + s, prof, run.counter)
                    eds.append(res.final_ed2)
                per_target.append(float(np.median(eds)))
                rows.append((tau, frac, tgt.name, float(np.median(eds))))
            table[(tau, frac)] = geo_mean(per_target)
    best = min(table, key=table.get)
    run.log(f"  grid best: tau*={best[0]}, eta*={best[1]}*tau "
            f"(geo-mean ED2 {table[best]:.5f})")
    # reference arm: literal paper temperature set (averaged field)
    ref_scores = []
    for tgt in targets:
        eds = []
        for s in range(prof.tune_seeds):

            class PaperTemps(Policy):
                name = "paper-temps"

                def decide(self, step, steps, obs):
                    tau = config["paper_temps"][step % 3]
                    return StepDecision(tau, best[1] * tau, True)

            res = run_one(tgt, PaperTemps(), 10_000 + s, prof, run.counter)
            eds.append(res.final_ed2)
        ref_scores.append(float(np.median(eds)))
        rows.append(("paper-temps", best[1], tgt.name,
                     float(np.median(eds))))
    run.log(f"  paper-temperature reference geo-mean ED2: "
            f"{geo_mean(ref_scores):.5f}")
    frozen = {
        "tau_star": best[0],
        "eta_frac": best[1],
        "eta_abs": best[0] * best[1],
        "mask": True,
        "score": table[best],
        "score_table": {f"{k[0]}_{k[1]}": v for k, v in table.items()},
        "tuning_kernel_pairs": run.counter.kernel_pairs,
    }
    run.save_json("baseline_frozen.json", frozen)
    run.save_csv("d0_grid.csv", ["tau", "eta_frac", "target", "median_ed2"],
                 rows)
    run.log(f"  frozen baseline: tau*={frozen['tau_star']} "
            f"eta*={frozen['eta_abs']:.4f} mask=on; "
            f"tuning cost {frozen['tuning_kernel_pairs']:.3e} kernel pairs")
    run.finish()


def load_frozen(pattern: str, name: str) -> dict:
    cands = sorted(glob.glob(str(RUNROOT / pattern / name)))
    if not cands:
        raise FileNotFoundError(f"no {name} found; run earlier stage first")
    with open(cands[-1], encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# D1: exact-estimator ablations
# ---------------------------------------------------------------------------


def D1(prof: Profile) -> None:
    base = load_frozen("*D0-baseline*", "baseline_frozen.json")
    run = Run("D1-ablations", {"profile": prof.name, "baseline": base})
    run.log(f"D1: one-factor ablations (exact estimator); baseline "
            f"tau*={base['tau_star']} eta*={base['eta_abs']:.4f}")
    tau_s, eta_s = base["tau_star"], base["eta_abs"]

    def arms():
        yield "base", lambda: FixedPolicy("base", tau_s, eta_s, True)
        yield "coarse-only", lambda: CoarsePolicy(eta_s, True)
        yield "c2f-only", lambda: C2FPolicy(eta_s, True)
        for v in (0.05, 0.1, 0.25):
            yield f"step-ceil-{v}", \
                lambda v=v: StepOnlyPolicy(tau_s, "ceil", v)
        for v in (0.1, 0.5):
            yield f"step-taumult-{v}", \
                lambda v=v: StepOnlyPolicy(tau_s, "taumult", v)
        yield "mask-only", lambda: MaskPolicy(tau_s, eta_s)
        yield "combined", lambda: C2FPolicy(eta_s, "auto", joint_eta=True,
                                            eta_cap=0.25)

    rows = []
    aggregates = {}
    for arm_name, make in arms():
        per_target = []
        for tgt in validation_targets():
            eds, pairs = [], []
            for s in range(prof.d1_seeds):
                res = run_one(tgt, make(), 20_000 + s, prof, run.counter)
                eds.append(res.final_ed2)
                pairs.append(res.kernel_pairs)
                rows.append((arm_name, tgt.name, s, res.final_ed2,
                             res.final_sw1, res.coverage, res.mass_error,
                             res.event_time, int(res.censored),
                             res.kernel_pairs, int(res.diverged)))
            per_target.append(float(np.median(eds)))
        aggregates[arm_name] = geo_mean(per_target)
        run.log(f"  {arm_name:18s}: geo-mean ED2 = {aggregates[arm_name]:.5f}"
                f"  (median pairs/run {np.median(pairs):.2e})")
    run.save_csv("d1_results.csv",
                 ["arm", "target", "seed", "ed2", "sw1", "coverage",
                  "mass_err", "event_time", "censored", "kernel_pairs",
                  "diverged"], rows)
    run.save_json("d1_aggregates.json", aggregates)
    improved = [a for a, v in aggregates.items()
                if a != "base" and v < aggregates["base"]]
    run.log(f"  gate D1: arms beating base on validation aggregate: "
            f"{improved if improved else 'NONE - record non-transfer'}")
    run.finish()


# ---------------------------------------------------------------------------
# D2: adaptive trigger selection + policy freeze
# ---------------------------------------------------------------------------


def D2(prof: Profile) -> None:
    base = load_frozen("*D0-baseline*", "baseline_frozen.json")
    run = Run("D2-policy", {"profile": prof.name, "baseline": base})
    run.log("D2: adaptive annealing trigger selection (validation targets)")
    rows = []
    aggregates = {}
    pair_costs = {}
    for trig in ("fixed70", "plateau", "agree"):
        per_target = []
        costs = []
        for tgt in validation_targets():
            eds = []
            for s in range(prof.d2_seeds):
                res = run_one(tgt, AdaptivePolicy(trig, eta_cap=0.25),
                              30_000 + s, prof, run.counter)
                eds.append(res.final_ed2)
                costs.append(res.kernel_pairs)
                rows.append((trig, tgt.name, s, res.final_ed2,
                             res.event_time, int(res.censored),
                             res.kernel_pairs))
            per_target.append(float(np.median(eds)))
        aggregates[trig] = geo_mean(per_target)
        pair_costs[trig] = float(np.median(costs))
        run.log(f"  {trig:8s}: geo-mean ED2 = {aggregates[trig]:.5f}  "
                f"median pairs/run = {pair_costs[trig]:.3e}")
    # select: best aggregate; must beat-or-match fixed70 at its charged cost
    winner = min(aggregates, key=aggregates.get)
    if aggregates[winner] > aggregates["fixed70"] * 1.02:
        winner = "fixed70"
    run.log(f"  selected trigger: {winner}")
    policy_frozen = {
        "type": "adaptive",
        "trigger": winner,
        "initial_bandwidth": "L_hat (k-means estimate from setup sample)",
        "anneal": "geometric L_hat -> sigma_hat over remaining budget",
        "step_rule": "eta = min(0.1*tau_t, 0.25*eta_ceiling(surrogate))",
        "mask_rule": "mask on iff N >= 8*K_hat",
        "fallback": "fixed70 anneal if trigger never fires",
        "eta_cap": 0.25,
        "validation_aggregate": aggregates[winner],
        "baseline_aggregate_reference": None,
    }
    run.save_json("policy_frozen.json", policy_frozen)
    run.save_csv("d2_results.csv",
                 ["trigger", "target", "seed", "ed2", "event_time",
                  "censored", "kernel_pairs"], rows)
    run.finish()


# ---------------------------------------------------------------------------
# D3: held-out paired benchmark + gate
# ---------------------------------------------------------------------------


def D3(prof: Profile) -> None:
    base = load_frozen("*D0-baseline*", "baseline_frozen.json")
    pol = load_frozen("*D2-policy*", "policy_frozen.json")
    run = Run("D3-heldout", {"profile": prof.name, "baseline": base,
                             "policy": pol})
    run.log(f"D3: held-out benchmark; base=(tau*={base['tau_star']}, "
            f"eta*={base['eta_abs']:.4f}, mask on) vs modified="
            f"adaptive-{pol['trigger']}")
    rows = []
    cell_ratios: dict[str, float] = {}
    all_pairs_mod, all_pairs_base = [], []
    fam_of_cell: dict[str, str] = {}
    times = {"base": ([], []), "mod": ([], [])}
    for tgt in heldout_targets():
        for init_kind in ("missing", "covered"):
            cell = f"{tgt.name}/{init_kind}"
            fam_of_cell[cell] = tgt.family
            mod_eds, base_eds = [], []
            for s in range(prof.d3_seeds):
                res_b = run_one(tgt, FixedPolicy("base", base["tau_star"],
                                                 base["eta_abs"], True),
                                40_000 + s, prof, run.counter, init_kind)
                res_m = run_one(tgt, AdaptivePolicy(pol["trigger"],
                                                    pol["eta_cap"]),
                                40_000 + s, prof, run.counter, init_kind)
                eb = min(res_b.final_ed2, 1e6)
                em = min(res_m.final_ed2, 1e6)
                base_eds.append(eb)
                mod_eds.append(em)
                all_pairs_base.append(eb)
                all_pairs_mod.append(em)
                times["base"][0].append(res_b.event_time)
                times["base"][1].append(res_b.censored)
                times["mod"][0].append(res_m.event_time)
                times["mod"][1].append(res_m.censored)
                rows.append((cell, tgt.family, s, eb, em,
                             res_b.final_sw1, res_m.final_sw1,
                             res_b.event_time, int(res_b.censored),
                             res_m.event_time, int(res_m.censored),
                             res_b.kernel_pairs, res_m.kernel_pairs,
                             int(res_b.diverged), int(res_m.diverged)))
            ratio = float(np.median(mod_eds) / max(np.median(base_eds),
                                                   1e-12))
            cell_ratios[cell] = ratio
            run.log(f"  {cell:28s}: base {np.median(base_eds):.5f}  "
                    f"mod {np.median(mod_eds):.5f}  ratio {ratio:.3f}")
    run.save_csv("d3_results.csv",
                 ["cell", "family", "seed", "base_ed2", "mod_ed2",
                  "base_sw1", "mod_sw1", "base_time", "base_cens",
                  "mod_time", "mod_cens", "base_pairs", "mod_pairs",
                  "base_div", "mod_div"], rows)
    # ---- gate evaluation (pre-declared) ----
    mod = np.asarray(all_pairs_mod)
    bas = np.asarray(all_pairs_base)
    point, lo, hi = paired_bootstrap_ratio(mod, bas)
    degraded = [c for c, r in cell_ratios.items() if r > 1.10]
    km_base = km_median(*times["base"])
    km_mod = km_median(*times["mod"])
    nong = [c for c in cell_ratios
            if fam_of_cell[c] != "gauss" or "overlap" in c or "hetero" in c]
    # non-Gaussian robustness: aggregate ratio over those cells' seeds
    ng_rows = [r for r in rows if r[0] in nong]
    ng_point, ng_lo, ng_hi = paired_bootstrap_ratio(
        np.asarray([r[4] for r in ng_rows]),
        np.asarray([r[3] for r in ng_rows]))
    gate = {
        "aggregate_ratio": point,
        "ratio_ci": [lo, hi],
        "crit1_ratio_le_0.8": point <= 0.8,
        "crit2_ci_hi_lt_1": hi < 1.0,
        "degraded_cells": degraded,
        "crit3_degraded_le_20pct": len(degraded) <= 0.2 * len(cell_ratios),
        "km_base": km_base,
        "km_mod": km_mod,
        "crit4_time_not_worse": (km_mod or 10**9) <= (km_base or 10**9),
        "nongauss_ratio": ng_point,
        "nongauss_ci": [ng_lo, ng_hi],
        "crit5_nongauss_holds": ng_hi < 1.0,
    }
    gate["PASS"] = all(gate[k] for k in
                       ("crit1_ratio_le_0.8", "crit2_ci_hi_lt_1",
                        "crit3_degraded_le_20pct", "crit4_time_not_worse",
                        "crit5_nongauss_holds"))
    run.save_json("d3_gate.json", gate)
    run.log(f"  aggregate paired ratio {point:.3f} CI[{lo:.3f},{hi:.3f}]  "
            f"degraded cells {len(degraded)}/{len(cell_ratios)}  "
            f"KM base {km_base} mod {km_mod}  "
            f"non-gauss ratio {ng_point:.3f} CI[{ng_lo:.3f},{ng_hi:.3f}]")
    run.log(f"  GATE D3: {'PASS' if gate['PASS'] else 'FAIL'} "
            f"({[k for k in gate if k.startswith('crit') and not gate[k]]}"
            f" failing)" if not gate["PASS"] else "  GATE D3: PASS")
    run.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["D0", "D1", "D2", "D3"])
    ap.add_argument("--profile", default="standard",
                    choices=list(PROFILES))
    args = ap.parse_args()
    prof = PROFILES[args.profile]
    {"D0": D0, "D1": D1, "D2": D2, "D3": D3}[args.stage](prof)


if __name__ == "__main__":
    main()
