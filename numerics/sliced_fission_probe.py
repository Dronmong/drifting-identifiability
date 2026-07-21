"""Exploratory probe: sliced-rank fission warm-start for drifting generators.

This is NOT a frozen scientific gate.  It is a cheap mechanism discriminator
motivated by the characterized negative in ``M1_headroom.md``:

* local/coarse Laplace fields can reach additional basins but do not partition
  a concentrated swarm among them;
* a sliced-Wasserstein rank coupling is globally mass-aware and supplies a
  distinct target quantile to each generated sample;
* after a short rank-coupled fission phase, exact paper Algorithm 2 resumes as
  the local refinement field.

The probe compares a tau-tuned paper baseline, sliced-only training, and
rank-fission warm starts on held-out evaluation draws.  It intentionally uses
the repository's existing NumPy TanhMLP/Adam and exact bi-softmax field.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NUMERICS = ROOT / "numerics"
if str(NUMERICS) not in sys.path:
    sys.path.insert(0, str(NUMERICS))

from identifiability_drift import compute_field  # noqa: E402
from lowdim_drift import controlled_means, energy_distance2, sliced_w1  # noqa: E402
from mode_recovery import (  # noqa: E402
    basin_radius,
    coverage,
    mass_calibration,
    sigma_radius,
)
from run_identifiability_generator import TanhMLP  # noqa: E402


@dataclass(frozen=True)
class RingMixture:
    name: str
    d: int
    scale: float
    means: np.ndarray
    sigma: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        ids = rng.integers(0, len(self.means), size=n)
        return self.means[ids] + self.sigma * rng.normal(size=(n, self.d))


def ring_mixture(k: int, d: int, sigma: float, radius: float = 1.0) \
        -> RingMixture:
    angles = 2.0 * np.pi * np.arange(k) / k
    means = np.zeros((k, d))
    means[:, 0] = radius * np.cos(angles)
    means[:, 1] = radius * np.sin(angles)
    return RingMixture(f"ring-K{k}-d{d}", d, radius, means, sigma)


def line_mixture(k: int, sigma: float, spacing: float) -> RingMixture:
    means = controlled_means(k, 1, spacing)
    scale = float(np.linalg.norm(means, axis=1).max() + 3 * sigma)
    return RingMixture(f"line-K{k}-s{spacing:g}-w{sigma:g}", 1, scale,
                       means, sigma)


def sliced_rank_field(x: np.ndarray, target: np.ndarray, latent: np.ndarray,
                      n_proj: int, rng: np.random.Generator,
                      tie_scale: float = 1e-10) -> np.ndarray:
    """Negative empirical SW2 gradient, represented as a stop-grad field.

    For every direction, generated and target projections are sorted and
    paired by rank.  The latent-dependent infinitesimal term selects a stable
    subgradient only when generated projections tie; it is far below ordinary
    floating-point-scale geometry and does not otherwise reorder the cloud.
    Multiplication by ``d`` removes E[theta theta^T] = I/d attenuation.
    """
    if x.shape != target.shape:
        raise ValueError("sliced rank field requires equal batch shapes")
    n, d = x.shape
    theta = rng.normal(size=(n_proj, d))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True)
    latent_dirs = rng.normal(size=(n_proj, latent.shape[1]))
    latent_dirs /= np.linalg.norm(latent_dirs, axis=1, keepdims=True)
    xp = x @ theta.T
    yp = target @ theta.T
    tie = latent @ latent_dirs.T
    tie /= np.maximum(np.std(tie, axis=0, keepdims=True), 1e-12)
    out = np.zeros_like(x)
    for ell in range(n_proj):
        ox = np.argsort(xp[:, ell] + tie_scale * tie[:, ell], kind="stable")
        oy = np.argsort(yp[:, ell], kind="stable")
        displacement = yp[oy, ell] - xp[ox, ell]
        out[ox] += displacement[:, None] * theta[ell]
    return (d / n_proj) * out


def rank_field_self_check() -> None:
    """Check that the 1-D implementation is exact rank displacement."""
    x = np.asarray([[2.0], [-1.0], [0.5], [4.0]])
    target = np.asarray([[3.0], [-2.0], [1.0], [0.0]])
    latent = np.asarray([[0.2, -0.1], [0.7, 0.4], [-0.3, 0.8], [1.1, -0.5]])
    got = sliced_rank_field(
        x, target, latent, 9, np.random.default_rng(20260720))
    ox = np.argsort(x[:, 0], kind="stable")
    oy = np.argsort(target[:, 0], kind="stable")
    expected = np.zeros_like(x)
    expected[ox, 0] = target[oy, 0] - x[ox, 0]
    if not np.allclose(got, expected, rtol=1e-12, atol=1e-12):
        raise AssertionError("1-D sliced field is not exact rank displacement")


def train(target: RingMixture, seed: int, *, arm: str, tau: float,
          updates: int = 1000, batch: int = 128, n_proj: int = 32,
          warm_fraction: float = 0.35,
          init_kind: str = "missing", switch_threshold: float = 0.15,
          min_quantile_fraction: float = 0.20,
          max_quantile_fraction: float = 0.85) -> dict:
    model = TanhMLP(target, init_kind, seed * 10_000 + 1)
    latent_rng = np.random.default_rng(seed * 10_000 + 2)
    data_rng = np.random.default_rng(seed * 10_000 + 3)
    slice_rng = np.random.default_rng(seed * 10_000 + 4)
    warm_steps = int(round(warm_fraction * updates))
    min_quantile_steps = int(round(min_quantile_fraction * updates))
    max_quantile_steps = int(round(max_quantile_fraction * updates))
    residual_ema: float | None = None
    switched_at: int | None = None

    for step in range(updates):
        z = latent_rng.normal(size=(batch, model.latent_dim))
        x, cache = model.forward(z, want_cache=True)
        positive = target.sample(batch, data_rng)
        if arm == "paper":
            field = compute_field(
                x, positive, tau=tau, gain="paper", mask=True,
                on_degenerate="zero").V
        elif arm == "sliced":
            field = sliced_rank_field(x, positive, z, n_proj, slice_rng)
        elif arm == "warm-paper":
            if step < warm_steps:
                field = sliced_rank_field(x, positive, z, n_proj, slice_rng)
            else:
                field = compute_field(
                    x, positive, tau=tau, gain="paper", mask=True,
                    on_degenerate="zero").V
        elif arm == "adaptive-paper":
            if switched_at is None:
                quantile = sliced_rank_field(
                    x, positive, z, n_proj, slice_rng)
                # The trigger uses only the observed target minibatch.  At
                # equality the two independent empirical batches still have a
                # nonzero rank residual, so an EMA and a nonzero threshold are
                # essential.  Standard deviation makes the statistic invariant
                # to a common rescaling of data space.
                target_scale = max(float(np.std(positive)), 1e-12)
                relative_residual = (
                    math.sqrt(float(np.mean(quantile ** 2))) / target_scale)
                residual_ema = (relative_residual if residual_ema is None else
                                0.95 * residual_ema +
                                0.05 * relative_residual)
                if ((step >= min_quantile_steps and
                     residual_ema <= switch_threshold) or
                        step >= max_quantile_steps):
                    switched_at = step
            if switched_at is None:
                field = quantile
            else:
                field = compute_field(
                    x, positive, tau=tau, gain="paper", mask=True,
                    on_degenerate="zero").V
        elif arm == "blend-paper":
            paper = compute_field(
                x, positive, tau=tau, gain="paper", mask=True,
                on_degenerate="zero").V
            sliced = sliced_rank_field(x, positive, z, n_proj, slice_rng)
            lam = max(0.0, 1.0 - step / max(warm_steps, 1))
            # RMS matching prevents the mixture coefficient from being only a
            # unit-choice artifact; Adam still sees the changed direction.
            prms = math.sqrt(float(np.mean(np.sum(paper * paper, axis=1))))
            srms = math.sqrt(float(np.mean(np.sum(sliced * sliced, axis=1))))
            field = paper + lam * sliced * (prms / max(srms, 1e-12))
        else:
            raise ValueError(arm)
        model.stopgrad_step(cache, field)

    eval_rng = np.random.default_rng(seed * 10_000 + 5)
    q = model.forward(eval_rng.normal(size=(4096, model.latent_dim)))
    p = target.sample(4096, np.random.default_rng(seed * 10_000 + 6))
    weights = np.full(len(target.means), 1.0 / len(target.means))
    reach = coverage(q, target.means, basin_radius(target.means), weights)
    resolve = coverage(
        q, target.means,
        sigma_radius(target.sigma, K=len(target.means)), weights)
    calibration = mass_calibration(
        q, target.means, basin_radius(target.means), weights)
    metric_rng = np.random.default_rng(seed * 10_000 + 7)
    # ED2 uses a smaller paired reference to avoid a 4096^2 temporary matrix.
    pick = metric_rng.choice(len(q), size=1024, replace=False)
    pickp = metric_rng.choice(len(p), size=1024, replace=False)
    return {
        "arm": arm,
        "tau": tau,
        "init": init_kind,
        "seed": seed,
        "reach": reach["unweighted"],
        "resolve": resolve["unweighted"],
        "mass_l1": calibration["l1_all"],
        "ed2": max(0.0, energy_distance2(q[pick], p[pickp])),
        "sw1": sliced_w1(q, p, 64, metric_rng),
        "switched_at": switched_at,
    }


def pilot() -> None:
    target = ring_mixture(k=32, d=2, sigma=0.02)
    rows: list[dict] = []
    # Tune the paper baseline over the same hard-regime range already used by
    # M1; the candidate is evaluated for every tau rather than compared only to
    # a deliberately poor paper setting.
    for seed in range(4):
        for tau in (0.2, 0.4, 0.8):
            for arm in ("paper", "warm-paper", "blend-paper"):
                rows.append(train(target, seed, arm=arm, tau=tau))
        rows.append(train(target, seed, arm="sliced", tau=0.4))

    print("exploratory sliced-rank fission probe (median over 4 seeds)")
    print("arm          tau   reach resolve massL1      ED2      SW1")
    for arm in ("paper", "warm-paper", "blend-paper", "sliced"):
        taus = (0.2, 0.4, 0.8) if arm != "sliced" else (0.4,)
        for tau in taus:
            group = [r for r in rows if r["arm"] == arm and r["tau"] == tau]
            med = {key: float(np.median([r[key] for r in group]))
                   for key in ("reach", "resolve", "mass_l1", "ed2", "sw1")}
            print(f"{arm:12s} {tau:4.1f}  {med['reach']:5.2f}  "
                  f"{med['resolve']:6.2f}  {med['mass_l1']:6.3f}  "
                  f"{med['ed2']:7.4f}  {med['sw1']:7.4f}")


def fresh_robustness_screen() -> None:
    """Post-selection robustness screen; exploratory, not pre-registered."""
    targets = (
        line_mixture(8, 0.08, 0.75),
        line_mixture(12, 0.04, 1.20),
        line_mixture(20, 0.06, 0.80),
        line_mixture(24, 0.03, 1.50),
    )
    paper_taus = (0.2, 0.5, 1.0, 2.0, 4.0)
    print("post-selection robustness screen (6 seeds; oracle tau per target)")
    print("target                    paperED candidateED ratio  paperSW candSW")
    ratios = []
    for target in targets:
        paper_rows = {
            tau: [train(target, seed, arm="paper", tau=tau, updates=1200,
                        batch=128, n_proj=8, warm_fraction=0.7)
                  for seed in range(6)]
            for tau in paper_taus
        }
        # Choose the paper tau by median ED2 on this very target.  This is an
        # oracle advantage for the baseline, appropriate for a robustness
        # screen but not a substitute for a frozen validation/test protocol.
        best_tau = min(
            paper_taus,
            key=lambda tau: np.median([r["ed2"] for r in paper_rows[tau]]))
        candidate = [
            train(target, seed, arm="warm-paper", tau=0.5, updates=1200,
                  batch=128, n_proj=8, warm_fraction=0.7)
            for seed in range(6)
        ]
        ped = float(np.median([r["ed2"] for r in paper_rows[best_tau]]))
        ced = float(np.median([r["ed2"] for r in candidate]))
        psw = float(np.median([r["sw1"] for r in paper_rows[best_tau]]))
        csw = float(np.median([r["sw1"] for r in candidate]))
        ratios.append(ced / max(ped, 1e-12))
        print(f"{target.name:25s} {ped:7.4f} {ced:11.4f} "
              f"{ced / max(ped, 1e-12):5.2f} {psw:8.4f} {csw:6.4f} "
              f"(paper tau={best_tau:g})")
    print(f"target-balanced geometric ED2 ratio: "
          f"{float(np.exp(np.mean(np.log(ratios)))):.3f}")


def initialization_screen() -> None:
    """Frozen-candidate stress screen over all repository initializations.

    This remains exploratory: it was added after the 70%/tau=.5 candidate was
    selected.  The paper arm again receives an oracle tau in every cell.
    """
    targets = (
        line_mixture(6, 0.10, 0.90),
        line_mixture(10, 0.07, 1.00),
        line_mixture(18, 0.05, 1.10),
        line_mixture(28, 0.04, 0.70),
    )
    inits = ("broad", "missing", "far", "concentrated")
    paper_taus = (0.2, 0.5, 1.0, 2.0, 4.0)
    ratios: dict[str, list[float]] = {init: [] for init in inits}
    print("post-selection initialization screen "
          "(4 seeds; oracle tau per cell)")
    print("target                    init          paperED candidateED ratio")
    for target in targets:
        for init_kind in inits:
            paper_rows = {
                tau: [train(
                    target, seed, arm="paper", tau=tau, updates=1200,
                    batch=128, n_proj=8, warm_fraction=0.7,
                    init_kind=init_kind)
                      for seed in range(4)]
                for tau in paper_taus
            }
            best_tau = min(
                paper_taus,
                key=lambda tau: np.median(
                    [r["ed2"] for r in paper_rows[tau]]))
            candidate = [train(
                target, seed, arm="warm-paper", tau=0.5, updates=1200,
                batch=128, n_proj=8, warm_fraction=0.7,
                init_kind=init_kind)
                         for seed in range(4)]
            ped = float(np.median(
                [r["ed2"] for r in paper_rows[best_tau]]))
            ced = float(np.median([r["ed2"] for r in candidate]))
            ratio = ced / max(ped, 1e-12)
            ratios[init_kind].append(ratio)
            print(f"{target.name:25s} {init_kind:12s} {ped:7.4f} "
                  f"{ced:11.4f} {ratio:5.2f} (paper tau={best_tau:g})")
    all_ratios = [ratio for values in ratios.values() for ratio in values]
    print("geometric ED2 ratios by initialization")
    for init_kind in inits:
        aggregate = float(np.exp(np.mean(np.log(ratios[init_kind]))))
        print(f"  {init_kind:12s}: {aggregate:.3f}")
    print("  target-balanced: "
          f"{float(np.exp(np.mean(np.log(all_ratios)))):.3f}")


def failed_cell_repair_screen() -> None:
    """Development-only test of the residual-triggered switch.

    Uses only the two far-start cells on which the fixed 70/30 candidate failed;
    it may select a trigger but cannot provide confirmatory evidence.
    """
    targets = (
        line_mixture(18, 0.05, 1.10),
        line_mixture(28, 0.04, 0.70),
    )
    print("failed-cell repair screen (4 seeds, far initialization)")
    print("target                    arm/threshold       ED2 switch")
    for target in targets:
        configs = (
            ("paper", None),
            ("warm-paper", None),
            ("sliced", None),
            ("adaptive-paper", 0.10),
            ("adaptive-paper", 0.15),
            ("adaptive-paper", 0.20),
        )
        for arm, threshold in configs:
            rows = [train(
                target, seed, arm=arm, tau=0.5, updates=1200, batch=128,
                n_proj=8, warm_fraction=0.7, init_kind="far",
                switch_threshold=0.15 if threshold is None else threshold)
                    for seed in range(4)]
            ed2 = float(np.median([r["ed2"] for r in rows]))
            switch_values = [r["switched_at"] for r in rows
                             if r["switched_at"] is not None]
            switch = (float(np.median(switch_values))
                      if switch_values else math.nan)
            label = arm if threshold is None else f"{arm}/{threshold:.2f}"
            print(f"{target.name:25s} {label:20s} {ed2:7.4f} "
                  f"{switch:6.0f}")


def main() -> None:
    rank_field_self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--init-screen", action="store_true")
    parser.add_argument("--repair-screen", action="store_true")
    args = parser.parse_args()
    if args.repair_screen:
        failed_cell_repair_screen()
    elif args.init_screen:
        initialization_screen()
    elif args.fresh:
        fresh_robustness_screen()
    else:
        pilot()


if __name__ == "__main__":
    main()
