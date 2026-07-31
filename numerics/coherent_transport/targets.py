"""Structured-geometry 2-D targets (research plan section 6.1).

Each probes a distinct support failure mode: disconnected support
(checkerboard, separated modes), curved manifolds (swiss roll, moons, pinwheel),
holes (rings/annulus). Samplers return (n, 2). Mixtures also expose oracle
modes/sigmas/weights for rare-mass evaluation (EVALUATION ONLY)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class GeoTarget:
    name: str
    sampler: Callable[[int, np.random.Generator], np.ndarray]
    kind: str
    modes: np.ndarray | None = None
    sigmas: np.ndarray | None = None
    weights: np.ndarray | None = None
    scale: float = 1.0


def checkerboard(name="checkerboard", cells=4, noise=0.0) -> GeoTarget:
    def s(n, rng):
        pts = np.empty((n, 2))
        c = 0
        while c < n:
            xy = rng.uniform(0, cells, size=(n, 2))
            idx = np.floor(xy).astype(int)
            keep = xy[((idx[:, 0] + idx[:, 1]) % 2 == 0)]
            take = min(len(keep), n - c)
            pts[c:c + take] = keep[:take]
            c += take
        return pts + rng.normal(size=(n, 2)) * noise
    return GeoTarget(name, s, "checkerboard", scale=float(cells))


def swiss_roll(name="swiss_roll", noise=0.05) -> GeoTarget:
    def s(n, rng):
        t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=n))
        x = t * np.cos(t); y = t * np.sin(t)
        return np.stack([x, y], 1) / 10.0 + rng.normal(size=(n, 2)) * noise
    return GeoTarget(name, s, "manifold", scale=1.5)


def two_moons(name="moons", noise=0.06) -> GeoTarget:
    def s(n, rng):
        w = rng.integers(0, 2, n); th = rng.uniform(0, np.pi, n)
        x = np.where(w == 0, np.cos(th), 1 - np.cos(th))
        y = np.where(w == 0, np.sin(th), 0.5 - np.sin(th))
        return np.stack([x, y], 1) + rng.normal(size=(n, 2)) * noise
    return GeoTarget(name, s, "manifold", scale=1.5)


def rings(name="rings", radii=(0.4, 0.9), width=0.03) -> GeoTarget:
    def s(n, rng):
        which = rng.integers(0, len(radii), n)
        r = np.array(radii)[which] + rng.normal(size=n) * width
        th = rng.uniform(0, 2 * np.pi, n)
        return np.stack([r * np.cos(th), r * np.sin(th)], 1)
    return GeoTarget(name, s, "hole", scale=float(max(radii)))


def pinwheel(name="pinwheel", blades=5, noise=0.04) -> GeoTarget:
    def s(n, rng):
        b = rng.integers(0, blades, n)
        r = rng.uniform(0.1, 1.0, n)
        th = b * (2 * np.pi / blades) + r * 2.0 + rng.normal(size=n) * noise
        return np.stack([r * np.cos(th), r * np.sin(th)], 1)
    return GeoTarget(name, s, "manifold", scale=1.0)


def separated_modes(name="sep_modes", K=9, L=1.0, sigma=0.03,
                    rare=False) -> GeoTarget:
    ang = 2 * np.pi * np.arange(K) / K
    modes = np.stack([np.cos(ang), np.sin(ang)], 1) * (L * K / (2 * np.pi))
    if rare:
        if K < 3:
            raise ValueError("rare separated-modes target requires K >= 3")
        # Exact requested mixture probabilities, rather than assigning raw
        # weights .02/.05 and renormalising them down to .25%/.62%.
        w = np.full(K, (1.0 - 0.02 - 0.05) / (K - 2))
        w[1] = 0.02
        w[2] = 0.05
    else:
        w = np.full(K, 1.0)
    w = w / w.sum()
    sig = np.full(K, sigma)

    def s(n, rng):
        idx = rng.choice(K, size=n, p=w)
        return modes[idx] + rng.normal(size=(n, 2)) * sigma
    return GeoTarget(name, s, "disconnected", modes, sig, w,
                     scale=float(L * K / (2 * np.pi)))


def suite() -> list[GeoTarget]:
    return [checkerboard(), swiss_roll(), two_moons(), rings(), pinwheel(),
            separated_modes(), separated_modes("sep_modes_rare", rare=True)]


def _tests(log=print) -> None:
    failures = []

    def check(name, ok):
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failures.append(name)

    names = [target.name for target in suite()]
    check("1. structured suite includes Swiss roll", "swiss_roll" in names)
    rare = separated_modes("rare-test", rare=True)
    check("2. rare target has exact 2% and 5% probabilities",
          rare.weights is not None
          and abs(float(rare.weights[1]) - 0.02) < 1e-15
          and abs(float(rare.weights[2]) - 0.05) < 1e-15
          and abs(float(rare.weights.sum()) - 1.0) < 1e-15)
    samples = rare.sampler(100, np.random.default_rng(1))
    check("3. target samplers return finite (n,2) arrays",
          samples.shape == (100, 2) and np.all(np.isfinite(samples)))
    if failures:
        raise SystemExit(f"target tests FAILED: {failures}")
    log("all target tests passed")


if __name__ == "__main__":
    _tests()
