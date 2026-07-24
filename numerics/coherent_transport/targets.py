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
        w = np.ones(K); w[0] = 0.02 * K  # will renormalise; make mode 0 common
        w = np.full(K, 1.0); w[1] = 0.02; w[2] = 0.05  # rare modes 1,2
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
    return [checkerboard(), two_moons(), rings(), pinwheel(),
            separated_modes(), separated_modes("sep_modes_rare", rare=True)]
