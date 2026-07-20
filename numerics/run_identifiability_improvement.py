"""Frozen NCJ low-dimensional particle study.

Protocol: numerics/IdentifiabilityImprovementProtocol.md

Commands:

    uv run --with numpy --with scipy python \
      numerics/run_identifiability_improvement.py validation --profile smoke

    uv run --with numpy --with scipy python \
      numerics/run_identifiability_improvement.py validation --profile standard

    uv run --with numpy --with scipy python \
      numerics/run_identifiability_improvement.py test --profile standard

    uv run --with numpy --with scipy python \
      numerics/run_identifiability_improvement.py generator --profile standard

The test registry is never instantiated during validation.  The generator
command is a gate: it refuses unless the standard particle E4 result records
PASS=true, after which a separate generator protocol must be frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import scipy
from numpy.linalg import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from identifiability_drift import (  # noqa: E402
    DegenerateFieldError,
    compute_field,
    invariant_tests,
)
from lowdim_drift import (  # noqa: E402
    WorkCounter,
    controlled_means,
    energy_distance2,
    km_median,
    sliced_w1,
)

MASTER = 20260719
TAU = 0.35
BASE_ETA = 0.0525
NORM_CLIP = 2.0
SIGMA_RATIOS = (0.0, 0.10, 0.25, 0.50)
ETA_GRID = (0.025, 0.0525)
INITS = ("covered", "missing", "far", "concentrated")

PROTOCOL = HERE / "IdentifiabilityImprovementProtocol.md"
PLAN = HERE / "IdentifiabilityDrivenImprovementPlan.md"
FIELD_SOURCE = HERE / "identifiability_drift.py"
COMMON_SOURCE = HERE / "lowdim_drift.py"
VALIDATION_REGISTRY = HERE / "identifiability_validation_registry.json"
TEST_REGISTRY = HERE / "identifiability_test_registry.json"
RUNROOT = HERE / "identifiability_runs"

FROZEN_REGISTRY_HASHES = {
    VALIDATION_REGISTRY.name:
        "1DAAB55F47D4C557EB10A3960171CAA39AD2FD5BB482D11339B423CE8FBB5B97",
    TEST_REGISTRY.name:
        "938B51C36178313D145DEA05A3E6FED04A033C4977B49D48CB92903D10D4A2B5",
}


# ---------------------------------------------------------------------------
# Profiles and arm specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    name: str
    steps: int
    N: int
    batch: int
    validation_seeds: int
    test_seeds: int
    ref_final: int
    ref_cross: int


PROFILES = {
    "smoke": Profile("smoke", 60, 24, 32, 1, 2, 256, 128),
    "standard": Profile("standard", 400, 48, 64, 8, 20, 1024, 256),
}


@dataclass(frozen=True)
class Arm:
    label: str
    gain: str
    crossfit: bool
    mask: bool
    sigma_ratio: float
    eta: float
    norm_clip: float | None

    def payload(self) -> dict:
        return asdict(self)


def _tag(x: float) -> str:
    return str(x).replace(".", "p")


PAPER = Arm("paper", "paper", False, True, 0.0, BASE_ETA, None)


def validation_arms() -> list[Arm]:
    arms = [PAPER]
    arms.extend(
        Arm(f"normalized__eta-{_tag(eta)}", "constant", False, True,
            0.0, eta, NORM_CLIP)
        for eta in ETA_GRID
    )
    arms.append(Arm("crossfit-only", "paper", True, False, 0.0,
                    BASE_ETA, None))
    arms.extend(
        Arm(f"jitter-only__sigma-{_tag(sig)}", "paper", False, True,
            sig, BASE_ETA, None)
        for sig in SIGMA_RATIOS if sig > 0
    )
    arms.extend(
        Arm(f"normalized-crossfit__eta-{_tag(eta)}", "constant", True,
            False, 0.0, eta, NORM_CLIP)
        for eta in ETA_GRID
    )
    arms.extend(
        Arm(f"ncj__eta-{_tag(eta)}__sigma-{_tag(sig)}", "constant", True,
            False, sig, eta, NORM_CLIP)
        for eta in ETA_GRID for sig in SIGMA_RATIOS
    )
    assert len({arm.label for arm in arms}) == len(arms) == 17
    return arms


def test_arms(frozen: dict) -> list[Arm]:
    eta = float(frozen["eta"])
    sig = float(frozen["sigma_ratio"])
    return [
        PAPER,
        Arm("normalized-only", "constant", False, True, 0.0, eta,
            NORM_CLIP),
        Arm("crossfit-only", "paper", True, False, 0.0, BASE_ETA, None),
        Arm("jitter-only", "paper", False, True, sig, BASE_ETA, None),
        Arm("normalized-crossfit", "constant", True, False, 0.0, eta,
            NORM_CLIP),
        Arm("ncj", "constant", True, False, sig, eta, NORM_CLIP),
    ]


# ---------------------------------------------------------------------------
# Frozen target registry and samplers
# ---------------------------------------------------------------------------


@dataclass
class StudyTarget:
    name: str
    d: int
    family: str
    params: dict
    sampler: Callable[[int, np.random.Generator], np.ndarray]
    scale: float
    means: np.ndarray | None = None
    weights: np.ndarray | None = None
    first_component: Callable[[int, np.random.Generator], np.ndarray] | None = None

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.sampler(n, rng)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def verify_registry_hashes() -> dict[str, str]:
    realized = {
        path.name: sha256_file(path)
        for path in (VALIDATION_REGISTRY, TEST_REGISTRY)
    }
    if realized != FROZEN_REGISTRY_HASHES:
        raise RuntimeError(
            f"frozen registry hash mismatch: expected={FROZEN_REGISTRY_HASHES}, "
            f"realized={realized}")
    return realized


def _weights(spec, K: int) -> np.ndarray:
    if spec == "equal":
        return np.full(K, 1.0 / K)
    if spec == "ascending":
        raw = np.arange(1, K + 1, dtype=float)
        return raw / raw.sum()
    if spec == "descending":
        raw = np.arange(K, 0, -1, dtype=float)
        return raw / raw.sum()
    out = np.asarray(spec, dtype=float)
    if out.shape != (K,) or np.any(out <= 0):
        raise ValueError(f"invalid weights {spec}")
    return out / out.sum()


def _rotation2(theta: float) -> np.ndarray:
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])


def build_target(obj: dict) -> StudyTarget:
    name, family, d = obj["name"], obj["family"], int(obj["d"])
    p = obj["params"]

    if family == "gaussian":
        if "std" in p:
            stds = np.full(d, float(p["std"]))
            R = np.eye(d)
        else:
            stds = np.asarray(p["stds"], dtype=float)
            R = _rotation2(float(p.get("rotation", 0.0)))
        def sample(n, rng):
            return rng.normal(size=(n, d)) @ (R @ np.diag(stds)).T
        return StudyTarget(name, d, family, p, sample,
                           float(3 * stds.max()))

    if family == "student":
        df, scl = float(p["df"]), float(p["scale"])
        def sample(n, rng):
            return rng.standard_t(df, size=(n, d)) * scl
        return StudyTarget(name, d, family, p, sample, 4.0 * scl)

    if family == "gauss_mixture":
        K, L = int(p["K"]), float(p["L"])
        means = controlled_means(K, d, L)
        sigmas = np.full(K, float(p["sigmas"])) if np.isscalar(p["sigmas"]) \
            else np.asarray(p["sigmas"], dtype=float)
        weights = _weights(p["weights"], K)
        def sample(n, rng):
            comp = rng.choice(K, size=n, p=weights)
            return means[comp] + rng.normal(size=(n, d)) * sigmas[comp, None]
        def first(n, rng):
            return means[0] + rng.normal(size=(n, d)) * sigmas[0]
        scale = float(norm(means, axis=1).max() + 3 * sigmas.max())
        return StudyTarget(name, d, family, p, sample, scale, means, weights,
                           first)

    if family == "grid_mixture":
        side, spacing, sigma = int(p["side"]), float(p["spacing"]), \
            float(p["sigma"])
        axis = (np.arange(side) - (side - 1) / 2) * spacing
        means = np.asarray([(x, y) for x in axis for y in axis], dtype=float)
        if p["weights"] == "checker":
            raw = np.asarray([1.0 + ((i // side + i % side) % 2)
                              for i in range(side * side)])
            weights = raw / raw.sum()
        else:
            weights = np.full(side * side, 1 / (side * side))
        def sample(n, rng):
            comp = rng.choice(len(means), size=n, p=weights)
            return means[comp] + rng.normal(size=(n, 2)) * sigma
        def first(n, rng):
            return means[0] + rng.normal(size=(n, 2)) * sigma
        return StudyTarget(name, 2, family, p, sample,
                           float(norm(means, axis=1).max() + 3 * sigma),
                           means, weights, first)

    if family == "ring":
        radius, width = float(p["radius"]), float(p["width"])
        def sample(n, rng):
            th = rng.uniform(0, 2 * np.pi, n)
            rad = radius + rng.normal(size=n) * width
            return np.stack([rad * np.cos(th), rad * np.sin(th)], axis=1)
        return StudyTarget(name, 2, family, p, sample, radius + 3 * width)

    if family == "circles":
        r1, r2 = map(float, p["radii"])
        width, ow = float(p["width"]), float(p["outer_weight"])
        def sample(n, rng):
            outer = rng.random(n) < ow
            rad = np.where(outer, r2, r1) + rng.normal(size=n) * width
            th = rng.uniform(0, 2 * np.pi, n)
            return np.stack([rad * np.cos(th), rad * np.sin(th)], axis=1)
        return StudyTarget(name, 2, family, p, sample, r2 + 3 * width)

    if family == "moons":
        scl, noise, uw = float(p["scale"]), float(p["noise"]), \
            float(p["upper_weight"])
        def sample(n, rng):
            upper = rng.random(n) < uw
            th = rng.uniform(0, np.pi, n)
            x = np.where(upper, np.cos(th), 1 - np.cos(th)) * scl
            y = np.where(upper, np.sin(th), 0.5 - np.sin(th)) * scl
            return np.stack([x, y], axis=1) + \
                rng.normal(size=(n, 2)) * noise * scl
        return StudyTarget(name, 2, family, p, sample, 1.7 * scl)

    if family == "banana":
        curv, noise, xs = float(p["curvature"]), float(p["noise"]), \
            float(p["xscale"])
        def sample(n, rng):
            x = rng.normal(size=n) * xs
            y = curv * (x ** 2 - xs ** 2) + rng.normal(size=n) * noise
            return np.stack([x, y], axis=1)
        return StudyTarget(name, 2, family, p, sample,
                           3 * max(xs, curv * xs * xs))

    if family == "spiral":
        turns, noise, radius = float(p["turns"]), float(p["noise"]), \
            float(p["radius"])
        def sample(n, rng):
            t = rng.uniform(0.15, 2 * np.pi * turns, n)
            r = radius * t / (2 * np.pi * turns)
            pts = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
            return pts + rng.normal(size=(n, 2)) * noise
        return StudyTarget(name, 2, family, p, sample, radius + 3 * noise)

    if family == "sine":
        amp, freq, width, xr = float(p["amplitude"]), \
            float(p["frequency"]), float(p["width"]), float(p["xrange"])
        def sample(n, rng):
            x = rng.uniform(-xr, xr, n)
            y = amp * np.sin(freq * x) + rng.normal(size=n) * width
            return np.stack([x, y], axis=1)
        return StudyTarget(name, 2, family, p, sample,
                           float(math.hypot(xr, amp + 3 * width)))

    if family == "contaminated":
        core, frac, tail = float(p["core_std"]), \
            float(p["contamination"]), float(p["tail_std"])
        def sample(n, rng):
            is_tail = rng.random(n) < frac
            std = np.where(is_tail, tail, core)
            return rng.normal(size=(n, d)) * std[:, None]
        return StudyTarget(name, d, family, p, sample, 2.5 * tail)

    if family == "helix":
        turns, radius, height, noise = float(p["turns"]), \
            float(p["radius"]), float(p["height"]), float(p["noise"])
        def sample(n, rng):
            t = rng.uniform(0, 2 * np.pi * turns, n)
            z = height * (t / (2 * np.pi * turns) - 0.5)
            pts = np.stack([radius * np.cos(t), radius * np.sin(t), z],
                           axis=1)
            return pts + rng.normal(size=(n, 3)) * noise
        return StudyTarget(name, 3, family, p, sample,
                           float(math.hypot(radius, height / 2) + 3 * noise))

    if family == "sphere_shell":
        radius, width = float(p["radius"]), float(p["width"])
        def sample(n, rng):
            direction = rng.normal(size=(n, d))
            direction /= norm(direction, axis=1, keepdims=True)
            rad = radius + rng.normal(size=n) * width
            return direction * rad[:, None]
        return StudyTarget(name, d, family, p, sample, radius + 3 * width)

    raise ValueError(f"unknown family {family}")


def load_registry(path: Path) -> tuple[dict, list[StudyTarget]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj, [build_target(item) for item in obj["targets"]]


def init_cloud(kind: str, target: StudyTarget, N: int,
               rng: np.random.Generator) -> np.ndarray:
    if kind == "covered":
        return target.sample(N, rng) + \
            rng.normal(size=(N, target.d)) * 0.04 * target.scale
    if kind == "missing":
        if target.first_component is not None:
            return target.first_component(N, rng)
        return rng.normal(size=(N, target.d)) * 0.18 * target.scale
    if kind == "far":
        direction = np.ones(target.d) / math.sqrt(target.d)
        return rng.normal(size=(N, target.d)) * 0.08 * target.scale + \
            3.25 * target.scale * direction
    if kind == "concentrated":
        center = target.sample(4096, np.random.default_rng(772_001)).mean(0)
        return center + rng.normal(size=(N, target.d)) * 0.015 * target.scale
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Metrics and one paired trial
# ---------------------------------------------------------------------------


def mode_metrics(q: np.ndarray, target: StudyTarget) -> tuple[float, float]:
    if target.means is None or target.weights is None:
        return float("nan"), float("nan")
    labels = norm(q[:, None, :] - target.means[None, :, :], axis=2).argmin(1)
    mass = np.bincount(labels, minlength=len(target.means)) / len(q)
    coverage = float((mass >= 0.5 * target.weights).mean())
    return coverage, float(np.abs(mass - target.weights).sum())


def support_coverage(q: np.ndarray, reference: np.ndarray,
                     target: StudyTarget) -> float:
    cutoff = 0.12 * max(target.scale, 1e-8)
    mins = np.full(len(reference), np.inf)
    for start in range(0, len(reference), 256):
        block = reference[start:start + 256]
        mins[start:start + len(block)] = norm(
            block[:, None, :] - q[None, :, :], axis=2).min(1)
    return float(np.mean(mins <= cutoff))


def particle_spread(q: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((q - q.mean(0)) ** 2, axis=1))))


def mean_pairwise_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """The exact pairwise-distance reduction used by ``energy_distance2``.

    Kept local so the immutable reference-reference term can be cached across
    paired arms without changing the primary statistic or its summation
    order.
    """
    total = 0.0
    for start in range(0, len(P), 256):
        total += norm(
            P[start:start + 256, None, :] - Q[None, :, :], axis=2
        ).sum()
    return float(total / (len(P) * len(Q)))


def energy_distance2_with_reference_self(
        A: np.ndarray, B: np.ndarray, reference_self: float) -> float:
    """Energy distance with the immutable ``B`` self-term precomputed."""
    return (2 * mean_pairwise_distance(A, B) -
            mean_pairwise_distance(A, A) - reference_self)


def clip_vectors(V: np.ndarray, bound: float | None) -> np.ndarray:
    if bound is None:
        return V
    lengths = norm(V, axis=1)
    scale = np.minimum(1.0, bound / np.maximum(lengths, 1e-300))
    return V * scale[:, None]


def _nanquantile(x: np.ndarray, q: float) -> float:
    finite = np.asarray(x)[np.isfinite(x)]
    return float(np.quantile(finite, q)) if len(finite) else float("nan")


def diagnostic_values(diag: dict) -> dict[str, float]:
    self_lev = diag.get("self_leverage")
    return {
        "pq_p10": _nanquantile(diag["PQ"], 0.10),
        "pq_median": _nanquantile(diag["PQ"], 0.50),
        "delta_norm_median": _nanquantile(diag["delta_norm"], 0.50),
        "ess_pos_median": _nanquantile(diag["ESSpos"], 0.50),
        "ess_neg_median": _nanquantile(diag["ESSneg"], 0.50),
        "field_norm_median": _nanquantile(diag["field_norm"], 0.50),
        "self_leverage_mean": float(np.mean(self_lev))
            if self_lev is not None else float("nan"),
    }


def seed_base(registry_seed: int, target_index: int, init_index: int,
              seed: int) -> int:
    return int(registry_seed * 1_000_003 + target_index * 100_003 +
               init_index * 10_007 + seed * 101)


TRAJECTORY_COLUMNS = [
    "step", "ed2", "support_coverage", "mode_coverage", "mass_error",
    "spread_ratio", "pq_p10", "pq_median", "delta_norm_median",
    "ess_pos_median", "ess_neg_median", "self_leverage_mean",
    "field_norm_median", "degenerate_rows",
]


def run_trial(target: StudyTarget, target_index: int, init_kind: str,
              init_index: int, arm: Arm, seed: int, profile: Profile,
              registry_seed: int, counter: WorkCounter,
              reference_self_terms: tuple[float, float]) \
        -> tuple[dict, np.ndarray, np.ndarray]:
    wall0 = time.perf_counter()
    base = seed_base(registry_seed, target_index, init_index, seed)
    init_rng = np.random.default_rng(base + 1)
    data_rng = np.random.default_rng(base + 2)
    neg_rng = np.random.default_rng(base + 3)
    q = init_cloud(init_kind, target, profile.N, init_rng)
    ref_final = target.sample(profile.ref_final,
                              np.random.default_rng(base + 4))
    ref_cross = target.sample(profile.ref_cross,
                              np.random.default_rng(base + 5))
    ref_final_self, ref_cross_self = reference_self_terms
    target_spread = max(particle_spread(ref_final), 1e-12)
    sigma = arm.sigma_ratio * TAU
    track_every = max(1, profile.steps // 40)
    hist: list[list[float]] = []
    event_time: int | None = None
    degenerate_total = 0
    diverged = False
    training_pairs0 = counter.kernel_pairs
    last_diag: dict | None = None

    for step in range(1, profile.steps + 1):
        positives = target.sample(profile.batch, data_rng)
        negatives = q[neg_rng.integers(0, len(q), size=len(q))].copy() \
            if arm.crossfit else None
        tracked = step == 1 or step % track_every == 0 or \
            step == profile.steps
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = compute_field(
                    q, positives, negatives, tau=TAU, gain=arm.gain,
                    mask=arm.mask, jitter_sigma=sigma,
                    jitter_seed=base + 1_000_000 + step,
                    counter=counter, want_diagnostics=tracked,
                    on_degenerate="zero",
                )
        except DegenerateFieldError:
            diverged = True
            break
        degenerate_total += result.n_degenerate_rows
        V = clip_vectors(result.V, arm.norm_clip)
        q = q + arm.eta * V
        if not np.all(np.isfinite(q)) or \
                norm(q) > 1e6 * max(target.scale, 1.0):
            diverged = True
            break

        ed = max(0.0, energy_distance2_with_reference_self(
            q, ref_cross, ref_cross_self))
        if event_time is None and ed <= 0.05 * target.scale:
            event_time = step
        if tracked:
            assert result.diagnostics is not None
            last_diag = result.diagnostics
            dv = diagnostic_values(last_diag)
            cov, mass = mode_metrics(q, target)
            hist.append([
                float(step), ed, support_coverage(q, ref_cross, target), cov,
                mass, particle_spread(q) / target_spread,
                dv["pq_p10"], dv["pq_median"],
                dv["delta_norm_median"], dv["ess_pos_median"],
                dv["ess_neg_median"], dv["self_leverage_mean"],
                dv["field_norm_median"], float(result.n_degenerate_rows),
            ])

    training_pairs = counter.kernel_pairs - training_pairs0
    if diverged:
        # A failed trajectory is a recorded scientific outcome, not a reason
        # to abort the whole paired registry.  In particular, never feed a
        # non-finite cloud back into the field implementation merely to
        # manufacture final diagnostics.
        fdiag = {
            "pq_p10": float("nan"),
            "pq_median": float("nan"),
            "delta_norm_median": float("nan"),
            "ess_pos_median": float("nan"),
            "ess_neg_median": float("nan"),
            "field_norm_median": float("nan"),
            "self_leverage_mean": float("nan"),
        }
        residual = float("inf")
        final_ed = float("inf")
        final_sw = float("inf")
        coverage, mass_error = float("nan"), float("nan")
        support = float("nan")
        spread = float("nan")
    else:
        final_ref_neg = q[np.random.default_rng(base + 6).integers(
            0, len(q), size=len(q))].copy() if arm.crossfit else None
        final_data = target.sample(
            profile.batch, np.random.default_rng(base + 7))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            final_field = compute_field(
                q, final_data, final_ref_neg, tau=TAU, gain=arm.gain,
                mask=arm.mask, jitter_sigma=sigma,
                jitter_seed=base + 2_000_000, counter=counter,
                want_diagnostics=True, on_degenerate="zero",
            )
        degenerate_total += final_field.n_degenerate_rows
        fdiag = diagnostic_values(final_field.diagnostics)
        final_V = clip_vectors(final_field.V, arm.norm_clip)
        residual = float(np.sqrt(np.mean(np.sum(final_V ** 2, axis=1))))
        final_ed = max(0.0, energy_distance2_with_reference_self(
            q, ref_final, ref_final_self))
        final_sw = sliced_w1(
            q, ref_final, 32, np.random.default_rng(base + 8))
        coverage, mass_error = mode_metrics(q, target)
        support = support_coverage(q, ref_final, target)
        spread = particle_spread(q)
    row = {
        "arm": arm.label,
        "target": target.name,
        "family": target.family,
        "dimension": target.d,
        "init": init_kind,
        "cell": f"{target.name}/{init_kind}",
        "seed": seed,
        "ed2": final_ed,
        "sw1": final_sw,
        "mode_coverage": coverage,
        "mass_error": mass_error,
        "support_coverage": support,
        "residual": residual,
        "event_time": event_time if event_time is not None else profile.steps,
        "censored": int(event_time is None),
        "diverged": int(diverged),
        "degenerate_rows": degenerate_total,
        "training_kernel_pairs": training_pairs,
        "total_kernel_pairs": counter.kernel_pairs - training_pairs0,
        "generator_forwards": 0,
        "wall_seconds": time.perf_counter() - wall0,
        "particle_spread": spread,
        "spread_ratio": spread / target_spread,
        "tau": TAU,
        "eta": arm.eta,
        "sigma_ratio": arm.sigma_ratio,
        "sigma": sigma,
        "gain": arm.gain,
        "crossfit": int(arm.crossfit),
        "mask": int(arm.mask),
        "norm_clip": arm.norm_clip,
        **fdiag,
    }
    return row, np.asarray(hist, dtype=float), q


# ---------------------------------------------------------------------------
# Provenance and output
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def safe_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


class StudyRun:
    def __init__(self, stage: str, profile: Profile, config: dict,
                 registry_path: Path, expected_rows: int,
                 extra_sources: Iterable[Path] = ()) -> None:
        status = _git("status", "--porcelain")
        if profile.name == "standard" and status:
            raise RuntimeError(
                "standard scientific runs require a clean Git tree; "
                f"status was:\n{status}")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = RUNROOT / f"{stamp}-NCJ-{stage}-{profile.name}"
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "source_snapshots").mkdir()
        self.start = time.time()
        self.lines: list[str] = []
        self.counter = WorkCounter()
        self.expected_rows = expected_rows
        self.registry_path = registry_path
        sources = [Path(__file__), FIELD_SOURCE, COMMON_SOURCE, PROTOCOL,
                   PLAN, registry_path, *extra_sources]
        hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sources}
        for source in sources:
            (self.dir / "source_snapshots" / source.name).write_bytes(
                source.read_bytes())
        (self.dir / "source_hashes.json").write_text(
            json.dumps(hashes, indent=2), encoding="utf-8")
        self.manifest = {
            "stage": stage,
            "profile": asdict(profile),
            "master_seed": MASTER,
            "commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(status),
            "git_status": status.splitlines() if status else [],
            "cmdline": sys.argv,
            "config": config,
            "registry": registry_path.name,
            "registry_sha256": sha256_file(registry_path),
            "all_frozen_registry_hashes": verify_registry_hashes(),
            "source_hashes": hashes,
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "expected_rows": expected_rows,
            "start": stamp,
        }

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self.lines.append(message)

    def finish(self, realized_rows: int) -> None:
        if realized_rows != self.expected_rows:
            raise RuntimeError(
                f"row-count mismatch {realized_rows} != {self.expected_rows}")
        self.manifest.update({
            "realized_rows": realized_rows,
            "wall_seconds": round(time.time() - self.start, 3),
            "kernel_pairs": self.counter.kernel_pairs,
            "field_calls": self.counter.field_calls,
        })
        (self.dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8")
        (self.dir / "stdout.log").write_text(
            "\n".join(self.lines) + "\n", encoding="utf-8")


CSV_FIELDS = [
    "arm", "target", "family", "dimension", "init", "cell", "seed",
    "ed2", "sw1", "mode_coverage", "mass_error", "support_coverage",
    "residual", "event_time", "censored", "diverged", "degenerate_rows",
    "training_kernel_pairs", "total_kernel_pairs", "generator_forwards",
    "wall_seconds", "particle_spread", "spread_ratio", "tau", "eta",
    "sigma_ratio", "sigma", "gain", "crossfit", "mask", "norm_clip",
    "pq_p10", "pq_median", "delta_norm_median", "ess_pos_median",
    "ess_neg_median", "field_norm_median", "self_leverage_mean",
]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_grid(stage: str, profile: Profile, registry_path: Path,
             arms: list[Arm], seeds: int, extra_config: dict | None = None) \
        -> tuple[StudyRun, list[dict]]:
    return run_grid_with_sources(stage, profile, registry_path, arms, seeds,
                                 extra_config, ())


def run_grid_with_sources(stage: str, profile: Profile, registry_path: Path,
             arms: list[Arm], seeds: int, extra_config: dict | None = None,
             extra_sources: Iterable[Path] = ()) \
        -> tuple[StudyRun, list[dict]]:
    registry, targets = load_registry(registry_path)
    expected = len(targets) * len(INITS) * len(arms) * seeds
    config = {
        "protocol": PROTOCOL.name,
        "registry_name": registry["registry"],
        "targets": [target.name for target in targets],
        "initializations": list(INITS),
        "arms": [arm.payload() for arm in arms],
        "trajectory_columns": TRAJECTORY_COLUMNS,
        **(extra_config or {}),
    }
    run = StudyRun(stage, profile, config, registry_path, expected,
                   extra_sources)
    run.log(f"NCJ {stage}: {len(targets)} targets x {len(INITS)} inits x "
            f"{len(arms)} arms x {seeds} seeds = {expected} rows")
    invariant_tests(run.log)
    metric_rng = np.random.default_rng(MASTER + 909)
    metric_a = metric_rng.normal(size=(19, 3))
    metric_b = metric_rng.normal(size=(31, 3))
    metric_exact = energy_distance2(metric_a, metric_b)
    metric_cached = energy_distance2_with_reference_self(
        metric_a, metric_b, mean_pairwise_distance(metric_b, metric_b))
    if metric_cached != metric_exact:
        raise RuntimeError(
            "cached energy-distance path changed the frozen statistic: "
            f"{metric_cached} != {metric_exact}")
    run.log("  [PASS] cached ED2 is bitwise equal to frozen ED2")
    rows: list[dict] = []
    trajectories: dict[str, np.ndarray] = {}
    finals: dict[str, np.ndarray] = {}
    registry_seed = int(registry["master_seed"])
    for ti, target in enumerate(targets):
        target_start = time.perf_counter()
        for ii, init_kind in enumerate(INITS):
            # Final and threshold references are paired across arms.  Cache
            # only their immutable self-distance terms; each trial still
            # reconstructs the same reference arrays from its recorded seed.
            reference_self_by_seed: dict[int, tuple[float, float]] = {}
            for seed in range(seeds):
                base = seed_base(registry_seed, ti, ii, seed)
                ref_final = target.sample(
                    profile.ref_final, np.random.default_rng(base + 4))
                ref_cross = target.sample(
                    profile.ref_cross, np.random.default_rng(base + 5))
                reference_self_by_seed[seed] = (
                    mean_pairwise_distance(ref_final, ref_final),
                    mean_pairwise_distance(ref_cross, ref_cross),
                )
            for arm in arms:
                for seed in range(seeds):
                    row, trajectory, final_q = run_trial(
                        target, ti, init_kind, ii, arm, seed, profile,
                        registry_seed, run.counter,
                        reference_self_by_seed[seed])
                    rows.append(row)
                    key = safe_key(
                        f"{arm.label}__{target.name}__{init_kind}__s{seed}")
                    trajectories[key] = trajectory
                    finals[key] = final_q
        run.log(f"  completed {target.name} in "
                f"{time.perf_counter() - target_start:.1f}s")
    write_rows(run.dir / "rows.csv", rows)
    np.savez_compressed(run.dir / "trajectories.npz", **trajectories)
    np.savez_compressed(run.dir / "final_particles.npz", **finals)
    return run, rows


# ---------------------------------------------------------------------------
# Validation selection and E4 statistics
# ---------------------------------------------------------------------------


def geomean_log(values: Iterable[float]) -> float:
    x = np.maximum(np.asarray(list(values), dtype=float), 1e-12)
    return float(np.exp(np.mean(np.log(x))))


def cell_median_ed2(rows: list[dict], arm: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["arm"] == arm:
            grouped[row["cell"]].append(float(row["ed2"]))
    return {cell: float(np.median(values)) for cell, values in grouped.items()}


def validation_aggregates(rows: list[dict], arms: list[Arm]) -> dict[str, float]:
    return {
        arm.label: geomean_log(cell_median_ed2(rows, arm.label).values())
        for arm in arms
    }


def choose_ncj(arms: list[Arm], aggregates: dict[str, float]) -> Arm:
    candidates = [arm for arm in arms if arm.label.startswith("ncj__")]
    return sorted(candidates, key=lambda arm: (
        round(aggregates[arm.label], 12), arm.sigma_ratio, arm.eta, arm.label
    ))[0]


def latest_frozen(profile_name: str) -> tuple[Path, dict]:
    files = sorted(RUNROOT.glob(
        f"*-NCJ-validation-{profile_name}/ncj_policy_frozen.json"),
        reverse=True)
    if not files:
        raise FileNotFoundError(
            f"no frozen NCJ validation policy for profile={profile_name}")
    path = files[0]
    obj = json.loads(path.read_text(encoding="utf-8"))
    return path, obj


def paired_logs(rows: list[dict], arm: str,
                cell_filter: Callable[[dict], bool] | None = None) \
        -> tuple[dict[str, list[float]], dict[str, str]]:
    paired: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    family: dict[str, str] = {}
    for row in rows:
        if row["arm"] in ("paper", arm) and \
                (cell_filter is None or cell_filter(row)):
            key = (row["cell"], int(row["seed"]))
            paired[key][row["arm"]] = row
            family[row["cell"]] = row["family"]
    out: dict[str, list[float]] = defaultdict(list)
    for (cell, _), pair in paired.items():
        if "paper" in pair and arm in pair:
            a = max(float(pair[arm]["ed2"]), 1e-12)
            b = max(float(pair["paper"]["ed2"]), 1e-12)
            out[cell].append(float(np.log(a / b)))
    return dict(out), family


def hierarchical_stats(rows: list[dict], arm: str,
                       cell_filter: Callable[[dict], bool] | None = None,
                       n_boot: int = 10000, seed: int = MASTER) -> dict:
    by_cell, families = paired_logs(rows, arm, cell_filter)
    cells = sorted(by_cell)
    if not cells:
        raise ValueError(f"no paired cells for {arm}")
    cell_logs = {cell: float(np.median(by_cell[cell])) for cell in cells}
    point = float(np.exp(np.mean(list(cell_logs.values()))))
    rng = np.random.default_rng(seed)
    hier = np.empty(n_boot)
    cell_only = np.empty(n_boot)
    all_rows = np.asarray([v for c in cells for v in by_cell[c]])
    fixed_rows = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(cells, size=len(cells), replace=True)
        cell_only[b] = np.exp(np.mean([cell_logs[c] for c in chosen]))
        sampled_medians = []
        for cell in chosen:
            values = np.asarray(by_cell[cell])
            sampled = values[rng.integers(0, len(values), len(values))]
            sampled_medians.append(np.median(sampled))
        hier[b] = np.exp(np.mean(sampled_medians))
        fixed_rows[b] = np.exp(np.mean(all_rows[
            rng.integers(0, len(all_rows), len(all_rows))]))
    family_ratios = {}
    for fam in sorted(set(families.values())):
        vals = [v for c, v in cell_logs.items() if families[c] == fam]
        family_ratios[fam] = float(np.exp(np.median(vals)))
    return {
        "point_ratio": point,
        "hierarchical_ci": [float(x) for x in np.quantile(hier, [.025, .975])],
        "cell_only_ci": [float(x) for x in np.quantile(cell_only, [.025, .975])],
        "fixed_suite_row_ci": [float(x) for x in np.quantile(fixed_rows,
                                                               [.025, .975])],
        "cell_ratios": {c: float(np.exp(v)) for c, v in cell_logs.items()},
        "family_ratios": family_ratios,
    }


def km_missing_mixtures(rows: list[dict], arm: str) -> int | None:
    mixture = {"gauss_mixture", "grid_mixture"}
    selected = [row for row in rows if row["arm"] == arm and
                row["init"] == "missing" and row["family"] in mixture]
    return km_median([int(row["event_time"]) for row in selected],
                     [bool(int(row["censored"])) for row in selected])


def run_validation(profile: Profile) -> Path:
    arms = validation_arms()
    run, rows = run_grid(
        "validation", profile, VALIDATION_REGISTRY, arms,
        profile.validation_seeds,
        {"selection": "minimum target-balanced geometric mean of cell-median ED2",
         "candidate_eta": ETA_GRID, "candidate_sigma_ratio": SIGMA_RATIOS,
         "tie_break": ["smaller sigma", "smaller eta", "label"]})
    aggregates = validation_aggregates(rows, arms)
    winner = choose_ncj(arms, aggregates)
    frozen = {
        "profile": profile.name,
        "winner_label": winner.label,
        "eta": winner.eta,
        "sigma_ratio": winner.sigma_ratio,
        "tau": TAU,
        "norm_clip": NORM_CLIP,
        "validation_aggregate": aggregates[winner.label],
        "paper_aggregate": aggregates["paper"],
        "winner_to_paper_ratio": aggregates[winner.label] /
            aggregates["paper"],
        "all_aggregates": aggregates,
        "validation_registry_sha256": sha256_file(VALIDATION_REGISTRY),
        "test_registry_sha256": sha256_file(TEST_REGISTRY),
        "source_commit": run.manifest["commit"],
        "protocol_sha256": sha256_file(PROTOCOL),
    }
    (run.dir / "ncj_policy_frozen.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8")
    summary = {"stage": "validation", "frozen_policy": frozen,
               "aggregates": aggregates}
    (run.dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    for label, value in sorted(aggregates.items(), key=lambda x: x[1]):
        run.log(f"  {label:45s} ED2={value:.8f}")
    run.log(f"FROZEN: {winner.label}, ratio-to-paper="
            f"{frozen['winner_to_paper_ratio']:.4f}")
    run.finish(len(rows))
    return run.dir


def run_test(profile: Profile) -> Path:
    frozen_path, frozen = latest_frozen(profile.name)
    if profile.name == "standard" and _git("status", "--porcelain"):
        raise RuntimeError(
            "commit the frozen validation run before opening the test registry")
    if frozen["test_registry_sha256"] != sha256_file(TEST_REGISTRY):
        raise RuntimeError("test registry changed after validation freeze")
    arms = test_arms(frozen)
    extra = {
        "frozen_policy_path": str(frozen_path.relative_to(ROOT)),
        "frozen_policy_sha256": sha256_file(frozen_path),
        "frozen_policy": frozen,
        "gate": {
            "ratio_le": 0.80,
            "hierarchical_ci_hi_lt": 1.0,
            "winning_cells_fraction_ge": 0.60,
            "max_family_ratio_le": 1.10,
            "gaussian_mixture_ci_hi_lt": 1.0,
            "non_gaussian_ci_hi_lt": 1.0,
            "missing_km_not_worse": True,
            "divergence_increase_le": 0.02,
            "equal_kernel_pairs": True,
        },
    }
    run, rows = run_grid_with_sources(
        "test", profile, TEST_REGISTRY, arms, profile.test_seeds, extra,
        (frozen_path,))
    arm_stats = {
        arm.label: hierarchical_stats(rows, arm.label,
                                      seed=MASTER + i * 101)
        for i, arm in enumerate(arms) if arm.label != "paper"
    }
    primary = arm_stats["ncj"]
    mixture_families = {"gauss_mixture", "grid_mixture"}
    mix = hierarchical_stats(
        rows, "ncj", lambda row: row["family"] in mixture_families,
        seed=MASTER + 7001)
    nong = hierarchical_stats(
        rows, "ncj", lambda row: row["family"] not in
        {"gaussian", "gauss_mixture", "grid_mixture"},
        seed=MASTER + 7002)
    win_fraction = float(np.mean(
        np.asarray(list(primary["cell_ratios"].values())) < 1.0))
    max_family = max(primary["family_ratios"].values())
    km_paper = km_missing_mixtures(rows, "paper")
    km_ncj = km_missing_mixtures(rows, "ncj")
    paper_rows = [row for row in rows if row["arm"] == "paper"]
    ncj_rows = [row for row in rows if row["arm"] == "ncj"]
    paper_div = float(np.mean([row["diverged"] for row in paper_rows]))
    ncj_div = float(np.mean([row["diverged"] for row in ncj_rows]))
    declared_pairs = {int(row["total_kernel_pairs"]) for row in rows}
    gate = {
        "aggregate_ratio": primary["point_ratio"],
        "hierarchical_ci": primary["hierarchical_ci"],
        "crit1_ratio_le_0.80": primary["point_ratio"] <= 0.80,
        "crit2_hierarchical_ci_hi_lt_1": primary["hierarchical_ci"][1] < 1.0,
        "winning_cells_fraction": win_fraction,
        "crit3_winning_cells_ge_0.60": win_fraction >= 0.60,
        "max_family_ratio": max_family,
        "family_ratios": primary["family_ratios"],
        "crit4_no_family_gt_1.10": max_family <= 1.10,
        "gaussian_mixture": mix,
        "crit5_gaussian_mixture_ci_hi_lt_1": mix["hierarchical_ci"][1] < 1.0,
        "non_gaussian": nong,
        "crit6_non_gaussian_ci_hi_lt_1": nong["hierarchical_ci"][1] < 1.0,
        "missing_km_paper": km_paper,
        "missing_km_ncj": km_ncj,
        "crit7_missing_km_not_worse":
            (km_ncj if km_ncj is not None else 10**9) <=
            (km_paper if km_paper is not None else 10**9),
        "paper_divergence_rate": paper_div,
        "ncj_divergence_rate": ncj_div,
        "distinct_total_kernel_pair_counts": sorted(declared_pairs),
        "crit8_divergence_and_equal_cost":
            ncj_div <= paper_div + 0.02 and len(declared_pairs) == 1,
    }
    criteria = [key for key in gate if key.startswith("crit")]
    gate["PASS"] = all(bool(gate[key]) for key in criteria)
    summary = {"stage": "test", "frozen_policy": frozen,
               "arm_statistics": arm_stats, "gate": gate}
    (run.dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (run.dir / "e4_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8")
    for arm, stats in arm_stats.items():
        run.log(f"  {arm:22s}: ratio={stats['point_ratio']:.4f} "
                f"hierCI={stats['hierarchical_ci']}")
    run.log(f"  mixture CI={mix['hierarchical_ci']}; "
            f"nonGaussian CI={nong['hierarchical_ci']}")
    run.log(f"  winning cells={win_fraction:.3f}; max family={max_family:.3f}; "
            f"missing KM paper={km_paper} ncj={km_ncj}")
    run.log(f"E4 GATE: {'PASS' if gate['PASS'] else 'FAIL'}")
    run.finish(len(rows))
    return run.dir


def run_generator_gate(profile: Profile) -> None:
    gate_files = sorted(RUNROOT.glob(
        f"*-NCJ-test-{profile.name}/e4_gate.json"), reverse=True)
    if not gate_files:
        raise SystemExit("generator transfer refused: no particle E4 gate exists")
    gate = json.loads(gate_files[0].read_text(encoding="utf-8"))
    if not gate.get("PASS", False):
        raise SystemExit(
            "generator transfer refused by the frozen protocol: particle E4 "
            "gate failed; this is the correct terminal behavior")
    raise SystemExit(
        "particle E4 passed, but generator execution remains blocked until a "
        "separate generator protocol is frozen and committed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validation", "test", "generator"))
    parser.add_argument("--profile", choices=sorted(PROFILES),
                        default="standard")
    args = parser.parse_args()
    verify_registry_hashes()
    profile = PROFILES[args.profile]
    if args.stage == "validation":
        path = run_validation(profile)
        print(f"validation artifacts: {path}")
    elif args.stage == "test":
        path = run_test(profile)
        print(f"test artifacts: {path}")
    else:
        run_generator_gate(profile)


if __name__ == "__main__":
    main()
