"""Fresh target families for the sealed PSQT accumulator confirmation.

This module contains no candidate algorithm.  It is a deterministic sampler
from explicit JSON registry entries produced before confirmatory execution.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.linalg import norm


FAMILIES = (
    "gaussian-mixture",
    "disconnected-nongaussian",
    "rare-mode",
    "correlated-unimodal",
    "curved-connected",
    "multiple-curves",
    "skew-heavy",
    "dependence-trap",
)


def _rotation(angle: float) -> np.ndarray:
    return np.asarray([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])


def _transform(points: np.ndarray, angle: float,
               translation: list[float]) -> np.ndarray:
    return points @ _rotation(angle).T + np.asarray(translation, dtype=float)


def _mixture_labels(weights: np.ndarray, n: int,
                    rng: np.random.Generator) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if (weights.ndim != 1 or np.any(weights <= 0.0) or
            not np.isclose(weights.sum(), 1.0)):
        raise ValueError("mixture weights must be positive and sum to one")
    return rng.choice(len(weights), size=n, p=weights)


def sample_registry_target(entry: dict, n: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Sample one finite 2D target described entirely by ``entry``."""
    if n < 1:
        raise ValueError("sample count must be positive")
    family = entry["family"]
    if family not in FAMILIES:
        raise ValueError(f"unknown target family {family}")
    p = entry["parameters"]

    if family in {"gaussian-mixture", "rare-mode"}:
        means = np.asarray(p["means"], dtype=float)
        sigmas = np.asarray(p["sigmas"], dtype=float)
        weights = np.asarray(p["weights"], dtype=float)
        labels = _mixture_labels(weights, n, rng)
        points = means[labels] + rng.normal(size=(n, 2)) * sigmas[labels, None]

    elif family == "disconnected-nongaussian":
        centers = np.asarray(p["centers"], dtype=float)
        axes = np.asarray(p["axes"], dtype=float)
        angles = np.asarray(p["component_angles"], dtype=float)
        weights = np.asarray(p["weights"], dtype=float)
        labels = _mixture_labels(weights, n, rng)
        points = np.empty((n, 2), dtype=float)
        for component in range(len(weights)):
            mask = labels == component
            count = int(mask.sum())
            if count == 0:
                continue
            radius = np.sqrt(rng.random(count))
            theta = rng.uniform(0.0, 2.0 * np.pi, count)
            local = np.column_stack((radius * np.cos(theta),
                                     radius * np.sin(theta)))
            local *= axes[component]
            points[mask] = (
                local @ _rotation(float(angles[component])).T +
                centers[component])

    elif family == "correlated-unimodal":
        radial = p["radial"]
        if radial == "gaussian":
            base = rng.normal(size=(n, 2))
        elif radial == "student-t5":
            base = rng.standard_t(5, size=(n, 2)) / math.sqrt(5.0 / 3.0)
        elif radial == "laplace":
            base = rng.laplace(size=(n, 2)) / math.sqrt(2.0)
        else:
            raise ValueError(radial)
        scales = np.asarray(p["scales"], dtype=float)
        points = _transform(
            base * scales, float(p["angle"]), p["translation"])

    elif family == "curved-connected":
        kind = p["kind"]
        angle = float(p["angle"])
        translation = p["translation"]
        width = float(p["width"])
        if kind == "perturbed-ring":
            theta = rng.uniform(0.0, 2.0 * np.pi, n)
            radius = float(p["radius"]) * (
                1.0 + float(p["amplitude"]) * np.cos(
                    int(p["frequency"]) * theta + float(p["phase"])))
            radius += rng.normal(size=n) * width
            base = np.column_stack((radius * np.cos(theta),
                                    radius * np.sin(theta)))
        elif kind == "spiral":
            theta = rng.uniform(float(p["theta0"]), float(p["theta1"]), n)
            radius = float(p["r0"]) + float(p["slope"]) * (
                theta - float(p["theta0"]))
            radius += rng.normal(size=n) * width
            base = np.column_stack((radius * np.cos(theta),
                                    radius * np.sin(theta)))
        elif kind == "arc":
            theta = rng.uniform(float(p["theta0"]), float(p["theta1"]), n)
            radius = float(p["radius"]) + rng.normal(size=n) * width
            base = np.column_stack((radius * np.cos(theta),
                                    radius * np.sin(theta)))
        else:
            raise ValueError(kind)
        points = _transform(base, angle, translation)

    elif family == "multiple-curves":
        weights = np.asarray(p["weights"], dtype=float)
        labels = _mixture_labels(weights, n, rng)
        radii = np.asarray(p["radii"], dtype=float)
        widths = np.asarray(p["widths"], dtype=float)
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        radius = radii[labels] + rng.normal(size=n) * widths[labels]
        wobble = 1.0 + float(p["amplitude"]) * np.cos(
            int(p["frequency"]) * theta + float(p["phase"]))
        base = np.column_stack((radius * wobble * np.cos(theta),
                                radius * wobble * np.sin(theta)))
        points = _transform(base, float(p["angle"]), p["translation"])

    elif family == "skew-heavy":
        x = rng.lognormal(
            mean=float(p["log_mean"]), sigma=float(p["log_sigma"]), size=n)
        x -= math.exp(float(p["log_mean"]) +
                      0.5 * float(p["log_sigma"]) ** 2)
        y = rng.standard_t(float(p["df"]), size=n)
        y *= float(p["y_scale"])
        y += float(p["coupling"]) * np.tanh(x)
        base = np.column_stack((x * float(p["x_scale"]), y))
        points = _transform(base, float(p["angle"]), p["translation"])

    elif family == "dependence-trap":
        kind = p["kind"]
        if kind == "offaxis-binary":
            weights = np.asarray(p["weights"], dtype=float)
            labels = _mixture_labels(weights, n, rng)
            axis = np.asarray([
                math.cos(float(p["angle"])),
                math.sin(float(p["angle"])),
            ])
            orthogonal = np.asarray([-axis[1], axis[0]])
            signs = 2.0 * labels - 1.0
            base = (
                signs[:, None] * float(p["half_separation"]) * axis +
                rng.normal(size=(n, 1)) * float(p["axial_noise"]) * axis +
                rng.normal(size=(n, 1)) * float(p["orthogonal_noise"]) *
                orthogonal)
            points = base + np.asarray(p["translation"], dtype=float)
        elif kind == "checkerboard":
            pattern = int(p["parity"])
            first = rng.choice(np.asarray([-1.0, 1.0]), size=n)
            flip = rng.random(n) < float(p["flip_probability"])
            second = first * (1.0 if pattern > 0 else -1.0)
            second = np.where(flip, -second, second)
            base = np.column_stack((first, second)) * float(p["scale"])
            base += rng.normal(size=(n, 2)) * float(p["noise"])
            points = _transform(
                base, float(p["angle"]), p["translation"])
        elif kind == "nonlinear-sine":
            x = rng.uniform(-np.pi, np.pi, n)
            branch = rng.choice(np.asarray([-1.0, 1.0]), size=n)
            y = branch * np.sin(float(p["frequency"]) * x)
            base = np.column_stack((x * float(p["x_scale"]), y))
            base += rng.normal(size=(n, 2)) * float(p["noise"])
            points = _transform(
                base, float(p["angle"]), p["translation"])
        else:
            raise ValueError(kind)
    else:  # pragma: no cover - family list is exhaustive above
        raise AssertionError(family)

    if points.shape != (n, 2) or not np.all(np.isfinite(points)):
        raise FloatingPointError(f"invalid sample for target {entry['id']}")
    return points


def oracle_modes(entry: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Synthetic-only mode information, never exposed to candidate arms."""
    family = entry["family"]
    p = entry["parameters"]
    if family in {"gaussian-mixture", "rare-mode"}:
        return (np.asarray(p["means"], dtype=float),
                np.asarray(p["weights"], dtype=float))
    if family == "disconnected-nongaussian":
        return (np.asarray(p["centers"], dtype=float),
                np.asarray(p["weights"], dtype=float))
    if family == "dependence-trap" and p["kind"] == "offaxis-binary":
        axis = np.asarray([
            math.cos(float(p["angle"])),
            math.sin(float(p["angle"])),
        ])
        center = np.asarray(p["translation"], dtype=float)
        half = float(p["half_separation"])
        return (np.vstack((center - half * axis, center + half * axis)),
                np.asarray(p["weights"], dtype=float))
    return None, None


def bridge_fraction(points: np.ndarray, entry: dict) -> float:
    means, _ = oracle_modes(entry)
    if means is None or len(means) != 2:
        return float("nan")
    delta = means[1] - means[0]
    half = 0.5 * norm(delta)
    if half <= 0.0:
        return float("nan")
    direction = delta / norm(delta)
    midpoint = 0.5 * (means[0] + means[1])
    coordinate = (np.asarray(points) - midpoint) @ direction
    return float(np.mean(np.abs(coordinate) < 0.60 * half))


def mode_metrics(points: np.ndarray, entry: dict) \
        -> tuple[float, float, float, int]:
    means, weights = oracle_modes(entry)
    if means is None:
        return float("nan"), float("nan"), float("nan"), 0
    labels = norm(
        np.asarray(points)[:, None, :] - means[None, :, :], axis=2
    ).argmin(axis=1)
    masses = np.bincount(labels, minlength=len(means)) / len(points)
    coverage = float((masses >= 0.5 * weights).mean())
    mass_l1 = float(np.abs(masses - weights).sum())
    minority = int(np.argmin(weights))
    recovered = int(masses[minority] >= 0.5 * weights[minority])
    return coverage, mass_l1, float(masses[minority]), recovered

