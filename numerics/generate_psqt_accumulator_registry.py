"""Generate a fresh target-only registry for PSQT accumulator confirmation.

This file does not import any candidate or baseline algorithm.  It writes all
parameters before evaluation and refuses to overwrite an existing registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from numpy.linalg import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from psqt_confirmatory_targets import (  # noqa: E402
    FAMILIES,
    sample_registry_target,
)

SEALED_MASTER = 2026072501
SMOKE_MASTER = 2026072500
REGISTRY_VERSION = 1


def rounded(value):
    if isinstance(value, np.ndarray):
        return rounded(value.tolist())
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)):
        return round(float(value), 12)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def transform(points: np.ndarray, angle: float,
              translation: np.ndarray) -> np.ndarray:
    rotation = np.asarray([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    return points @ rotation.T + translation


def separated_centers(rng: np.random.Generator, count: int,
                      minimum: float, rejections: list[dict],
                      label: str) -> np.ndarray:
    for attempt in range(1000):
        centers = rng.uniform(-1.25, 1.25, size=(count, 2))
        pairwise = norm(
            centers[:, None, :] - centers[None, :, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)
        if float(pairwise.min()) >= minimum:
            if attempt:
                rejections.append({
                    "label": label,
                    "reason": "minimum center separation",
                    "rejected_draws": attempt,
                })
            return centers
    raise RuntimeError(f"could not generate separated centers for {label}")


def positive_weights(rng: np.random.Generator, count: int,
                     alpha: float, minimum: float,
                     rejections: list[dict], label: str) -> np.ndarray:
    for attempt in range(1000):
        weights = rng.dirichlet(np.full(count, alpha))
        if float(weights.min()) >= minimum:
            if attempt:
                rejections.append({
                    "label": label,
                    "reason": "minimum component weight",
                    "rejected_draws": attempt,
                })
            return weights
    raise RuntimeError(f"could not generate weights for {label}")


def translation(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-0.65, 0.65, size=2)


def gaussian_mixture(index: int, rng: np.random.Generator,
                     rejections: list[dict]) -> dict:
    count = int(rng.integers(2, 8))
    centers = separated_centers(
        rng, count, 0.48, rejections, f"F1-{index:02d}")
    angle = float(rng.uniform(0.0, np.pi))
    shift = translation(rng)
    means = transform(centers, angle, shift)
    alpha = float(rng.uniform(0.7, 2.0))
    weights = positive_weights(
        rng, count, alpha, 0.03, rejections, f"F1-{index:02d}")
    sigmas = rng.uniform(0.06, 0.24, size=count)
    scale = float(norm(means - shift, axis=1).max() + 3 * sigmas.max())
    return {
        "id": f"CF-F1-GMM-{index:02d}",
        "family": "gaussian-mixture",
        "scale": scale,
        "parameters": {
            "means": means, "sigmas": sigmas, "weights": weights,
            "generation_angle": angle, "translation": shift,
            "dirichlet_alpha": alpha,
        },
    }


def disconnected(index: int, rng: np.random.Generator,
                 rejections: list[dict]) -> dict:
    count = int(rng.integers(2, 5))
    centers = separated_centers(
        rng, count, 0.85, rejections, f"F2-{index:02d}")
    angle = float(rng.uniform(0.0, np.pi))
    shift = translation(rng)
    centers = transform(centers, angle, shift)
    axes = np.column_stack((
        rng.uniform(0.12, 0.32, count),
        rng.uniform(0.05, 0.16, count),
    ))
    weights = positive_weights(
        rng, count, float(rng.uniform(0.8, 2.0)), 0.05,
        rejections, f"F2-{index:02d}")
    scale = float(norm(centers - shift, axis=1).max() + axes.max())
    return {
        "id": f"CF-F2-ELLIPSES-{index:02d}",
        "family": "disconnected-nongaussian",
        "scale": scale,
        "parameters": {
            "centers": centers,
            "axes": axes,
            "component_angles": rng.uniform(0.0, np.pi, count),
            "weights": weights,
        },
    }


def rare_mode(index: int, rng: np.random.Generator) -> dict:
    minority = (0.02, 0.05, 0.10)[index % 3]
    angle = float(rng.uniform(0.0, np.pi))
    axis = np.asarray([math.cos(angle), math.sin(angle)])
    half = float(rng.uniform(0.75, 1.45))
    shift = translation(rng)
    means = np.vstack((shift - half * axis, shift + half * axis))
    if index % 2:
        weights = np.asarray([minority, 1.0 - minority])
    else:
        weights = np.asarray([1.0 - minority, minority])
    sigmas = rng.uniform(0.055, 0.18, size=2)
    return {
        "id": f"CF-F3-RARE-{int(100*minority):02d}-{index:02d}",
        "family": "rare-mode",
        "scale": float(half + 3 * sigmas.max()),
        "parameters": {
            "means": means, "sigmas": sigmas, "weights": weights,
            "minority": minority,
        },
    }


def correlated(index: int, rng: np.random.Generator) -> dict:
    major = float(rng.uniform(0.55, 1.15))
    ratio = float(rng.uniform(0.12, 0.55))
    radial = ("gaussian", "student-t5", "laplace")[index % 3]
    return {
        "id": f"CF-F4-CORRELATED-{index:02d}",
        "family": "correlated-unimodal",
        "scale": 3.5 * major,
        "parameters": {
            "radial": radial,
            "scales": [major, major * ratio],
            "angle": float(rng.uniform(0.0, np.pi)),
            "translation": translation(rng),
        },
    }


def curved(index: int, rng: np.random.Generator) -> dict:
    kind = ("perturbed-ring", "spiral", "arc")[index % 3]
    common = {
        "kind": kind,
        "angle": float(rng.uniform(0.0, np.pi)),
        "translation": translation(rng),
        "width": float(rng.uniform(0.025, 0.085)),
    }
    if kind == "perturbed-ring":
        common.update({
            "radius": float(rng.uniform(0.75, 1.25)),
            "amplitude": float(rng.uniform(0.05, 0.18)),
            "frequency": int(rng.integers(2, 6)),
            "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
        })
        scale = common["radius"] * (1 + common["amplitude"]) + \
            3 * common["width"]
    elif kind == "spiral":
        theta0 = float(rng.uniform(-0.5, 0.5))
        theta1 = theta0 + float(rng.uniform(1.5 * np.pi, 2.5 * np.pi))
        common.update({
            "theta0": theta0, "theta1": theta1,
            "r0": float(rng.uniform(0.15, 0.35)),
            "slope": float(rng.uniform(0.08, 0.16)),
        })
        scale = common["r0"] + common["slope"] * (theta1 - theta0) + \
            3 * common["width"]
    else:
        theta0 = float(rng.uniform(-np.pi, 0.0))
        theta1 = theta0 + float(rng.uniform(0.8 * np.pi, 1.7 * np.pi))
        common.update({
            "theta0": theta0, "theta1": theta1,
            "radius": float(rng.uniform(0.75, 1.35)),
        })
        scale = common["radius"] + 3 * common["width"]
    return {
        "id": f"CF-F5-{kind.upper()}-{index:02d}",
        "family": "curved-connected",
        "scale": float(scale),
        "parameters": common,
    }


def multiple_curves(index: int, rng: np.random.Generator) -> dict:
    inner = float(rng.uniform(0.32, 0.62))
    outer = float(rng.uniform(0.95, 1.45))
    weight = float(rng.uniform(0.25, 0.75))
    amplitude = float(rng.uniform(0.0, 0.10))
    return {
        "id": f"CF-F6-MULTICURVE-{index:02d}",
        "family": "multiple-curves",
        "scale": outer * (1 + amplitude) + 0.18,
        "parameters": {
            "radii": [inner, outer],
            "widths": rng.uniform(0.02, 0.06, size=2),
            "weights": [weight, 1.0 - weight],
            "amplitude": amplitude,
            "frequency": int(rng.integers(2, 7)),
            "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            "angle": float(rng.uniform(0.0, np.pi)),
            "translation": translation(rng),
        },
    }


def skew_heavy(index: int, rng: np.random.Generator) -> dict:
    log_sigma = float(rng.uniform(0.45, 0.85))
    x_scale = float(rng.uniform(0.35, 0.75))
    y_scale = float(rng.uniform(0.22, 0.65))
    return {
        "id": f"CF-F7-SKEW-{index:02d}",
        "family": "skew-heavy",
        "scale": float(4.0 * max(x_scale, y_scale)),
        "parameters": {
            "log_mean": float(rng.uniform(-0.15, 0.15)),
            "log_sigma": log_sigma,
            "x_scale": x_scale,
            "y_scale": y_scale,
            "df": float(rng.uniform(3.2, 7.0)),
            "coupling": float(rng.uniform(-0.7, 0.7)),
            "angle": float(rng.uniform(0.0, np.pi)),
            "translation": translation(rng),
        },
    }


def dependence(index: int, rng: np.random.Generator) -> dict:
    kind = ("offaxis-binary", "checkerboard", "nonlinear-sine")[index % 3]
    common = {
        "kind": kind,
        "angle": float(rng.uniform(0.08, np.pi - 0.08)),
        "translation": translation(rng),
    }
    if kind == "offaxis-binary":
        common.update({
            "half_separation": float(rng.uniform(0.75, 1.35)),
            "axial_noise": float(rng.uniform(0.04, 0.13)),
            "orthogonal_noise": float(rng.uniform(0.04, 0.16)),
            "weights": [0.5, 0.5],
        })
        scale = common["half_separation"] + 0.5
    elif kind == "checkerboard":
        common.update({
            "parity": 1 if index % 2 == 0 else -1,
            "flip_probability": float(rng.uniform(0.03, 0.18)),
            "scale": float(rng.uniform(0.65, 1.15)),
            "noise": float(rng.uniform(0.04, 0.13)),
        })
        scale = math.sqrt(2) * common["scale"] + 0.4
    else:
        common.update({
            "frequency": float(rng.uniform(0.8, 1.8)),
            "x_scale": float(rng.uniform(0.25, 0.5)),
            "noise": float(rng.uniform(0.03, 0.10)),
        })
        scale = 2.0
    return {
        "id": f"CF-F8-{kind.upper()}-{index:02d}",
        "family": "dependence-trap",
        "scale": float(scale),
        "parameters": common,
    }


BUILDERS = (
    gaussian_mixture,
    disconnected,
    rare_mode,
    correlated,
    curved,
    multiple_curves,
    skew_heavy,
    dependence,
)


def generate(kind: str) -> dict:
    if kind not in {"smoke", "sealed"}:
        raise ValueError(kind)
    master = SMOKE_MASTER if kind == "smoke" else SEALED_MASTER
    count = 1 if kind == "smoke" else 8
    sequence = np.random.SeedSequence(master)
    children = sequence.spawn(len(BUILDERS))
    targets = []
    rejections: list[dict] = []
    for family_index, (family, builder, child) in enumerate(zip(
            FAMILIES, BUILDERS, children), start=1):
        rng = np.random.default_rng(child)
        for index in range(count):
            if builder in {gaussian_mixture, disconnected}:
                target = builder(index, rng, rejections)
            else:
                target = builder(index, rng)
            if target["family"] != family:
                raise AssertionError("builder/family order mismatch")
            target["registry_seed"] = int(
                master + family_index * 100_003 + index * 1009)
            targets.append(rounded(target))
    ids = [target["id"] for target in targets]
    if len(ids) != len(set(ids)):
        raise AssertionError("registry target IDs are not unique")

    # Target-only structural validation; no candidate is imported or run.
    validations = []
    for target in targets:
        rng = np.random.default_rng(int(target["registry_seed"]))
        sample = sample_registry_target(target, 2048, rng)
        validations.append({
            "id": target["id"],
            "finite": bool(np.all(np.isfinite(sample))),
            "empirical_coordinate_scale": rounded(
                np.quantile(norm(sample - np.median(sample, axis=0), axis=1),
                            0.95)),
        })
    if not all(item["finite"] for item in validations):
        raise AssertionError("registry structural validation failed")
    return {
        "registry_version": REGISTRY_VERSION,
        "status": "disposable-smoke" if kind == "smoke" else "sealed-fresh",
        "master_seed": master,
        "targets_per_family": count,
        "families": list(FAMILIES),
        "target_count": len(targets),
        "construction_rejections": rejections,
        "structural_validation": validations,
        "targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("smoke", "sealed"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = generate(args.kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    args.output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(f"registry: {args.output}")
    print(f"targets: {payload['target_count']}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()

