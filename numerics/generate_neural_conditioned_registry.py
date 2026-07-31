"""Generate the fresh neural conditioned-transport confirmation registry.

This module is deliberately algorithm-independent.  It records target laws,
directions, and seeds but never imports a candidate or evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "neural_conditioned_confirmatory_registry_v3.json"
MASTER_SEED = 20260803
DIMENSIONS = (2, 4, 8, 16)
FAMILIES = ("balanced-gmm", "rare-gmm", "correlated-t", "nonlinear")
INSTANCES = 4
TRAIN_DIRECTIONS = 64
HELDOUT_DIRECTIONS = 128


def orthogonal_matrix(rng: np.random.Generator, dimension: int) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs[None, :]


def orthogonal_bank(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    blocks = [
        orthogonal_matrix(rng, dimension).T
        for _ in range((count + dimension - 1) // dimension)
    ]
    return np.concatenate(blocks, axis=0)[:count]


def centered_centers(
    rng: np.random.Generator, count: int, dimension: int, radius: float
) -> np.ndarray:
    centers = rng.normal(size=(count, dimension))
    centers -= centers.mean(axis=0, keepdims=True)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    return radius * centers


def parameters(family: str, dimension: int, rng: np.random.Generator) -> dict:
    if family == "balanced-gmm":
        count = int(rng.integers(3, 7))
        radius = float(rng.uniform(2.3, 3.2))
        return {
            "centers": centered_centers(rng, count, dimension, radius).tolist(),
            "weights": [1.0 / count] * count,
            "sigmas": rng.uniform(0.28, 0.58, size=count).tolist(),
        }
    if family == "rare-gmm":
        rare_mass = float(rng.choice([0.02, 0.05, 0.10]))
        direction = rng.normal(size=dimension)
        direction /= np.linalg.norm(direction)
        separation = float(rng.uniform(3.2, 4.5))
        return {
            "centers": np.stack([-0.2 * direction, separation * direction]).tolist(),
            "weights": [1.0 - rare_mass, rare_mass],
            "sigmas": [float(rng.uniform(0.45, 0.65)), float(rng.uniform(0.22, 0.35))],
        }
    if family == "correlated-t":
        maximum_scale = float(rng.uniform(1.25, 2.0))
        return {
            "rotation": orthogonal_matrix(rng, dimension).tolist(),
            "axis_scales": np.geomspace(
                float(rng.uniform(0.22, 0.40)), maximum_scale, dimension
            ).tolist(),
            "degrees_of_freedom": float(rng.uniform(3.5, 7.0)),
            "skew_strength": float(rng.uniform(0.10, 0.35)),
        }
    if family == "nonlinear":
        return {
            "rotation": orthogonal_matrix(rng, dimension).tolist(),
            "curve_strength": float(rng.uniform(0.55, 1.0)),
            "noise_scale": float(rng.uniform(0.15, 0.30)),
        }
    raise ValueError(f"unknown family {family}")


def build_registry(
    master_seed: int = MASTER_SEED, instances: int = INSTANCES
) -> dict:
    if instances < 1:
        raise ValueError("instances must be positive")
    cells = len(DIMENSIONS) * len(FAMILIES) * instances
    sequences = np.random.SeedSequence(master_seed).spawn(cells)
    targets: list[dict] = []
    index = 0
    for dimension in DIMENSIONS:
        for family in FAMILIES:
            for instance in range(instances):
                parameter_seq, train_seq, heldout_seq, seed_seq = sequences[
                    index
                ].spawn(4)
                seeds = seed_seq.generate_state(8, dtype=np.uint32)
                targets.append(
                    {
                        "index": index,
                        "name": f"NCT-d{dimension}-{family}-i{instance}",
                        "dimension": dimension,
                        "family": family,
                        "instance": instance,
                        "parameters": parameters(
                            family, dimension, np.random.default_rng(parameter_seq)
                        ),
                        "training_directions": orthogonal_bank(
                            np.random.default_rng(train_seq),
                            TRAIN_DIRECTIONS,
                            dimension,
                        ).tolist(),
                        "heldout_directions": orthogonal_bank(
                            np.random.default_rng(heldout_seq),
                            HELDOUT_DIRECTIONS,
                            dimension,
                        ).tolist(),
                        "seeds": {
                            "target_pool": int(seeds[0]),
                            "evaluation_target": int(seeds[1]),
                            "evaluation_latent": int(seeds[2]),
                            "model": int(seeds[3]),
                            "training_latent": int(seeds[4]),
                            "minibatch_order": int(seeds[5]),
                            "bootstrap": int(seeds[6]),
                            "reserved": int(seeds[7]),
                        },
                    }
                )
                index += 1
    bootstrap_seed = int(
        np.random.SeedSequence([master_seed, 999]).generate_state(
            1, dtype=np.uint32
        )[0]
    )
    return {
        "schema": "neural-conditioned-confirmatory-v1",
        "master_seed": master_seed,
        "bootstrap_seed": bootstrap_seed,
        "dimensions": list(DIMENSIONS),
        "families": list(FAMILIES),
        "instances_per_cell": instances,
        "training_direction_count": TRAIN_DIRECTIONS,
        "heldout_direction_count": HELDOUT_DIRECTIONS,
        "target_count": len(targets),
        "rejection_count": 0,
        "targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--instances", type=int, default=INSTANCES)
    args = parser.parse_args()
    output = args.output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite {output} or its sidecar")
    payload = (
        json.dumps(
            build_registry(args.master_seed, args.instances),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    output.write_bytes(payload)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
