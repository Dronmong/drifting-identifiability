"""Generate the fresh Phase-1 neural pooled-rank development registry.

The generator records all target parameters and projection directions before
any endpoint is evaluated.  It never imports or inspects an algorithm runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "neural_pooled_rank_development_registry.json"
MASTER_SEED = 20260725
DIMENSIONS = (2, 4, 8, 16)
FAMILIES = ("balanced-gmm", "rare-gmm", "correlated-t", "nonlinear")
TRAIN_DIRECTIONS = 64
HELDOUT_DIRECTIONS = 128
SMOKE_FAMILY = {
    2: "balanced-gmm",
    4: "rare-gmm",
    8: "correlated-t",
    16: "nonlinear",
}


def orthogonal_matrix(rng: np.random.Generator, dimension: int) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs[None, :]


def orthogonal_bank(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    while sum(len(block) for block in rows) < count:
        rows.append(orthogonal_matrix(rng, dimension).T)
    result = np.concatenate(rows, axis=0)[:count]
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def centered_random_centers(
    rng: np.random.Generator, count: int, dimension: int, radius: float
) -> np.ndarray:
    centers = rng.normal(size=(count, dimension))
    centers -= centers.mean(axis=0, keepdims=True)
    lengths = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / np.maximum(lengths, 1e-12) * radius
    return centers


def target_parameters(family: str, dimension: int, rng: np.random.Generator) -> dict:
    if family == "balanced-gmm":
        centers = centered_random_centers(rng, 4, dimension, 2.75)
        return {
            "centers": centers.tolist(),
            "weights": [0.25] * 4,
            "sigmas": [0.32, 0.38, 0.44, 0.50],
        }
    if family == "rare-gmm":
        direction = rng.normal(size=dimension)
        direction /= np.linalg.norm(direction)
        centers = np.stack(
            [
                -0.20 * direction,
                3.80 * direction,
            ]
        )
        return {
            "centers": centers.tolist(),
            "weights": [0.95, 0.05],
            "sigmas": [0.55, 0.28],
        }
    if family == "correlated-t":
        return {
            "rotation": orthogonal_matrix(rng, dimension).tolist(),
            "axis_scales": np.geomspace(0.30, 1.60, dimension).tolist(),
            "degrees_of_freedom": 4.0,
            "skew_strength": 0.20,
        }
    if family == "nonlinear":
        return {
            "rotation": orthogonal_matrix(rng, dimension).tolist(),
            "curve_strength": 0.75,
            "noise_scale": 0.20,
        }
    raise ValueError(f"unknown family {family}")


def build_registry(
    *,
    master_seed: int = MASTER_SEED,
    variants_per_cell: int = 1,
) -> dict:
    if variants_per_cell < 1:
        raise ValueError("variants_per_cell must be positive")
    root = np.random.SeedSequence(master_seed)
    target_sequences = root.spawn(
        len(DIMENSIONS) * len(FAMILIES) * variants_per_cell
    )
    targets = []
    index = 0
    for dimension in DIMENSIONS:
        for family in FAMILIES:
            for variant in range(variants_per_cell):
                sequence = target_sequences[index]
                parameter_seq, train_seq, heldout_seq, seed_seq = sequence.spawn(4)
                parameter_rng = np.random.default_rng(parameter_seq)
                train_rng = np.random.default_rng(train_seq)
                heldout_rng = np.random.default_rng(heldout_seq)
                seeds = seed_seq.generate_state(8, dtype=np.uint32)
                name = f"NPR-d{dimension}-{family}"
                if variants_per_cell > 1:
                    name += f"-v{variant:02d}"
                target = {
                    "index": index,
                    "name": name,
                    "dimension": dimension,
                    "family": family,
                    "smoke": (
                        variant == 0 and family == SMOKE_FAMILY[dimension]
                    ),
                    "parameters": target_parameters(family, dimension, parameter_rng),
                    "training_directions": orthogonal_bank(
                        train_rng, TRAIN_DIRECTIONS, dimension
                    ).tolist(),
                    "heldout_directions": orthogonal_bank(
                        heldout_rng, HELDOUT_DIRECTIONS, dimension
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
                if variants_per_cell > 1:
                    target["variant"] = variant
                targets.append(target)
                index += 1
    registry = {
        "schema": "neural-pooled-rank-development-v1",
        "master_seed": master_seed,
        "dimensions": list(DIMENSIONS),
        "families": list(FAMILIES),
        "training_direction_count": TRAIN_DIRECTIONS,
        "heldout_direction_count": HELDOUT_DIRECTIONS,
        "target_count": len(targets),
        "rejection_count": 0,
        "targets": targets,
    }
    if variants_per_cell > 1:
        registry["variants_per_cell"] = variants_per_cell
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--variants-per-cell", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if (output.exists() or sidecar.exists()) and not args.force:
        raise FileExistsError(f"refusing to overwrite registry or sidecar: {output}")
    payload = (
        json.dumps(
            build_registry(
                master_seed=args.master_seed,
                variants_per_cell=args.variants_per_cell,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
