"""Sealed evaluation for CAP-EMF-1 — protocol section 7.2.

Runs **once**, after the final checkpoint is frozen and hashed.  It is built and
hashed into the preflight *before* the training run, so it cannot be written
while looking at training curves and then tuned to flatter them.

**FID follows the standard CIFAR-10 protocol**: 50 000 generated samples against
the 50 000 training images.  That is the reason the target is unconditional
CIFAR-10 rather than a single class — the headline number is comparable to
published results instead of a small-sample-biased figure.  The 10 000 test
images are opened only here, and only for the held-out cross-check.

Every sample grid is written **beside its nearest-training-image grid**.  At
960 epochs a model that memorizes produces beautiful samples, and the sample
grid alone cannot tell the difference.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..appearance import precision_recall, spectrum_report
from ..device import configure, resolve_device
from ..diagnostics import write_json
from ..fid import frechet_from_features, inception_features, kid_from_features
from ..stage_b2.metrics import duplicate_rate, nearest_reference_distances
from ..stage_b25.evaluation import density_coverage
from . import CAP_PHASE, CAP_UNIT
from .artifacts import (
    DEFAULT_RESULT,
    HERE,
    assert_result_path_unused,
    file_sha256,
    load_checkpoint,
    verify_sidecar,
)
from .config import enable_tf32, profile
from .data import cifar10_train_pool, sealed_test_pool
from .diagnostics import effective_rank, endpoint_health
from .model import CAPPixelTransformer, one_step_sample

#: FID and KID use the full 50 000; the manifold statistics are O(n^2) in
#: memory, so they use a declared subsample.  10 000 x 10 000 float64 is
#: ~800 MB, which is the largest that stays comfortable on a rented box.
FID_SAMPLES = 50_000
MANIFOLD_SAMPLES = 10_000
GRID_ROWS, GRID_COLUMNS = 8, 16


@dataclass(frozen=True)
class EvaluationSeeds:
    samples: int = 20_260_804
    manifold: int = 20_260_805
    grid: int = 20_260_806


def generate(
    model: torch.nn.Module,
    count: int,
    model_config,
    seed: int,
    device: torch.device,
    batch: int = 500,
) -> tuple[torch.Tensor, int]:
    """One-call samples, with the forward count asserted, not assumed."""
    generator = torch.Generator().manual_seed(int(seed))
    shape = (model_config.channels, model_config.image_size, model_config.image_size)
    calls = {"n": 0}
    handle = model.register_forward_pre_hook(
        lambda module, args: calls.__setitem__("n", calls["n"] + 1)
    )
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, count, batch):
            size = min(batch, count - start)
            noise = torch.randn((size,) + shape, generator=generator).to(device)
            chunks.append(one_step_sample(model, noise).cpu())
    handle.remove()
    expected = (count + batch - 1) // batch
    if calls["n"] != expected:
        raise RuntimeError(
            f"sampler made {calls['n']} evaluations for {expected} batches; "
            "CAP-EMF-1 must be exactly one network call per batch"
        )
    return torch.cat(chunks)[:count], calls["n"]


def _features(images: torch.Tensor, device: torch.device, batch: int = 500):
    rows = []
    for start in range(0, len(images), batch):
        block = images[start : start + batch]
        rows.append(inception_features(block, device).double().numpy())
    return np.concatenate(rows)


def _to_uint8(images: torch.Tensor) -> np.ndarray:
    scaled = (images.clamp(-1.0, 1.0) + 1.0) * 127.5
    return scaled.round().to(torch.uint8).permute(0, 2, 3, 1).numpy()


def save_grid(images: torch.Tensor, path: Path, rows: int, columns: int) -> str:
    """Write an uncurated grid.  No selection of any kind happens here."""
    from PIL import Image

    needed = rows * columns
    if len(images) < needed:
        raise ValueError("not enough images for the declared grid")
    array = _to_uint8(images[:needed])
    height, width = array.shape[1:3]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index in range(needed):
        r, c = divmod(index, columns)
        canvas[r * height : (r + 1) * height, c * width : (c + 1) * width] = array[index]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)
    return file_sha256(path)


def memorization_audit(
    generated: torch.Tensor, train: torch.Tensor, device: torch.device
) -> dict:
    """Nearest training image per sample: the check the grids exist to support."""
    distances, indices = nearest_reference_distances(generated, train, device)
    distances = distances.cpu()
    scale = float(
        torch.cdist(
            train[:512].reshape(512, -1).float(),
            train[512:1024].reshape(512, -1).float(),
        ).median()
    )
    return {
        "nearest_train_distance_median": float(distances.median()),
        "nearest_train_distance_p05": float(distances.quantile(0.05)),
        "nearest_train_distance_min": float(distances.min()),
        "real_real_distance_median": scale,
        "nearest_over_real_median": float(distances.median()) / scale,
        "duplicate_rate": duplicate_rate(generated[:2048], scale),
        "distinct_nearest_train_fraction": float(
            len(torch.unique(indices.cpu())) / len(indices)
        ),
        "nearest_train_indices": indices.cpu()[: GRID_ROWS * GRID_COLUMNS].tolist(),
    }


def evaluate(
    model: torch.nn.Module,
    frozen,
    device: torch.device,
    *,
    data_root: str | None,
    output_dir: Path,
    seeds: EvaluationSeeds,
) -> dict:
    train = cifar10_train_pool(data_root)
    test = sealed_test_pool(data_root, acknowledge_sealed=True)

    generated, _ = generate(
        model, FID_SAMPLES, frozen.model, seeds.samples, device
    )

    generated_features = _features(generated, device)
    train_features = _features(train, device)
    test_features = _features(test, device)

    rng = np.random.default_rng(seeds.manifold)
    pick_g = rng.choice(len(generated_features), MANIFOLD_SAMPLES, replace=False)
    pick_r = rng.choice(len(train_features), MANIFOLD_SAMPLES, replace=False)
    manifold_generated = generated_features[pick_g]
    manifold_real = train_features[pick_r]

    grids = {}
    grid_images = generated[: GRID_ROWS * GRID_COLUMNS]
    grids["samples"] = save_grid(
        grid_images, output_dir / "samples_uncurated.png", GRID_ROWS, GRID_COLUMNS
    )
    memorization = memorization_audit(generated[:4096], train, device)
    neighbours = train[torch.as_tensor(memorization["nearest_train_indices"])]
    grids["nearest_train"] = save_grid(
        neighbours, output_dir / "samples_nearest_train.png", GRID_ROWS, GRID_COLUMNS
    )

    health = endpoint_health(generated[:8192], train[:8192])

    return {
        "fid_50k_vs_train": frechet_from_features(generated_features, train_features),
        "fid_vs_sealed_test": frechet_from_features(generated_features, test_features),
        "kid_vs_train": kid_from_features(manifold_generated, manifold_real),
        "precision_recall": precision_recall(manifold_generated, manifold_real),
        "density_coverage": density_coverage(manifold_generated, manifold_real),
        "effective_rank": effective_rank(generated[:8192]),
        "target_effective_rank": effective_rank(train[:8192]),
        "endpoint_health": health,
        "spectrum": spectrum_report(generated[:2048], train[:2048]),
        "memorization": memorization,
        "grids": grids,
        "counts": {
            "generated": int(len(generated)),
            "fid_reference_train": int(len(train)),
            "sealed_test": int(len(test)),
            "manifold_subsample": MANIFOLD_SAMPLES,
        },
        "protocol_note": (
            "FID uses the standard CIFAR-10 protocol: 50k generated against the "
            "50k training images. Manifold statistics use a declared 10k "
            "subsample because they are O(n^2) in memory. The sealed test set "
            "is a held-out cross-check, not the headline reference."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CAP-EMF-1 sealed evaluation")
    parser.add_argument("--unit", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--kind", default="ema", choices=("ema", "raw"))
    parser.add_argument("--out", type=Path, default=HERE / "cap_evaluation.json")
    parser.add_argument("--grids", type=Path, default=HERE / "grids")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--acknowledge-sealed",
        action="store_true",
        help="required; opening the test split is a deliberate one-time act",
    )
    args = parser.parse_args()

    if not args.acknowledge_sealed:
        raise SystemExit(
            "the CIFAR-10 test split is sealed. Re-invoke with "
            "--acknowledge-sealed once the final checkpoint is frozen; this "
            "evaluation runs exactly once."
        )
    assert_result_path_unused(args.out)

    digest = verify_sidecar(args.unit)
    unit = json.loads(args.unit.read_text(encoding="utf-8"))
    if unit.get("status") != "cap-emf1-unit":
        raise RuntimeError("not a CAP-EMF-1 unit artifact")

    frozen = profile("capability")
    step = args.step or frozen.train.checkpoint_updates[-1]
    if args.step is not None and args.step != frozen.train.checkpoint_updates[-1]:
        # Permitted only for a recorded budget stop; never for a better number.
        print(
            f"WARNING: evaluating step {args.step}, not the primary "
            f"{frozen.train.checkpoint_updates[-1]}. This is legitimate only as "
            "a recorded budget stop.",
            flush=True,
        )
    record = unit["checkpoints"][str(step)][args.kind]
    payload = load_checkpoint(Path(record["path"]), expected_sha=record["sha256"])

    device = resolve_device(args.device)
    settings = configure(device)
    precision = enable_tf32()
    model = CAPPixelTransformer(frozen.model, 0).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)

    seeds = EvaluationSeeds()
    results = evaluate(
        model,
        frozen,
        device,
        data_root=args.data_root,
        output_dir=args.grids,
        seeds=seeds,
    )

    result = {
        "status": "cap-emf1-evaluation",
        "phase": CAP_PHASE,
        "unit": CAP_UNIT,
        "development_only": True,
        "sealed_split_opened": True,
        "evaluated_step": step,
        "evaluated_kind": args.kind,
        "is_primary_checkpoint": step == frozen.train.checkpoint_updates[-1],
        "unit_artifact_sha256": digest,
        "checkpoint_sha256": record["sha256"],
        "device": settings,
        "precision": precision,
        "seeds": seeds.__dict__,
        "results": results,
        "limits": [
            "One developmental unit; no replication claim.",
            "FID is comparable to published CIFAR-10 numbers only with the "
            "compute difference stated: 48M images seen against DDPM's 102M.",
            "Manifold statistics condition on a 10k subsample and a fitted kNN "
            "manifold.",
            "This evaluation runs once. No metric here may select a checkpoint.",
        ],
    }
    written = write_json(args.out, result)
    print(f"FID-50k vs train : {results['fid_50k_vs_train']:.3f}")
    print(f"FID vs sealed test: {results['fid_vs_sealed_test']:.3f}")
    print(f"KID              : {results['kid_vs_train']:.5f}")
    print(f"precision/recall : {results['precision_recall']}")
    print(f"nearest-train/real median ratio: "
          f"{results['memorization']['nearest_over_real_median']:.3f}")
    print(f"wrote {args.out} sha256={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
