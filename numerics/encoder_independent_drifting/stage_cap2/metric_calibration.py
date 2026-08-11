"""One train-only real/real reference point for CAP2 image metrics.

This is deliberately *not* called a variance or noise calibration: one
disjoint partition supplies one finite-sample sanity point, not a sampling
distribution.  Both halves are also compared with the same published
CIFAR-10-train CleanFID/CleanKID reference used for model evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..device import configure, resolve_device
from ..stage_cap.data import cifar10_train_pool
from .artifacts import (
    assert_unused,
    source_manifest,
    write_json_atomic,
    write_npz_atomic,
)
from .standard_metrics import (
    DEFAULT_KID_SEED,
    DEFAULT_METRIC_WORKERS,
    DEFAULT_SAMPLE_COUNT,
    _require_cleanfid,
    clean_metrics_from_features,
    clean_pair_metrics_from_features,
    evaluation_provenance,
    extract_clean_features,
    feature_array_sha256,
    load_clean_feature_archive,
    validate_png_folder,
)


def revalidate_metric_calibration_evidence(
    calibration: dict,
    *,
    anchor: Path,
    expected_count: int = DEFAULT_SAMPLE_COUNT,
    expected_dimension: int = 2_048,
) -> dict[str, object]:
    """Recompute the real/real margin from the retained canonical features."""

    if not isinstance(calibration, dict):
        raise TypeError("metric calibration evidence must be a dictionary")
    metrics = calibration.get("metrics", {})
    reference = metrics.get("kid_reference", {}) if isinstance(metrics, dict) else {}
    if not isinstance(reference, dict):
        raise TypeError("metric calibration lacks its KID reference record")
    reference_path = Path(str(reference.get("path", "")))
    if not reference_path.is_absolute():
        reference_path = (anchor / reference_path).resolve()
    features, actual = load_clean_feature_archive(
        reference_path,
        expected_count=expected_count,
        expected_dimension=expected_dimension,
    )
    seed = calibration.get("seed")
    kid_seed = metrics.get("kid_seed") if isinstance(metrics, dict) else None
    samples_per_side = calibration.get("samples_per_side")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(kid_seed, bool)
        or not isinstance(kid_seed, int)
        or isinstance(samples_per_side, bool)
        or not isinstance(samples_per_side, int)
        or samples_per_side * 2 != expected_count
    ):
        raise RuntimeError("metric calibration seed/count metadata is malformed")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(expected_count, generator=generator)
    left_indices = indices[:samples_per_side]
    right_indices = indices[samples_per_side:]
    left = features[left_indices.numpy()]
    right = features[right_indices.numpy()]
    fid_module = _require_cleanfid()
    recomputed = {
        "direct_disjoint_pair": clean_pair_metrics_from_features(
            fid_module, left, right, kid_seed=kid_seed
        ),
        "matched_published_train_reference": {
            "left": clean_metrics_from_features(
                fid_module,
                left,
                kid_reference_features=features,
                kid_seed=kid_seed,
            ),
            "right": clean_metrics_from_features(
                fid_module,
                right,
                kid_reference_features=features,
                kid_seed=kid_seed,
            ),
        },
    }

    def same_metrics(recorded: object, observed: dict[str, float]) -> bool:
        return (
            isinstance(recorded, dict)
            and set(recorded) == set(observed)
            and all(
                isinstance(recorded.get(name), (int, float))
                and not isinstance(recorded.get(name), bool)
                and math.isfinite(float(recorded[name]))
                and math.isclose(
                    float(recorded[name]), value, rel_tol=1e-12, abs_tol=1e-12
                )
                for name, value in observed.items()
            )
        )

    recorded_matched = metrics.get("matched_published_train_reference", {})
    checks = {
        "canonical_feature_archive": all(
            reference.get(name) == actual.get(name)
            for name in ("sha256", "feature_sha256", "count", "dimension", "dtype")
        ),
        "left_indices_reconstructed": calibration.get("left_indices_sha256")
        == hashlib.sha256(left_indices.numpy().tobytes()).hexdigest(),
        "right_indices_reconstructed": calibration.get("right_indices_sha256")
        == hashlib.sha256(right_indices.numpy().tobytes()).hexdigest(),
        "direct_pair_recomputed": same_metrics(
            metrics.get("direct_disjoint_pair"), recomputed["direct_disjoint_pair"]
        ),
        "left_reference_scores_recomputed": same_metrics(
            recorded_matched.get("left")
            if isinstance(recorded_matched, dict)
            else None,
            recomputed["matched_published_train_reference"]["left"],
        ),
        "right_reference_scores_recomputed": same_metrics(
            recorded_matched.get("right")
            if isinstance(recorded_matched, dict)
            else None,
            recomputed["matched_published_train_reference"]["right"],
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "recomputed": recomputed,
        "kid_reference_archive": actual,
        "limit": (
            "recomputed from the retained canonical clean-Inception features; "
            "does not re-extract those features from CIFAR-10 PNG bytes"
        ),
    }


def export_images(images: torch.Tensor, folder: Path) -> dict[str, object]:
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(f"refusing to mix calibration images in {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    arrays = (
        ((images.clamp(-1, 1) + 1) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    for index, array in enumerate(arrays):
        Image.fromarray(array).save(
            folder / f"{index:06d}.png", format="PNG", compress_level=0
        )
    return validate_png_folder(
        folder,
        expected_count=len(images),
        require_sequential_names=True,
    )


def assemble_reference_features(
    left_features: np.ndarray,
    right_features: np.ndarray,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    *,
    expected_count: int = DEFAULT_SAMPLE_COUNT,
) -> np.ndarray:
    """Restore two feature partitions to canonical dataset-index order."""

    if left_features.ndim != 2 or right_features.ndim != 2:
        raise RuntimeError("CleanFID returned a non-matrix feature population")
    if left_features.shape[1:] != right_features.shape[1:]:
        raise RuntimeError("calibration feature dimensions differ between halves")
    if len(left_features) != len(left_indices) or len(right_features) != len(
        right_indices
    ):
        raise RuntimeError("feature rows do not match their recorded dataset indices")
    left_order = left_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    right_order = right_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    order = np.concatenate((left_order, right_order))
    if len(order) != expected_count or not np.array_equal(
        np.sort(order), np.arange(expected_count, dtype=np.int64)
    ):
        raise RuntimeError(
            "KID reference construction requires an exact partition of all "
            f"{expected_count:,} dataset indices"
        )
    combined = np.concatenate((left_features, right_features), axis=0)
    reference_features = np.empty_like(combined)
    reference_features[order] = combined
    if not np.isfinite(reference_features).all():
        raise RuntimeError("KID reference contains non-finite CleanFID features")
    return reference_features


def compute_real_real(
    left: Path,
    right: Path,
    *,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    kid_reference_out: Path,
    device: torch.device,
    batch: int,
    workers: int,
    kid_seed: int,
    verbose: bool = True,
) -> dict[str, object]:
    fid_module, left_features = extract_clean_features(
        left,
        device=device,
        batch=batch,
        workers=workers,
        verbose=verbose,
        expected_count=len(left_indices),
    )
    second_module, right_features = extract_clean_features(
        right,
        device=device,
        batch=batch,
        workers=workers,
        verbose=verbose,
        expected_count=len(right_indices),
    )
    if second_module is not fid_module:
        raise RuntimeError("CleanFID module changed between matched feature passes")
    reference_features = assemble_reference_features(
        left_features, right_features, left_indices, right_indices
    )
    if reference_features.shape[1] != 2_048:
        raise RuntimeError(
            f"expected 2,048 CleanFID features, found {reference_features.shape[1]}"
        )

    # The two folders are a random partition, but KID must use one canonical
    # population independent of that partition.  Restore dataset-index order
    # and seal the resulting 50k CleanFID feature matrix as a shared artifact.
    archive_sha256 = write_npz_atomic(kid_reference_out, features=reference_features)
    kid_reference = {
        "path": str(kid_reference_out.resolve()),
        "sha256": archive_sha256,
        "feature_sha256": feature_array_sha256(reference_features),
        "count": len(reference_features),
        "dimension": reference_features.shape[1],
        "dtype": str(reference_features.dtype),
        "population": "all 50,000 CIFAR-10 train images in dataset-index order",
        "preprocessing": "clean-fid 0.1.35 clean Inception preprocessing",
    }
    return {
        "direct_disjoint_pair": clean_pair_metrics_from_features(
            fid_module,
            left_features,
            right_features,
            kid_seed=kid_seed,
        ),
        "matched_published_train_reference": {
            "left": clean_metrics_from_features(
                fid_module,
                left_features,
                kid_reference_features=reference_features,
                kid_seed=kid_seed,
            ),
            "right": clean_metrics_from_features(
                fid_module,
                right_features,
                kid_reference_features=reference_features,
                kid_seed=kid_seed,
            ),
        },
        "backend": "clean-fid",
        "cleanfid_version": evaluation_provenance(device)["packages"]["clean-fid"],
        "mode": "clean",
        "reference": "cifar10/train/32",
        "kid_seed": kid_seed,
        "feature_counts": {
            "left": len(left_features),
            "right": len(right_features),
        },
        "kid_reference": kid_reference,
        "comparison": "one pair of disjoint CIFAR-10 training subsets",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--samples-per-side", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=20_260_806)
    parser.add_argument("--kid-seed", type=int, default=DEFAULT_KID_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-batch", type=int, default=128)
    parser.add_argument("--metric-workers", type=int, default=DEFAULT_METRIC_WORKERS)
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--kid-reference-features-out", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("metric_calibration.json")
    )
    args = parser.parse_args()
    assert_unused(args.out)
    assert_unused(args.kid_reference_features_out)
    device = resolve_device(args.device)
    numerical_settings = configure(device, allow_tf32=False)
    torch.use_deterministic_algorithms(True)
    pool = cifar10_train_pool(args.data_root)
    if args.samples_per_side != DEFAULT_SAMPLE_COUNT // 2:
        raise ValueError(
            "the canonical KID reference requires exactly 25,000 samples per side"
        )
    if 2 * args.samples_per_side != len(pool):
        raise ValueError("the two subsets must partition all CIFAR-10 train images")
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(pool), generator=generator)[
        : 2 * args.samples_per_side
    ]
    left_indices = indices[: args.samples_per_side]
    right_indices = indices[args.samples_per_side :]
    left = export_images(pool[left_indices], args.left_dir)
    right = export_images(pool[right_indices], args.right_dir)
    metrics = compute_real_real(
        args.left_dir,
        args.right_dir,
        left_indices=left_indices,
        right_indices=right_indices,
        kid_reference_out=args.kid_reference_features_out,
        device=device,
        batch=args.metric_batch,
        workers=args.metric_workers,
        kid_seed=args.kid_seed,
    )
    metrics.update(
        {
            "metric_batch": args.metric_batch,
            "metric_workers": args.metric_workers,
        }
    )
    result = {
        "status": "cap-emf2-real-real-calibration",
        "decision": "COMPLETE",
        "calibration_kind": "single_real_real_reference_point",
        "seed": args.seed,
        "samples_per_side": args.samples_per_side,
        "left_indices_sha256": hashlib.sha256(
            left_indices.numpy().tobytes()
        ).hexdigest(),
        "right_indices_sha256": hashlib.sha256(
            right_indices.numpy().tobytes()
        ).hexdigest(),
        "left": left,
        "right": right,
        "metrics": metrics,
        "limits": [
            "This is one observed real/real reference point, not an estimate of metric variance or a confidence interval.",
            "The published train reference contains the same population as these subsets; matched-reference scores are sanity checks, not independent holdout scores.",
            "Only CIFAR-10 train is used; the local test split remains closed.",
        ],
        "provenance": {
            **evaluation_provenance(device),
            "numerical_settings": numerical_settings,
            "deterministic_algorithms": True,
        },
        "source_sha256": source_manifest(),
    }
    digest = write_json_atomic(args.out, result)
    print(json.dumps(result, indent=2))
    print(f"wrote {args.out} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
