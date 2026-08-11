"""Standard, train-reference-only CleanFID/CleanKID evaluation.

Generated PNGs are retained, decoded, shape-checked, and hashed so every
reported number is reconstructible.  The standardized metrics use CleanFID's
published CIFAR-10 *training* reference; this module never opens the local
CIFAR-10 test split.

The external-folder path is intentionally strict.  A positive control must
declare its sample count and a human-readable source citation, and every file
must be a decodable 32x32 RGB PNG.  An anonymous folder is not evidence.

Every evaluation also runs one shared, fixed-subset repository diagnostic:
ImageNet1K-V1 Inception precision/recall and unbiased KID on 2,048 generated
and train images, plus a 256-sample literal-pixel memorization check against
all 50,000 training images and their horizontal flips.  CAP1 baselines and
CAP2 candidates call this exact function with the same seeds; these auxiliary
numbers are matched-arm diagnostics, not published benchmark values.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..appearance import precision_recall
from ..device import configure, resolve_device
from ..fid import inception_features, kid_from_features
from ..stage_b2.metrics import nearest_reference_distances
from ..stage_cap.config import CAPModelConfig
from ..stage_cap.data import cifar10_train_pool
from ..stage_cap.model import CAPPixelTransformer, one_step_sample
from .artifacts import (
    assert_unused,
    file_sha256,
    source_manifest,
    verify_file,
    verify_json,
    write_json_atomic,
    write_npz_atomic,
)

CIFAR_IMAGE_SIZE = (32, 32)
DEFAULT_SAMPLE_COUNT = 50_000
DEFAULT_GENERATION_SEED = 20_260_804
DEFAULT_KID_SEED = 20_260_831
DEFAULT_METRIC_WORKERS = 0 if sys.platform.startswith("win") else 4
KID_SUBSETS = 100
KID_MAX_SUBSET_SIZE = 1_000
REPOSITORY_AUXILIARY_PROTOCOL = "train-only-repository-auxiliary-v1"
REPOSITORY_REFERENCE_SEED = 20_260_832
REPOSITORY_FEATURE_SAMPLES = 2_048
REPOSITORY_MEMORIZATION_SAMPLES = 256
REPOSITORY_MEMORIZATION_SCALE_SAMPLES = 256


def tensor_content_sha256(value: torch.Tensor, *, chunk: int = 1_024) -> str:
    """Hash tensor values, dtype, and shape without one giant byte copy."""
    if chunk <= 0:
        raise ValueError("tensor hash chunk must be positive")
    tensor = value.detach().cpu()
    digest = hashlib.sha256(f"{tensor.dtype}:{tuple(tensor.shape)}:".encode())
    for start in range(0, len(tensor), chunk):
        block = tensor[start : start + chunk].contiguous()
        digest.update(block.numpy().tobytes(order="C"))
    return digest.hexdigest()


def feature_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{array.dtype}:{array.shape}:".encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_standard_protocol(
    *,
    count: int,
    kid_seed: int,
    generation_seed: int,
    checkpoint_mode: bool,
) -> None:
    """Reject protocol drift before any model/data/metric work begins."""
    if count != DEFAULT_SAMPLE_COUNT:
        raise ValueError(
            f"standard evaluation is fixed at {DEFAULT_SAMPLE_COUNT} samples"
        )
    if kid_seed != DEFAULT_KID_SEED:
        raise ValueError(f"standard CleanKID seed is fixed at {DEFAULT_KID_SEED}")
    if checkpoint_mode and generation_seed != DEFAULT_GENERATION_SEED:
        raise ValueError(
            f"standard checkpoint generation seed is fixed at {DEFAULT_GENERATION_SEED}"
        )


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def evaluation_provenance(device: torch.device) -> dict[str, object]:
    """Numerical/package provenance needed to reproduce an evaluation."""
    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "clean-fid": _package_version("clean-fid"),
            "numpy": _package_version("numpy"),
            "Pillow": _package_version("Pillow"),
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
        },
        "device": str(device),
        "gpu_name": gpu_name,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "command": sys.argv,
        "image_quantization": "clip [-1,1], round 255*(x+1)/2 to uint8 RGB PNG",
    }


def _load_model(path: Path, device: torch.device):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(path, map_location="cpu")
    config = CAPModelConfig(**payload["profile"]["model"])
    model = CAPPixelTransformer(config, seed=1).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model, config


def _uint8(images: torch.Tensor) -> np.ndarray:
    if not bool(torch.isfinite(images).all()):
        raise RuntimeError("generator produced a non-finite image")
    return (
        ((images.clamp(-1, 1) + 1) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def validate_png_folder(
    folder: Path,
    *,
    expected_count: int,
    expected_size: tuple[int, int] = CIFAR_IMAGE_SIZE,
    require_sequential_names: bool = False,
) -> dict[str, object]:
    """Decode and hash an exact PNG set; reject silent folder contamination."""
    if expected_count <= 0:
        raise ValueError("expected PNG count must be positive")
    if not folder.is_dir():
        raise RuntimeError(f"PNG folder does not exist: {folder}")
    entries = sorted(folder.iterdir())
    invalid_entries = [
        path.name
        for path in entries
        if not path.is_file() or path.suffix.lower() != ".png"
    ]
    if invalid_entries:
        raise RuntimeError(
            f"PNG folder contains non-PNG entries: {invalid_entries[:5]}"
        )
    files = entries
    if len(files) != expected_count:
        raise RuntimeError(
            f"expected exactly {expected_count} PNGs in {folder}, found {len(files)}"
        )
    if require_sequential_names:
        expected_names = [f"{index:06d}.png" for index in range(expected_count)]
        actual_names = [path.name for path in files]
        if actual_names != expected_names:
            raise RuntimeError(
                "generated PNG names are not the declared sequential set"
            )

    digest = hashlib.sha256()
    for path in files:
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            image_format = image.format
            image_mode = image.mode
            image_size = image.size
            image.verify()
        if image_format != "PNG" or image_mode != "RGB" or image_size != expected_size:
            raise RuntimeError(
                f"{path} is {image_format}/{image_mode}/{image_size}, expected "
                f"PNG/RGB/{expected_size}"
            )
        digest.update(path.name.encode("utf-8"))
        digest.update(raw)
    return {
        "directory": str(folder.resolve()),
        "count": len(files),
        "image_size": list(expected_size),
        "mode": "RGB",
        "png_manifest_sha256": digest.hexdigest(),
        "validation": "all files decoded; exact count, format, mode, and size checked",
    }


def generate_png_folder(
    model: CAPPixelTransformer,
    config: CAPModelConfig,
    output: Path,
    *,
    count: int,
    batch: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    if count <= 0 or batch <= 0:
        raise ValueError("generation count and batch must be positive")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to mix generated samples in {output}")
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    calls = {"count": 0}

    def count_forward(_module, _arguments) -> None:
        calls["count"] += 1

    handle = model.register_forward_pre_hook(count_forward)
    try:
        with torch.no_grad():
            for start in range(0, count, batch):
                size = min(batch, count - start)
                noise = torch.randn(
                    (size, config.channels, config.image_size, config.image_size),
                    generator=generator,
                ).to(device)
                images = _uint8(one_step_sample(model, noise))
                for offset, array in enumerate(images):
                    Image.fromarray(array).save(
                        output / f"{start + offset:06d}.png",
                        format="PNG",
                        compress_level=0,
                    )
    finally:
        handle.remove()
    expected_calls = (count + batch - 1) // batch
    if calls["count"] != expected_calls:
        raise RuntimeError(
            f"one-step generation made {calls['count']} model calls; "
            f"expected {expected_calls}"
        )
    manifest = validate_png_folder(
        output,
        expected_count=count,
        expected_size=(config.image_size, config.image_size),
        require_sequential_names=True,
    )
    return {
        **manifest,
        "batch": batch,
        "seed": seed,
        "model_calls": calls["count"],
        "expected_model_calls": expected_calls,
        "one_model_call_per_batch": True,
    }


def existing_png_folder(
    folder: Path,
    *,
    expected_count: int,
    source_citation: str,
) -> dict[str, object]:
    if not source_citation.strip():
        raise ValueError("an external positive control requires a source citation")
    return {
        **validate_png_folder(folder, expected_count=expected_count),
        "source": "pre-existing positive-control folder",
        "source_citation": source_citation.strip(),
    }


def bind_positive_control_provenance(
    samples: dict[str, object], provenance_path: Path
) -> dict[str, object]:
    """Bind an external folder to its immutable source-generation record."""

    provenance = verify_json(
        provenance_path, "cap-emf2-stylegan2ada-positive-control-source"
    )
    images = provenance.get("images", {})
    if provenance.get("decision") != "COMPLETE":
        raise RuntimeError("positive-control source generation is incomplete")
    if provenance.get("source_sha256") != source_manifest():
        raise RuntimeError("positive-control source was generated from stale CAP2 code")
    if (
        Path(str(images.get("directory", ""))).resolve()
        != Path(str(samples.get("directory", ""))).resolve()
    ):
        raise RuntimeError(
            "positive-control source record names a different PNG folder"
        )
    if images.get("count") != samples.get("count") or images.get(
        "png_manifest_sha256"
    ) != samples.get("png_manifest_sha256"):
        raise RuntimeError("positive-control PNGs differ from their source record")
    citation = provenance.get("citation")
    if not isinstance(citation, str) or citation.strip() != samples.get(
        "source_citation"
    ):
        raise RuntimeError("positive-control citation differs from its source record")
    return {
        **samples,
        "source_provenance": str(provenance_path.resolve()),
        "source_provenance_sha256": provenance["artifact_sha256"],
    }


def load_png_tensor(folder: Path, count: int) -> torch.Tensor:
    """Load the lexicographically first fixed-PNG subset into ``[-1, 1]``."""
    if count <= 0:
        raise ValueError("PNG tensor count must be positive")
    files = sorted(folder.glob("*.png"))
    if len(files) < count:
        raise RuntimeError(f"fixed PNG subset needs {count} files, found {len(files)}")
    arrays = []
    for path in files[:count]:
        with Image.open(path) as image:
            arrays.append(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())
    values = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float()
    return values / 127.5 - 1.0


def fixed_reference_indices(pool_size: int, count: int) -> torch.Tensor:
    if count <= 0 or count > pool_size:
        raise ValueError("invalid fixed-reference subset size")
    generator = torch.Generator().manual_seed(REPOSITORY_REFERENCE_SEED)
    return torch.randperm(pool_size, generator=generator)[:count]


def repository_feature_metrics(
    generated: torch.Tensor,
    reference: torch.Tensor,
    *,
    device: torch.device,
    feature_batch: int,
) -> dict[str, object]:
    """Fixed same-backend P/R/KID used for both CAP1 and CAP2."""
    if feature_batch <= 0:
        raise ValueError("repository feature batch must be positive")
    generated_features = (
        inception_features(generated, device=device, batch=feature_batch)
        .double()
        .numpy()
    )
    reference_features = (
        inception_features(reference, device=device, batch=feature_batch)
        .double()
        .numpy()
    )
    manifold = precision_recall(generated_features, reference_features)
    return {
        "backend": "torchvision Inception-v3 ImageNet1K_V1 pool features",
        "scope": (
            "same-backend train-only arm comparison; not numerically comparable "
            "with CleanFID or published precision/recall"
        ),
        "samples_generated": len(generated_features),
        "samples_reference": len(reference_features),
        "precision": manifold["precision"],
        "recall": manifold["recall"],
        "pr_f1": manifold["f1"],
        "neighbours": manifold["k"],
        "unbiased_kid": kid_from_features(generated_features, reference_features),
        "feature_batch": feature_batch,
        "generated_feature_sha256": feature_array_sha256(generated_features),
        "reference_feature_sha256": feature_array_sha256(reference_features),
    }


def _typical_pixel_distance(
    train: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    indices = fixed_reference_indices(
        len(train), 2 * REPOSITORY_MEMORIZATION_SCALE_SAMPLES
    )
    left = train[indices[:REPOSITORY_MEMORIZATION_SCALE_SAMPLES]].to(device)
    right = train[indices[REPOSITORY_MEMORIZATION_SCALE_SAMPLES:]].to(device)
    return float(
        torch.cdist(left.flatten(1).float(), right.flatten(1).float()).median().cpu()
    )


def pixel_memorization_metrics(
    generated: torch.Tensor,
    train: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Small literal-pixel copying audit against train images and their flips."""
    if len(generated) != REPOSITORY_MEMORIZATION_SAMPLES:
        raise ValueError("memorization audit received the wrong fixed subset size")
    direct_distance, direct_claim = nearest_reference_distances(
        generated, train, device
    )
    flipped_distance, flipped_claim = nearest_reference_distances(
        generated, torch.flip(train, dims=(-1,)), device
    )
    use_flip = flipped_distance < direct_distance
    distance = torch.where(use_flip, flipped_distance, direct_distance)
    claim = torch.where(use_flip, flipped_claim + len(train), direct_claim)
    normalizer = _typical_pixel_distance(train, device=device)
    unique_outputs = torch.unique(generated.flatten(1), dim=0).shape[0]
    return {
        "space": "quantized RGB pixel L2, including horizontal-flip augmentation",
        "scope": (
            "fixed small audit for literal/near pixel copying; it does not rule "
            "out semantic memorization"
        ),
        "samples_generated": len(generated),
        "reference_train_images": len(train),
        "typical_real_pair_distance": normalizer,
        "nearest_train_or_flip_median": float(distance.median()),
        "nearest_train_or_flip_p05": float(torch.quantile(distance, 0.05)),
        "nearest_train_or_flip_min": float(distance.min()),
        "nearest_median_over_real_pair": float(distance.median()) / normalizer,
        "nearest_p05_over_real_pair": float(torch.quantile(distance, 0.05))
        / normalizer,
        "exact_pixel_copy_fraction": float((distance <= 1e-6).float().mean()),
        "flip_claim_fraction": float(use_flip.float().mean()),
        "distinct_train_or_flip_claim_fraction": float(torch.unique(claim).numel())
        / len(generated),
        "exact_generated_duplicate_fraction": 1.0
        - float(unique_outputs) / len(generated),
    }


def compute_repository_auxiliary_metrics(
    folder: Path,
    *,
    sample_manifest: dict[str, object],
    device: torch.device,
    data_root: str | None,
    feature_batch: int,
) -> dict[str, object]:
    """Run the identical fixed-PNG, train-only diagnostics for any model arm.

    ``sample_manifest`` must be returned by this module's strict PNG validator
    or generator in the same process.  Requiring it binds these smaller
    diagnostics to the exact 50k folder scored by CleanFID/CleanKID without a
    second decode of all 50,000 files.
    """
    if Path(str(sample_manifest.get("directory", ""))).resolve() != folder.resolve():
        raise RuntimeError("auxiliary metric folder differs from its PNG manifest")
    if int(sample_manifest.get("count", -1)) != DEFAULT_SAMPLE_COUNT:
        raise RuntimeError("auxiliary metrics require the fixed 50k PNG manifest")
    manifest_hash = sample_manifest.get("png_manifest_sha256")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise RuntimeError("auxiliary metrics require a hashed PNG manifest")

    train = cifar10_train_pool(data_root)
    if len(train) != DEFAULT_SAMPLE_COUNT:
        raise RuntimeError(
            f"expected the 50k CIFAR-10 training pool, found {len(train)} rows"
        )
    generated_subset = load_png_tensor(folder, REPOSITORY_FEATURE_SAMPLES)
    reference_indices = fixed_reference_indices(len(train), REPOSITORY_FEATURE_SAMPLES)
    reference_subset = train[reference_indices]
    feature_metrics = repository_feature_metrics(
        generated_subset,
        reference_subset,
        device=device,
        feature_batch=feature_batch,
    )
    memorization = pixel_memorization_metrics(
        generated_subset[:REPOSITORY_MEMORIZATION_SAMPLES],
        train,
        device=device,
    )
    return {
        "protocol": REPOSITORY_AUXILIARY_PROTOCOL,
        "png_manifest_sha256": manifest_hash,
        "generated_subset": {
            "selection": (
                "lexicographically first PNGs from the exact hashed 50k folder; "
                "no curation"
            ),
            "count": REPOSITORY_FEATURE_SAMPLES,
            "tensor_sha256": tensor_content_sha256(generated_subset),
        },
        "reference_subset": {
            "split": "cifar10/train",
            "seed": REPOSITORY_REFERENCE_SEED,
            "count": REPOSITORY_FEATURE_SAMPLES,
            "indices_sha256": hashlib.sha256(
                reference_indices.numpy().tobytes()
            ).hexdigest(),
            "tensor_sha256": tensor_content_sha256(reference_subset),
        },
        "cifar10_train_tensor_sha256": tensor_content_sha256(train),
        "repository_feature_metrics": feature_metrics,
        "memorization": memorization,
        "method_sources": {
            "precision_recall": (
                "Kynkaanniemi et al. (2019), Improved Precision and Recall "
                "Metric for Assessing Generative Models, arXiv:1904.06991"
            ),
            "kid": (
                "Binkowski et al. (2018), Demystifying MMD GANs, "
                "ICLR 2018 / arXiv:1801.01401"
            ),
            "feature_extractor": (
                "torchvision Inception_V3_Weights.IMAGENET1K_V1; bilinear "
                "299x299 resize and the weights' ImageNet normalization"
            ),
        },
        "limits": (
            "repository-feature metrics are matched-arm diagnostics, not a "
            "published benchmark; the pixel audit cannot exclude semantic memorization"
        ),
    }


def _require_cleanfid():
    try:
        from cleanfid import fid
    except ImportError as error:
        raise RuntimeError(
            "CleanFID is required: install the pinned clean-fid dependency "
            "before evaluation"
        ) from error
    return fid


def _deterministic_kernel_distance(
    fid_module,
    reference_features: np.ndarray,
    candidate_features: np.ndarray,
    *,
    seed: int,
) -> float:
    """Run CleanFID's KID estimator under a recorded NumPy random seed."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return float(
            fid_module.kernel_distance(
                reference_features,
                candidate_features,
                num_subsets=KID_SUBSETS,
                max_subset_size=KID_MAX_SUBSET_SIZE,
            )
        )
    finally:
        np.random.set_state(state)


def _moments(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return features.mean(axis=0), np.cov(features, rowvar=False)


def clean_metrics_from_features(
    fid_module,
    features: np.ndarray,
    *,
    kid_reference_features: np.ndarray,
    kid_seed: int = DEFAULT_KID_SEED,
) -> dict[str, float]:
    """CleanFID plus KID against one sealed full CIFAR-10 train population.

    CleanFID publishes train-set FID moments, but version 0.1.35 does not
    publish a CIFAR-10 train KID feature archive.  The caller therefore passes
    the exact 50,000 locally extracted clean-Inception reference features.
    """
    reference_mean, reference_covariance = fid_module.get_reference_statistics(
        "cifar10", 32, mode="clean", split="train", metric="FID"
    )
    mean, covariance = _moments(features)
    return {
        "clean_fid_cifar10_train": float(
            fid_module.frechet_distance(
                mean,
                covariance,
                reference_mean,
                reference_covariance,
            )
        ),
        "clean_kid_cifar10_train": _deterministic_kernel_distance(
            fid_module,
            kid_reference_features,
            features,
            seed=kid_seed,
        ),
    }


def clean_pair_metrics_from_features(
    fid_module,
    left_features: np.ndarray,
    right_features: np.ndarray,
    *,
    kid_seed: int = DEFAULT_KID_SEED,
) -> dict[str, float]:
    """Clean-preprocessed FID/KID between two explicitly supplied sets."""
    left_mean, left_covariance = _moments(left_features)
    right_mean, right_covariance = _moments(right_features)
    return {
        "clean_fid": float(
            fid_module.frechet_distance(
                left_mean,
                left_covariance,
                right_mean,
                right_covariance,
            )
        ),
        "clean_kid": _deterministic_kernel_distance(
            fid_module,
            left_features,
            right_features,
            seed=kid_seed,
        ),
    }


def extract_clean_features(
    folder: Path,
    *,
    device: torch.device,
    batch: int,
    workers: int,
    verbose: bool,
    expected_count: int = DEFAULT_SAMPLE_COUNT,
) -> tuple[object, np.ndarray]:
    if batch <= 0 or workers < 0:
        raise ValueError("metric batch must be positive and workers nonnegative")
    fid_module = _require_cleanfid()
    extractor = fid_module.build_feature_extractor(
        "clean", device, use_dataparallel=False
    )
    # CleanFID 0.1.35 lists both lowercase and uppercase extensions. On a
    # case-insensitive filesystem its recursive glob therefore returns every
    # PNG twice. Feed the exact, unique, already validated population instead.
    files = unique_png_files(folder, expected_count=expected_count)
    features = fid_module.get_files_features(
        files,
        extractor,
        num_workers=workers,
        batch_size=batch,
        device=device,
        mode="clean",
        description=f"Clean metrics {folder.name}: ",
        verbose=verbose,
    )
    features = np.asarray(features)
    if features.ndim != 2 or len(features) != expected_count:
        raise RuntimeError(
            "CleanFID feature extraction returned shape "
            f"{features.shape}, expected ({expected_count}, feature_dimension)"
        )
    if (
        features.dtype not in (np.float32, np.float64)
        or not np.isfinite(features).all()
    ):
        raise RuntimeError("CleanFID returned non-finite or non-floating features")
    return fid_module, features


def unique_png_files(folder: Path, *, expected_count: int) -> list[str]:
    """Enumerate an exact PNG population once on every filesystem."""

    files = [
        str(path)
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() == ".png"
    ]
    if len(files) != expected_count:
        raise RuntimeError(
            f"Clean metrics require exactly {expected_count} unique PNGs, "
            f"found {len(files)}"
        )
    return files


def load_clean_feature_archive(
    path: Path,
    *,
    expected_count: int = DEFAULT_SAMPLE_COUNT,
    expected_dimension: int = 2_048,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load and content-hash one immutable clean-Inception population."""
    digest = verify_file(path)
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["features"]:
            raise RuntimeError("feature archive must contain only 'features'")
        features = np.asarray(archive["features"])
    if features.shape != (expected_count, expected_dimension):
        raise RuntimeError(
            "feature archive must have shape "
            f"({expected_count}, {expected_dimension}), found {features.shape}"
        )
    if (
        features.dtype not in (np.float32, np.float64)
        or not np.isfinite(features).all()
    ):
        raise RuntimeError("feature archive must be finite float32/float64")
    return features, {
        "path": str(path.resolve()),
        "sha256": digest,
        "feature_sha256": feature_array_sha256(features),
        "count": len(features),
        "dimension": features.shape[1],
        "dtype": str(features.dtype),
    }


def load_kid_reference_features(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Load the exact full-train CleanFID feature population used for KID."""

    features, metadata = load_clean_feature_archive(path)
    return features, {
        **metadata,
        "population": "all 50,000 CIFAR-10 train images in dataset-index order",
        "preprocessing": "clean-fid 0.1.35 clean Inception preprocessing",
    }


def revalidate_clean_evaluation_evidence(
    evaluation: dict,
    *,
    anchor: Path,
    expected_count: int = DEFAULT_SAMPLE_COUNT,
    expected_dimension: int = 2_048,
) -> dict[str, object]:
    """Revalidate the retained population and recompute its standard metrics.

    The evaluation JSON is not treated as evidence by itself.  This routine
    resolves its portable references, decodes and re-hashes every generated
    PNG, strict-loads both full clean-Inception archives (including their SHA
    sidecars), and recomputes FID/KID from the retained feature arrays.  It does
    not re-extract clean-Inception features from the PNGs; the immutable JSON
    instead binds the generated archive to the exact PNG manifest and fixed
    generation seed produced in the same source-bound evaluation call.
    """

    if not isinstance(evaluation, dict):
        raise TypeError("clean evaluation evidence must be a dictionary")

    def resolve(reference: object, *, label: str) -> Path:
        if not isinstance(reference, str) or not reference.strip():
            raise RuntimeError(f"clean evaluation has no {label} reference")
        path = Path(reference)
        return path.resolve() if path.is_absolute() else (anchor / path).resolve()

    samples = evaluation.get("samples", {})
    metrics = evaluation.get("standard_train_reference_metrics") or evaluation.get(
        "metrics", {}
    )
    generated_record = metrics.get("generated_features", {})
    kid_record = metrics.get("kid_reference", {})
    if not all(
        isinstance(record, dict)
        for record in (samples, metrics, generated_record, kid_record)
    ):
        raise TypeError("clean evaluation evidence has malformed records")

    sample_path = resolve(samples.get("directory"), label="PNG population")
    generated_path = resolve(
        generated_record.get("path"), label="generated feature archive"
    )
    kid_path = resolve(kid_record.get("path"), label="KID reference archive")
    png_actual = validate_png_folder(
        sample_path,
        expected_count=expected_count,
        expected_size=CIFAR_IMAGE_SIZE,
        require_sequential_names=True,
    )
    generated_features, generated_actual = load_clean_feature_archive(
        generated_path,
        expected_count=expected_count,
        expected_dimension=expected_dimension,
    )
    kid_features, kid_actual = load_clean_feature_archive(
        kid_path,
        expected_count=expected_count,
        expected_dimension=expected_dimension,
    )

    kid_seed = metrics.get("kid_seed")
    if isinstance(kid_seed, bool) or not isinstance(kid_seed, int):
        raise TypeError("clean evaluation KID seed must be an integer")
    recomputed = clean_metrics_from_features(
        _require_cleanfid(),
        generated_features,
        kid_reference_features=kid_features,
        kid_seed=kid_seed,
    )
    bound_fields = ("sha256", "feature_sha256", "count", "dimension", "dtype")

    def metric_matches(key: str) -> bool:
        recorded = metrics.get(key)
        actual = recomputed.get(key)
        return (
            isinstance(recorded, (int, float))
            and not isinstance(recorded, bool)
            and isinstance(actual, (int, float))
            and math.isfinite(float(recorded))
            and math.isfinite(float(actual))
            and math.isclose(
                float(recorded), float(actual), rel_tol=1e-12, abs_tol=1e-12
            )
        )

    checks = {
        "exact_generated_png_population": all(
            samples.get(key) == png_actual.get(key)
            for key in ("count", "image_size", "mode", "png_manifest_sha256")
        ),
        "generated_feature_archive": all(
            generated_record.get(key) == generated_actual.get(key)
            for key in bound_fields
        )
        and generated_record.get("population")
        == "all fixed-seed generated images in sequential PNG order"
        and generated_record.get("preprocessing")
        == "clean-fid 0.1.35 clean Inception preprocessing"
        and generated_record.get("source_png_manifest_sha256")
        == png_actual.get("png_manifest_sha256")
        and generated_record.get("generation_seed") == samples.get("seed"),
        "kid_reference_feature_archive": all(
            kid_record.get(key) == kid_actual.get(key) for key in bound_fields
        )
        and kid_record.get("population")
        == "all 50,000 CIFAR-10 train images in dataset-index order"
        and kid_record.get("preprocessing")
        == "clean-fid 0.1.35 clean Inception preprocessing",
        "clean_fid_recomputed": metric_matches("clean_fid_cifar10_train"),
        "clean_kid_recomputed": metric_matches("clean_kid_cifar10_train"),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "recomputed": recomputed,
        "generated_archive": generated_actual,
        "kid_reference_archive": kid_actual,
        "png_population": png_actual,
        "limit": (
            "feature archive is source-bound to the PNG manifest; this pass "
            "does not rerun clean-Inception feature extraction from PNG bytes"
        ),
    }


def compute_clean_metrics(
    folder: Path,
    *,
    kid_reference_path: Path,
    generated_feature_path: Path | None = None,
    device: torch.device,
    batch: int = 128,
    workers: int = DEFAULT_METRIC_WORKERS,
    kid_seed: int = DEFAULT_KID_SEED,
    include_legacy_fid: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    fid_module, features = extract_clean_features(
        folder,
        device=device,
        batch=batch,
        workers=workers,
        verbose=verbose,
    )
    kid_reference_features, kid_reference = load_kid_reference_features(
        kid_reference_path
    )
    metrics: dict[str, object] = clean_metrics_from_features(
        fid_module,
        features,
        kid_reference_features=kid_reference_features,
        kid_seed=kid_seed,
    )
    if generated_feature_path is not None:
        archive_sha = write_npz_atomic(generated_feature_path, features=features)
        metrics["generated_features"] = {
            "path": str(generated_feature_path.resolve()),
            "sha256": archive_sha,
            "feature_sha256": feature_array_sha256(features),
            "count": len(features),
            "dimension": int(features.shape[1]),
            "dtype": str(features.dtype),
            "population": ("all fixed-seed generated images in sequential PNG order"),
            "preprocessing": "clean-fid 0.1.35 clean Inception preprocessing",
        }
    if include_legacy_fid:
        metrics["legacy_tensorflow_fid_cifar10_train"] = float(
            fid_module.compute_fid(
                str(folder),
                dataset_name="cifar10",
                dataset_res=32,
                dataset_split="train",
                mode="legacy_tensorflow",
                device=device,
                batch_size=batch,
                num_workers=workers,
                use_dataparallel=False,
                verbose=verbose,
            )
        )
    metrics.update(
        {
            "backend": "clean-fid",
            "cleanfid_version": _package_version("clean-fid"),
            "reference": "cifar10/train/32",
            "mode": "clean",
            "feature_count": len(features),
            "kid_seed": kid_seed,
            "kid_subsets": KID_SUBSETS,
            "kid_max_subset_size": KID_MAX_SUBSET_SIZE,
            "metric_batch": batch,
            "metric_workers": workers,
            "kid_reference": kid_reference,
        }
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--existing-png-dir", type=Path, default=None)
    parser.add_argument("--external-source-citation", default=None)
    parser.add_argument("--external-source-provenance", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--metric-batch", type=int, default=128)
    parser.add_argument("--metric-workers", type=int, default=DEFAULT_METRIC_WORKERS)
    parser.add_argument("--feature-batch", type=int, default=128)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_GENERATION_SEED)
    parser.add_argument("--kid-seed", type=int, default=DEFAULT_KID_SEED)
    parser.add_argument("--kid-reference-features", type=Path, required=True)
    parser.add_argument(
        "--generated-features",
        type=Path,
        default=None,
        help="retained full clean-Inception feature archive; defaults beside --out",
    )
    parser.add_argument("--include-legacy-fid", action="store_true")
    parser.add_argument("--png-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_unused(args.out)
    generated_feature_path = args.generated_features or args.out.with_name(
        f"{args.out.stem}_clean_features.npz"
    )
    assert_unused(generated_feature_path)
    if (args.checkpoint is None) == (args.existing_png_dir is None):
        raise ValueError("provide exactly one of --checkpoint or --existing-png-dir")
    validate_standard_protocol(
        count=args.count,
        kid_seed=args.kid_seed,
        generation_seed=args.seed,
        checkpoint_mode=args.checkpoint is not None,
    )
    device = resolve_device(args.device)
    numerical_settings = configure(device, allow_tf32=False)
    torch.use_deterministic_algorithms(True)
    if args.checkpoint is not None:
        if args.png_dir is None:
            raise ValueError("checkpoint evaluation requires --png-dir")
        if args.external_source_citation is not None:
            raise ValueError("checkpoint evaluation cannot take an external citation")
        if args.external_source_provenance is not None:
            raise ValueError("checkpoint evaluation cannot take external provenance")
        payload, model, config = _load_model(args.checkpoint, device)
        samples = generate_png_folder(
            model,
            config,
            args.png_dir,
            count=args.count,
            batch=args.batch,
            seed=args.seed,
            device=device,
        )
        checkpoint = str(args.checkpoint.resolve())
        checkpoint_sha256 = file_sha256(args.checkpoint)
        checkpoint_step = payload.get("step")
        checkpoint_kind = payload.get("kind")
        metric_folder = args.png_dir
    else:
        samples = existing_png_folder(
            args.existing_png_dir,
            expected_count=args.count,
            source_citation=args.external_source_citation or "",
        )
        if args.external_source_provenance is None:
            raise ValueError(
                "external positive-control evaluation requires immutable source provenance"
            )
        samples = bind_positive_control_provenance(
            samples, args.external_source_provenance
        )
        checkpoint = None
        checkpoint_sha256 = None
        checkpoint_step = None
        checkpoint_kind = "external-positive-control"
        metric_folder = args.existing_png_dir
    standard = compute_clean_metrics(
        metric_folder,
        kid_reference_path=args.kid_reference_features,
        generated_feature_path=generated_feature_path,
        device=device,
        batch=args.metric_batch,
        workers=args.metric_workers,
        kid_seed=args.kid_seed,
        include_legacy_fid=args.include_legacy_fid,
    )
    standard["generated_features"].update(
        {
            "source_png_manifest_sha256": samples["png_manifest_sha256"],
            "generation_seed": samples.get("seed"),
        }
    )
    auxiliary = compute_repository_auxiliary_metrics(
        metric_folder,
        sample_manifest=samples,
        device=device,
        data_root=args.data_root,
        feature_batch=args.feature_batch,
    )
    result = {
        "status": "cap-emf-standard-evaluation",
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "checkpoint_kind": checkpoint_kind,
        "samples": samples,
        "metrics": standard,
        "repository_auxiliary": auxiliary,
        # Symmetric aliases make CAP1 and CAP2 comparison consumers identical.
        "repository_feature_metrics": auxiliary["repository_feature_metrics"],
        "memorization": auxiliary["memorization"],
        "selection_scope": (
            "CIFAR-10 train-reference development/report metric; local test split "
            "never opened"
        ),
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
