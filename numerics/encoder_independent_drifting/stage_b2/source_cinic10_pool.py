"""Build the frozen B2 external pool from CINIC-10's ImageNet contribution.

The official CINIC-10 archive combines all 60,000 CIFAR-10 images with
210,000 downsampled ImageNet images.  The two sources are recoverable from the
filenames.  This utility accepts only the ImageNet naming convention, samples
the ten CIFAR-compatible classes evenly, rejects decoded-pixel duplicates, and
checks every retained image against the complete local CIFAR-10 corpus.

The output is deliberately a generated ``.npz`` artifact (ignored by git).
Its adjacent JSON provenance record binds the archive and output hashes,
selection seed, source paths, class counts, and overlap checks.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import random
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

ARCHIVE_URL = (
    "https://datashare.ed.ac.uk/bitstreams/"
    "e8e186cc-2688-48f1-aa5a-3fda1a43f8b6/download"
)
DATASET_DOI = "https://doi.org/10.7488/ds/2448"
LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
OFFICIAL_ARCHIVE_MD5 = "6ee4d0c996905fe93221de577967a372"
SELECTION_SEED = 2_026_07_31
SAMPLES_PER_CLASS = 600
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
_IMAGENET_NAME = re.compile(r"^n[0-9]+_[0-9]+\.png$", re.IGNORECASE)
_CIFAR_PREFIXES = ("cifar10-", "cifar-10-")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_md5(path: Path) -> str:
    """Return the repository ETag digest; MD5 is used only for provenance."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(image: np.ndarray) -> str:
    if image.shape != (32, 32, 3) or image.dtype != np.uint8:
        raise ValueError("pixel hashes require uint8 32x32 RGB images")
    return hashlib.sha256(image.tobytes(order="C")).hexdigest()


def _load_cifar_batch(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    flat = np.asarray(payload[b"data"], dtype=np.uint8)
    return flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def complete_cifar_pixel_hashes(root: Path) -> set[str]:
    """Hash the official 50k train and 10k test images without torchvision."""
    batch_dir = root / "cifar-10-batches-py"
    paths = [batch_dir / f"data_batch_{index}" for index in range(1, 6)]
    paths.append(batch_dir / "test_batch")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "the complete local CIFAR-10 corpus is required for overlap checks; "
            f"missing {missing}"
        )
    hashes: set[str] = set()
    count = 0
    for path in paths:
        batch = _load_cifar_batch(path)
        count += len(batch)
        hashes.update(pixel_sha256(image) for image in batch)
    if count != 60_000 or len(hashes) != 60_000:
        raise RuntimeError(
            "expected 60,000 distinct official CIFAR-10 decoded images, got "
            f"{count} rows and {len(hashes)} hashes"
        )
    return hashes


def classify_members(
    archive: tarfile.TarFile,
    minimum_per_class: int,
) -> tuple[dict[str, list[tarfile.TarInfo]], Counter[str]]:
    candidates: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
    exclusions: Counter[str] = Counter()
    for member in archive.getmembers():
        if not member.isfile() or not member.name.lower().endswith(".png"):
            continue
        parts = PurePosixPath(member.name).parts
        if len(parts) < 3:
            exclusions["unexpected_path"] += 1
            continue
        class_name = parts[-2].lower()
        filename = parts[-1]
        if class_name not in CLASSES:
            exclusions["unexpected_class"] += 1
            continue
        if filename.lower().startswith(_CIFAR_PREFIXES):
            exclusions["cifar_filename"] += 1
            continue
        if not _IMAGENET_NAME.fullmatch(filename):
            exclusions["unexpected_non_cifar_filename"] += 1
            continue
        candidates[class_name].append(member)
    missing = [name for name in CLASSES if len(candidates[name]) < minimum_per_class]
    if missing:
        raise RuntimeError(f"too few verified ImageNet candidates for {missing}")
    return dict(candidates), exclusions


def decode_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
    handle = archive.extractfile(member)
    if handle is None:
        raise RuntimeError(f"could not read archive member {member.name}")
    with Image.open(io.BytesIO(handle.read())) as image:
        rgb = image.convert("RGB")
        if rgb.size != (32, 32):
            raise ValueError(f"unexpected image size {rgb.size} for {member.name}")
        return np.asarray(rgb, dtype=np.uint8).copy()


def build_pool(
    archive_path: Path,
    output_path: Path,
    provenance_path: Path,
    cifar_root: Path,
    seed: int,
    samples_per_class: int,
) -> dict:
    if samples_per_class <= 0:
        raise ValueError("samples per class must be positive")
    archive_md5 = file_md5(archive_path)
    if archive_md5 != OFFICIAL_ARCHIVE_MD5:
        raise RuntimeError(
            "CINIC-10 archive does not match the official repository ETag: "
            f"expected {OFFICIAL_ARCHIVE_MD5}, got {archive_md5}"
        )
    archive_sha256 = file_sha256(archive_path)
    cifar_hashes = complete_cifar_pixel_hashes(cifar_root)
    selected_images: list[np.ndarray] = []
    selected_labels: list[int] = []
    selected_paths: list[str] = []
    retained_hashes: set[str] = set()
    rejected = Counter()
    candidate_counts: dict[str, int] = {}

    # A gzip tarball supports forward seeking efficiently but a randomized
    # extraction order repeatedly decompresses from the beginning.  Choose a
    # deterministic oversupply first, decode it once in archive-offset order,
    # and only then apply the shuffled ranks.  Two hundred spare candidates per
    # class is far more than needed for the duplicate checks in this dataset.
    oversupply = 200
    with tarfile.open(archive_path, mode="r:gz") as archive:
        candidates, filename_exclusions = classify_members(
            archive, samples_per_class + oversupply
        )
        chosen: list[tuple[int, str, int, tarfile.TarInfo]] = []
        for label, class_name in enumerate(CLASSES):
            class_candidates = sorted(
                candidates[class_name], key=lambda item: item.name
            )
            candidate_counts[class_name] = len(class_candidates)
            random.Random(f"{seed}:{class_name}").shuffle(class_candidates)
            for rank, member in enumerate(
                class_candidates[: samples_per_class + oversupply]
            ):
                chosen.append((label, class_name, rank, member))

        decoded: dict[str, tuple[np.ndarray, str]] = {}
        for _, _, _, member in sorted(chosen, key=lambda item: item[3].offset_data):
            image = decode_member(archive, member)
            decoded[member.name] = (image, pixel_sha256(image))

        for label, class_name in enumerate(CLASSES):
            ranked = sorted(
                (item for item in chosen if item[0] == label), key=lambda item: item[2]
            )
            kept = 0
            for _, _, _, member in ranked:
                image, image_hash = decoded[member.name]
                if image_hash in cifar_hashes:
                    rejected["decoded_pixel_match_to_cifar10"] += 1
                    continue
                if image_hash in retained_hashes:
                    rejected["decoded_pixel_duplicate_in_pool"] += 1
                    continue
                retained_hashes.add(image_hash)
                selected_images.append(image)
                selected_labels.append(label)
                selected_paths.append(member.name)
                kept += 1
                if kept == samples_per_class:
                    break
            if kept != samples_per_class:
                raise RuntimeError(
                    f"only {kept} unique non-CIFAR images survived for {class_name}; "
                    "increase the deterministic oversupply"
                )

    images = np.stack(selected_images).astype(np.uint8, copy=False)
    labels = np.asarray(selected_labels, dtype=np.uint8)
    source_paths = np.asarray(selected_paths)
    if images.shape != (len(CLASSES) * samples_per_class, 32, 32, 3):
        raise AssertionError(f"unexpected derived pool shape {images.shape}")
    if len(retained_hashes) != len(images):
        raise AssertionError("derived external pool contains duplicate decoded images")
    if Counter(labels.tolist()) != Counter(
        {label: samples_per_class for label in range(len(CLASSES))}
    ):
        raise AssertionError("derived external pool is not class balanced")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        images=images,
        labels=labels,
        class_names=np.asarray(CLASSES),
        source_paths=source_paths,
    )
    record = {
        "schema": "drifting-identifiability-b2-external-pool-v1",
        "source_id": (
            "cinic10-imagenet-only-balanced-600-per-class-seed-20260731"
            if seed == SELECTION_SEED and samples_per_class == SAMPLES_PER_CLASS
            else f"cinic10-imagenet-only-balanced-{samples_per_class}-seed-{seed}"
        ),
        "dataset": "CINIC-10 Is Not ImageNet or CIFAR-10",
        "dataset_doi": DATASET_DOI,
        "archive_url": ARCHIVE_URL,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "archive_path": str(archive_path.resolve()),
        "archive_md5_matching_official_etag": archive_md5,
        "archive_sha256": archive_sha256,
        "output_path": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
        "selection_seed": seed,
        "samples_per_class": samples_per_class,
        "samples": len(images),
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "classes": list(CLASSES),
        "candidate_counts": candidate_counts,
        "filename_exclusions": dict(filename_exclusions),
        "rejected_after_decode": dict(rejected),
        "selection_guarantees": {
            "only_imagenet_style_filenames": True,
            "all_cifar_style_filenames_excluded": True,
            "decoded_pixel_overlap_with_complete_cifar10": 0,
            "decoded_pixel_duplicates_within_pool": 0,
            "class_balanced": True,
            "all_source_paths_recorded_inside_npz": True,
        },
        "scientific_scope": (
            "Fresh class-aligned 32x32 ImageNet-derived distribution-shift pool. "
            "It is not an in-domain replacement test set for CIFAR-10."
        ),
        "reuse_attestation": (
            "Repository search before sourcing found no prior CINIC-10 use; this "
            "pool was built after B2 design/preflight and before any B2 baseline "
            "or candidate training."
        ),
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument(
        "--cifar-root",
        type=Path,
        default=Path.home() / ".cache" / "cifar",
    )
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
    args = parser.parse_args()
    provenance = args.provenance or args.output.with_suffix(".provenance.json")
    record = build_pool(
        args.archive,
        args.output,
        provenance,
        args.cifar_root,
        args.seed,
        args.samples_per_class,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
