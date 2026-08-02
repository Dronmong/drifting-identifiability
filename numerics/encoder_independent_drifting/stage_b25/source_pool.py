"""Create a B2.5 CINIC pool provably disjoint from consumed B2 bytes.

A different sampling seed is not enough: two samples from the same archive can
overlap.  This builder excludes both source paths and decoded-pixel hashes from
every supplied prior pool and records those exclusions in provenance.
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

from ..stage_b2.source_cinic10_pool import (
    ARCHIVE_URL,
    CLASSES,
    DATASET_DOI,
    LICENSE,
    LICENSE_URL,
    OFFICIAL_ARCHIVE_MD5,
    classify_members,
    complete_cifar_pixel_hashes,
    decode_member,
    file_md5,
    file_sha256,
    pixel_sha256,
)

B25_SELECTION_SEED = 2_026_08_01
B25_SAMPLES_PER_CLASS = 600


def _load_exclusions(paths: list[Path]) -> tuple[set[str], set[str], list[dict]]:
    source_paths: set[str] = set()
    pixel_hashes: set[str] = set()
    records: list[dict] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"B2.5 exclusion pool does not exist: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if "images" not in payload or "source_paths" not in payload:
                raise ValueError(f"exclusion pool lacks images/source_paths: {path}")
            images = np.asarray(payload["images"])
            names = np.asarray(payload["source_paths"]).astype(str)
        if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ValueError(f"unexpected exclusion image shape in {path}")
        if images.dtype != np.uint8 or len(images) != len(names):
            raise ValueError(f"invalid exclusion pool encoding in {path}")
        source_paths.update(names.tolist())
        pixel_hashes.update(pixel_sha256(image) for image in images)
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "samples": len(images),
            }
        )
    return source_paths, pixel_hashes, records


def build_disjoint_pool(
    archive_path: Path,
    output_path: Path,
    provenance_path: Path,
    cifar_root: Path,
    exclusion_pools: list[Path],
    *,
    seed: int = B25_SELECTION_SEED,
    samples_per_class: int = B25_SAMPLES_PER_CLASS,
) -> dict:
    if output_path.exists() or provenance_path.exists():
        raise FileExistsError("refusing to overwrite a B2.5 data artifact")
    if samples_per_class <= 0:
        raise ValueError("samples per class must be positive")
    archive_md5 = file_md5(archive_path)
    if archive_md5 != OFFICIAL_ARCHIVE_MD5:
        raise RuntimeError("CINIC archive does not match the official ETag")

    excluded_paths, excluded_pixels, exclusion_records = _load_exclusions(
        exclusion_pools
    )
    cifar_hashes = complete_cifar_pixel_hashes(cifar_root)
    retained_hashes: set[str] = set()
    selected_images: list[np.ndarray] = []
    selected_labels: list[int] = []
    selected_paths: list[str] = []
    rejected: Counter[str] = Counter()
    candidate_counts: dict[str, int] = {}
    oversupply = 300

    with tarfile.open(archive_path, mode="r:gz") as archive:
        candidates, filename_exclusions = classify_members(
            archive, samples_per_class + oversupply
        )
        chosen: list[tuple[int, str, int, tarfile.TarInfo]] = []
        for label, class_name in enumerate(CLASSES):
            class_candidates = [
                member
                for member in sorted(candidates[class_name], key=lambda item: item.name)
                if member.name not in excluded_paths
            ]
            candidate_counts[class_name] = len(class_candidates)
            random.Random(f"b25:{seed}:{class_name}").shuffle(class_candidates)
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
                if image_hash in excluded_pixels:
                    rejected["decoded_pixel_match_to_exclusion_pool"] += 1
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
                    f"only {kept} disjoint images survived for {class_name}"
                )

    images = np.stack(selected_images).astype(np.uint8, copy=False)
    labels = np.asarray(selected_labels, dtype=np.uint8)
    paths = np.asarray(selected_paths)
    expected = len(CLASSES) * samples_per_class
    if images.shape != (expected, 32, 32, 3):
        raise AssertionError(f"unexpected B2.5 pool shape {images.shape}")
    if set(selected_paths) & excluded_paths:
        raise AssertionError("B2.5 source paths overlap a consumed pool")
    if retained_hashes & excluded_pixels:
        raise AssertionError("B2.5 pixels overlap a consumed pool")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        images=images,
        labels=labels,
        class_names=np.asarray(CLASSES),
        source_paths=paths,
    )
    record = {
        "schema": "drifting-identifiability-b25-external-pool-v1",
        "source_id": (
            f"cinic10-imagenet-only-b25-disjoint-balanced-"
            f"{samples_per_class}-seed-{seed}"
        ),
        "dataset": "CINIC-10 Is Not ImageNet or CIFAR-10",
        "dataset_doi": DATASET_DOI,
        "archive_url": ARCHIVE_URL,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "archive_path": str(archive_path.resolve()),
        "archive_md5_matching_official_etag": archive_md5,
        "archive_sha256": file_sha256(archive_path),
        "output_path": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
        "selection_seed": int(seed),
        "samples_per_class": int(samples_per_class),
        "samples": len(images),
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "classes": list(CLASSES),
        "candidate_counts_after_path_exclusion": candidate_counts,
        "filename_exclusions": dict(filename_exclusions),
        "rejected_after_decode": dict(rejected),
        "excluded_pools": exclusion_records,
        "selection_guarantees": {
            "only_imagenet_style_filenames": True,
            "all_cifar_style_filenames_excluded": True,
            "decoded_pixel_overlap_with_complete_cifar10": 0,
            "source_path_overlap_with_excluded_pools": 0,
            "decoded_pixel_overlap_with_excluded_pools": 0,
            "decoded_pixel_duplicates_within_pool": 0,
            "class_balanced": True,
            "all_source_paths_recorded_inside_npz": True,
        },
        "scientific_scope": (
            "Development-only class-aligned shifted evaluation pool. It is "
            "not an in-domain replacement for the CIFAR-10 target law."
        ),
    }
    provenance_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--exclude-pool", type=Path, action="append", required=True)
    parser.add_argument(
        "--cifar-root", type=Path, default=Path.home() / ".cache" / "cifar"
    )
    parser.add_argument("--seed", type=int, default=B25_SELECTION_SEED)
    parser.add_argument("--samples-per-class", type=int, default=600)
    args = parser.parse_args()
    record = build_disjoint_pool(
        args.archive,
        args.output,
        args.provenance,
        args.cifar_root,
        args.exclude_pool,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
