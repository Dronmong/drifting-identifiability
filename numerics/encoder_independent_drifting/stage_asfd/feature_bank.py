"""Build and verify durable, role-separated ASFD target feature banks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from ..device import configure, resolve_device
from ..stage_cap.data import cifar10_train_labels, cifar10_train_pool, flip_batch
from ..stage_cap2.artifacts import (
    verify_json as verify_cap2_json,
)
from ..stage_cap2.artifacts import (
    write_json_atomic,
    write_sha256_sidecar_atomic,
)
from ..stage_cap2.standard_metrics import _load_model, tensor_content_sha256
from .artifacts import (
    assert_no_inherited_freeze,
    file_sha256,
    source_manifest,
)
from .config import asfd_config
from .features import LevelNormalization, encode, freeze_trunk

STATUS = "asfd-feature-bank"
QUALIFICATION_STATUS = "asfd-target-only-qualification"
BANK_SEED = 20_260_851
IMAGES_PER_CLASS = 500
ROLES = ("train_positive", "train_probe", "fresh_positive", "fresh_probe")
VIEWS_PER_ROLE = 4


def dataset_binding(pool: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    """Exact CIFAR population identity used by cached and replayed roles.

    Feature shards alone do not bind the raw images later recovered by index.
    Recording both tensors closes that gap: a different or corrupted data root
    cannot silently pair frozen descriptors with another pixel population.
    """
    if pool.shape != (50_000, 3, 32, 32) or pool.dtype != torch.float32:
        raise RuntimeError("unexpected CIFAR-10 training tensor for ASFD banking")
    if labels.shape != (50_000,) or labels.dtype != torch.int64:
        raise RuntimeError("unexpected CIFAR-10 label tensor for ASFD banking")
    return {
        "train_tensor_sha256": tensor_content_sha256(pool),
        "train_tensor_shape": list(pool.shape),
        "train_tensor_dtype": str(pool.dtype),
        "train_labels_sha256": tensor_content_sha256(labels),
        "train_labels_shape": list(labels.shape),
        "train_labels_dtype": str(labels.dtype),
    }


def verify_dataset_binding(
    recorded: object,
    pool: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    """Fail closed unless current raw replay data equal the bank population."""
    live = dataset_binding(pool, labels)
    if recorded != live:
        raise RuntimeError(
            "ASFD feature bank belongs to a different CIFAR-10 population"
        )
    return live


def _resolve(reference: object, anchor: Path) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise RuntimeError("feature-bank input contains an empty path")
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (anchor / path).resolve()


def _portable(path: Path, anchor: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def _normalization(payload: dict) -> dict[str, LevelNormalization]:
    result = {}
    for name, record in payload.items():
        scale = record.get("channel_scale")
        result[name] = LevelNormalization(
            channel_scale=None if scale is None else torch.tensor(scale),
            level_scale=float(record["level_scale"]),
            pc1_share=float(record["pc1_share"]),
            per_channel_applied=bool(record["per_channel_applied"]),
        )
    return result


def stratified_partitions(
    labels: torch.Tensor, excluded: set[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two fixed, class-balanced, image-disjoint 5k populations."""
    generator = torch.Generator().manual_seed(BANK_SEED)
    train, fresh = [], []
    for label in range(10):
        candidates = torch.nonzero(labels == label, as_tuple=True)[0]
        if excluded:
            excluded_tensor = torch.tensor(sorted(excluded), dtype=candidates.dtype)
            candidates = candidates[~torch.isin(candidates, excluded_tensor)]
        order = candidates[torch.randperm(len(candidates), generator=generator)]
        required = 2 * IMAGES_PER_CLASS
        if len(order) < required:
            raise RuntimeError(f"class {label} lacks {required} unexcluded images")
        train.append(order[:IMAGES_PER_CLASS])
        fresh.append(order[IMAGES_PER_CLASS:required])
    return torch.cat(train).sort().values, torch.cat(fresh).sort().values


def _index_hash(values: torch.Tensor) -> str:
    return hashlib.sha256(
        values.numpy().astype("<i8", copy=False).tobytes()
    ).hexdigest()


def _role_indices(role: str, train: torch.Tensor, fresh: torch.Tensor) -> torch.Tensor:
    base = train if role.startswith("train_") else fresh
    return base.repeat(VIEWS_PER_ROLE)


def build_banks(
    qualification_path: Path,
    *,
    data_root: str | None,
    device: torch.device,
    batch: int,
    output_dir: Path,
) -> dict:
    qualification = verify_cap2_json(qualification_path, QUALIFICATION_STATUS)
    if qualification.get("decision") != "GO" or not isinstance(
        qualification.get("selected"), dict
    ):
        raise RuntimeError("feature banks require a GO target-only qualification")
    if qualification.get("source_sha256") != source_manifest():
        raise RuntimeError("ASFD source changed after target-only qualification")
    checkpoint = _resolve(
        qualification.get("teacher_checkpoint", {}).get("path"),
        qualification_path.parent,
    )
    if file_sha256(checkpoint) != qualification["teacher_checkpoint"].get("sha256"):
        raise RuntimeError("qualified teacher checkpoint changed")
    _payload, model, _model_config = _load_model(checkpoint, device)
    trunk = freeze_trunk(model)
    config = asfd_config()
    selected = qualification["selected"]
    t_f = float(selected["t_f"])
    normalization = _normalization(selected["normalization"])
    pool = cifar10_train_pool(data_root)
    labels = cifar10_train_labels(data_root)
    population_binding = dataset_binding(pool, labels)
    excluded = {int(value) for value in qualification.get("sample_indices", [])}
    train_indices, fresh_indices = stratified_partitions(labels, excluded)
    if set(train_indices.tolist()) & set(fresh_indices.tolist()):
        raise RuntimeError("train and fresh feature banks overlap")

    output_dir.mkdir(parents=True, exist_ok=False)
    records = {}
    for role_index, role in enumerate(ROLES):
        indices = _role_indices(role, train_indices, fresh_indices)
        rows = len(indices)
        role_files = {}
        maps = {
            level: open_memmap(
                output_dir / f"{role}_{level}.npy",
                mode="w+",
                dtype=np.float16,
                shape=(rows, config.features.vectors_per_level, _model_config.width),
            )
            for level in config.features.levels
        }
        flip_bits = torch.empty(rows, dtype=torch.bool)
        view_seeds = []
        for view in range(VIEWS_PER_ROLE):
            view_seed = BANK_SEED + 10_000 * role_index + view
            view_seeds.append(
                {
                    "view": view,
                    "feature_noise_seed": view_seed,
                    "horizontal_flip_seed": view_seed + 1_000,
                }
            )
            noise_generator = torch.Generator().manual_seed(view_seed)
            flip_generator = torch.Generator().manual_seed(view_seed + 1_000)
            base = train_indices if role.startswith("train_") else fresh_indices
            for start in range(0, len(base), batch):
                stop = min(start + batch, len(base))
                row_start = view * len(base) + start
                row_stop = view * len(base) + stop
                images = pool[base[start:stop]].clone()
                flips = torch.rand(stop - start, generator=flip_generator) < 0.5
                images = flip_batch(images, flips).to(device)
                encoded = encode(
                    trunk,
                    images,
                    t_f,
                    config.features,
                    normalization,
                    generator=noise_generator,
                )
                for level, values in encoded.items():
                    maps[level][row_start:row_stop] = (
                        values.detach().cpu().numpy().astype(np.float16, copy=False)
                    )
                flip_bits[row_start:row_stop] = flips
        for mapping in maps.values():
            mapping.flush()
        # On Windows an open mmap can prevent a subsequent integrity reader
        # from obtaining a stable view of the file.  Drop *all* references
        # before hashing any shard, not just the loop variable.
        maps.clear()
        del mapping
        gc.collect()
        for level in config.features.levels:
            path = output_dir / f"{role}_{level}.npy"
            digest = write_sha256_sidecar_atomic(path)
            role_files[level] = {
                "path": path.name,
                "sha256": digest,
                "shape": [
                    rows,
                    config.features.vectors_per_level,
                    _model_config.width,
                ],
                "dtype": "float16",
                "bytes": path.stat().st_size,
            }
        records[role] = {
            "base_population": "train" if role.startswith("train_") else "fresh",
            "rows": rows,
            "images": rows // VIEWS_PER_ROLE,
            "views_per_image": VIEWS_PER_ROLE,
            "image_indices": indices.tolist(),
            "image_indices_sha256": _index_hash(indices),
            "view_seeds": view_seeds,
            # Replay needs the actual augmentation decisions, not merely a
            # digest that can detect but cannot reconstruct them.
            "horizontal_flip_bits": flip_bits.tolist(),
            "flip_bits_sha256": hashlib.sha256(flip_bits.numpy().tobytes()).hexdigest(),
            "files": role_files,
        }
    result = {
        "status": STATUS,
        "decision": "GO",
        "qualification": {
            "path": _portable(qualification_path, output_dir),
            "sha256": qualification["artifact_sha256"],
        },
        "teacher_checkpoint": {
            "path": _portable(checkpoint, output_dir),
            "sha256": file_sha256(checkpoint),
        },
        "dataset": "CIFAR-10 official training split, all classes",
        "dataset_binding": population_binding,
        "allocation": {
            "images_per_class_per_population": IMAGES_PER_CLASS,
            "train_indices": train_indices.tolist(),
            "train_indices_sha256": _index_hash(train_indices),
            "fresh_indices": fresh_indices.tolist(),
            "fresh_indices_sha256": _index_hash(fresh_indices),
            "qualification_indices_excluded": True,
            "train_fresh_disjoint": True,
        },
        "t_f": t_f,
        "normalization": selected["normalization"],
        "roles": records,
        "source_sha256": source_manifest(),
        "limits": [
            "Banks contain target-training features only; no test image is opened.",
            "Each role has independent noise/flip streams and train/fresh populations are image-disjoint.",
            "Float16 stores frozen targets; kernel distances and energies remain FP32 at use time.",
        ],
    }
    metadata = output_dir / "feature_bank.json"
    digest = write_json_atomic(metadata, result)
    result["artifact_sha256"] = digest
    return result


class FeatureBank:
    """Read-only mmap bank with full startup verification."""

    def __init__(self, metadata_path: Path) -> None:
        self.metadata_path = metadata_path.resolve()
        self.payload = verify_cap2_json(metadata_path, STATUS)
        if self.payload.get("source_sha256") != source_manifest():
            raise RuntimeError("ASFD source changed after feature-bank construction")
        self.arrays: dict[str, dict[str, np.memmap]] = {}
        for role, record in self.payload["roles"].items():
            self.arrays[role] = {}
            for level, file_record in record["files"].items():
                path = metadata_path.parent / file_record["path"]
                if file_sha256(path) != file_record["sha256"]:
                    raise RuntimeError(f"feature-bank shard changed: {path}")
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                if (
                    list(values.shape) != file_record["shape"]
                    or str(values.dtype) != file_record["dtype"]
                ):
                    raise RuntimeError(f"feature-bank shard metadata differs: {path}")
                self.arrays[role][level] = values

    def sample(
        self,
        role: str,
        indices: np.ndarray,
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if role not in self.arrays:
            raise ValueError(f"unknown feature-bank role {role!r}")
        result = {}
        for level, values in self.arrays[role].items():
            # Advanced indexing materializes only the requested rows.
            block = np.array(values[indices], dtype=np.float32, copy=True)
            result[level] = torch.from_numpy(block).to(device)
        return result

    def raw_batch(
        self,
        role: str,
        indices: np.ndarray,
        pool: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Replay the raw-image endpoint paired with cached feature rows."""
        if role not in self.payload["roles"]:
            raise ValueError(f"unknown feature-bank role {role!r}")
        record = self.payload["roles"][role]
        image_indices = record.get("image_indices")
        flip_bits = record.get("horizontal_flip_bits")
        if not isinstance(image_indices, list) or not isinstance(flip_bits, list):
            raise TypeError(f"feature-bank role {role!r} lacks replay metadata")
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or len(selected) == 0:
            raise ValueError("feature-bank row selection must be nonempty and flat")
        if selected.min() < 0 or selected.max() >= len(image_indices):
            raise IndexError("feature-bank row selection is outside the bank")
        source = torch.tensor(
            [image_indices[int(index)] for index in selected], dtype=torch.long
        )
        flips = torch.tensor(
            [flip_bits[int(index)] for index in selected], dtype=torch.bool
        )
        return flip_batch(pool[source].clone(), flips).to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert_no_inherited_freeze()
    if args.output_dir.exists():
        raise RuntimeError("refusing to reuse a feature-bank directory")
    device = resolve_device(args.device)
    configure(device, allow_tf32=False)
    torch.use_deterministic_algorithms(True)
    result = build_banks(
        args.qualification,
        data_root=args.data_root,
        device=device,
        batch=args.batch,
        output_dir=args.output_dir,
    )
    print(
        f"wrote {args.output_dir / 'feature_bank.json'} "
        f"sha256={result['artifact_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
