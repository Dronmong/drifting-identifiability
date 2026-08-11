"""Focused integrity tests for CAP2 train-only development evaluation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from ...diagnostics import write_json
from .. import development_evaluation, metric_calibration, standard_metrics
from ..artifacts import (
    save_checkpoint,
    source_manifest,
    write_json_atomic,
    write_npz_atomic,
)
from ..development_evaluation import (
    DEVELOPMENT_SAMPLES,
    GENERATION_SEED,
    portable_reference,
    verify_final_ema,
    write_uncurated_grid,
)
from ..metric_calibration import (
    assemble_reference_features,
    revalidate_metric_calibration_evidence,
)
from ..promotion import _grid_integrity
from ..standard_metrics import (
    REPOSITORY_AUXILIARY_PROTOCOL,
    REPOSITORY_FEATURE_SAMPLES,
    REPOSITORY_MEMORIZATION_SAMPLES,
    REPOSITORY_REFERENCE_SEED,
    bind_positive_control_provenance,
    clean_metrics_from_features,
    compute_repository_auxiliary_metrics,
    existing_png_folder,
    feature_array_sha256,
    fixed_reference_indices,
    generate_png_folder,
    load_clean_feature_archive,
    revalidate_clean_evaluation_evidence,
    unique_png_files,
    validate_png_folder,
    validate_standard_protocol,
)


class _TinyCleanFID:
    @staticmethod
    def get_reference_statistics(*_args, **_kwargs):
        return np.zeros(2), np.eye(2)

    @staticmethod
    def frechet_distance(mean_a, _cov_a, mean_b, _cov_b):
        return float(np.square(mean_a - mean_b).sum())

    @staticmethod
    def kernel_distance(left, right, **_kwargs):
        return float(np.square(left.mean(axis=0) - right.mean(axis=0)).sum())


def _assert_raises(error_type, function, *args, **kwargs) -> Exception:
    try:
        function(*args, **kwargs)
    except error_type as error:
        return error
    raise AssertionError(f"expected {error_type.__name__}")


def _write_rgb(path: Path, value: int = 0) -> None:
    array = np.full((32, 32, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path, format="PNG", compress_level=0)


def _screen_fixture(
    root: Path,
    *,
    final_step: int = 150_000,
    recorded_step: int | None = None,
    checkpoint_kind: str = "ema",
    corrupt_recorded_hash: bool = False,
) -> Path:
    recorded_step = final_step if recorded_step is None else recorded_step
    profile = {
        "name": "cap2-ordered_uniform-local_1000_d0002_fp32",
        "model": {"image_size": 32},
        "train": {
            "updates": final_step,
            "checkpoint_updates": [50_000, final_step],
        },
    }
    realized_profile = {
        **profile,
        "train": {**profile["train"]},
        "realized_device": "cpu",
    }
    preflight_hash = "a" * 64
    run_identity = "b" * 64
    unit_seed = 0
    checkpoint_path = root / f"step{recorded_step}_ema.pt"
    recorded_hash = save_checkpoint(
        checkpoint_path,
        {"weight": torch.ones(1)},
        step=recorded_step,
        kind=checkpoint_kind,
        arm="ordered_uniform",
        declared_profile=profile,
        realized_profile=realized_profile,
        preflight_sha256=preflight_hash,
        run_identity_sha256=run_identity,
        unit_seed=unit_seed,
    )
    if corrupt_recorded_hash:
        recorded_hash = "0" * 64
    unit_path = root / "result.json"
    write_json(
        unit_path,
        {
            "status": "cap-emf2-screen-unit",
            "development_only": True,
            "arm": "ordered_uniform",
            "preflight_sha256": preflight_hash,
            "run_identity_sha256": run_identity,
            "unit_seed": unit_seed,
            "declared_profile": profile,
            "realized_profile": realized_profile,
            "training": {"optimizer_updates": final_step},
            "checkpoints": {
                str(recorded_step): {
                    "ema": {
                        "path": str(checkpoint_path.resolve()),
                        "sha256": recorded_hash,
                    }
                }
            },
        },
    )
    return unit_path


def test_verified_unit_selects_only_declared_final_ema():
    with TemporaryDirectory() as directory:
        unit_path = _screen_fixture(Path(directory))
        verified = verify_final_ema(unit_path)
    assert verified.step == 150_000
    assert verified.checkpoint["kind"] == "ema"
    assert verified.unit["artifact_sha256"]


def test_verified_unit_accepts_only_an_explicit_recorded_intermediate_ema():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        profile = {
            "name": "cap2-ordered_uniform-local_1000_d0002_fp32",
            "model": {"image_size": 32},
            "train": {
                "updates": 750_000,
                "checkpoint_updates": [50_000, 650_000, 750_000],
            },
        }
        realized_profile = {
            **profile,
            "train": {**profile["train"]},
            "realized_device": "cpu",
        }
        records = {}
        for step in (650_000, 750_000):
            path = root / f"step{step}_ema.pt"
            digest = save_checkpoint(
                path,
                {"weight": torch.full((1,), float(step))},
                step=step,
                kind="ema",
                arm="ordered_uniform",
                declared_profile=profile,
                realized_profile=realized_profile,
                preflight_sha256="a" * 64,
                run_identity_sha256="b" * 64,
                unit_seed=0,
            )
            records[str(step)] = {
                "ema": {"path": str(path.resolve()), "sha256": digest}
            }
        unit_path = root / "result.json"
        write_json(
            unit_path,
            {
                "status": "cap-emf2-screen-unit",
                "development_only": True,
                "arm": "ordered_uniform",
                "preflight_sha256": "a" * 64,
                "run_identity_sha256": "b" * 64,
                "unit_seed": 0,
                "declared_profile": profile,
                "realized_profile": realized_profile,
                "training": {"optimizer_updates": 750_000},
                "checkpoints": records,
            },
        )
        verified = verify_final_ema(unit_path, step=650_000)
        assert verified.step == 650_000
        error = _assert_raises(RuntimeError, verify_final_ema, unit_path, step=700_000)
        assert "declared checkpoint" in str(error)


def test_verified_unit_rejects_raw_checkpoint_under_ema_label():
    with TemporaryDirectory() as directory:
        unit_path = _screen_fixture(Path(directory), checkpoint_kind="raw")
        error = _assert_raises(RuntimeError, verify_final_ema, unit_path)
    assert "kind" in str(error)


def test_verified_unit_rejects_intermediate_or_unrecorded_final():
    with TemporaryDirectory() as directory:
        unit_path = _screen_fixture(Path(directory), recorded_step=100_000)
        error = _assert_raises(RuntimeError, verify_final_ema, unit_path)
    assert "requested checkpoint" in str(error)


def test_verified_unit_rejects_checkpoint_hash_mismatch():
    with TemporaryDirectory() as directory:
        unit_path = _screen_fixture(Path(directory), corrupt_recorded_hash=True)
        error = _assert_raises(RuntimeError, verify_final_ema, unit_path)
    assert "hash" in str(error)


def test_verified_unit_rejects_a_recorded_step_after_declared_final():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        unit_path = _screen_fixture(root)
        unit = json.loads(unit_path.read_text(encoding="utf-8"))
        unit["checkpoints"]["300000"] = unit["checkpoints"]["150000"]
        write_json(unit_path, unit)
        error = _assert_raises(RuntimeError, verify_final_ema, unit_path)
    assert "last recorded" in str(error)


def test_png_validation_is_exact_and_decodes_every_file():
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        _write_rgb(folder / "000000.png", 10)
        _write_rgb(folder / "000001.png", 20)
        manifest = validate_png_folder(
            folder, expected_count=2, require_sequential_names=True
        )
        assert manifest["count"] == 2
        assert len(manifest["png_manifest_sha256"]) == 64
        _write_rgb(folder / "extra.png", 30)
        _assert_raises(
            RuntimeError,
            validate_png_folder,
            folder,
            expected_count=2,
            require_sequential_names=True,
        )


def test_png_validation_rejects_wrong_mode():
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(folder / "000000.png")
        error = _assert_raises(
            RuntimeError,
            validate_png_folder,
            folder,
            expected_count=1,
            require_sequential_names=True,
        )
    assert "RGB" in str(error)


def test_cleanfid_file_enumeration_is_case_insensitive_without_duplicates():
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        _write_rgb(folder / "lower.png")
        _write_rgb(folder / "upper.PNG")
        files = unique_png_files(folder, expected_count=2)
    assert len(files) == 2
    assert len(set(files)) == 2


def test_external_positive_control_requires_citation():
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        _write_rgb(folder / "sample.png")
        _assert_raises(
            ValueError,
            existing_png_folder,
            folder,
            expected_count=1,
            source_citation="",
        )
        record = existing_png_folder(
            folder,
            expected_count=1,
            source_citation="Example et al., DOI:10.0000/example",
        )
    assert record["source_citation"].startswith("Example")


def test_external_positive_control_binds_immutable_source_record():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        folder = root / "pngs"
        folder.mkdir()
        _write_rgb(folder / "000000.png")
        citation = "pinned public model, checkpoint, and sampler"
        samples = existing_png_folder(
            folder, expected_count=1, source_citation=citation
        )
        provenance_path = root / "source.json"
        write_json_atomic(
            provenance_path,
            {
                "status": "cap-emf2-stylegan2ada-positive-control-source",
                "decision": "COMPLETE",
                "citation": citation,
                "images": samples,
                "source_sha256": source_manifest(),
            },
        )
        bound = bind_positive_control_provenance(samples, provenance_path)
        assert len(bound["source_provenance_sha256"]) == 64
        altered = {**samples, "source_citation": "different"}
        _assert_raises(
            RuntimeError,
            bind_positive_control_provenance,
            altered,
            provenance_path,
        )


def test_uncurated_grid_uses_fixed_prefix():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        folder = root / "pngs"
        folder.mkdir()
        for index in range(4):
            _write_rgb(folder / f"{index:06d}.png", 30 * index)
        grid = root / "grid.png"
        record = write_uncurated_grid(folder, grid, rows=2, columns=2)
        with Image.open(grid) as image:
            assert image.size == (64, 64)
        assert record["rows"] == 2
        assert "no curation" in record["selection"]


def test_uncurated_grid_reference_survives_artifact_tree_relocation():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        original = root / "windows_evaluator"
        folder = original / "pngs"
        folder.mkdir(parents=True)
        for index in range(8 * 16):
            _write_rgb(folder / f"{index:06d}.png", index % 256)
        record = write_uncurated_grid(
            folder,
            original / "grid.png",
            reference_anchor=original,
        )
        assert record["path"] == "grid.png"
        assert _grid_integrity({"uncurated_grid": record}, anchor=original)

        relocated = root / "linux_training_host"
        shutil.move(str(original), relocated)
        assert _grid_integrity({"uncurated_grid": record}, anchor=relocated)


def test_all_evaluation_leaf_references_can_be_relocated():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        original = root / "original"
        original.mkdir()
        leaves = {
            name: original / relative
            for name, relative in {
                "unit": "result.json",
                "checkpoint": "checkpoints/final.pt",
                "samples": "pngs",
                "generated_features": "features/generated.npz",
                "kid_reference": "features/reference.npz",
            }.items()
        }
        for name, path in leaves.items():
            if name == "samples":
                path.mkdir(parents=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
        references = {
            name: portable_reference(path, original) for name, path in leaves.items()
        }

        relocated = root / "relocated"
        shutil.move(str(original), relocated)
        for name, reference in references.items():
            assert (relocated / reference).resolve().exists(), name


def test_development_protocol_fixes_count_and_seed():
    assert DEVELOPMENT_SAMPLES == 50_000
    assert GENERATION_SEED == 20_260_804


def test_clean_feature_archive_is_content_bound_and_shape_checked():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "features.npz"
        expected = np.arange(12, dtype=np.float32).reshape(3, 4)
        digest = write_npz_atomic(path, features=expected)
        loaded, metadata = load_clean_feature_archive(
            path, expected_count=3, expected_dimension=4
        )
        assert np.array_equal(loaded, expected)
        assert metadata["sha256"] == digest
        assert metadata["count"] == 3
        assert metadata["dimension"] == 4
        _assert_raises(
            RuntimeError,
            load_clean_feature_archive,
            path,
            expected_count=4,
            expected_dimension=4,
        )


def test_clean_evaluation_leaf_population_and_metrics_are_recomputed(monkeypatch):
    class FakeCleanFID:
        @staticmethod
        def get_reference_statistics(*_args, **_kwargs):
            return np.zeros(4), np.eye(4)

        @staticmethod
        def frechet_distance(*_args, **_kwargs):
            return 1.25

        @staticmethod
        def kernel_distance(*_args, **_kwargs):
            return 0.0125

    monkeypatch.setattr(standard_metrics, "_require_cleanfid", FakeCleanFID)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        pngs = root / "pngs"
        pngs.mkdir()
        for index in range(3):
            _write_rgb(pngs / f"{index:06d}.png", 30 * index)
        png_record = validate_png_folder(
            pngs, expected_count=3, require_sequential_names=True
        )
        generated_path = root / "generated.npz"
        reference_path = root / "reference.npz"
        write_npz_atomic(
            generated_path,
            features=np.arange(12, dtype=np.float32).reshape(3, 4),
        )
        write_npz_atomic(
            reference_path,
            features=np.arange(12, 24, dtype=np.float32).reshape(3, 4),
        )
        _generated, generated_record = load_clean_feature_archive(
            generated_path, expected_count=3, expected_dimension=4
        )
        _reference, reference_record = load_clean_feature_archive(
            reference_path, expected_count=3, expected_dimension=4
        )
        generated_record.update(
            {
                "path": generated_path.name,
                "population": (
                    "all fixed-seed generated images in sequential PNG order"
                ),
                "preprocessing": ("clean-fid 0.1.35 clean Inception preprocessing"),
                "source_png_manifest_sha256": png_record["png_manifest_sha256"],
                "generation_seed": 17,
            }
        )
        reference_record.update(
            {
                "path": reference_path.name,
                "population": (
                    "all 50,000 CIFAR-10 train images in dataset-index order"
                ),
                "preprocessing": ("clean-fid 0.1.35 clean Inception preprocessing"),
            }
        )
        evaluation = {
            "samples": {
                **png_record,
                "directory": pngs.name,
                "seed": 17,
            },
            "standard_train_reference_metrics": {
                "kid_seed": 19,
                "clean_fid_cifar10_train": 1.25,
                "clean_kid_cifar10_train": 0.0125,
                "generated_features": generated_record,
                "kid_reference": reference_record,
            },
        }
        verified = revalidate_clean_evaluation_evidence(
            evaluation, anchor=root, expected_count=3, expected_dimension=4
        )
        assert verified["valid"] is True
        assert all(verified["checks"].values())

        evaluation["standard_train_reference_metrics"]["clean_fid_cifar10_train"] = 9.0
        rejected = revalidate_clean_evaluation_evidence(
            evaluation, anchor=root, expected_count=3, expected_dimension=4
        )
        assert rejected["valid"] is False
        assert rejected["checks"]["clean_fid_recomputed"] is False


def test_cap1_and_cap2_share_one_fixed_auxiliary_implementation():
    assert REPOSITORY_AUXILIARY_PROTOCOL == "train-only-repository-auxiliary-v1"
    assert REPOSITORY_FEATURE_SAMPLES == 2_048
    assert REPOSITORY_MEMORIZATION_SAMPLES == 256
    assert REPOSITORY_REFERENCE_SEED == 20_260_832
    assert (
        development_evaluation.compute_repository_auxiliary_metrics
        is standard_metrics.compute_repository_auxiliary_metrics
    )


def test_repository_reference_subset_is_seeded_and_fixed():
    first = fixed_reference_indices(50_000, REPOSITORY_FEATURE_SAMPLES)
    torch.manual_seed(999)
    second = fixed_reference_indices(50_000, REPOSITORY_FEATURE_SAMPLES)
    assert torch.equal(first, second)
    assert len(torch.unique(first)) == REPOSITORY_FEATURE_SAMPLES


def test_repository_auxiliary_rejects_non_50k_manifest_before_data_access():
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        error = _assert_raises(
            RuntimeError,
            compute_repository_auxiliary_metrics,
            folder,
            sample_manifest={
                "directory": str(folder.resolve()),
                "count": 49_999,
                "png_manifest_sha256": "0" * 64,
            },
            device=torch.device("cpu"),
            data_root=None,
            feature_batch=2,
        )
    assert "50k" in str(error)


def test_standard_baseline_protocol_rejects_sample_or_seed_drift():
    validate_standard_protocol(
        count=50_000,
        kid_seed=20_260_831,
        generation_seed=20_260_804,
        checkpoint_mode=True,
    )
    _assert_raises(
        ValueError,
        validate_standard_protocol,
        count=49_999,
        kid_seed=20_260_831,
        generation_seed=20_260_804,
        checkpoint_mode=True,
    )
    _assert_raises(
        ValueError,
        validate_standard_protocol,
        count=50_000,
        kid_seed=20_260_831,
        generation_seed=1,
        checkpoint_mode=True,
    )


def test_generator_counts_actual_one_step_forwards():
    class IdentityGenerator(torch.nn.Module):
        def forward(self, state, _t, _interval):
            return state

    with TemporaryDirectory() as directory:
        record = generate_png_folder(
            IdentityGenerator(),
            SimpleNamespace(channels=3, image_size=32),
            Path(directory),
            count=3,
            batch=2,
            seed=5,
            device=torch.device("cpu"),
        )
    assert record["model_calls"] == 2
    assert record["one_model_call_per_batch"] is True


def test_clean_kid_is_seeded_and_reported_with_fid():
    class FakeFID:
        @staticmethod
        def get_reference_statistics(_name, _resolution, *, mode, split, metric):
            assert mode == "clean" and split == "train"
            if metric == "FID":
                return np.zeros(2), np.eye(2)
            return np.arange(24, dtype=np.float64).reshape(12, 2) / 10

        @staticmethod
        def frechet_distance(*_arguments):
            return 7.0

        @staticmethod
        def kernel_distance(_left, _right, *, num_subsets, max_subset_size):
            assert num_subsets == 100 and max_subset_size == 1_000
            return float(np.random.random())

    features = np.arange(20, dtype=np.float64).reshape(10, 2) / 10
    reference = np.arange(24, dtype=np.float64).reshape(12, 2) / 10
    first = clean_metrics_from_features(
        FakeFID, features, kid_reference_features=reference, kid_seed=31
    )
    np.random.seed(999)
    second = clean_metrics_from_features(
        FakeFID, features, kid_reference_features=reference, kid_seed=31
    )
    assert first == second
    assert first["clean_fid_cifar10_train"] == 7.0
    assert "clean_kid_cifar10_train" in first


def test_kid_reference_partition_is_restored_to_dataset_order():
    left = np.asarray([[30.0, 31.0], [10.0, 11.0]])
    right = np.asarray([[0.0, 1.0], [20.0, 21.0]])
    restored = assemble_reference_features(
        left,
        right,
        torch.tensor([3, 1]),
        torch.tensor([0, 2]),
        expected_count=4,
    )
    assert restored.tolist() == [
        [0.0, 1.0],
        [10.0, 11.0],
        [20.0, 21.0],
        [30.0, 31.0],
    ]


def test_kid_reference_rejects_duplicate_or_missing_dataset_indices():
    error = _assert_raises(
        RuntimeError,
        assemble_reference_features,
        np.zeros((2, 2)),
        np.ones((2, 2)),
        torch.tensor([0, 1]),
        torch.tensor([1, 3]),
        expected_count=4,
    )
    assert "exact partition" in str(error)


def test_real_real_margin_is_recomputed_from_canonical_archive(monkeypatch, tmp_path):
    count, dimension, seed, kid_seed = 8, 2, 17, 23
    features = np.arange(count * dimension, dtype=np.float64).reshape(count, dimension)
    archive = tmp_path / "reference.npz"
    archive_sha = write_npz_atomic(archive, features=features)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(count, generator=generator)
    left_indices, right_indices = indices[: count // 2], indices[count // 2 :]
    left, right = features[left_indices.numpy()], features[right_indices.numpy()]
    direct = standard_metrics.clean_pair_metrics_from_features(
        _TinyCleanFID, left, right, kid_seed=kid_seed
    )
    matched = {
        side: standard_metrics.clean_metrics_from_features(
            _TinyCleanFID,
            values,
            kid_reference_features=features,
            kid_seed=kid_seed,
        )
        for side, values in (("left", left), ("right", right))
    }
    calibration = {
        "seed": seed,
        "samples_per_side": count // 2,
        "left_indices_sha256": hashlib.sha256(
            left_indices.numpy().tobytes()
        ).hexdigest(),
        "right_indices_sha256": hashlib.sha256(
            right_indices.numpy().tobytes()
        ).hexdigest(),
        "metrics": {
            "kid_seed": kid_seed,
            "kid_reference": {
                "path": str(archive),
                "sha256": archive_sha,
                "feature_sha256": feature_array_sha256(features),
                "count": count,
                "dimension": dimension,
                "dtype": str(features.dtype),
            },
            "direct_disjoint_pair": direct,
            "matched_published_train_reference": matched,
        },
    }
    monkeypatch.setattr(metric_calibration, "_require_cleanfid", lambda: _TinyCleanFID)
    verified = revalidate_metric_calibration_evidence(
        calibration,
        anchor=tmp_path,
        expected_count=count,
        expected_dimension=dimension,
    )
    assert verified["valid"] is True

    calibration["metrics"]["direct_disjoint_pair"]["clean_fid"] += 1.0
    rejected = revalidate_metric_calibration_evidence(
        calibration,
        anchor=tmp_path,
        expected_count=count,
        expected_dimension=dimension,
    )
    assert rejected["checks"]["direct_pair_recomputed"] is False


def _run_all() -> int:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    print(f"=== stage_cap2 development evaluation ({len(tests)} tests) ===")
    failures = 0
    for name, function in tests:
        try:
            function()
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"  [FAIL] {name}: {type(error).__name__}: {error}")
        else:
            print(f"  [PASS] {name}")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
