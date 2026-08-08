"""Focused integrity tests for CAP2 train-only development evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from ...diagnostics import write_json
from .. import development_evaluation, standard_metrics
from ..artifacts import save_checkpoint, source_manifest, write_json_atomic
from ..development_evaluation import (
    DEVELOPMENT_SAMPLES,
    GENERATION_SEED,
    verify_final_ema,
    write_uncurated_grid,
)
from ..metric_calibration import assemble_reference_features
from ..standard_metrics import (
    REPOSITORY_AUXILIARY_PROTOCOL,
    REPOSITORY_FEATURE_SAMPLES,
    REPOSITORY_MEMORIZATION_SAMPLES,
    REPOSITORY_REFERENCE_SEED,
    bind_positive_control_provenance,
    clean_metrics_from_features,
    compute_repository_auxiliary_metrics,
    existing_png_folder,
    fixed_reference_indices,
    generate_png_folder,
    unique_png_files,
    validate_png_folder,
    validate_standard_protocol,
)


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


def test_verified_unit_rejects_raw_checkpoint_under_ema_label():
    with TemporaryDirectory() as directory:
        unit_path = _screen_fixture(Path(directory), checkpoint_kind="raw")
        error = _assert_raises(RuntimeError, verify_final_ema, unit_path)
    assert "kind" in str(error)


def test_verified_unit_rejects_intermediate_or_unrecorded_final():
    with TemporaryDirectory() as directory:
        unit_path = _screen_fixture(Path(directory), recorded_step=100_000)
        error = _assert_raises(RuntimeError, verify_final_ema, unit_path)
    assert "final checkpoint" in str(error)


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


def test_development_protocol_fixes_count_and_seed():
    assert DEVELOPMENT_SAMPLES == 50_000
    assert GENERATION_SEED == 20_260_804


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
