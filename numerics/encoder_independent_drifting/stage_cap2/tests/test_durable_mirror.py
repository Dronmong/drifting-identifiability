from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ..artifacts import verify_json, write_json_atomic
from ..durable_mirror import (
    ORPHAN_STORE,
    PROBE_STORE,
    DurableMirror,
    probe_root,
    provision_root,
    restore_recovery,
    restore_tree,
)
from ..run_screen import _require_durable_workspace


def _write(path: Path, value: int) -> None:
    write_json_atomic(path, {"status": "mirror-test", "value": value})


def _replace(path: Path, value: int) -> None:
    path.unlink()
    path.with_suffix(path.suffix + ".sha256").unlink()
    _write(path, value)


def _provision(path: Path, *, storage_id: str = "volume-test-001") -> dict:
    path.mkdir(parents=True)
    return provision_root(
        path,
        storage_id=storage_id,
        attest_instance_independent=True,
    )


def test_root_must_be_preexisting_explicitly_attested_and_probeable() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "run"
        missing = root / "missing-volume"
        with pytest.raises(RuntimeError, match="pre-existing provisioned"):
            DurableMirror(source, missing)
        assert not missing.exists()

        mirror_root = root / "persistent"
        mirror_root.mkdir()
        with pytest.raises(RuntimeError, match="lacks its explicit"):
            DurableMirror(source, mirror_root)
        with pytest.raises(RuntimeError, match="explicit instance-independent"):
            provision_root(
                mirror_root,
                storage_id="volume-test-001",
                attest_instance_independent=False,
            )

        attestation = provision_root(
            mirror_root,
            storage_id="volume-test-001",
            attest_instance_independent=True,
        )
        mirror = DurableMirror(source, mirror_root)
        assert mirror.attestation["artifact_sha256"] == attestation["artifact_sha256"]

        raw = bytes(range(256)) * 2
        probe = probe_root(mirror_root, payload=raw)
        assert probe["storage_id"] == "volume-test-001"
        assert probe["bytes"] == len(raw)
        assert probe["roundtrip_verified"] is True
        assert probe["probe_removed"] is True
        assert list((mirror_root / PROBE_STORE).iterdir()) == []
        assert mirror.probe(byte_count=31)["bytes"] == 31


def test_authorization_workspace_requires_attestation_and_closed_layout() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        workspace = root / "workspace"
        _provision(workspace, storage_id="workspace-volume")
        run = workspace / "runs" / "arm"
        preflight = workspace / "gates" / "preflight.json"
        record = _require_durable_workspace(
            workspace,
            required_paths={"output directory": run, "preflight": preflight},
        )
        assert record["attestation"]["storage_id"] == "workspace-volume"
        assert record["live_roundtrip_probe"]["roundtrip_verified"] is True

        with pytest.raises(RuntimeError, match="outside"):
            _require_durable_workspace(
                workspace,
                required_paths={"selection": root / "outside" / "selection.json"},
            )


def test_versioned_recovery_round_trip_latest_and_chosen_step() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "run"
        mirror_root = root / "persistent"
        latest_restore = root / "latest-restored"
        chosen_restore = root / "chosen-restored"
        _provision(mirror_root)

        immutable = source / "checkpoints" / "step50000.json"
        recovery = source / "checkpoints" / "recovery.json"
        _write(immutable, 10)
        _write(recovery, 1)
        mirror = DurableMirror(source, mirror_root)
        immutable_record = mirror.mirror(immutable)
        first = mirror.mirror(recovery, mutable=True, recovery_step=50_000)
        assert first["recovery_step"] == 50_000

        _replace(recovery, 2)
        second = mirror.mirror_recovery(recovery, recovery_step=150_000)
        assert second["sha256"] != first["sha256"]
        assert mirror.recovery_steps(recovery) == [50_000, 150_000]
        assert mirror.verify(immutable) == immutable_record
        assert mirror.verify_recovery(recovery) == second
        assert mirror.verify_recovery(recovery, recovery_step=150_000) == second

        latest_records = restore_tree(mirror_root, latest_restore)
        assert {
            (record["relative_path"], record.get("recovery_step"))
            for record in latest_records
        } == {
            ("checkpoints/recovery.json", 150_000),
            ("checkpoints/step50000.json", None),
        }
        assert (
            verify_json(
                latest_restore / "checkpoints" / "recovery.json", "mirror-test"
            )["value"]
            == 2
        )

        chosen = restore_recovery(
            mirror_root,
            chosen_restore,
            relative_path=Path("checkpoints/recovery.json"),
            recovery_step=50_000,
        )
        assert chosen["recovery_step"] == 50_000
        assert (
            verify_json(
                chosen_restore / "checkpoints" / "recovery.json", "mirror-test"
            )["value"]
            == 1
        )
        with pytest.raises(RuntimeError, match="lacks recovery step 123"):
            restore_recovery(
                mirror_root,
                root / "missing-step",
                relative_path=Path("checkpoints/recovery.json"),
                recovery_step=123,
            )


def test_recovery_requires_a_step_and_a_step_commit_is_immutable() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "run"
        mirror_root = root / "persistent"
        _provision(mirror_root)
        recovery = source / "checkpoints" / "recovery.json"
        _write(recovery, 1)
        mirror = DurableMirror(source, mirror_root)

        with pytest.raises(ValueError, match="requires recovery_step"):
            mirror.mirror(recovery, mutable=True)
        with pytest.raises(ValueError, match="nonnegative"):
            mirror.mirror_recovery(recovery, recovery_step=-1)

        first = mirror.mirror_recovery(recovery, recovery_step=10)
        _replace(recovery, 2)
        with pytest.raises(RuntimeError, match="step commit changed"):
            mirror.mirror_recovery(recovery, recovery_step=10)
        assert mirror.recovery_steps(recovery) == [10]
        with pytest.raises(RuntimeError, match="local recovery differs"):
            mirror.verify_recovery(recovery, recovery_step=10)
        assert first["recovery_step"] == 10


def test_incomplete_new_commit_falls_back_and_retry_repairs_it() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "run"
        mirror_root = root / "persistent"
        _provision(mirror_root)
        recovery = source / "checkpoints" / "recovery.json"
        _write(recovery, 1)
        mirror = DurableMirror(source, mirror_root)
        mirror.mirror_recovery(recovery, recovery_step=100)

        _replace(recovery, 2)
        second = mirror.mirror_recovery(recovery, recovery_step=200)
        commit = mirror_root / str(second["commit_relative_path"])
        commit_sidecar = commit.with_suffix(commit.suffix + ".sha256")
        commit_sidecar.unlink()

        # A commit is visible only as a complete pair.  The earlier numeric
        # step remains the safe latest recovery after a simulated pointer loss.
        assert mirror.recovery_steps(recovery) == [100]
        restored = root / "fallback"
        record = restore_recovery(
            mirror_root,
            restored,
            relative_path=Path("checkpoints/recovery.json"),
        )
        assert record["recovery_step"] == 100
        assert verify_json(restored / "checkpoints" / "recovery.json")["value"] == 1

        # A rerun from step 100 has different elapsed-time metadata in reality,
        # hence a different recovery digest at the same numeric step.  The lone
        # commit JSON is uncommitted and must not poison step 200 forever.
        _replace(recovery, 3)
        repaired = mirror.mirror_recovery(recovery, recovery_step=200)
        assert repaired["sha256"] != second["sha256"]
        assert repaired["recovery_step"] == second["recovery_step"]
        assert commit_sidecar.is_file()
        assert mirror.recovery_steps(recovery) == [100, 200]


def test_committed_version_corruption_fails_closed_but_orphan_version_is_ignored() -> (
    None
):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "run"
        mirror_root = root / "persistent"
        _provision(mirror_root)
        recovery = source / "checkpoints" / "recovery.json"
        _write(recovery, 1)
        mirror = DurableMirror(source, mirror_root)
        mirror.mirror_recovery(recovery, recovery_step=100)

        _replace(recovery, 2)
        second = mirror.mirror_recovery(recovery, recovery_step=200)
        version = mirror_root / str(second["version_relative_path"])
        version_sidecar = version.with_suffix(version.suffix + ".sha256")
        version_sidecar.unlink()
        with pytest.raises(RuntimeError, match="sidecar missing"):
            mirror.recovery_steps(recovery)

        # Retrying the identical step safely completes the partial version.
        assert mirror.mirror_recovery(recovery, recovery_step=200) == second
        assert version_sidecar.is_file()

        # A fully written version without a commit is an ignored orphan, as it
        # would be after a crash immediately before commit publication.
        _replace(recovery, 3)
        orphan = mirror.mirror_recovery(recovery, recovery_step=300)
        orphan_commit = mirror_root / str(orphan["commit_relative_path"])
        orphan_commit.unlink()
        orphan_commit.with_suffix(orphan_commit.suffix + ".sha256").unlink()
        assert mirror.recovery_steps(recovery) == [100, 200]
        restored = root / "orphan-fallback"
        record = restore_recovery(
            mirror_root,
            restored,
            relative_path=Path("checkpoints/recovery.json"),
        )
        assert record["recovery_step"] == 200
        assert verify_json(restored / "checkpoints" / "recovery.json")["value"] == 2


def test_restore_excludes_and_rerun_replaces_only_precommit_future_artifacts() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "run"
        mirror_root = root / "persistent"
        _provision(mirror_root)
        mirror = DurableMirror(source, mirror_root)
        recovery = source / "checkpoints" / "recovery.json"
        committed = source / "checkpoints" / "cap2_ordered_uniform_step100_raw.pt"
        future = source / "checkpoints" / "cap2_ordered_uniform_step150_raw.pt"

        _write(committed, 100)
        mirror.mirror(committed)
        _write(recovery, 1)
        mirror.mirror_recovery(recovery, recovery_step=100)

        # Simulate termination after the immutable checkpoint reached remote
        # storage but before recovery step 150 was committed.
        _write(future, 150)
        first_future = mirror.mirror(future)
        restored = root / "restored-at-100"
        restore_tree(mirror_root, restored)
        assert (restored / "checkpoints" / committed.name).is_file()
        assert not (restored / "checkpoints" / future.name).exists()

        # A deterministic rerun can differ at the serialization-byte level.
        # Because step 150 is beyond the latest commit, preserve the orphan and
        # replace the active mirror pair.
        _replace(future, 151)
        replacement = mirror.mirror(future)
        assert replacement["sha256"] != first_future["sha256"]
        assert any((mirror_root / ORPHAN_STORE).rglob("payload-*.bin"))

        _replace(recovery, 2)
        mirror.mirror_recovery(recovery, recovery_step=150)
        _replace(future, 152)
        with pytest.raises(RuntimeError, match="immutable mirror artifact changed"):
            mirror.mirror(future)


def test_immutable_behavior_repairs_exact_incomplete_pair_and_rejects_changes() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "run"
        mirror_root = root / "persistent"
        _provision(mirror_root)
        mirror = DurableMirror(source, mirror_root)

        artifact = source / "artifact.json"
        _write(artifact, 1)
        mirror.mirror(artifact)
        _replace(artifact, 2)
        with pytest.raises(RuntimeError, match="immutable mirror artifact changed"):
            mirror.mirror(artifact)

        other = source / "other.json"
        _write(other, 1)
        incomplete = mirror_root / "other.json"
        incomplete.write_bytes(other.read_bytes())
        repaired_payload_first = mirror.mirror(other)
        assert (
            repaired_payload_first["sha256"]
            == verify_json(other, "mirror-test")["artifact_sha256"]
        )

        sidecar_first = source / "sidecar-first.json"
        _write(sidecar_first, 2)
        mirrored_sidecar = mirror_root / "sidecar-first.json.sha256"
        mirrored_sidecar.write_bytes(
            sidecar_first.with_suffix(".json.sha256").read_bytes()
        )
        repaired_sidecar_first = mirror.mirror(sidecar_first)
        assert (
            repaired_sidecar_first["sha256"]
            == verify_json(sidecar_first, "mirror-test")["artifact_sha256"]
        )

        ambiguous = source / "ambiguous.json"
        _write(ambiguous, 3)
        (mirror_root / ambiguous.name).write_bytes(b"wrong bytes")
        replaced_incomplete = mirror.mirror(ambiguous)
        assert (
            replaced_incomplete["sha256"]
            == verify_json(ambiguous, "mirror-test")["artifact_sha256"]
        )
        assert any((mirror_root / ORPHAN_STORE).rglob("payload-*.bin"))

        nested_mirror = source / "mounted"
        _provision(nested_mirror, storage_id="nested-test")
        with pytest.raises(RuntimeError, match="non-nested"):
            DurableMirror(source, nested_mirror)
