"""Attested, hash-verified off-instance mirroring for paid CAP2 screens.

The mirror root is an explicit trust boundary.  It must already exist and carry
an operator-created attestation stating that its storage survives deletion of
the training instance.  Constructing :class:`DurableMirror` never creates that
root, so a missing mount cannot silently fall back to the pod filesystem.

Immutable artifacts retain their source-relative names.  Rolling recoveries
are different: overwriting ``recovery.pt`` and then its SHA sidecar can destroy
the last good remote pair if the process dies between those writes.  Recovery
payloads are therefore stored as immutable, digest-named versions.  A numeric
step commit is published only after its version is complete.  Latest recovery
means the greatest valid committed numeric step; incomplete newer work is
ignored, while corruption of a completed commit fails closed.

The attestation is necessarily an operator assertion: a filesystem API cannot
prove a provider's retention policy.  ``probe_root`` complements it with a
live byte-for-byte write/read/delete check against the mounted storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .artifacts import file_sha256, verify_file, verify_json, write_json_atomic

ATTESTATION_STATUS = "cap-emf2-durable-root-attestation"
ATTESTATION_FILENAME = ".cap2-durable-root.json"
ATTESTATION_SCHEMA_VERSION = 1
RECOVERY_COMMIT_STATUS = "cap-emf2-durable-recovery-commit"
RECOVERY_STORE = ".cap2-recovery-versions"
PROBE_STORE = ".cap2-durable-probes"
ORPHAN_STORE = ".cap2-uncommitted-orphans"

_STEP_ARTIFACT_PATTERNS = (
    re.compile(r"^cap2_.+_step(?P<step>\d+)_(?:raw|ema)\.(?:pt|png)$"),
    re.compile(r"^cap2_.+_snapshot_step(?P<step>\d+)\.pt$"),
    re.compile(r"^result_(?P<step>\d+)\.json$"),
    re.compile(r"^early_admission_(?P<step>\d+)\.json$"),
    re.compile(r"^promotion_(?P<step>\d+)k_to_\d+k\.json$"),
)


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _hash_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _numeric_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("recovery step must be a nonnegative integer")
    return value


def _strict_relative(path: Path, root: Path) -> Path:
    path = path.resolve()
    root = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"artifact {path} is outside run root {root}") from error
    if not relative.parts:
        raise RuntimeError("the run root itself is not an artifact")
    return relative


def _separate_roots(source_root: Path, mirror_root: Path) -> tuple[Path, Path]:
    source = source_root.resolve()
    mirror = mirror_root.resolve()
    if source == mirror or source in mirror.parents or mirror in source.parents:
        raise RuntimeError(
            "durable mirror and run output must be separate, non-nested roots"
        )
    return source, mirror


def _require_existing_directory(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise RuntimeError(f"{label} must be a pre-existing provisioned directory")
    if not resolved.is_dir():
        raise RuntimeError(f"{label} is not a directory: {resolved}")
    return resolved


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_bytes(raw: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as writer:
            writer.write(raw)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _known_artifact_step(relative: Path) -> int | None:
    """Extract the horizon from an immutable CAP2 step artifact.

    Only names emitted by the guarded runner and its immediate admission /
    promotion tools are recognized.  Unknown immutable files remain subject to
    the original fail-closed behavior and are never silently discarded.
    """

    for pattern in _STEP_ARTIFACT_PATTERNS:
        matched = pattern.fullmatch(relative.name)
        if matched is not None:
            return int(matched.group("step"))
    return None


def provision_root(
    mirror_root: Path,
    *,
    storage_id: str,
    attest_instance_independent: bool,
) -> dict[str, object]:
    """Attest an already-mounted root; never create the root itself.

    ``storage_id`` should be a provider volume, bucket, or filesystem identity
    that remains stable when a training instance is replaced.  The function
    records an explicit operator assertion, not an unverifiable automatic
    durability claim.
    """

    root = _require_existing_directory(mirror_root, label="durable mirror root")
    if not attest_instance_independent:
        raise RuntimeError(
            "durable root provisioning requires explicit instance-independent "
            "storage attestation"
        )
    if not isinstance(storage_id, str) or not storage_id.strip():
        raise ValueError("durable storage id must be a nonempty string")
    attestation_path = root / ATTESTATION_FILENAME
    payload: dict[str, object] = {
        "status": ATTESTATION_STATUS,
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "storage_id": storage_id.strip(),
        "instance_independent": True,
        "attestation": (
            "operator attests that this pre-provisioned storage survives "
            "deletion or replacement of the CAP2 training instance"
        ),
    }
    digest = write_json_atomic(attestation_path, payload)
    return {**payload, "artifact_sha256": digest}


def load_root_attestation(mirror_root: Path) -> dict[str, object]:
    root = _require_existing_directory(mirror_root, label="durable mirror root")
    path = root / ATTESTATION_FILENAME
    if not path.is_file() or not _sidecar(path).is_file():
        raise RuntimeError(
            "durable mirror root lacks its explicit instance-independent attestation"
        )
    payload = verify_json(path, ATTESTATION_STATUS)
    if (
        payload.get("schema_version") != ATTESTATION_SCHEMA_VERSION
        or payload.get("instance_independent") is not True
        or not isinstance(payload.get("storage_id"), str)
        or not str(payload["storage_id"]).strip()
        or not isinstance(payload.get("attestation"), str)
        or "survives" not in str(payload["attestation"]).lower()
    ):
        raise RuntimeError("durable mirror root attestation is incomplete")
    return payload


def probe_root(
    mirror_root: Path,
    *,
    byte_count: int = 4096,
    payload: bytes | None = None,
) -> dict[str, object]:
    """Perform a live byte-for-byte write/read/delete round trip."""

    attestation = load_root_attestation(mirror_root)
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
    ):
        raise ValueError("probe byte count must be a positive integer")
    raw = payload if payload is not None else os.urandom(byte_count)
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("probe payload must be nonempty bytes")
    probe_dir = mirror_root.resolve() / PROBE_STORE
    probe_dir.mkdir(parents=True, exist_ok=True)
    path = probe_dir / f"probe-{secrets.token_hex(16)}.bin"
    try:
        _atomic_write_bytes(raw, path)
        restored = path.read_bytes()
        if restored != raw:
            raise RuntimeError(
                "durable mirror byte roundtrip changed the probe payload"
            )
        digest = _hash_bytes(restored)
    finally:
        path.unlink(missing_ok=True)
    return {
        "status": "cap-emf2-durable-root-probe",
        "storage_id": attestation["storage_id"],
        "attestation_sha256": attestation["artifact_sha256"],
        "bytes": len(raw),
        "sha256": digest,
        "roundtrip_verified": True,
        "probe_removed": not path.exists(),
    }


@dataclass(frozen=True)
class MirrorRecord:
    relative_path: str
    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class RecoveryMirrorRecord(MirrorRecord):
    recovery_step: int
    version_relative_path: str
    commit_relative_path: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            **super().as_dict(),
            "recovery_step": self.recovery_step,
            "version_relative_path": self.version_relative_path,
            "commit_relative_path": self.commit_relative_path,
        }


def _recovery_stream_root(mirror_root: Path, relative: Path) -> Path:
    return mirror_root / RECOVERY_STORE / relative


def _version_name(step: int, digest: str, suffix: str) -> str:
    return f"{step:012d}-{digest}{suffix}"


def _commit_path(stream_root: Path, step: int) -> Path:
    return stream_root / "commits" / f"{step:012d}.json"


def _publish_recovery_version(source: Path, destination: Path, digest: str) -> None:
    """Complete or verify one immutable content-addressed payload pair."""

    destination_sidecar = _sidecar(destination)
    source_sidecar = _sidecar(source)
    if destination.exists() and _hash_bytes(destination.read_bytes()) != digest:
        raise RuntimeError(f"recovery version payload is corrupt: {destination}")
    if destination_sidecar.exists():
        words = destination_sidecar.read_text(encoding="utf-8").split()
        if not words or words[0] != digest:
            raise RuntimeError(f"recovery version sidecar is corrupt: {destination}")
    if not destination.exists():
        _atomic_copy(source, destination)
    if not destination_sidecar.exists():
        _atomic_copy(source_sidecar, destination_sidecar)
    verify_file(destination, digest)
    if destination.stat().st_size != source.stat().st_size:
        raise RuntimeError(f"recovery version size differs: {destination}")


def _publish_recovery_commit(path: Path, payload: dict[str, object]) -> str:
    """Publish or repair an immutable numeric commit pair.

    A process crash can leave exactly one member of the pair.  Retrying the
    same step repairs it only when the surviving member is byte-identical to
    the expected commit; a different commit at the same step fails closed.
    """

    raw = _canonical_json(payload)
    digest = _hash_bytes(raw)
    sidecar = _sidecar(path)
    expected_sidecar = f"{digest}  {path.name}\n".encode()
    complete = path.is_file() and sidecar.is_file()
    if complete:
        # Once both members have been published the numeric step is immutable.
        # This branch deliberately rejects even a self-consistently rehashed
        # replacement: a complete commit is the transaction boundary.
        if path.read_bytes() != raw:
            raise RuntimeError(f"recovery step commit changed: {path}")
        if sidecar.read_bytes() != expected_sidecar:
            raise RuntimeError(f"recovery step commit sidecar changed: {path}")
        verify_file(path, digest)
        return digest

    # A lone payload or sidecar is not a commit.  It can be left by termination
    # between the two atomic writes.  A rerun from the previous committed
    # recovery legitimately has different wall-clock metadata and therefore a
    # different digest, so replace the incomplete pair rather than permanently
    # poisoning this numeric step.  Removing a lone sidecar first ensures no
    # transient complete pair can authenticate the replacement payload.
    sidecar.unlink(missing_ok=True)
    _atomic_write_bytes(raw, path)
    _atomic_write_bytes(expected_sidecar, sidecar)
    verify_file(path, digest)
    return digest


def _safe_version_path(stream_root: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise RuntimeError("recovery commit has an empty version reference")
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("recovery commit version reference escapes its stream")
    candidate = (stream_root / relative).resolve()
    try:
        candidate.relative_to(stream_root.resolve())
    except ValueError as error:
        raise RuntimeError(
            "recovery commit version reference escapes its stream"
        ) from error
    return candidate


def _load_commit(
    path: Path, *, stream_root: Path, expected_relative: Path
) -> dict[str, object] | None:
    sidecar = _sidecar(path)
    if not path.is_file() or not sidecar.is_file():
        return None
    payload = verify_json(path, RECOVERY_COMMIT_STATUS)
    try:
        filename_step = int(path.stem)
        step = _numeric_step(payload.get("recovery_step"))
        size = int(payload.get("bytes", -1))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid recovery commit metadata: {path}") from error
    digest = payload.get("sha256")
    if (
        payload.get("schema_version") != 1
        or filename_step != step
        or payload.get("relative_path") != expected_relative.as_posix()
        or not _hash64(digest)
        or size < 0
    ):
        raise RuntimeError(f"invalid recovery commit binding: {path}")
    version = _safe_version_path(stream_root, payload.get("version"))
    verify_file(version, str(digest))
    if version.stat().st_size != size:
        raise RuntimeError(f"committed recovery version size differs: {version}")
    payload["version_path"] = version
    payload["commit_path"] = path
    return payload


def _commits(stream_root: Path, relative: Path) -> dict[int, dict[str, object]]:
    commits_dir = stream_root / "commits"
    if not commits_dir.is_dir():
        return {}
    records: dict[int, dict[str, object]] = {}
    for path in sorted(commits_dir.glob("*.json")):
        record = _load_commit(
            path,
            stream_root=stream_root,
            expected_relative=relative,
        )
        if record is None:
            continue
        step = int(record["recovery_step"])
        if step in records:
            raise RuntimeError(f"duplicate committed recovery step {step}")
        records[step] = record
    return records


def _latest_committed_recovery_step(mirror_root: Path) -> int | None:
    """Return the greatest valid commit across this run namespace.

    CAP2 provisions one namespace per arm and therefore normally has one
    recovery stream.  Taking the maximum across all streams is intentionally
    conservative: no immutable artifact at or below *any* committed recovery
    may be replaced.
    """

    steps: list[int] = []
    for relative, stream_root in _recovery_streams(mirror_root):
        steps.extend(_commits(stream_root, relative))
    return max(steps) if steps else None


def _quarantine_uncommitted_pair(
    destination: Path, *, relative: Path, mirror_root: Path
) -> None:
    """Preserve and remove a pre-commit immutable pair before replacement."""

    quarantine = mirror_root / ORPHAN_STORE / relative
    for label, source in (
        ("payload", destination),
        ("sidecar", _sidecar(destination)),
    ):
        if not source.is_file():
            continue
        digest = _hash_bytes(source.read_bytes())
        archived = quarantine / f"{label}-{digest}.bin"
        if archived.exists():
            if _hash_bytes(archived.read_bytes()) != digest:
                raise RuntimeError(f"uncommitted quarantine is corrupt: {archived}")
        else:
            _atomic_copy(source, archived)
        if _hash_bytes(archived.read_bytes()) != digest:
            raise RuntimeError(f"uncommitted quarantine copy differs: {archived}")

    # Remove the authenticity companion first.  A crash after this point leaves
    # an incomplete, and therefore still replaceable, uncommitted pair.
    _sidecar(destination).unlink(missing_ok=True)
    destination.unlink(missing_ok=True)


def _select_commit(
    stream_root: Path, relative: Path, recovery_step: int | None
) -> dict[str, object]:
    records = _commits(stream_root, relative)
    if not records:
        raise RuntimeError(
            f"durable mirror has no committed recovery for {relative.as_posix()}"
        )
    step = max(records) if recovery_step is None else _numeric_step(recovery_step)
    if step not in records:
        raise RuntimeError(
            f"durable mirror lacks recovery step {step} for {relative.as_posix()}"
        )
    return records[step]


def _record_from_commit(
    record: dict[str, object], *, mirror_root: Path
) -> RecoveryMirrorRecord:
    version = record["version_path"]
    commit = record["commit_path"]
    assert isinstance(version, Path)
    assert isinstance(commit, Path)
    return RecoveryMirrorRecord(
        relative_path=str(record["relative_path"]),
        sha256=str(record["sha256"]),
        bytes=int(record["bytes"]),
        recovery_step=int(record["recovery_step"]),
        version_relative_path=version.relative_to(mirror_root).as_posix(),
        commit_relative_path=commit.relative_to(mirror_root).as_posix(),
    )


class DurableMirror:
    """Synchronously mirror and verify one CAP2 run tree."""

    def __init__(self, source_root: Path, mirror_root: Path) -> None:
        mirror = _require_existing_directory(mirror_root, label="durable mirror root")
        self.source_root, self.mirror_root = _separate_roots(source_root, mirror)
        self.attestation = load_root_attestation(self.mirror_root)

    def probe(self, *, byte_count: int = 4096) -> dict[str, object]:
        return probe_root(self.mirror_root, byte_count=byte_count)

    def mirror(
        self,
        path: Path,
        *,
        mutable: bool = False,
        recovery_step: int | None = None,
    ) -> dict[str, str | int]:
        if mutable:
            if recovery_step is None:
                raise ValueError("mutable recovery mirroring requires recovery_step")
            return self.mirror_recovery(path, recovery_step=recovery_step)
        if recovery_step is not None:
            raise ValueError("recovery_step is valid only for mutable recovery")
        relative = _strict_relative(path, self.source_root)
        digest = verify_file(path)
        destination = self.mirror_root / relative
        destination_sidecar = _sidecar(destination)
        if destination.exists() or destination_sidecar.exists():
            complete = destination.is_file() and destination_sidecar.is_file()
            # A process can die between the two atomic renames that publish an
            # immutable payload/sidecar pair.  Repair directly when the
            # surviving half is byte-identical to the verified source half.
            if not complete and destination.is_file():
                if (
                    file_sha256(destination) == digest
                    and destination.stat().st_size == path.stat().st_size
                ):
                    _atomic_copy(_sidecar(path), destination_sidecar)
                    mirrored = verify_file(destination, digest)
                    if mirrored != digest:
                        raise RuntimeError(
                            f"mirror repair failed for {relative.as_posix()}"
                        )
                    return MirrorRecord(
                        relative.as_posix(), digest, path.stat().st_size
                    ).as_dict()
            elif (
                not complete
                and destination_sidecar.is_file()
                and destination_sidecar.read_bytes() == _sidecar(path).read_bytes()
            ):
                _atomic_copy(path, destination)
                mirrored = verify_file(destination, digest)
                if (
                    mirrored != digest
                    or destination.stat().st_size != path.stat().st_size
                ):
                    raise RuntimeError(
                        f"mirror repair failed for {relative.as_posix()}"
                    )
                return MirrorRecord(
                    relative.as_posix(), digest, path.stat().st_size
                ).as_dict()
            if not complete:
                # A lone half was never a committed immutable artifact.  A
                # deterministic finalization retry may legitimately change
                # incidental metadata, so preserve the orphan for diagnosis
                # and publish the new complete pair even when the training
                # recovery step itself is already committed.
                _quarantine_uncommitted_pair(
                    destination, relative=relative, mirror_root=self.mirror_root
                )
            else:
                existing: str | None = None
                try:
                    existing = verify_file(destination)
                except RuntimeError:
                    # Corruption is replaceable only for a recognized artifact
                    # beyond every committed recovery.  The check below keeps
                    # committed history fail-closed.
                    existing = None
                if existing == digest:
                    return MirrorRecord(
                        relative.as_posix(), digest, path.stat().st_size
                    ).as_dict()

                artifact_step = _known_artifact_step(relative)
                committed_step = _latest_committed_recovery_step(self.mirror_root)
                replaceable = artifact_step is not None and (
                    committed_step is None or artifact_step > committed_step
                )
                if not replaceable:
                    raise RuntimeError(
                        f"immutable mirror artifact changed at {relative.as_posix()}"
                    )
                _quarantine_uncommitted_pair(
                    destination, relative=relative, mirror_root=self.mirror_root
                )
        _atomic_copy(path, destination)
        _atomic_copy(_sidecar(path), destination_sidecar)
        mirrored = verify_file(destination, digest)
        if mirrored != digest or destination.stat().st_size != path.stat().st_size:
            raise RuntimeError(f"mirror verification failed for {relative.as_posix()}")
        return MirrorRecord(relative.as_posix(), digest, path.stat().st_size).as_dict()

    def mirror_recovery(
        self, path: Path, *, recovery_step: int
    ) -> dict[str, str | int]:
        step = _numeric_step(recovery_step)
        relative = _strict_relative(path, self.source_root)
        digest = verify_file(path)
        size = path.stat().st_size
        stream_root = _recovery_stream_root(self.mirror_root, relative)
        suffix = "".join(path.suffixes) or ".bin"
        version = stream_root / "versions" / _version_name(step, digest, suffix)
        _publish_recovery_version(path, version, digest)
        commit = _commit_path(stream_root, step)
        commit_payload: dict[str, object] = {
            "status": RECOVERY_COMMIT_STATUS,
            "schema_version": 1,
            "relative_path": relative.as_posix(),
            "recovery_step": step,
            "sha256": digest,
            "bytes": size,
            "version": version.relative_to(stream_root).as_posix(),
        }
        _publish_recovery_commit(commit, commit_payload)
        selected = _select_commit(stream_root, relative, step)
        return _record_from_commit(selected, mirror_root=self.mirror_root).as_dict()

    def recovery_steps(self, path: Path) -> list[int]:
        relative = _strict_relative(path, self.source_root)
        stream_root = _recovery_stream_root(self.mirror_root, relative)
        return sorted(_commits(stream_root, relative))

    def verify(self, path: Path) -> dict[str, str | int]:
        relative = _strict_relative(path, self.source_root)
        digest = verify_file(path)
        destination = self.mirror_root / relative
        verify_file(destination, digest)
        if destination.stat().st_size != path.stat().st_size:
            raise RuntimeError(f"mirror size differs for {relative.as_posix()}")
        return MirrorRecord(relative.as_posix(), digest, path.stat().st_size).as_dict()

    def verify_recovery(
        self, path: Path, *, recovery_step: int | None = None
    ) -> dict[str, str | int]:
        relative = _strict_relative(path, self.source_root)
        stream_root = _recovery_stream_root(self.mirror_root, relative)
        record = _select_commit(stream_root, relative, recovery_step)
        digest = verify_file(path)
        if digest != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise RuntimeError(
                f"local recovery differs from committed step {record['recovery_step']}"
            )
        return _record_from_commit(record, mirror_root=self.mirror_root).as_dict()


def _internal_mirror_path(path: Path, mirror_root: Path) -> bool:
    relative = path.relative_to(mirror_root)
    return bool(relative.parts) and relative.parts[0] in {
        RECOVERY_STORE,
        PROBE_STORE,
        ORPHAN_STORE,
    }


def _immutable_payloads(mirror_root: Path) -> list[Path]:
    attestation = mirror_root / ATTESTATION_FILENAME
    return sorted(
        path
        for path in mirror_root.rglob("*")
        if path.is_file()
        and path != attestation
        and path != _sidecar(attestation)
        and not path.name.endswith(".sha256")
        and not _internal_mirror_path(path, mirror_root)
    )


def _recovery_streams(mirror_root: Path) -> list[tuple[Path, Path]]:
    store = mirror_root / RECOVERY_STORE
    if not store.is_dir():
        return []
    streams: list[tuple[Path, Path]] = []
    for commits_dir in sorted(path for path in store.rglob("commits") if path.is_dir()):
        stream_root = commits_dir.parent
        relative = stream_root.relative_to(store)
        if not relative.parts:
            raise RuntimeError("durable recovery stream has an empty relative path")
        streams.append((relative, stream_root))
    return streams


def _restore_pair(source: Path, target: Path, digest: str) -> None:
    if target.exists() or _sidecar(target).exists():
        raise RuntimeError(f"restore destination is not clean at {target}")
    _atomic_copy(source, target)
    _atomic_write_bytes(f"{digest}  {target.name}\n".encode(), _sidecar(target))
    verify_file(target, digest)


def restore_recovery(
    mirror_root: Path,
    destination_root: Path,
    *,
    relative_path: Path,
    recovery_step: int | None = None,
) -> dict[str, str | int]:
    """Restore one chosen, or the latest, committed numeric recovery step."""

    mirror = _require_existing_directory(mirror_root, label="durable mirror root")
    load_root_attestation(mirror)
    destination = destination_root.resolve()
    if (
        mirror == destination
        or mirror in destination.parents
        or destination in mirror.parents
    ):
        raise RuntimeError("restore source and destination must be separate roots")
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError("recovery restore path must be a strict relative path")
    stream_root = _recovery_stream_root(mirror, relative)
    commit = _select_commit(stream_root, relative, recovery_step)
    version = commit["version_path"]
    assert isinstance(version, Path)
    target = destination / relative
    _restore_pair(version, target, str(commit["sha256"]))
    return _record_from_commit(commit, mirror_root=mirror).as_dict()


def restore_tree(
    mirror_root: Path,
    destination_root: Path,
    *,
    recovery_step: int | None = None,
) -> list[dict[str, str | int]]:
    """Restore immutable artifacts plus chosen/latest committed recoveries."""

    mirror = _require_existing_directory(mirror_root, label="durable mirror root")
    load_root_attestation(mirror)
    destination = destination_root.resolve()
    if (
        mirror == destination
        or mirror in destination.parents
        or destination in mirror.parents
    ):
        raise RuntimeError("restore source and destination must be separate roots")
    records: list[dict[str, str | int]] = []
    recovery_commits: list[tuple[Path, dict[str, object]]] = []
    for relative, stream_root in _recovery_streams(mirror):
        recovery_commits.append(
            (relative, _select_commit(stream_root, relative, recovery_step))
        )
    # A CAP2 namespace is one arm and normally contains exactly one stream.  If
    # an operator has placed several streams in one namespace, the minimum
    # selected step is the only conservative global cutoff for shared
    # immutable artifacts.
    committed_cutoff = (
        min(int(commit["recovery_step"]) for _relative, commit in recovery_commits)
        if recovery_commits
        else None
    )
    for source in _immutable_payloads(mirror):
        digest = verify_file(source)
        relative = source.relative_to(mirror)
        artifact_step = _known_artifact_step(relative)
        if (
            committed_cutoff is not None
            and artifact_step is not None
            and artifact_step > committed_cutoff
        ):
            # This is a complete immutable artifact published immediately
            # before a recovery commit that never landed.  Leave it un-restored
            # in the mirror namespace; a deterministic rerun may verify/reuse
            # or safely replace it while it remains beyond the commit boundary.
            continue
        target = destination / relative
        _restore_pair(source, target, digest)
        records.append(
            MirrorRecord(relative.as_posix(), digest, source.stat().st_size).as_dict()
        )
    for relative, commit in recovery_commits:
        version = commit["version_path"]
        assert isinstance(version, Path)
        _restore_pair(version, destination / relative, str(commit["sha256"]))
        records.append(_record_from_commit(commit, mirror_root=mirror).as_dict())
    if not records:
        raise RuntimeError("durable mirror has no committed artifacts")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    provision = subparsers.add_parser("provision")
    provision.add_argument("--mirror-dir", type=Path, required=True)
    provision.add_argument("--storage-id", required=True)
    provision.add_argument(
        "--i-attest-instance-independent-storage", action="store_true"
    )

    probe = subparsers.add_parser("probe")
    probe.add_argument("--mirror-dir", type=Path, required=True)
    probe.add_argument("--bytes", type=int, default=4096)

    restore = subparsers.add_parser("restore")
    restore.add_argument("--mirror-dir", type=Path, required=True)
    restore.add_argument("--output-dir", type=Path, required=True)
    restore.add_argument("--recovery-step", type=int, default=None)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--mirror-dir", type=Path, required=True)
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument("--recovery-step", type=int, default=None)

    args = parser.parse_args()
    if args.command == "provision":
        result = provision_root(
            args.mirror_dir,
            storage_id=args.storage_id,
            attest_instance_independent=args.i_attest_instance_independent_storage,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "probe":
        result = probe_root(args.mirror_dir, byte_count=args.bytes)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "restore":
        records = restore_tree(
            args.mirror_dir,
            args.output_dir,
            recovery_step=args.recovery_step,
        )
    else:
        mirror = DurableMirror(args.output_dir, args.mirror_dir)
        records = []
        for path in sorted(args.output_dir.rglob("*")):
            if not path.is_file() or path.name.endswith(".sha256"):
                continue
            if mirror.recovery_steps(path):
                records.append(
                    mirror.verify_recovery(path, recovery_step=args.recovery_step)
                )
            else:
                records.append(mirror.verify(path))
    print(f"{args.command}: verified {len(records)} mirrored artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
