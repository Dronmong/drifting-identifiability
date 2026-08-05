"""Artifact boundary for CAP-EMF-1.

**The dependency manifest is an explicit list, never a directory glob.**

This is the direct lesson of the B2.5 blocker.  ``b1_freeze.py`` builds its
manifest with ``PACKAGE.glob("*.py")`` and ``(PACKAGE / "tests").glob("*.py")``,
so two unrelated test files added months later permanently invalidated a
completed confirmation — and the recorded bytes proved unrecoverable from git,
which cost the whole B2.5 continuation.  A glob silently binds a stage to every
file some future author will legitimately add next to it.

Adding a module to this package therefore does **not** invalidate a CAP-EMF-1
preflight unless the module is named in ``_DEPENDENCIES``.  Removing or editing
a named one does, which is the intended behaviour.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ..diagnostics import ROOT

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
PROTOCOL = ROOT / "numerics" / "EncoderIndependentCAPEMF1Protocol.md"
DEFAULT_PREFLIGHT = HERE / "cap_preflight.json"
DEFAULT_RESULT = HERE / "cap_emf1_unit.json"
CHECKPOINTS = HERE / "checkpoints"

# Every file whose bytes change the meaning of a CAP-EMF-1 result.  Test
# modules are deliberately absent: they do not execute during training, and
# binding them is exactly what broke B1.
_DEPENDENCIES: tuple[str, ...] = (
    "stage_cap/__init__.py",
    "stage_cap/artifacts.py",
    "stage_cap/config.py",
    "stage_cap/data.py",
    "stage_cap/diagnostics.py",
    # Hashed deliberately: the sealed evaluation is written and frozen BEFORE
    # the run, so it cannot be authored while looking at training curves.
    "stage_cap/evaluation.py",
    "stage_cap/model.py",
    "stage_cap/objective.py",
    "stage_cap/preflight.py",
    "stage_cap/training.py",
    "config.py",
    "device.py",
    "diagnostics.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not sidecar.exists():
        raise RuntimeError(f"artifact or SHA sidecar missing: {path}")
    actual = file_sha256(path)
    recorded = sidecar.read_text(encoding="utf-8").split()[0]
    if actual != recorded:
        raise RuntimeError(f"SHA mismatch for {path.name}")
    return actual


def source_manifest() -> dict[str, str]:
    paths = [PACKAGE / name for name in _DEPENDENCIES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"declared CAP dependency missing: {missing}")
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
        for path in sorted(set(paths))
    }


def manifest_difference(recorded: dict[str, str]) -> dict[str, list[str]]:
    """Added / removed / changed against the live manifest.

    Whole-dict equality is what gates a run; reporting the three categories
    separately matters because they have different causes and different fixes.
    """
    live = source_manifest()
    return {
        "added": sorted(set(live) - set(recorded)),
        "removed": sorted(set(recorded) - set(live)),
        "changed": sorted(
            name for name in set(live) & set(recorded) if live[name] != recorded[name]
        ),
    }


def profile_payload(profile) -> dict:
    profile.validate()
    return json.loads(
        json.dumps(
            {
                "name": profile.name,
                "purpose": profile.purpose,
                "model": asdict(profile.model),
                "objective": asdict(profile.objective),
                "train": asdict(profile.train),
                "gate": asdict(profile.gate),
            },
            sort_keys=True,
        )
    )


def assert_result_path_unused(path: Path) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise RuntimeError(f"refusing to overwrite a consumed CAP artifact: {path}")


def load_preflight(path: Path = DEFAULT_PREFLIGHT) -> dict:
    digest = verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "cap-emf1-preflight":
        raise RuntimeError("not a CAP-EMF-1 preflight artifact")
    if payload.get("verdict", {}).get("decision") != "GO":
        raise RuntimeError("CAP-EMF-1 preflight did not return GO")
    if payload.get("protocol_sha256") != file_sha256(PROTOCOL):
        raise RuntimeError("CAP-EMF-1 protocol changed after preflight")
    if payload.get("source_sha256") != source_manifest():
        raise RuntimeError(
            "CAP-EMF-1 executable sources changed after preflight: "
            f"{manifest_difference(payload.get('source_sha256', {}))}"
        )
    payload["artifact_sha256"] = digest
    return payload


def checkpoint_path(step: int, kind: str) -> Path:
    if kind not in {"raw", "ema"}:
        raise ValueError("checkpoint kind must be 'raw' or 'ema'")
    return CHECKPOINTS / f"cap_emf1_step{step}_{kind}.pt"


def save_checkpoint(
    path: Path,
    state_dict: dict,
    *,
    step: int,
    kind: str,
    profile: dict,
    preflight_sha: str,
    parameter_count: int,
) -> str:
    import torch

    if path.exists():
        raise RuntimeError(f"refusing to overwrite CAP checkpoint {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "cap-emf-1",
            "step": step,
            "kind": kind,
            "profile": profile,
            "preflight_sha256": preflight_sha,
            "parameter_count": parameter_count,
            "state_dict": {
                name: value.detach().cpu() for name, value in state_dict.items()
            },
        },
        path,
    )
    return file_sha256(path)


def load_checkpoint(path: Path, *, expected_sha: str | None = None) -> dict:
    import torch

    if expected_sha is not None and file_sha256(path) != expected_sha:
        raise RuntimeError(f"CAP checkpoint changed after training: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(path, map_location="cpu")
    if payload.get("stage") != "cap-emf-1":
        raise RuntimeError("not a CAP-EMF-1 checkpoint")
    return payload
