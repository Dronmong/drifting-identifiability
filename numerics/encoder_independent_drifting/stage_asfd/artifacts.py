"""Artifact boundary for ASFD.

**The dependency manifest is an explicit list, never a directory glob.**

``b1_freeze.py`` globs its package *and* its tests directory, so two unrelated
test modules added months later permanently invalidated a completed
confirmation and the recorded bytes proved unrecoverable from git -- which cost
the entire B2.5 continuation.  Adding a module beside this stage therefore does
not invalidate an ASFD preflight unless the module is named below.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ..diagnostics import ROOT

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SPECIFICATION = ROOT / "numerics" / "AnchoredSelfFeatureDriftingSpecification.md"
DEFAULT_QUALIFICATION = HERE / "asfd_qualification.json"
DEFAULT_PREFLIGHT = HERE / "asfd_preflight.json"
DEFAULT_RESULT = HERE / "asfd_development.json"
BANKS = HERE / "banks"

_DEPENDENCIES: tuple[str, ...] = (
    "stage_asfd/__init__.py",
    "stage_asfd/artifacts.py",
    "stage_asfd/calibration.py",
    "stage_asfd/config.py",
    "stage_asfd/correction.py",
    "stage_asfd/continuation.py",
    "stage_asfd/features.py",
    "stage_asfd/evaluation.py",
    "stage_asfd/final_report.py",
    "stage_asfd/final_visual_review.py",
    "stage_asfd/feature_bank.py",
    "stage_asfd/field.py",
    "stage_asfd/gradients.py",
    "stage_asfd/qualify.py",
    "stage_asfd/qualification.py",
    "stage_asfd/preflight.py",
    "stage_asfd/recovery.py",
    # The frozen trunk and its objective are part of ASFD's meaning: a
    # different trunk is a different feature geometry.
    "stage_cap/__init__.py",
    "stage_cap/config.py",
    "stage_cap/data.py",
    "stage_cap/diagnostics.py",
    "stage_cap/model.py",
    "stage_cap/monitoring.py",
    "stage_cap/objective.py",
    "stage_cap/training.py",
    "stage_cap2/artifacts.py",
    "stage_cap2/benchmark.py",
    "stage_cap2/budget.py",
    "stage_cap2/config.py",
    "stage_cap2/development_evaluation.py",
    "stage_cap2/durable_mirror.py",
    "stage_cap2/early_admission.py",
    "stage_cap2/gate_calibration.py",
    "stage_cap2/hardware.py",
    "stage_cap2/metric_calibration.py",
    "stage_cap2/numerical_admission.py",
    "stage_cap2/preflight.py",
    "stage_cap2/preview.py",
    "stage_cap2/promotion.py",
    "stage_cap2/runpod_bootstrap.sh",
    "stage_cap2/runpod_pipeline.sh",
    "stage_cap2/run_screen.py",
    "stage_cap2/selection.py",
    "stage_cap2/standard_metrics.py",
    # Transitive implementations used by the retained final-evaluation
    # population.  Hashing only standard_metrics.py would let its numerical
    # meaning change after continuation without invalidating ASFD evidence.
    "stage_b2/metrics.py",
    "appearance.py",
    "fid.py",
    "spectral_anchor.py",
    "config.py",
    "device.py",
    "diagnostics.py",
)

#: Freeze artifacts that must never be loaded from this package. Their
#: constants were calibrated against a different architecture, a different loss
#: scale and a different data subset; only the procedure transfers.
FORBIDDEN_FREEZES: tuple[str, ...] = (
    "b1_freeze.json",
    "stage_b2/b2_freeze.json",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not sidecar.exists():
        raise RuntimeError(f"artifact or SHA sidecar missing: {path}")
    actual = file_sha256(path)
    if actual != sidecar.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError(f"SHA mismatch for {path.name}")
    return actual


def source_manifest() -> dict[str, str]:
    paths = [PACKAGE / name for name in _DEPENDENCIES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"declared ASFD dependency missing: {missing}")
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
        for path in sorted(set(paths))
    }


def manifest_difference(recorded: dict[str, str]) -> dict[str, list[str]]:
    live = source_manifest()
    return {
        "added": sorted(set(live) - set(recorded)),
        "removed": sorted(set(recorded) - set(live)),
        "changed": sorted(
            name for name in set(live) & set(recorded) if live[name] != recorded[name]
        ),
    }


def config_payload(config) -> dict:
    config.validate()
    return json.loads(json.dumps(asdict(config), sort_keys=True, default=str))


def assert_result_path_unused(path: Path) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise RuntimeError(f"refusing to overwrite a consumed ASFD artifact: {path}")


def assert_no_inherited_freeze() -> None:
    """Fail loudly if a B1/B2 constant is reachable from this package.

    This module is excluded from its own scan: it is where the forbidden names
    are declared, so including it would guarantee a false positive.
    """
    this_module = "stage_asfd/artifacts.py"
    for name in _DEPENDENCIES:
        if not name.startswith("stage_asfd/") or name == this_module:
            continue
        text = (PACKAGE / name).read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_FREEZES:
            stem = Path(forbidden).name
            for number, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if stem in stripped and not stripped.startswith("#"):
                    raise RuntimeError(
                        f"{name}:{number} references the inherited freeze "
                        f"{forbidden}; ASFD re-derives every constant"
                    )
