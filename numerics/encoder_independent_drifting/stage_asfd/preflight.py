"""Outcome-blind ASFD calibration and measured 500-update continuation preflight."""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..config import AnchorConfig
from ..device import configure, resolve_device
from ..spectral_anchor import projected_scale
from ..stage_cap.config import CAPModelConfig, CAPObjectiveConfig
from ..stage_cap.data import cifar10_train_pool, flip_batch
from ..stage_cap.model import CAPPixelTransformer
from ..stage_cap.objective import emf_loss, sample_time_triangle
from ..stage_cap.training import load_recovery_payload, train_cap_unit
from ..stage_cap2.artifacts import (
    load_preflight as load_cap2_preflight,
)
from ..stage_cap2.artifacts import (
    verify_json,
    write_json_atomic,
)
from ..stage_cap2.config import apply_calibrated_gate, screen_profile
from .artifacts import (
    assert_no_inherited_freeze,
    assert_result_path_unused,
    file_sha256,
    source_manifest,
)
from .config import asfd_config
from .correction import ASFDCorrection
from .gradients import gradient_norm, snapshot
from .recovery import fork_foundation_recovery

STATUS = "asfd-preflight"
CALIBRATION_EVENTS = 50
SMOKE_UPDATES = 500
PREFLIGHT_SEED = 20_260_941
_FOUNDATION_RUNTIME_KEYS = (
    "device",
    "torch_version",
    "cuda_version",
    "gpu_name",
    "gpu_memory_gib",
    "capability",
)


def _require_foundation_runtime_and_rate(
    cap2_preflight: dict, live: dict
) -> dict[str, object]:
    """Bind the ASFD smoke to the foundation benchmark's hardware and rate.

    ASFD deliberately disables TF32 even though the foundation benchmark used
    it, so ``allow_tf32`` is not part of this equality. Every field that
    identifies the CUDA/PyTorch stack or physical GPU *is* part of it. Without
    this check, a smoke measured on another GPU could be priced using the
    foundation provider's hourly rate and would not be valid budget evidence.
    """

    benchmark = cap2_preflight.get("inputs", {}).get("benchmark")
    if not isinstance(benchmark, dict):
        raise TypeError("CAP2 preflight lacks its production benchmark")
    recorded = benchmark.get("device")
    if not isinstance(recorded, dict):
        raise TypeError("CAP2 benchmark lacks its runtime binding")
    changed = [
        key for key in _FOUNDATION_RUNTIME_KEYS if recorded.get(key) != live.get(key)
    ]
    if changed:
        raise RuntimeError(
            "ASFD preflight runtime differs from the foundation benchmark: "
            + ", ".join(changed)
        )
    if live.get("device") != "cuda" or live.get("allow_tf32") is not False:
        raise RuntimeError("ASFD preflight requires CUDA with TF32 disabled")

    benchmark_rate = float(benchmark.get("hourly_rate", math.nan))
    budget_rate = float(cap2_preflight.get("budget", {}).get("hourly_rate", math.nan))
    if (
        not math.isfinite(benchmark_rate)
        or benchmark_rate <= 0
        or benchmark_rate != budget_rate
    ):
        raise RuntimeError(
            "ASFD preflight's provider rate differs from the measured benchmark"
        )
    return {
        "matched_fields": list(_FOUNDATION_RUNTIME_KEYS),
        "foundation_runtime": {
            key: recorded.get(key) for key in _FOUNDATION_RUNTIME_KEYS
        },
        "asfd_allow_tf32": False,
        "foundation_allow_tf32": recorded.get("allow_tf32"),
        "hourly_rate": budget_rate,
    }


def _resolve(reference: object, anchor: Path) -> Path:
    if not isinstance(reference, str) or not reference:
        raise RuntimeError("ASFD preflight input contains an empty path")
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (anchor / path).resolve()


def _portable(path: Path, anchor: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def continuation_profile(cap2_preflight: dict, updates: int):
    if updates <= 750_000:
        raise ValueError("an ASFD continuation horizon must exceed 750k")
    base = apply_calibrated_gate(
        screen_profile("ordered_uniform", cap2_preflight["candidate"], updates=750_000),
        cap2_preflight["inputs"]["gate_calibration"],
    )
    checkpoints = tuple(range(50_000, 750_001, 50_000)) + (updates,)
    result = replace(
        base,
        train=replace(
            base.train,
            updates=updates,
            checkpoint_updates=checkpoints,
            snapshot_every=10_000,
        ),
    )
    result.validate()
    return result


def _calibration_primary_gradient(
    model: CAPPixelTransformer,
    pool: torch.Tensor,
    objective: CAPObjectiveConfig,
    *,
    micro_batch: int,
    accumulation: int,
    device: torch.device,
    generators: dict[str, torch.Generator],
) -> list[torch.Tensor | None]:
    model.zero_grad(set_to_none=True)
    for _ in range(accumulation):
        order = torch.randint(
            0, len(pool), (micro_batch,), generator=generators["data"]
        )
        clean = pool[order].clone()
        flips = torch.rand(micro_batch, generator=generators["flip"]) < 0.5
        clean = flip_batch(clean, flips).to(device)
        noise = torch.randn(
            clean.shape, generator=generators["noise"], dtype=clean.dtype
        ).to(device)
        triangle = sample_time_triangle(
            micro_batch,
            objective,
            generators["time"],
            device,
            diagonal_generator=generators["diagonal"],
        )
        result = emf_loss(model, clean, noise, triangle, objective)
        (result.loss / accumulation).backward()
    return snapshot(model)


def _coefficient_calibration(
    model: CAPPixelTransformer,
    correction: ASFDCorrection,
    pool: torch.Tensor,
    objective: CAPObjectiveConfig,
    *,
    micro_batch: int,
    accumulation: int,
    device: torch.device,
    events: int,
) -> tuple[dict[str, float], list[dict]]:
    generators = {
        name: torch.Generator().manual_seed(PREFLIGHT_SEED + index)
        for index, name in enumerate(("data", "flip", "noise", "time", "diagonal"))
    }
    caps = {
        "b1": correction.config.gradients.cap_b1,
        "raw": correction.config.gradients.cap_raw,
        "self": correction.config.gradients.cap_self,
    }
    suggestions = {name: [] for name in caps}
    rows = []
    for event in range(events):
        primary = _calibration_primary_gradient(
            model,
            pool,
            objective,
            micro_batch=micro_batch,
            accumulation=accumulation,
            device=device,
            generators=generators,
        )
        primary_norm = gradient_norm(primary)
        components, stats = correction.compute_components(
            750_010 + 10 * event, model, include_health=False
        )
        norms = {name: gradient_norm(values) for name, values in components.items()}
        if primary_norm <= 0 or any(value <= 0 for value in norms.values()):
            raise RuntimeError("ASFD coefficient calibration saw a zero gradient")
        for name, cap in caps.items():
            suggestions[name].append(cap * primary_norm / norms[name])
        rows.append(
            {
                "event": event + 1,
                "primary_norm": primary_norm,
                "component_norm": norms,
                "losses": stats["losses"],
            }
        )
        model.zero_grad(set_to_none=True)
    coefficients = {
        name: float(np.median(values)) for name, values in suggestions.items()
    }
    return coefficients, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-gate", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--feature-bank", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--calibration-events", type=int, default=CALIBRATION_EVENTS)
    parser.add_argument("--smoke-updates", type=int, default=SMOKE_UPDATES)
    args = parser.parse_args()
    assert_result_path_unused(args.out)
    assert_no_inherited_freeze()
    if args.work_dir.exists():
        raise RuntimeError("refusing to reuse an ASFD preflight work directory")
    if (
        args.calibration_events != CALIBRATION_EVENTS
        or args.smoke_updates != SMOKE_UPDATES
    ):
        raise ValueError(
            "production ASFD preflight requires exactly 50 events and 500 updates"
        )
    args.work_dir.mkdir(parents=True)
    foundation = verify_json(args.foundation_gate, "cap-emf2-750k-foundation-gate")
    qualification = verify_json(args.qualification, "asfd-target-only-qualification")
    if foundation.get("decision") != "GO" or qualification.get("decision") != "GO":
        raise RuntimeError(
            "ASFD preflight requires GO foundation and qualification gates"
        )
    if (
        qualification.get("foundation_gate", {}).get("sha256")
        != foundation["artifact_sha256"]
    ):
        raise RuntimeError("ASFD qualification belongs to another foundation")
    bank_payload = verify_json(args.feature_bank, "asfd-feature-bank")
    if (
        bank_payload.get("qualification", {}).get("sha256")
        != qualification["artifact_sha256"]
    ):
        raise RuntimeError("ASFD feature bank belongs to another qualification")
    live_sources = source_manifest()
    if (
        qualification.get("source_sha256") != live_sources
        or bank_payload.get("source_sha256") != live_sources
    ):
        raise RuntimeError("ASFD source changed after qualification or bank creation")
    teacher = _resolve(
        qualification["teacher_checkpoint"]["path"], args.qualification.parent
    )
    recovery = _resolve(
        foundation["foundation"]["recovery"], args.foundation_gate.parent
    )
    cap2_preflight_path = _resolve(
        foundation["inputs"]["preflight"], args.foundation_gate.parent
    )
    cap2_preflight = load_cap2_preflight(cap2_preflight_path)
    if (
        file_sha256(teacher) != qualification["teacher_checkpoint"]["sha256"]
        or bank_payload.get("teacher_checkpoint", {}).get("sha256")
        != qualification["teacher_checkpoint"]["sha256"]
    ):
        raise RuntimeError("qualified teacher and feature-bank teacher differ")
    device = resolve_device(args.device)
    settings = configure(device, allow_tf32=False)
    foundation_runtime_binding = _require_foundation_runtime_and_rate(
        cap2_preflight, settings
    )
    torch.use_deterministic_algorithms(True)
    pool = cifar10_train_pool(args.data_root)

    teacher_payload = torch.load(teacher, map_location="cpu", weights_only=True)
    model_config = teacher_payload["profile"]["model"]
    objective = CAPObjectiveConfig(**teacher_payload["profile"]["objective"])
    train_record = teacher_payload["profile"]["train"]
    online_payload, online_digest = load_recovery_payload(
        recovery, require_sidecar=True, validate_counters=True
    )
    if online_digest != foundation["foundation"]["recovery_sha256"]:
        raise RuntimeError("foundation recovery changed after the 750k gate")
    model = CAPPixelTransformer(CAPModelConfig(**model_config), seed=1).to(device)
    model.load_state_dict(online_payload["model"])
    model.train()

    anchor_config = AnchorConfig()
    scale_indices = torch.randperm(
        len(pool), generator=torch.Generator().manual_seed(PREFLIGHT_SEED)
    )[:256]
    spectral_scale = projected_scale(
        pool[scale_indices].flatten(1),
        anchor_config,
        torch.Generator().manual_seed(PREFLIGHT_SEED + 1),
    )
    calibration_correction = ASFDCorrection(
        teacher_checkpoint=teacher,
        bank_metadata=args.feature_bank,
        qualification=qualification,
        coefficients={"b1": 1.0, "raw": 1.0, "self": 1.0},
        spectral_scale=spectral_scale,
        device=device,
        data_root=args.data_root,
        identity_binding={"purpose": "outcome-blind coefficient calibration"},
        # Calibration must not tune coefficients on the exact correction
        # draws subsequently reused by the production-prefix smoke.
        stream_seed_offset=1_000_000,
    )
    coefficients, calibration_rows = _coefficient_calibration(
        model,
        calibration_correction,
        pool,
        objective,
        micro_batch=int(train_record["micro_batch"]),
        accumulation=int(train_record["accumulation_steps"]),
        device=device,
        events=args.calibration_events,
    )
    del calibration_correction, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    smoke_extension = ASFDCorrection(
        teacher_checkpoint=teacher,
        bank_metadata=args.feature_bank,
        qualification=qualification,
        coefficients=coefficients,
        spectral_scale=spectral_scale,
        device=device,
        data_root=args.data_root,
        # Keep the 500-update run on the *production* 50k progress schedule.
        # Shortening continuation_updates here would compress the spectral
        # curriculum by 100x and cease to test a true production prefix.
        continuation_updates=asfd_config().continuation_updates,
        identity_binding={
            "foundation_gate_sha256": foundation["artifact_sha256"],
            "qualification_sha256": qualification["artifact_sha256"],
            "feature_bank_sha256": bank_payload["artifact_sha256"],
            "mode": "measured-preflight",
        },
    )
    profile = continuation_profile(cap2_preflight, 750_000 + args.smoke_updates)
    fork_path = args.work_dir / "preflight_recovery.pt"
    external = {
        "status": "asfd-preflight-run-identity",
        "foundation_gate_sha256": foundation["artifact_sha256"],
        "source_sha256": source_manifest(),
    }
    fork_foundation_recovery(
        recovery,
        fork_path,
        profile=profile,
        external_identity=external,
        extension=smoke_extension,
        expected_sha256=foundation["foundation"]["recovery_sha256"],
    )
    started = time.time()
    outcome = train_cap_unit(
        pool,
        profile,
        device,
        recovery_path=fork_path,
        recovery_identity=external,
        training_extension=smoke_extension,
    )
    wall = time.time() - started
    events = len(outcome.auxiliary_history)
    projected_hours = (
        wall / args.smoke_updates * asfd_config().continuation_updates / 3600
    )
    hourly_rate = float(cap2_preflight["budget"]["hourly_rate"])
    projected_cost = projected_hours * hourly_rate * 1.15
    foundation_upper = float(cap2_preflight["budget"]["authorized_upper_cost"])
    reserved_asfd = float(
        cap2_preflight["budget"].get("post_foundation_training_reserve", 0.0)
    )
    aggregate_upper = foundation_upper - reserved_asfd + projected_cost
    maximum = float(cap2_preflight["budget"]["max_total_cost"])
    checks = {
        "foundation_and_qualification_bound": True,
        "source_manifest_current": bank_payload.get("source_sha256") == live_sources,
        "calibration_complete": len(calibration_rows) == CALIBRATION_EVENTS,
        "coefficients_positive": all(
            value > 0 and math.isfinite(value) for value in coefficients.values()
        ),
        "smoke_updates_complete": outcome.optimizer_updates == 750_000 + SMOKE_UPDATES,
        "smoke_correction_events_complete": events
        == SMOKE_UPDATES // asfd_config().gradients.cadence,
        "smoke_finite": outcome.nonfinite_updates == 0,
        "one_call_inference": outcome.inference_forward_calls == 1,
        "asfd_cost_within_frozen_reserve": projected_cost <= reserved_asfd,
        "aggregate_budget_within_ceiling": aggregate_upper <= maximum,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "status": STATUS,
        "decision": "GO" if not failed else "NO_GO",
        "failed": failed,
        "checks": checks,
        "foundation_gate": {
            "path": _portable(args.foundation_gate, args.out.parent),
            "sha256": foundation["artifact_sha256"],
        },
        "qualification": {
            "path": _portable(args.qualification, args.out.parent),
            "sha256": qualification["artifact_sha256"],
        },
        "feature_bank": {
            "path": _portable(args.feature_bank, args.out.parent),
            "sha256": bank_payload["artifact_sha256"],
        },
        "teacher_checkpoint": {
            "path": _portable(teacher, args.out.parent),
            "sha256": qualification["teacher_checkpoint"]["sha256"],
        },
        "foundation_recovery": {
            "path": _portable(recovery, args.out.parent),
            "sha256": foundation["foundation"]["recovery_sha256"],
        },
        "coefficients": coefficients,
        "spectral_scale": spectral_scale,
        "calibration": {
            "events": len(calibration_rows),
            "rows": calibration_rows,
            "scale_indices": scale_indices.tolist(),
        },
        "measured_smoke": {
            "updates": args.smoke_updates,
            "correction_events": events,
            "wall_seconds": wall,
            "peak_memory_bytes": outcome.peak_memory_bytes,
        },
        "budget": {
            "foundation_authorized_upper_cost_including_asfd_reserve": foundation_upper,
            "frozen_asfd_training_reserve": reserved_asfd,
            "projected_asfd_cost_with_15pct_contingency": projected_cost,
            "aggregate_upper_cost_after_measured_replacement": aggregate_upper,
            "maximum_total_cost": maximum,
        },
        "device": settings,
        "foundation_runtime_binding": foundation_runtime_binding,
        "source_sha256": live_sources,
    }
    digest = write_json_atomic(args.out, result)
    print(f"wrote {args.out} sha256={digest} decision={result['decision']}")
    return 0 if result["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
