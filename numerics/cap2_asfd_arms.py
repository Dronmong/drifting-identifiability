"""Run the three ASFD arms as continuations of an ungated foundation.

``stage_asfd.continuation`` is the gated path: it verifies a 750k foundation
gate, an ASFD preflight and their SHA bindings, and it only ever runs the full
``EMF-ASFD`` arm. This runs the same correction machinery against a foundation
that no gate vouches for, and it drives all three declared arms.

**The control arm is the point.** ``ARMS`` declares control / raw / ASFD but
nothing in the package ever runs the first two, so a naive comparison would put
a corrected continuation against an uncorrected *foundation* -- confounding the
correction with 50,000 extra updates. Every arm here continues the same
recovery for the same number of updates from the same optimizer, EMA and RNG
state, so the only difference between them is which correction gradients are
added.

What this is not: a gated result. The qualification it consumes carries
``teacher_provenance`` saying so, and nothing here writes an artifact any
sealed consumer will accept. It is evidence for a research question, which is
what it was asked to be.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from numerics.cap2_candidate_diagnostic import (  # noqa: E402
    checkpoint_ladder,
    diagnostic_profile,
)
from numerics.encoder_independent_drifting.stage_cap.data import (  # noqa: E402
    cifar10_train_pool,
)
from numerics.encoder_independent_drifting.stage_cap.model import (  # noqa: E402
    CAPPixelTransformer,
)
from numerics.encoder_independent_drifting.stage_cap.training import (  # noqa: E402
    train_cap_unit,
)
from numerics.encoder_independent_drifting.stage_asfd.correction import (  # noqa: E402
    ASFDCorrection,
)
from numerics.encoder_independent_drifting.stage_asfd.config import (  # noqa: E402
    asfd_config,
)
from numerics.encoder_independent_drifting.stage_cap2.artifacts import (  # noqa: E402
    verify_json,
)
from numerics.encoder_independent_drifting.stage_asfd.recovery import (  # noqa: E402
    fork_foundation_recovery,
)
from numerics.encoder_independent_drifting.stage_asfd.artifacts import (  # noqa: E402
    file_sha256,
)
from numerics.encoder_independent_drifting.spectral_anchor import (  # noqa: E402
    projected_scale,
)
from numerics.encoder_independent_drifting.config import AnchorConfig  # noqa: E402

#: One seed for the whole comparison. Every arm shares it, so the arms differ
#: only in their caps.
ARMS_SEED = 20_260_921

#: Arms and how each one weights the three correction components. ``None``
#: means no correction at all -- the control that isolates the correction from
#: the extra updates it rides along with.
#:
#: The raw arm disables the self-feature branch by capping it at a negligible
#: fraction rather than at zero, because a zero cap is rejected by
#: ``GradientConfig.validate``. The component is still computed, so the arms
#: stay on one code path and the cost is identical; only its contribution is
#: suppressed. That is the whole claim under test: raw-pixel similarity versus
#: the same field read in learned self-features.
ARM_CAPS: dict[str, dict[str, float] | None] = {
    "control": None,
    "raw": {"b1": 0.15, "raw": 0.10, "self": 1e-6},
    "asfd": {"b1": 0.15, "raw": 0.10, "self": 0.10},
}


def foundation_profile(step: int, extra_updates: int):
    """The 650k run's exact profile, extended only in its planned horizon.

    ``_recovery_identity`` pops ``updates`` and ``checkpoint_updates`` and keeps
    everything else, so the profile must be rebuilt at the *original* horizon
    and then extended -- rebuilding it at the new horizon would change
    ``snapshot_every`` and ``log_every`` and the resume would be refused.
    """
    ladder = checkpoint_ladder(step, 13)
    _candidate, prof = diagnostic_profile(
        "ordered_uniform", "smooth_100_d001_fp32", step, ladder,
        loss_weight_floor=1.0, gradient_clip=15.0,
    )
    final = step + extra_updates
    return replace(
        prof,
        train=replace(prof.train, updates=final, checkpoint_updates=(final,)),
    )


def extract_teacher(recovery: Path, out: Path) -> int:
    """Write the foundation's EMA weights as the frozen descriptor teacher."""
    payload = torch.load(recovery, map_location="cpu", weights_only=True)
    step = int(payload["completed_updates"])
    prof = foundation_profile(step, 0)
    # EMAState.state_dict() is shadow | buffers, and the recovery payload keeps
    # the two apart. Taking shadow alone silently drops every buffer and the
    # load then fails on missing keys.
    ema = payload["ema"]
    if "shadow" not in ema:
        raise SystemExit("recovery EMA payload has no shadow weights")
    state = {name: value.clone() for name, value in ema["shadow"].items()}
    state.update(
        {name: value.clone() for name, value in ema.get("buffers", {}).items()}
    )
    # Prove the weights load into the declared architecture before writing.
    CAPPixelTransformer(prof.model, seed=1).load_state_dict(state)
    torch.save(
        {
            "stage": "cap-emf-2-candidate-audit",
            "kind": "ema",
            "step": step,
            "state_dict": state,
            "profile": {
                "model": asdict(prof.model),
                "objective": asdict(prof.objective),
            },
            "provenance": (
                "EMA weights read from an ungated foundation recovery; no "
                "foundation gate vouches for this teacher"
            ),
        },
        out,
    )
    print(f"teacher written at step {step}: {out}")
    return step


def calibrate(correction: ASFDCorrection, model, pool, objective, *, device,
              micro_batch, accumulation, events, step_base):
    """Coefficients that put each component at its cap, as preflight does.

    ``step_base`` must be this foundation's step. The band schedule reads
    progress from the step, so calibrating anywhere but the start of the
    schedule measures the anchor against a bank the early run never sees.
    """
    from numerics.encoder_independent_drifting.stage_asfd.preflight import (
        _coefficient_calibration,
    )

    return _coefficient_calibration(
        model, correction, pool, objective,
        micro_batch=micro_batch, accumulation=accumulation,
        device=device, events=events, step_base=step_base,
    )


def build_correction(*, teacher: Path, bank: Path, qualification: dict,
                     caps: dict[str, float], coefficients: dict[str, float],
                     spectral_scale: float, device, data_root, start: int,
                     updates: int, seed_offset: int) -> ASFDCorrection:
    config = asfd_config()
    config = replace(
        config,
        gradients=replace(
            config.gradients,
            cap_b1=caps["b1"], cap_raw=caps["raw"], cap_self=caps["self"],
        ),
    )
    return ASFDCorrection(
        teacher_checkpoint=teacher,
        bank_metadata=bank,
        qualification=qualification,
        coefficients=coefficients,
        spectral_scale=spectral_scale,
        device=device,
        data_root=data_root,
        continuation_start=start,
        continuation_updates=updates,
        config=config,
        stream_seed_offset=seed_offset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-recovery", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--extract-teacher", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--feature-bank", type=Path)
    parser.add_argument("--arms", nargs="+", default=["control", "raw", "asfd"],
                        choices=tuple(ARM_CAPS))
    parser.add_argument("--updates", type=int, default=50_000)
    parser.add_argument("--calibration-events", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.extract_teacher:
        extract_teacher(args.foundation_recovery, args.teacher)
        return

    device = torch.device(args.device)
    start = int(
        torch.load(args.foundation_recovery, map_location="cpu", weights_only=True)[
            "completed_updates"
        ]
    )
    prof = foundation_profile(start, args.updates)
    pool = cifar10_train_pool(args.data_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"continuing {start} -> {start + args.updates}, arms={args.arms}")

    needs_correction = any(ARM_CAPS[a] is not None for a in args.arms)
    if needs_correction and (args.qualification is None or args.feature_bank is None):
        raise SystemExit(
            "--qualification and --feature-bank are required for the raw and "
            "asfd arms; only the control arm runs without them"
        )
    # verify_json, not json.loads: it checks the file against its .sha256
    # sidecar and attaches ``artifact_sha256``, which ASFDCorrection compares
    # against the bank's recorded binding. Reading the JSON directly skips both
    # the verification and the field, and the binding check then fails on a
    # None it was never given.
    qualification = (
        verify_json(args.qualification, "asfd-target-only-qualification")
        if needs_correction
        else {}
    )
    coefficients = None
    spectral_scale = None
    if needs_correction:
        # AnchorConfig is not a field of ASFDConfig -- ASFDCorrection builds its
        # own instance -- so it must be constructed here too, and the sampling
        # mirrors the gated preflight: a random 256-image subset, not a prefix.
        anchor_config = AnchorConfig()
        scale_indices = torch.randperm(
            len(pool), generator=torch.Generator().manual_seed(ARMS_SEED)
        )[:256]
        # Left on CPU: sample_directions draws from a CPU generator, so the
        # probe directions are CPU tensors and a CUDA target would mismatch in
        # the projection matmul. The gated preflight passes CPU here too.
        spectral_scale = projected_scale(
            pool[scale_indices].flatten(1),
            anchor_config,
            torch.Generator().manual_seed(ARMS_SEED + 1),
        )
        print(f"spectral scale (target-only): {spectral_scale:.6f}")
        probe_model = CAPPixelTransformer(prof.model, seed=1).to(device)
        probe_model.load_state_dict(
            torch.load(args.foundation_recovery, map_location="cpu",
                       weights_only=True)["model"]
        )
        probe = build_correction(
            teacher=args.teacher, bank=args.feature_bank,
            qualification=qualification, caps=ARM_CAPS["asfd"],
            coefficients={"b1": 1.0, "raw": 1.0, "self": 1.0},
            spectral_scale=spectral_scale, device=device,
            data_root=args.data_root, start=start, updates=args.updates,
            seed_offset=0,
        )
        coefficients, rows = calibrate(
            probe, probe_model, pool, prof.objective, device=device,
            micro_batch=prof.train.micro_batch,
            accumulation=prof.train.accumulation_steps,
            events=args.calibration_events, step_base=start,
        )
        print("calibrated coefficients:", coefficients)
        (args.out_dir / "calibration.json").write_text(
            json.dumps({"coefficients": coefficients,
                        "spectral_scale": spectral_scale,
                        "events": rows}, indent=2, default=float)
        )
        del probe, probe_model
        torch.cuda.empty_cache()

    summary = {}
    for arm in args.arms:
        arm_dir = args.out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        recovery = arm_dir / "recovery.pt"
        caps = ARM_CAPS[arm]
        extension = None
        if caps is not None:
            # Every arm uses the SAME stream seed. stream_seed_offset seeds the
            # spectral bank and every correction RNG stream, so varying it per
            # arm would make raw and asfd differ in their random draws as well
            # as their caps -- and any difference between them could no longer
            # be attributed to the self-feature branch, which is the only thing
            # the pair exists to measure.
            extension = build_correction(
                teacher=args.teacher, bank=args.feature_bank,
                qualification=qualification, caps=caps,
                coefficients=coefficients, spectral_scale=spectral_scale,
                device=device, data_root=args.data_root, start=start,
                updates=args.updates, seed_offset=0,
            )

        if not recovery.exists():
            if extension is None:
                # No extension: a plain copy is right, and a fork would inject a
                # replay state that train_cap_unit then refuses as unrequested.
                shutil.copy2(args.foundation_recovery, recovery)
                sidecar = args.foundation_recovery.with_suffix(
                    args.foundation_recovery.suffix + ".sha256"
                )
                if sidecar.exists():
                    shutil.copy2(
                        sidecar, recovery.with_suffix(recovery.suffix + ".sha256")
                    )
            else:
                # Attaching a correction at a resume boundary needs the
                # extension's replay state already in the recovery -- a plain
                # copy fails with "extended recovery lacks its replay state".
                # fork_foundation_recovery is the audited way to do it: all
                # scientific state is retained and only the continuation
                # identity and the freshly declared extension state change.
                fork_foundation_recovery(
                    args.foundation_recovery,
                    recovery,
                    profile=prof,
                    external_identity=None,
                    extension=extension,
                    expected_sha256=file_sha256(args.foundation_recovery),
                    unit_seed=901,
                    foundation_step=start,
                )
        print(f"\n=== arm {arm}: caps={caps} ===", flush=True)
        captured: dict[int, dict] = {}
        started = time.time()
        outcome = train_cap_unit(
            pool, prof, device,
            recovery_path=recovery,
            checkpoint=lambda step, raw, ema: captured.__setitem__(
                int(step), {k: v.detach().to("cpu", copy=True) for k, v in raw.items()}
            ),
            training_extension=extension,
            unit_seed=901,
            progress=lambda line: print(f"  {line}", flush=True),
        )
        elapsed = time.time() - started
        for step, state in captured.items():
            torch.save(
                {"stage": "cap-emf-2-candidate-audit", "kind": "raw", "step": step,
                 "state_dict": state,
                 "profile": {"model": asdict(prof.model),
                             "objective": asdict(prof.objective)},
                 "arm": arm,
                 "provenance": "ungated ASFD arm continuation"},
                arm_dir / f"{arm}_step{step}_raw.pt",
            )
        summary[arm] = {
            "caps": caps,
            "wall_seconds": round(elapsed, 1),
            "clipped_updates": outcome.clipped_updates,
            "nonfinite_updates": outcome.nonfinite_updates,
            "checkpoints": sorted(captured),
            "health": outcome.health[-3:],
            "auxiliary": outcome.auxiliary_history[-3:],
        }
        print(f"  arm {arm} done in {elapsed/3600:.2f} h", flush=True)
        (args.out_dir / "arms.json").write_text(
            json.dumps({"start": start, "updates": args.updates,
                        "coefficients": coefficients,
                        "spectral_scale": spectral_scale,
                        "arms": summary}, indent=2, default=float)
        )

    print("\n=== all arms complete ===")


if __name__ == "__main__":
    main()
