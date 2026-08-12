"""Non-promoting conditioning diagnostic for CAP-EMF-2 numerical candidates.

Production admission (``stage_cap2.numerical_admission``) refuses to run when a
candidate's embedding scale differs from the trained checkpoint it is handed::

    candidate embedding scale differs from the trained checkpoint;
    run a short model trained with that embedding before admission

That guard is correct, and it is why ``smooth_100_d001_fp32`` is registered as
"requires a short trained-model audit ... cannot be certified from the
historical checkpoint alone".  No scale-100 checkpoint exists, so the candidate
cannot be evaluated at all until one is trained.

This module trains that short model.  For each requested candidate it builds the
*production* architecture at that candidate's embedding scale, trains a matched
short run (identical seed, data order and horizon across candidates -- only
``embedding_scale`` and ``delta`` differ), and then measures every audit stratum
with ``numerical_admission.audit_stratum``: the same measurement code the real
gate uses, so the rows are directly comparable to a production admission record.

What this is NOT
----------------
This is a diagnostic, not an admission.  Its output can never promote a
candidate:

* it lives outside ``stage_cap2`` and is absent from the sealed dependency
  manifest, so no preflight, gate or promotion record can be bound to it;
* it writes ``"promoting": false`` and carries no hardware binding, so the rows
  are not production-GPU certified;
* its horizon is far shorter than production, and its warmup is compressed to
  match, so the trained function is only representative enough to answer a
  conditioning question -- not a quality question.

Its single purpose is to decide, before any paid GPU time, whether a candidate
has any prospect of passing admission.  A candidate that fails the interior
strata here will very likely fail them on an A40 too, and the campaign can be
re-planned for the price of local electricity instead of a rented pod.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

# ``production_numerical_mode`` enables deterministic algorithms, which CuBLAS
# refuses to honour unless this is set before CUDA initializes.  The value
# matches ``runpod_pipeline.sh`` so the diagnostic runs in the production
# numerical environment rather than a laxer local one.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402  -- must follow the CuBLAS workspace declaration

from numerics.encoder_independent_drifting.stage_cap.config import (
    profile as cap1_profile,
)
from numerics.encoder_independent_drifting.stage_cap.data import cifar10_train_pool
from numerics.encoder_independent_drifting.stage_cap.model import CAPPixelTransformer
from numerics.encoder_independent_drifting.stage_cap.training import (
    replicated_cap_seed,
    train_cap_unit,
)
from numerics.encoder_independent_drifting.stage_cap2.config import (
    SAMPLER_ARMS,
    numerical_candidate,
)
from numerics.encoder_independent_drifting.stage_cap2.numerical_admission import (
    ASSEMBLED_TARGET_COSINE_MIN,
    ASSEMBLED_TARGET_RELATIVE_RMS_MAX,
    AUDIT_SOURCES,
    AUDIT_STRATA,
    TARGET_COSINE_MIN,
    TARGET_RELATIVE_RMS_MAX,
    audit_stratum,
    production_numerical_mode,
)

# Deliberately distinct from the production audit seed base (20_260_820) so a
# diagnostic row can never be mistaken for, or replayed as, a gate row.
DIAGNOSTIC_SEED_BASE = 20_260_901


def checkpoint_ladder(updates: int, count: int = 4) -> tuple[int, ...]:
    """Evenly spaced measurement points ending at the full horizon.

    A single short-horizon point is weak evidence: the 200-update calibration
    failed a different set of strata than the 650k production checkpoint did,
    so conditioning clearly moves as the function sharpens.  Measuring a ladder
    inside one training run costs nothing extra and shows whether the gap
    between candidates widens or closes with training, which is the part that
    extrapolates.
    """
    if updates < count:
        return (updates,)
    step = updates // count
    return tuple(sorted({step * index for index in range(1, count + 1)} | {updates}))


def diagnostic_profile(arm: str, candidate_name: str, updates: int,
                       checkpoints: tuple[int, ...]):
    """The production architecture at one candidate's scale, on a short horizon.

    ``screen_profile`` pins ``updates`` to the production ladder (50k .. 750k),
    which is right for a screen and unusable for a pre-flight diagnostic.  This
    mirrors its construction against the same ``capability`` base and compresses
    only the schedule -- never the architecture, the sampler or the numerical
    candidate, since those are what is being measured.
    """
    candidate = numerical_candidate(candidate_name)
    base = cap1_profile("capability")
    sampler_mode, sampled_r_floor, coefficient_floor = SAMPLER_ARMS[arm]
    return candidate, replace(
        base,
        name=f"cap2-diagnostic-{arm}-{candidate.name}",
        purpose=(
            "non-promoting numerical conditioning diagnostic; short horizon, "
            "no sealed-test access, never eligible for selection or promotion"
        ),
        model=replace(base.model, scalar_embedding_scale=candidate.embedding_scale),
        objective=replace(
            base.objective,
            sampler_mode=sampler_mode,
            sampled_r_floor=sampled_r_floor,
            coefficient_denominator_floor=coefficient_floor,
            diagonal_sampling="fixed_count_first_draw",
            emf_delta=candidate.delta,
            stopped_evaluation=candidate.stopped_evaluation,
        ),
        train=replace(
            base.train,
            updates=updates,
            checkpoint_updates=checkpoints,
            # A 5,000-update warmup over a 3,000-update diagnostic would never
            # leave the ramp, leaving an essentially untrained function -- the
            # CAP-EMF-1 trap of measuring a finite difference against nothing.
            warmup_updates=max(1, min(base.train.warmup_updates, updates // 10)),
            snapshot_every=max(1, updates),
            log_every=max(1, updates // 10),
            # Health at each measured step: the effective-rank and HH ratios
            # record how non-trivial the function had become when its
            # conditioning was measured.
            health_every=max(1, checkpoints[0]),
            recovery_every=max(1, updates),
        ),
    )


def train_short(prof, pool, device, unit_seed, announce):
    """Train the diagnostic horizon, returning the raw state at each checkpoint.

    ``TrainOutcome`` does not carry the model, so states are taken through the
    checkpoint callback -- the same raw (non-EMA) state the production 50k
    readmission audits (``..._step50000_raw.pt``).  Over a short horizon a
    0.9999 EMA is still dominated by the initialization, so raw weights are the
    only meaningful choice here in any case.

    States are held on CPU: the capability model is ~151 MB, and keeping a
    four-rung ladder resident on a 6 GB card would compete with training.
    """
    captured: dict[int, dict] = {}

    def _capture(step, raw_state, ema_state):
        captured[int(step)] = {
            key: value.detach().to("cpu", copy=True)
            for key, value in raw_state.items()
        }
        return None

    outcome = train_cap_unit(
        pool, prof, device, checkpoint=_capture, unit_seed=unit_seed,
        progress=announce,
    )
    if not captured:
        raise RuntimeError("diagnostic training produced no checkpoint state")
    return captured, outcome


def measure(model, objective, candidate, *, device, batch, repeats, real_pool,
            include_gradient):
    """Every audit stratum x source, using the production gate's own measurer."""
    rows = []
    with production_numerical_mode(device):
        for repeat in range(repeats):
            for source_index, source in enumerate(AUDIT_SOURCES):
                for stratum_index, (name, t, r) in enumerate(AUDIT_STRATA):
                    row = audit_stratum(
                        model,
                        objective,
                        t_value=t,
                        r_value=r,
                        batch=batch,
                        seed=(
                            DIAGNOSTIC_SEED_BASE
                            + 10_000 * repeat
                            + 100 * source_index
                            + stratum_index
                        ),
                        delta=candidate.delta,
                        evaluation_mode=candidate.stopped_evaluation,
                        device=device,
                        include_gradient=include_gradient,
                        source=source,
                        real_pool=real_pool,
                        stratum_name=name,
                    )
                    row["repeat"] = repeat
                    rows.append(row)
    return rows


def summarize(rows):
    """Per-stratum pass rate and worst-case fidelity across sources/repeats."""
    out = {}
    for name, _t, _r in AUDIT_STRATA:
        subset = [row for row in rows if row["stratum"] == name]
        if not subset:
            continue
        out[name] = {
            "passed": sum(1 for row in subset if row["verdict"] == "PASS"),
            "total": len(subset),
            # The gate thresholds the worst row, so the diagnostic reports the
            # worst row too: mean fidelity would hide a single failing source.
            "quotient_cosine_min": min(
                row["quotient"]["cosine_min"] for row in subset
            ),
            "quotient_relative_rms_max": max(
                row["quotient"]["relative_rms_max"] for row in subset
            ),
            "assembled_cosine_min": min(
                row["assembled_target"]["cosine_min"] for row in subset
            ),
            "assembled_relative_rms_max": max(
                row["assembled_target"]["relative_rms_max"] for row in subset
            ),
            "failing_checks": sorted(
                {
                    check
                    for row in subset
                    for check, ok in row["admission_checks"].items()
                    if ok is False
                }
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["local_1000_d0002_fp32", "smooth_100_d001_fp32"],
        help="candidates to compare under an identical training run",
    )
    parser.add_argument("--arm", default="ordered_uniform", choices=tuple(SAMPLER_ARMS))
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument(
        "--rungs",
        type=int,
        default=4,
        help="measurement points spread across the horizon (trend, not a point)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--unit-seed", type=int, default=901)
    # Without gradient metrics every strict check that mentions the gradient is
    # False, so every verdict would read FAIL for a reason unrelated to the
    # candidate.  Gradients are therefore on by default here.
    parser.add_argument(
        "--no-gradient", dest="include_gradient", action="store_false"
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="measure at initialization; a fast smoke path, not evidence",
    )
    parser.add_argument(
        "--save-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "write each rung as a checkpoint numerical_admission can load. "
            "This is the certification path for a candidate whose embedding "
            "scale has no trained model: produce the checkpoint here, then run "
            "the real gate on it, on the production GPU, unmodified."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    real_pool = cifar10_train_pool(args.data_root)
    started = time.time()

    ladder = checkpoint_ladder(args.updates, args.rungs)
    results = {}
    for candidate_name in args.candidates:
        candidate, prof = diagnostic_profile(
            args.arm, candidate_name, args.updates, ladder
        )
        if args.micro_batch is not None:
            prof = replace(
                prof, train=replace(prof.train, micro_batch=args.micro_batch)
            )
        print(
            f"\n=== {candidate.name}: scale={candidate.embedding_scale:g} "
            f"delta={candidate.delta:g} "
            f"max_phase_step={candidate.maximum_phase_step:g} rad ===",
            flush=True,
        )

        train_seconds = 0.0
        health: list[dict] = []
        stability: dict = {}
        model = CAPPixelTransformer(prof.model, seed=1).to(device)
        if args.no_train:
            states = {
                0: CAPPixelTransformer(
                    prof.model, replicated_cap_seed("model-init", args.unit_seed)
                ).state_dict()
            }
        else:
            train_started = time.time()
            states, outcome = train_short(
                prof,
                real_pool,
                device,
                args.unit_seed,
                lambda line: print(f"  {line}", flush=True),
            )
            train_seconds = time.time() - train_started
            # A candidate that passes the numerical gate and then diverges is
            # not a viable foundation, so training stability is recorded beside
            # the conditioning it is supposed to explain.
            health = outcome.health
            moments = [record.get("second_moment_ratio", 0.0) for record in health]
            best_moment = max(moments) if moments else 0.0
            final_moment = moments[-1] if moments else 0.0
            # Collapse is judged on second-moment ratio only.  Every health
            # ratio targets 1.0 with a two-sided band, and at a 2,000-update
            # horizon the rank and Haar-HH ratios sit in the *tens* -- far above
            # their 1.50/1.75 ceilings -- so those falling is convergence toward
            # real statistics, not deterioration.  Reading a falling HH as
            # collapse flags a healthy run as a failure.  The protocol's own
            # H3b rank-retention rule is likewise inapplicable here: it assumes
            # a rank ratio near 1 and penalizes any drop.
            stability = {
                "clipped_updates": outcome.clipped_updates,
                "clipped_fraction": (
                    outcome.clipped_updates / max(1, outcome.optimizer_updates)
                ),
                # H7 thresholds a *windowed* clip fraction, not the cumulative
                # one: clipping is worst early, so a cumulative figure over a
                # long run dilutes a broken tail and a short run exaggerates a
                # normal warmup.  This is the number the gate actually applies.
                "clipped_updates_final_window": outcome.clipped_updates_final_window,
                "final_window_updates": outcome.final_window_updates,
                "clipped_fraction_final_window": (
                    outcome.clipped_updates_final_window
                    / max(1, outcome.final_window_updates)
                ),
                "h7_clip_fraction_max": 0.05,
                "optimizer_updates": outcome.optimizer_updates,
                "nonfinite_updates": outcome.nonfinite_updates,
                "best_rank_ratio": outcome.best_rank_ratio,
                "best_second_moment": best_moment,
                "final_second_moment": final_moment,
                "gate_second_moment_floor": prof.gate.second_moment_ratio,
                "final_haar_HH": health[-1].get("haar_HH_ratio") if health else None,
                "collapsed": bool(
                    health
                    and final_moment < 0.6 * best_moment
                    and final_moment < prof.gate.second_moment_ratio
                ),
            }

        by_step = {}
        saved = {}
        for step in sorted(states):
            model.load_state_dict(states[step])
            model.eval()
            if args.save_checkpoint_dir is not None:
                # The layout ``numerical_admission._load_checkpoint`` expects:
                # raw state plus the profile that produced it, so the gate can
                # verify the embedding scale it is being asked to audit.
                args.save_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                path = (
                    args.save_checkpoint_dir
                    / f"cap2_{args.arm}_{candidate.name}_step{step}_raw.pt"
                )
                torch.save(
                    {
                        "state_dict": states[step],
                        "profile": {
                            "model": asdict(prof.model),
                            "objective": asdict(prof.objective),
                        },
                        "step": step,
                        "arm": args.arm,
                        "candidate": candidate.name,
                        "unit_seed": args.unit_seed,
                        "provenance": (
                            "cap2_candidate_diagnostic short trained-model "
                            "audit; not a screen unit and not selectable"
                        ),
                    },
                    path,
                )
                saved[str(step)] = str(path)
                print(f"  saved {path.name}", flush=True)
            rows = measure(
                model,
                prof.objective,
                candidate,
                device=device,
                batch=args.batch,
                repeats=args.repeats,
                real_pool=real_pool,
                include_gradient=args.include_gradient,
            )
            summary = summarize(rows)
            by_step[str(step)] = {"summary": summary, "rows": rows}
            print(f"  -- step {step} --", flush=True)
            for name, stats in summary.items():
                flag = "ok  " if stats["passed"] == stats["total"] else "FAIL"
                print(
                    f"    {flag} {name:22s} {stats['passed']}/{stats['total']}  "
                    f"q_cos>={stats['quotient_cosine_min']:.4f}  "
                    f"q_rms<={stats['quotient_relative_rms_max']:.4f}  "
                    f"a_cos>={stats['assembled_cosine_min']:.5f}",
                    flush=True,
                )

        results[candidate.name] = {
            "candidate": asdict(candidate),
            "profile_name": prof.name,
            "updates": 0 if args.no_train else args.updates,
            "checkpoints": sorted(states),
            "saved_checkpoints": saved,
            "train_seconds": round(train_seconds, 1),
            "stability": stability,
            "health": health,
            "by_step": by_step,
        }
        if stability.get("collapsed"):
            print(
                f"  WARNING: {candidate.name} training collapsed (second moment "
                f"{stability['final_second_moment']:.3f} from a best of "
                f"{stability['best_second_moment']:.3f}, gate floor "
                f"{stability['gate_second_moment_floor']:.2f}); conditioning at "
                f"the last rung describes a degenerate model",
                flush=True,
            )

    payload = {
        "kind": "cap2_candidate_conditioning_diagnostic",
        "promoting": False,
        "admission": False,
        "not_evidence_for": (
            "candidate promotion, preflight, early admission or any gate; this "
            "record carries no hardware binding and no sealed source manifest"
        ),
        "arm": args.arm,
        "updates": 0 if args.no_train else args.updates,
        "checkpoint_ladder": list(ladder),
        "trained": not args.no_train,
        "unit_seed": args.unit_seed,
        "batch": args.batch,
        "repeats": args.repeats,
        "include_gradient": bool(args.include_gradient),
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "host": platform.node(),
        "torch": torch.__version__,
        "thresholds": {
            "quotient_cosine_min": TARGET_COSINE_MIN,
            "quotient_relative_rms_max": TARGET_RELATIVE_RMS_MAX,
            "assembled_target_cosine_min": ASSEMBLED_TARGET_COSINE_MIN,
            "assembled_target_relative_rms_max": ASSEMBLED_TARGET_RELATIVE_RMS_MAX,
        },
        "wall_seconds": round(time.time() - started, 1),
        "candidates": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    # The trend is the deliverable.  A candidate that is merely passing at the
    # shortest rung proves little; one whose margin holds or improves as the
    # function sharpens is the one worth paying an A40 to certify.
    for stratum in ("interior_high", "large_coefficient", "exact_inference"):
        print(f"\n=== {stratum} across training ===")
        for name, record in results.items():
            cells = []
            for step in record["checkpoints"]:
                stats = record["by_step"][str(step)]["summary"].get(stratum)
                if stats is None:
                    continue
                mark = "ok" if stats["passed"] == stats["total"] else "FAIL"
                cells.append(
                    f"{step}:{mark} q_cos={stats['quotient_cosine_min']:.4f}"
                )
            print(f"  {name:24s} " + "  |  ".join(cells))

    print("\n=== overall pass rate per candidate at the final rung ===")
    for name, record in results.items():
        final = record["by_step"][str(record["checkpoints"][-1])]["summary"]
        passed = sum(stats["passed"] for stats in final.values())
        total = sum(stats["total"] for stats in final.values())
        failing = sorted(
            {
                name_
                for name_, stats in final.items()
                if stats["passed"] != stats["total"]
            }
        )
        print(f"  {name:24s} {passed}/{total} rows; failing strata: {failing}")


if __name__ == "__main__":
    main()
