"""Secondary, exploratory: synthesize longer EMA horizons from raw snapshots.

The declared 0.9999 EMA has a ~10,000-update window, about 1.3% of a 750,000
update run, and the right horizon is not knowable in advance.  In CAP-EMF-1
averaging the trailing eight snapshots into a 200,000-update window moved
FID-50k from 112.94 to 83.65 -- a 26% improvement for no additional GPU time --
and the snapshots were then lost with the Pod, so the finding could never be
reproduced or extended.  Here the snapshots are durably mirrored and this module
turns them into a result.

**This is secondary by construction and may never select anything.**

Three properties enforce that, because a longer-EMA sweep is exactly the shape
of an accidental post-hoc selection on the primary metric:

* the windows are **predeclared** in :data:`DECLARED_WINDOWS` and every declared
  window is evaluated and reported.  There is no "best window" search, and the
  CLI cannot request a different set;
* metrics are computed against the **training** reference, the same reference
  the protocol already uses for its headline FID.  The sealed CIFAR-10 test
  split is never opened here -- this module has no code path that can open it;
* the artifact records ``eligible_for_selection: False`` and the primary
  checkpoint it must not displace, so a reader cannot mistake a better number
  here for the declared result.

The synthesis is a uniform average over a trailing window of raw snapshots,
which is the construction that produced the CAP-EMF-1 gain.  It is a boxcar
rather than Karras et al.'s power-function profile: with snapshots only every
25,000 updates the extra fidelity of a fitted profile is not resolvable, and a
uniform mean has no free parameters to tune.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from ..diagnostics import write_json
from .artifacts import file_sha256, load_snapshot, verify_json
from .development_evaluation import (
    VerifiedEMA,
    evaluate,
    portable_reference,
    verify_final_ema,
)
from .promotion import FID_KEY, KID_KEY

#: Trailing snapshot counts.  At the production 25,000-update cadence these are
#: 100k, 200k and 400k update windows against the declared EMA's ~10k.  200k is
#: the CAP-EMF-1 window that produced the measured 26% gain; the neighbours
#: bracket it so the reported curve shows whether the effect is a plateau or a
#: peak.  Predeclared: every one is reported, none is selected.
DECLARED_WINDOWS: tuple[int, ...] = (4, 8, 16)


@dataclass(frozen=True)
class SynthesizedEMA:
    window: int
    first_step: int
    last_step: int
    update_span: int
    snapshot_count: int
    path: Path
    sha256: str


def _verified_snapshots(unit: dict, unit_path: Path) -> list[tuple[int, Path, str]]:
    """Snapshot ledger from the sealed unit, integrity-checked against it."""
    ledger = unit.get("raw_snapshots")
    if not isinstance(ledger, list) or not ledger:
        raise RuntimeError("CAP2 unit records no raw snapshots")
    anchor = unit_path.parent
    entries: list[tuple[int, Path, str]] = []
    for record in ledger:
        if not isinstance(record, dict):
            raise TypeError("malformed CAP2 snapshot ledger entry")
        step = int(record["step"])
        path = (anchor / str(record["path"])).resolve()
        recorded = str(record["sha256"])
        if not path.is_file():
            raise RuntimeError(f"snapshot missing for step {step}: {path}")
        actual = file_sha256(path)
        if actual != recorded:
            raise RuntimeError(
                f"snapshot for step {step} changed after the unit was sealed: "
                f"{actual} != {recorded}"
            )
        entries.append((step, path, recorded))
    entries.sort(key=lambda item: item[0])
    return entries


def synthesize(
    unit: dict,
    unit_path: Path,
    window: int,
    output_dir: Path,
) -> SynthesizedEMA:
    """Uniform average of the trailing ``window`` raw snapshots.

    Accumulates in float64 and casts once at the end: averaging 16 float32
    tensors in float32 loses low-order bits exactly where a long average is
    supposed to be suppressing noise.
    """
    entries = _verified_snapshots(unit, unit_path)
    if window < 2:
        raise ValueError("a post-hoc window needs at least two snapshots")
    if window > len(entries):
        raise ValueError(
            f"window {window} exceeds the {len(entries)} available snapshots"
        )
    chosen = entries[-window:]

    arm = unit["arm"]
    declared = unit["declared_profile"]
    accumulator: dict[str, torch.Tensor] | None = None
    for step, path, recorded in chosen:
        payload = load_snapshot(
            path,
            expected_sha=recorded,
            step=step,
            arm=arm,
            preflight_sha256=unit.get("preflight_sha256"),
            run_identity_sha256=unit.get("run_identity_sha256"),
            unit_seed=unit.get("unit_seed"),
        )
        state = payload["state_dict"]
        if accumulator is None:
            accumulator = {name: value.double().clone() for name, value in state.items()}
        else:
            if set(accumulator) != set(state):
                raise RuntimeError(f"snapshot {step} has a different parameter set")
            for name, value in state.items():
                accumulator[name] += value.double()
    assert accumulator is not None
    averaged = {name: (value / window).float() for name, value in accumulator.items()}

    first_step, last_step = chosen[0][0], chosen[-1][0]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cap2_{arm}_posthoc_ema_w{window}_step{last_step}.pt"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite a synthesized average: {path}")
    torch.save(
        {
            "stage": "cap-emf-2-posthoc-ema",
            "step": int(last_step),
            "kind": "posthoc-ema",
            "arm": arm,
            # `profile` is the key `_load_model` reads; keeping the declared
            # profile means the averaged weights load through exactly the same
            # path as a real checkpoint.
            "profile": declared,
            "declared_profile": declared,
            "window_snapshots": window,
            "window_first_step": int(first_step),
            "window_last_step": int(last_step),
            "source_snapshot_sha256": [record for _, _, record in chosen],
            "secondary_exploratory": True,
            "eligible_for_selection": False,
            "state_dict": averaged,
        },
        path,
    )
    return SynthesizedEMA(
        window=window,
        first_step=first_step,
        last_step=last_step,
        update_span=last_step - first_step,
        snapshot_count=len(entries),
        path=path,
        sha256=file_sha256(path),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--kid-reference-features", type=Path, required=True)
    parser.add_argument("--generation-batch", type=int, default=500)
    parser.add_argument("--metric-batch", type=int, default=128)
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument("--feature-batch", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists() or Path(f"{args.out}.sha256").exists():
        raise RuntimeError(f"refusing to overwrite a consumed artifact: {args.out}")

    unit = verify_json(args.unit, "cap-emf2-screen-unit")
    primary = verify_final_ema(args.unit)
    device = torch.device(args.device)
    anchor = args.out.parent

    available = len(_verified_snapshots(unit, args.unit))
    # A budget stop truncates the run and therefore the snapshot ledger, and
    # CAP-EMF-1 ended on exactly such a stop. Skipping an unaffordable window
    # and recording it is more useful than aborting every window because the
    # widest one does not fit.
    skipped = [window for window in DECLARED_WINDOWS if window > available]
    usable = [window for window in DECLARED_WINDOWS if window <= available]
    if not usable:
        raise RuntimeError(
            f"only {available} snapshots exist; no declared window fits"
        )

    windows: list[dict] = []
    for window in usable:
        synthesized = synthesize(unit, args.unit, window, args.work_dir / "weights")
        work = args.work_dir / f"w{window}"
        result = evaluate(
            # The averaged weights are evaluated through the *same* function as
            # the primary checkpoint, so nothing about the metric path differs.
            VerifiedEMA(
                unit=primary.unit,
                unit_path=primary.unit_path,
                checkpoint={"path": str(synthesized.path)},
                checkpoint_path=synthesized.path,
                step=synthesized.last_step,
            ),
            device=device,
            data_root=args.data_root,
            png_dir=work / "png",
            grid_path=work / "grid.png",
            evaluation_anchor=anchor,
            generation_batch=args.generation_batch,
            metric_batch=args.metric_batch,
            metric_workers=args.metric_workers,
            kid_reference_features=args.kid_reference_features,
            generated_feature_path=work / "features.npz",
            feature_batch=args.feature_batch,
        )
        windows.append(
            {
                "window_snapshots": synthesized.window,
                "window_first_step": synthesized.first_step,
                "window_last_step": synthesized.last_step,
                "window_update_span": synthesized.update_span,
                "weights": portable_reference(synthesized.path, anchor),
                "weights_sha256": synthesized.sha256,
                "evaluation": result,
            }
        )

    payload = {
        "status": "cap-emf2-posthoc-ema",
        "development_only": True,
        "secondary_exploratory": True,
        "eligible_for_selection": False,
        "primary_checkpoint_step": primary.step,
        "primary_unit": portable_reference(args.unit, anchor),
        "declared_windows": list(DECLARED_WINDOWS),
        "available_snapshots": available,
        "skipped_windows": skipped,
        "windows": windows,
        "reference": "training split; the sealed test split is not opened here",
        "limits": [
            "Secondary and exploratory. The declared 0.9999 EMA at the primary "
            "checkpoint remains the result of this experiment.",
            "Every declared window is reported. No window was selected by its "
            "metrics, and this artifact may not be used to choose one.",
            "A uniform trailing average, not a fitted EMA profile: at a 25,000 "
            "update snapshot cadence a fitted profile is not resolvable.",
        ],
    }
    digest = write_json(args.out, payload)
    for record in windows:
        # The metric keys are the clean-fid train-reference names the gate uses,
        # not bare "fid"/"kid" -- reading the wrong key would silently print nan
        # and look like a computation that ran.
        standard = record["evaluation"]["standard_train_reference_metrics"]
        print(
            f"window {record['window_snapshots']:>3} snapshots "
            f"({record['window_update_span']:,} updates): "
            f"FID {float(standard[FID_KEY]):.3f}  "
            f"KID {float(standard[KID_KEY]):.5f}"
        )
    print(f"wrote {args.out} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
