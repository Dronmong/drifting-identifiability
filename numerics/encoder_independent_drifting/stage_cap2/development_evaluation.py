"""Train-only, fixed-protocol development evaluation for a CAP2 unit.

By default the only accepted model is the *final recorded EMA* of an immutable
CAP2 unit result.  A predeclared ``--step`` may select an earlier EMA only when
that exact step belongs to the result's immutable checkpoint ledger.  This is
used by the 750k foundation protocol to compare its prospectively declared
650k and 750k checkpoints; arbitrary, raw, hand-picked, and unrecorded states
still have no command-line path into this evaluator.

Standard CleanFID uses its published CIFAR-10 train moments; CleanKID uses a
hash-sealed clean-Inception extraction of all 50,000 training images.  Both use
50,000 fixed-noise one-call model samples.  Smaller, explicitly scoped subsets are
used for repository-backend precision/recall and pixel-space memorization so
the audit remains practical without opening the CIFAR-10 test split.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from ..device import configure, resolve_device
from ..diagnostics import ROOT
from .artifacts import (
    assert_unused,
    file_sha256,
    load_checkpoint,
    source_manifest,
    verify_json,
    write_json_atomic,
)
from .standard_metrics import (
    DEFAULT_GENERATION_SEED,
    DEFAULT_KID_SEED,
    DEFAULT_METRIC_WORKERS,
    DEFAULT_SAMPLE_COUNT,
    REPOSITORY_FEATURE_SAMPLES,
    REPOSITORY_MEMORIZATION_SAMPLES,
    REPOSITORY_REFERENCE_SEED,
    _load_model,
    compute_clean_metrics,
    compute_repository_auxiliary_metrics,
    evaluation_provenance,
    generate_png_folder,
)

DEVELOPMENT_SAMPLES = DEFAULT_SAMPLE_COUNT
GENERATION_SEED = DEFAULT_GENERATION_SEED
REFERENCE_SEED = REPOSITORY_REFERENCE_SEED
MANIFOLD_SAMPLES = REPOSITORY_FEATURE_SAMPLES
MEMORIZATION_SAMPLES = REPOSITORY_MEMORIZATION_SAMPLES
GRID_ROWS = 8
GRID_COLUMNS = 16


@dataclass(frozen=True)
class VerifiedEMA:
    unit: dict
    unit_path: Path
    checkpoint: dict
    checkpoint_path: Path
    step: int


def portable_reference(path: Path, anchor: Path) -> str:
    """Record an artifact relative to the evaluation JSON that binds it."""

    return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()


def verify_final_ema(unit_path: Path, *, step: int | None = None) -> VerifiedEMA:
    """Bind a unit result to its final or explicitly recorded EMA checkpoint."""
    unit = verify_json(unit_path, "cap-emf2-screen-unit")
    if unit.get("development_only") is not True:
        raise RuntimeError("CAP2 unit is not marked development-only")
    arm = unit.get("arm")
    profile = unit.get("declared_profile")
    realized_profile = unit.get("realized_profile")
    run_identity = unit.get("run_identity_sha256")
    unit_seed = unit.get("unit_seed")
    if not isinstance(arm, str) or not arm:
        raise RuntimeError("CAP2 unit has no arm identity")
    if not isinstance(profile, dict) or not isinstance(profile.get("train"), dict):
        raise TypeError("CAP2 unit lacks a declared training profile")
    if not isinstance(realized_profile, dict):
        raise TypeError("CAP2 unit lacks its realized training profile")
    if not isinstance(run_identity, str) or not run_identity:
        raise RuntimeError("CAP2 unit lacks its immutable run identity")
    if not isinstance(unit_seed, int):
        raise TypeError("CAP2 unit lacks its integer unit seed")
    final_step = int(profile["train"].get("updates", 0))
    if final_step <= 0:
        raise RuntimeError("CAP2 unit has no positive declared final step")
    checkpoint_steps = tuple(profile["train"].get("checkpoint_updates", ()))
    if not checkpoint_steps or int(checkpoint_steps[-1]) != final_step:
        raise RuntimeError("declared final step is not the final checkpoint step")
    training = unit.get("training", {})
    if int(training.get("optimizer_updates", -1)) != final_step:
        raise RuntimeError("screen unit did not complete its declared final step")
    requested_step = final_step if step is None else int(step)
    if requested_step <= 0 or requested_step not in {
        int(value) for value in checkpoint_steps
    }:
        raise RuntimeError("requested EMA step is not a declared checkpoint")

    checkpoints = unit.get("checkpoints")
    if not isinstance(checkpoints, dict) or str(requested_step) not in checkpoints:
        raise RuntimeError("screen unit does not record the requested checkpoint")
    try:
        recorded_steps = tuple(int(value) for value in checkpoints)
    except (TypeError, ValueError) as error:
        raise RuntimeError("screen unit has a malformed checkpoint step") from error
    if not recorded_steps or max(recorded_steps) != final_step:
        raise RuntimeError("declared final checkpoint is not the last recorded step")
    selected_record = checkpoints[str(requested_step)]
    if not isinstance(selected_record, dict) or "ema" not in selected_record:
        raise RuntimeError("screen unit does not record the requested EMA checkpoint")
    ema_record = selected_record["ema"]
    if not isinstance(ema_record, dict):
        raise TypeError("requested EMA checkpoint record is malformed")
    recorded_path = Path(ema_record.get("path", ""))
    if not recorded_path.is_absolute():
        recorded_path = unit_path.parent / recorded_path
    recorded_path = recorded_path.resolve()
    if not recorded_path.is_file():
        raise RuntimeError(f"recorded EMA checkpoint is missing: {recorded_path}")
    recorded_hash = ema_record.get("sha256")
    actual_hash = file_sha256(recorded_path)
    if not isinstance(recorded_hash, str) or recorded_hash != actual_hash:
        raise RuntimeError("EMA checkpoint hash does not match the unit result")

    checkpoint = load_checkpoint(
        recorded_path,
        expected_sha=recorded_hash,
        step=requested_step,
        kind="ema",
        arm=arm,
        declared_profile=profile,
        realized_profile=realized_profile,
        preflight_sha256=unit.get("preflight_sha256"),
        run_identity_sha256=run_identity,
        unit_seed=unit_seed,
    )
    return VerifiedEMA(
        unit=unit,
        unit_path=unit_path.resolve(),
        checkpoint=checkpoint,
        checkpoint_path=recorded_path,
        step=requested_step,
    )


def write_uncurated_grid(
    png_folder: Path,
    output: Path,
    *,
    rows: int = GRID_ROWS,
    columns: int = GRID_COLUMNS,
    reference_anchor: Path | None = None,
) -> dict[str, object]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite fixed grid {output}")
    count = rows * columns
    images = []
    for index in range(count):
        path = png_folder / f"{index:06d}.png"
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    width, height = images[0].size
    canvas = Image.new("RGB", (columns * width, rows * height))
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas.paste(image, (column * width, row * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", compress_level=0)
    recorded_path = (
        portable_reference(output, reference_anchor)
        if reference_anchor is not None
        else str(output.resolve())
    )
    return {
        "path": recorded_path,
        "sha256": file_sha256(output),
        "rows": rows,
        "columns": columns,
        "selection": (
            f"first {count} samples from fixed generation seed {GENERATION_SEED}; "
            "no curation or metric selection"
        ),
    }


def evaluate(
    verified: VerifiedEMA,
    *,
    device: torch.device,
    data_root: str | None,
    png_dir: Path,
    grid_path: Path,
    evaluation_anchor: Path,
    generation_batch: int,
    metric_batch: int,
    metric_workers: int,
    kid_reference_features: Path,
    generated_feature_path: Path,
    feature_batch: int,
) -> dict[str, object]:
    _payload, model, config = _load_model(verified.checkpoint_path, device)
    samples = generate_png_folder(
        model,
        config,
        png_dir,
        count=DEVELOPMENT_SAMPLES,
        batch=generation_batch,
        seed=GENERATION_SEED,
        device=device,
    )
    standard = compute_clean_metrics(
        png_dir,
        kid_reference_path=kid_reference_features,
        generated_feature_path=generated_feature_path,
        device=device,
        batch=metric_batch,
        workers=metric_workers,
        kid_seed=DEFAULT_KID_SEED,
        include_legacy_fid=False,
    )
    standard["generated_features"].update(
        {
            "source_png_manifest_sha256": samples["png_manifest_sha256"],
            "generation_seed": samples["seed"],
        }
    )
    auxiliary = compute_repository_auxiliary_metrics(
        png_dir,
        sample_manifest=samples,
        device=device,
        data_root=data_root,
        feature_batch=feature_batch,
    )
    grid = write_uncurated_grid(
        png_dir,
        grid_path,
        reference_anchor=evaluation_anchor,
    )
    return {
        "samples": samples,
        "standard_train_reference_metrics": standard,
        "repository_auxiliary": auxiliary,
        # Compatibility fields consumed by the promotion certificate.  They are
        # aliases of the shared CAP1/CAP2 auxiliary protocol, not recomputations.
        "repository_feature_metrics": auxiliary["repository_feature_metrics"],
        "memorization": auxiliary["memorization"],
        "uncurated_grid": grid,
        "reference_subset": auxiliary["reference_subset"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help=(
            "prospectively declared EMA checkpoint step; default is the final "
            "recorded EMA"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--png-dir", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--generation-batch", type=int, default=500)
    parser.add_argument("--metric-batch", type=int, default=128)
    parser.add_argument("--metric-workers", type=int, default=DEFAULT_METRIC_WORKERS)
    parser.add_argument("--kid-reference-features", type=Path, required=True)
    parser.add_argument(
        "--generated-features",
        type=Path,
        default=None,
        help=("immutable clean-Inception feature archive; defaults beside --out"),
    )
    parser.add_argument("--feature-batch", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_unused(args.out)
    generated_feature_path = args.generated_features or args.out.with_name(
        f"{args.out.stem}_clean_features.npz"
    )
    assert_unused(generated_feature_path)
    verified = verify_final_ema(args.unit, step=args.step)
    device = resolve_device(args.device)
    numerical_settings = configure(device, allow_tf32=False)
    torch.use_deterministic_algorithms(True)
    results = evaluate(
        verified,
        device=device,
        data_root=args.data_root,
        png_dir=args.png_dir,
        grid_path=args.grid,
        evaluation_anchor=args.out.parent,
        generation_batch=args.generation_batch,
        metric_batch=args.metric_batch,
        metric_workers=args.metric_workers,
        kid_reference_features=args.kid_reference_features,
        generated_feature_path=generated_feature_path,
        feature_batch=args.feature_batch,
    )
    # Every filesystem reference is relative to the immutable evaluation JSON.
    # This permits an intact run/evaluation tree to be relocated and audited on
    # a different machine without weakening any content-hash binding.
    results["samples"]["directory"] = portable_reference(args.png_dir, args.out.parent)
    standard = results["standard_train_reference_metrics"]
    standard["kid_reference"]["path"] = portable_reference(
        args.kid_reference_features, args.out.parent
    )
    standard["generated_features"]["path"] = portable_reference(
        generated_feature_path, args.out.parent
    )
    live_sources = source_manifest()
    this_file = Path(__file__).resolve()
    live_sources[str(this_file.relative_to(ROOT)).replace("\\", "/")] = file_sha256(
        this_file
    )
    result = {
        "status": "cap-emf2-development-evaluation",
        "development_only": True,
        "selection_reference": "CIFAR-10 train only; test split never opened",
        "checkpoint_selection": (
            "final recorded EMA"
            if args.step is None
            else "explicit prospectively declared recorded EMA step"
        ),
        "arm": verified.unit["arm"],
        "step": verified.step,
        "unit": {
            "path": portable_reference(verified.unit_path, args.out.parent),
            "sha256": verified.unit["artifact_sha256"],
            "preflight_sha256": verified.unit["preflight_sha256"],
        },
        "checkpoint": {
            "path": portable_reference(verified.checkpoint_path, args.out.parent),
            "sha256": file_sha256(verified.checkpoint_path),
            "kind": "ema",
            "step": verified.step,
        },
        "fixed_protocol": {
            "generated_samples": DEVELOPMENT_SAMPLES,
            "generation_seed": GENERATION_SEED,
            "clean_kid_seed": DEFAULT_KID_SEED,
            "manifold_samples": MANIFOLD_SAMPLES,
            "memorization_samples": MEMORIZATION_SAMPLES,
            "numerical_settings": numerical_settings,
        },
        **results,
        "provenance": {
            **evaluation_provenance(device),
            "numerical_settings": numerical_settings,
            "deterministic_algorithms": True,
        },
        "source_sha256": live_sources,
        "limits": [
            "This is a train-reference development evaluation, not a sealed-test or confirmation result.",
            "Only CleanFID/CleanKID use the published backend; repository precision/recall uses a separately identified feature extractor.",
            "The memorization subset detects literal or near pixel copying, not semantic memorization.",
            "One arm and one training seed cannot support a general performance claim.",
        ],
    }
    digest = write_json_atomic(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "arm": result["arm"],
                "step": result["step"],
                "standard_train_reference_metrics": result[
                    "standard_train_reference_metrics"
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {args.out} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
