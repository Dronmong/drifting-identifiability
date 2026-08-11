"""Fixed 50k-sample train-reference evaluation of the final ASFD EMA."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from ..device import configure, resolve_device
from ..stage_cap2.artifacts import (
    assert_unused,
    file_sha256,
    load_checkpoint,
    verify_json,
    write_json_atomic,
)
from ..stage_cap2.development_evaluation import (
    DEVELOPMENT_SAMPLES,
    GENERATION_SEED,
    MANIFOLD_SAMPLES,
    MEMORIZATION_SAMPLES,
    evaluate,
)
from ..stage_cap2.standard_metrics import (
    DEFAULT_KID_SEED,
    DEFAULT_METRIC_WORKERS,
    evaluation_provenance,
)
from .artifacts import source_manifest

STATUS = "asfd-final-evaluation"
FINAL_STEP = 800_000


def _portable(path: Path, anchor: Path) -> str:
    return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()


@dataclass(frozen=True)
class VerifiedASFD:
    result: dict
    result_path: Path
    checkpoint: dict
    checkpoint_path: Path


def verify_final_ema(result_path: Path) -> VerifiedASFD:
    result = verify_json(result_path, "asfd-continuation")
    if (
        result.get("decision") != "GO"
        or int(result.get("final_step", -1)) != FINAL_STEP
    ):
        raise RuntimeError("ASFD evaluation requires a completed GO continuation")
    if result.get("source_sha256") != source_manifest():
        raise RuntimeError("ASFD source changed after continuation")
    record = result.get("checkpoints", {}).get(str(FINAL_STEP), {}).get("ema")
    if not isinstance(record, dict):
        raise TypeError("ASFD continuation lacks its final EMA checkpoint")
    checkpoint_path = Path(record.get("path", ""))
    if not checkpoint_path.is_absolute():
        checkpoint_path = result_path.parent / checkpoint_path
    profile = result["profile"]
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_sha=record.get("sha256"),
        step=FINAL_STEP,
        kind="ema",
        arm="EMF-ASFD",
        declared_profile=profile,
        realized_profile=profile,
        preflight_sha256=result["preflight"]["sha256"],
        run_identity_sha256=result["run_identity_sha256"],
        unit_seed=0,
    )
    return VerifiedASFD(
        result=result,
        result_path=result_path.resolve(),
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path.resolve(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--png-dir", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--generation-batch", type=int, default=500)
    parser.add_argument("--metric-batch", type=int, default=128)
    parser.add_argument("--metric-workers", type=int, default=DEFAULT_METRIC_WORKERS)
    parser.add_argument("--kid-reference-features", type=Path, required=True)
    parser.add_argument("--generated-features", type=Path, required=True)
    parser.add_argument("--feature-batch", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.out, args.generated_features):
        assert_unused(path)
    verified = verify_final_ema(args.result)
    device = resolve_device(args.device)
    settings = configure(device, allow_tf32=False)
    torch.use_deterministic_algorithms(True)
    results = evaluate(
        # The shared evaluator only reads these four attributes.
        type(
            "EvaluationInput",
            (),
            {
                "checkpoint_path": verified.checkpoint_path,
                "step": FINAL_STEP,
            },
        )(),
        device=device,
        data_root=args.data_root,
        png_dir=args.png_dir,
        grid_path=args.grid,
        evaluation_anchor=args.out.parent,
        generation_batch=args.generation_batch,
        metric_batch=args.metric_batch,
        metric_workers=args.metric_workers,
        kid_reference_features=args.kid_reference_features,
        generated_feature_path=args.generated_features,
        feature_batch=args.feature_batch,
    )
    results["samples"]["directory"] = _portable(args.png_dir, args.out.parent)
    standard = results["standard_train_reference_metrics"]
    standard["kid_reference"]["path"] = _portable(
        args.kid_reference_features, args.out.parent
    )
    standard["generated_features"]["path"] = _portable(
        args.generated_features, args.out.parent
    )
    result = {
        "status": STATUS,
        "selection_reference": "CIFAR-10 train only; test split never opened",
        "arm": "EMF-ASFD",
        "step": FINAL_STEP,
        "continuation": {
            "path": _portable(verified.result_path, args.out.parent),
            "sha256": verified.result["artifact_sha256"],
            "preflight_sha256": verified.result["preflight"]["sha256"],
        },
        "checkpoint": {
            "path": _portable(verified.checkpoint_path, args.out.parent),
            "sha256": file_sha256(verified.checkpoint_path),
            "kind": "ema",
            "step": FINAL_STEP,
        },
        "fixed_protocol": {
            "generated_samples": DEVELOPMENT_SAMPLES,
            "generation_seed": GENERATION_SEED,
            "clean_kid_seed": DEFAULT_KID_SEED,
            "manifold_samples": MANIFOLD_SAMPLES,
            "memorization_samples": MEMORIZATION_SAMPLES,
            "numerical_settings": settings,
        },
        **results,
        "provenance": {
            **evaluation_provenance(device),
            "numerical_settings": settings,
            "deterministic_algorithms": True,
        },
        "source_sha256": source_manifest(),
        "limits": [
            "This is a one-foundation, one-continuation proof-of-concept evaluation.",
            "Without a matched 750k-to-800k raw continuation, pre/post differences do not isolate ASFD from 50k additional training.",
            "CleanFID/CleanKID use the published CIFAR-10 train reference; the test split remains sealed.",
        ],
    }
    digest = write_json_atomic(args.out, result)
    print(f"wrote {args.out} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
