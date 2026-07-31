"""Run the frozen, paired B1 confirmation exactly once."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import torch

from . import cifar
from .b1 import (
    B1_CONFIRMATION_UNITS,
    b1_config,
    evaluation_prior_seed,
    paired_seed_manifest,
    train_b1,
)
from .b1_evaluation import (
    adjudicate_b1,
    anchor_audit_suite,
    apply_b1_vetoes,
    compare_anchor_audits,
    confirmation_images,
    evaluation_allocation,
    memorization_statistics_augmented,
    summarize_anchor_audit,
)
from .b1_freeze import (
    DEFAULT_FREEZE,
    HERE,
    load_compatible_artifacts,
    load_freeze,
    verify_sidecar,
)
from .device import configure, resolve_device
from .diagnose_phase20 import save_grid
from .diagnostics import provenance, write_json
from .f3b import sample_model
from .f3b_evaluation import generated_metrics
from .f3b_freeze import profile_from_payload
from .fid import inception_features


def _save_ema_checkpoint(
    path: Path, outcome, frozen_profile: dict, unit: int, freeze_sha: str
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with outcome.ema.average_parameters(outcome.model):
        torch.save(
            {
                "unit": unit,
                "profile": frozen_profile,
                "b1_freeze_sha256": freeze_sha,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in outcome.model.state_dict().items()
                },
            },
            path,
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE / "b1_checkpoints")
    parser.add_argument("--out", type=Path, default=HERE / "b1_confirmatory.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    freeze = load_freeze(args.freeze)
    freeze_sha = verify_sidecar(args.freeze)
    b0, calibration, baseline = load_compatible_artifacts(args.freeze, freeze)
    selected = profile_from_payload(freeze["profile"])
    config = b1_config()
    allocation = evaluation_allocation(
        config,
        selected.evaluation.generated_samples,
        selected.evaluation.reference_samples,
        len(B1_CONFIRMATION_UNITS),
    )
    if allocation.digests != freeze["allocation_digests"]:
        raise RuntimeError("B1 evaluation allocation changed after freeze")
    train = cifar.cifar_pool(selected.model.image_size, "train", args.data_root)
    reference, controls, target_pairs = confirmation_images(
        allocation, selected.model.image_size, args.data_root
    )
    reference_features = inception_features(reference, device).double().numpy()
    baseline_by_unit = {int(row["unit"]): row for row in baseline["rows"]}

    started = time.time()
    rows = []
    checkpoints = {}
    for index, unit in enumerate(B1_CONFIRMATION_UNITS):
        print(f"\n=== B1 CONFIRMATION unit {unit} ===", flush=True)
        outcome = train_b1(
            train,
            selected.model,
            selected.train,
            unit,
            device,
            float(freeze["scale"]),
            float(freeze["lambda_event"]),
            config,
        )
        checkpoint_path = args.checkpoint_dir / f"b1_unit_{unit}_ema.pt"
        checkpoints[str(unit)] = {
            "path": str(checkpoint_path),
            "sha256": _save_ema_checkpoint(
                checkpoint_path, outcome, freeze["profile"], unit, freeze_sha
            ),
        }
        with outcome.ema.average_parameters(outcome.model):
            prior_seed = evaluation_prior_seed(unit)
            generated = sample_model(
                outcome.model,
                selected.evaluation.generated_samples,
                selected.model,
                selected.evaluation.nfe_ladder[0],
                prior_seed,
                device,
            )
            audits = anchor_audit_suite(
                outcome.model,
                selected.model,
                selected.evaluation.nfe_ladder[0],
                unit,
                float(freeze["scale"]),
                target_pairs,
                device,
                config,
            )
        if prior_seed != baseline_by_unit[unit]["main_evaluation_prior_seed"]:
            raise RuntimeError("B0/B1 main evaluation priors are not paired")
        metrics = generated_metrics(
            generated,
            reference_features,
            controls[index],
            device,
            include_fid=True,
        )
        memory = memorization_statistics_augmented(
            generated,
            train,
            float(calibration["normalizer"]),
            device,
        )
        veto = apply_b1_vetoes(metrics, memory, calibration["thresholds"])
        comparison = compare_anchor_audits(
            audits, baseline_by_unit[unit]["anchor_audit"], config
        )
        save_grid(
            generated[:64].cpu(),
            HERE
            / (
                f"b1_u{unit}_step{selected.train.steps}_"
                f"nfe{selected.evaluation.nfe_ladder[0]}.png"
            ),
        )
        b0_training = next(
            row["training"] for row in b0["rows"] if int(row["unit"]) == unit
        )
        training = {
            "history": outcome.history,
            "wall_seconds": outcome.wall_seconds,
            "peak_memory_bytes": outcome.peak_memory_bytes,
            "examples_seen": outcome.examples_seen,
            "optimizer_updates": outcome.optimizer_updates,
            "model_parameters": outcome.model.parameter_count(),
            "anchor_events": outcome.anchor_events,
            "anchor_refreshes": outcome.anchor_refreshes,
            "anchor_model_forwards": outcome.anchor_model_forwards,
            "lambda_event": outcome.lambda_event,
            "extra_forward_equivalents_over_b0": outcome.anchor_model_forwards,
            "wall_time_ratio_to_historical_b0": (
                outcome.wall_seconds / max(float(b0_training["wall_seconds"]), 1e-12)
            ),
            "historical_b0_wall_seconds": b0_training["wall_seconds"],
            "compute_caveat": (
                "Forward-equivalent count does not equal FLOPs or wall time; "
                "B1 also backpropagates through each anchor trajectory."
            ),
        }
        row = {
            "unit": unit,
            "training": training,
            "main_evaluation_prior_seed": prior_seed,
            "metrics": metrics,
            "baseline_metrics": baseline_by_unit[unit]["metrics"],
            "memorization": memory,
            "veto": veto,
            "anchor_audit": audits,
            "anchor_summary": summarize_anchor_audit(audits),
            "anchor_comparison": comparison,
            "assigned_control_group": index,
            "seeds": paired_seed_manifest(unit),
        }
        rows.append(row)
        print(
            f"recall={metrics['recall']:.4f}; "
            f"B0={baseline_by_unit[unit]['metrics']['recall']:.4f}; "
            f"anchor gate={comparison['passes']}; veto={veto['passes']}",
            flush=True,
        )
        del outcome, generated
        if device.type == "cuda":
            torch.cuda.empty_cache()

    verdict = adjudicate_b1(
        rows, baseline["rows"], baseline["matched_real_controls"], config
    )
    payload = {
        "status": "b1-confirmatory",
        "confirmatory": True,
        "protocol": "numerics/EncoderIndependentB1Protocol.md",
        "provenance": provenance(),
        "device": settings,
        "freeze_sha256": freeze_sha,
        "calibration_sha256": calibration["artifact_sha256"],
        "baseline_sha256": baseline["artifact_sha256"],
        "b0_artifact_sha256": b0["artifact_sha256"],
        "profile": freeze["profile"],
        "b1_config": freeze["b1_config"],
        "lambda_event": freeze["lambda_event"],
        "effective_gradient_ratio": freeze["effective_gradient_ratio"],
        "allocation_digests": allocation.digests,
        "matched_real_controls": baseline["matched_real_controls"],
        "checkpoints": checkpoints,
        "rows": rows,
        "verdict": verdict,
        "elapsed_seconds": time.time() - started,
        "claim_scope": freeze["claim_scope"],
        "interpretive_limits": [
            "The finite random-feature training loss is not measure determining.",
            "The ideal characteristic-function anchor is the correctness authority.",
            "The comparison is paired to B0 but is not compute matched.",
            "Unbiased audit values are report-only and may be negative.",
        ],
    }
    digest = write_json(args.out, payload)
    print("\n=== B1 CONFIRMATION ===")
    print(verdict["reading"])
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
