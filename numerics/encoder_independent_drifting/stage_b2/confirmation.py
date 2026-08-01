"""Run the hash-frozen, fresh-source, paired Stage-B2 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import torch

from .. import cifar
from ..device import configure, resolve_device
from ..diagnostics import provenance, write_json
from ..f3b import sample_model
from ..f3b_freeze import profile_from_payload
from ..fid import inception_features
from .artifacts import (
    DEFAULT_FREEZE,
    load_b1_calibration,
    load_compatible_artifacts,
    load_freeze,
    verify_sidecar,
)
from .core import (
    B2_CONFIRMATION_UNITS,
    b2_config,
    b2_seed,
    evaluation_prior_seed,
    paired_seed_manifest,
    train_b2,
)
from .evaluation import (
    adjudicate_b2,
    compare_drift_audits,
    drift_energy_audit_suite,
    fresh_evaluation_allocation,
    summarize_drift_audits,
)
from .fresh_data import load_fresh_pool
from .metrics import apply_vetoes, generated_metrics, memorization_statistics_augmented

HERE = Path(__file__).resolve().parent


def _save_checkpoint(
    path: Path, outcome, frozen_profile: dict, unit: int, freeze_sha: str
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with outcome.ema.average_parameters(outcome.model):
        torch.save(
            {
                "unit": unit,
                "profile": frozen_profile,
                "b2_freeze_sha256": freeze_sha,
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
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--out", type=Path, default=HERE / "b2_confirmatory.json")
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(args.out.suffix + ".sha256").exists():
        raise RuntimeError(
            "B2 confirmation output already exists; refusing an adaptive rerun"
        )

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    freeze = load_freeze(args.freeze)
    freeze_sha = verify_sidecar(args.freeze)
    b0, b1, preflight, baseline = load_compatible_artifacts(args.freeze, freeze)
    selected = profile_from_payload(freeze["profile"])
    config = b2_config()
    fresh_path = Path(freeze["fresh_data"]["path"])
    fresh_pool, fresh_record = load_fresh_pool(
        fresh_path,
        freeze["fresh_data"]["source_id"],
        selected.model.image_size,
        freeze["fresh_data"]["requested_float_encoding"],
    )
    if fresh_record["sha256"] != freeze["fresh_data"]["sha256"]:
        raise RuntimeError("fresh B2 confirmation data changed after freeze")
    allocation = fresh_evaluation_allocation(
        len(fresh_pool),
        fresh_record["source_id"],
        config,
        generated_samples=selected.evaluation.generated_samples,
        reference_samples=selected.evaluation.reference_samples,
        control_groups=len(B2_CONFIRMATION_UNITS),
    )
    if allocation.digests != freeze["allocation_digests"]:
        raise RuntimeError("B2 fresh-data allocation changed after freeze")
    reference = fresh_pool[torch.as_tensor(allocation.reference)]
    controls = tuple(
        fresh_pool[torch.as_tensor(indices)] for indices in allocation.controls
    )
    reference_features = inception_features(reference, device).double().numpy()
    train = cifar.cifar_pool(selected.model.image_size, "train", args.data_root)
    b1_calibration = load_b1_calibration(b1)
    baseline_by_unit = {int(row["unit"]): row for row in baseline["rows"]}

    rows = []
    checkpoints = {}
    started = time.time()
    for index, unit in enumerate(B2_CONFIRMATION_UNITS):
        print(f"\n=== B2 CONFIRMATION unit {unit} ===", flush=True)
        outcome = train_b2(
            train,
            selected.model,
            selected.train,
            unit,
            device,
            float(freeze["tau"]),
            float(freeze["lambda_event"]),
            config,
        )
        checkpoint_path = args.checkpoint_dir / f"b2_unit_{unit}_ema.pt"
        checkpoints[str(unit)] = {
            "path": str(checkpoint_path),
            "sha256": _save_checkpoint(
                checkpoint_path, outcome, freeze["profile"], unit, freeze_sha
            ),
        }
        with outcome.ema.average_parameters(outcome.model):
            main_seed = b2_seed("confirmation-evaluation", unit, "metric-prior")
            generated = sample_model(
                outcome.model,
                selected.evaluation.generated_samples,
                selected.model,
                selected.evaluation.nfe_ladder[0],
                main_seed,
                device,
            )
            audit_batches = [
                sample_model(
                    outcome.model,
                    config.audit_batch,
                    selected.model,
                    selected.evaluation.nfe_ladder[0],
                    evaluation_prior_seed(unit, replicate),
                    device,
                )
                for replicate in range(config.audit_replicates)
            ]
        if main_seed != baseline_by_unit[unit]["main_evaluation_prior_seed"]:
            raise RuntimeError("B0/B2 main evaluation priors are not paired")
        audits = drift_energy_audit_suite(
            audit_batches,
            fresh_pool,
            allocation,
            float(freeze["tau"]),
            unit,
            config,
            device,
        )
        comparison = compare_drift_audits(
            audits, baseline_by_unit[unit]["drift_audit"], config
        )
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
            float(b1_calibration["normalizer"]),
            device,
        )
        veto = apply_vetoes(metrics, memory, b1_calibration["thresholds"])
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
            "correction_events": outcome.correction_events,
            "correction_model_forwards": outcome.correction_model_forwards,
            "lambda_event": outcome.lambda_event,
            "wall_time_ratio_to_historical_b0": (
                outcome.wall_seconds / max(float(b0_training["wall_seconds"]), 1e-12)
            ),
            "historical_b0_wall_seconds": b0_training["wall_seconds"],
            "compute_caveat": (
                "wall time is authoritative; forward counts omit pairwise-kernel "
                "FLOPs and backward cost"
            ),
        }
        rows.append(
            {
                "unit": unit,
                "training": training,
                "main_evaluation_prior_seed": main_seed,
                "metrics": metrics,
                "baseline_metrics": baseline_by_unit[unit]["metrics"],
                "b1_metrics_report_only": baseline_by_unit[unit]["b1_report_only"][
                    "metrics"
                ],
                "b1_drift_summary_report_only": baseline_by_unit[unit][
                    "b1_report_only"
                ]["drift_summary"],
                "memorization": memory,
                "veto": veto,
                "drift_audit": audits,
                "drift_summary": summarize_drift_audits(audits),
                "drift_comparison": comparison,
                "assigned_control_group": index,
                "seeds": paired_seed_manifest(unit),
            }
        )
        print(
            f"recall={metrics['recall']:.4f}; "
            f"B0={baseline_by_unit[unit]['metrics']['recall']:.4f}; "
            f"drift gate={comparison['passes']}; veto={veto['passes']}",
            flush=True,
        )
        del outcome, generated, audit_batches
        if device.type == "cuda":
            torch.cuda.empty_cache()

    verdict = adjudicate_b2(
        rows, baseline["rows"], baseline["matched_real_controls"], config
    )
    payload = {
        "status": "b2-confirmatory",
        "confirmatory": True,
        "verdict": verdict,
        "profile": freeze["profile"],
        "b2_config": freeze["b2_config"],
        "tau": float(freeze["tau"]),
        "lambda_event": float(freeze["lambda_event"]),
        "effective_gradient_ratio": freeze["effective_gradient_ratio"],
        "fresh_data": fresh_record | {"operator_attested_unused": True},
        "allocation_digests": allocation.digests,
        "rows": rows,
        "matched_real_controls": baseline["matched_real_controls"],
        "checkpoints": checkpoints,
        "freeze_sha256": freeze_sha,
        "preflight_sha256": preflight["artifact_sha256"],
        "baseline_sha256": freeze["baseline_sha256"],
        "b0_artifact_sha256": b0["artifact_sha256"],
        "b1_artifact_sha256": b1["artifact_sha256"],
        "elapsed_seconds": time.time() - started,
        "claim_scope": freeze["claim_scope"],
        "interpretive_limits": [
            "The finite sample-split loss is a stochastic population-energy surrogate.",
            "PASS would not prove that the learned law has exactly zero drift.",
            "The external source may shift the image distribution relative to CIFAR-10.",
            "Inception metrics are report-only and never enter the correction loss.",
        ],
        "provenance": provenance(),
        "device": settings,
    }
    write_json(args.out, payload)
    print(f"\nB2 verdict: {verdict['decision']} — {verdict['reading']}", flush=True)


if __name__ == "__main__":
    main()
