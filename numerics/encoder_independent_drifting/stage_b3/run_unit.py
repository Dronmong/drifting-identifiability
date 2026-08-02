"""Train and evaluate one hash-bound Stage-B3 unit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch

from .. import cifar
from ..device import configure, resolve_device
from ..diagnostics import write_json
from ..fid import inception_features
from ..models import sample_latent
from ..stage_b2.artifacts import file_sha256
from ..stage_b2.core import b2_config
from ..stage_b2.fresh_data import load_fresh_pool
from .artifacts import (
    DEFAULT_DATA,
    DEFAULT_PREFLIGHT,
    assert_unused,
    load_preflight,
    load_reference_artifacts,
)
from .core import (
    B3_ARMS,
    B3Config,
    b3_seed,
    build_generator,
    calibrate_operator,
    configure_deterministic_execution,
    kernel_payload,
    seed_manifest,
    train_b3_arm,
)
from .evaluation import (
    compare_cross_architecture,
    compare_within_b3,
    drift_energy_audit_suite,
    evaluation_allocation,
    metrics_from_features,
    summarize_drift_audits,
)
from .references import load_reference_models, sample_reference_models

HERE = Path(__file__).resolve().parent


def _checkpoint_path(directory: Path, unit: int, arm: str, step: int) -> Path:
    safe_arm = arm.lower().replace("-", "_")
    return directory / f"b3_unit_{unit}_{safe_arm}_step_{step}.pt"


def _recovery_path(directory: Path, unit: int, arm: str) -> Path:
    safe_arm = arm.lower().replace("-", "_")
    return directory / f"b3_unit_{unit}_{safe_arm}_recovery.pt"


def _atomic_torch_save(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary_sidecar = path.with_name(path.name + ".sha256.tmp")
    torch.save(payload, temporary)
    digest = file_sha256(temporary)
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    os.replace(temporary, path)
    os.replace(temporary_sidecar, path.with_suffix(path.suffix + ".sha256"))
    return digest


def _save_checkpoint(
    path: Path,
    model,
    *,
    unit: int,
    arm,
    step: int,
    preflight_sha: str,
) -> str:
    return _atomic_torch_save(
        path,
        {
            "stage": "B3",
            "unit": int(unit),
            "arm": arm.name,
            "width": int(arm.width),
            "field_cloud": int(arm.field_cloud),
            "backward_microbatch": int(arm.backward_microbatch),
            "step": int(step),
            "preflight_sha256": preflight_sha,
            "raw_non_ema": True,
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        },
    )


def _save_recovery(
    path: Path,
    model,
    optimizer,
    target_rng,
    history,
    wall_seconds,
    *,
    unit: int,
    arm,
    step: int,
    preflight_sha: str,
    device: torch.device,
) -> str:
    return _atomic_torch_save(
        path,
        {
            "schema": 1,
            "stage": "B3-recovery",
            "unit": int(unit),
            "arm": arm.name,
            "width": int(arm.width),
            "field_cloud": int(arm.field_cloud),
            "backward_microbatch": int(arm.backward_microbatch),
            "step": int(step),
            "preflight_sha256": preflight_sha,
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "target_rng_state": copy.deepcopy(target_rng.bit_generator.state),
            "history": copy.deepcopy(history),
            "wall_seconds": float(wall_seconds),
            "peak_memory_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
            "peak_memory_reserved_bytes": (
                torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else None
            ),
        },
    )


def _load_recovery(path: Path, unit, arm, config, preflight_sha) -> dict | None:
    if not path.exists():
        if path.with_suffix(path.suffix + ".sha256").exists():
            raise RuntimeError(f"orphaned B3 recovery sidecar: {path}")
        return None
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"B3 recovery is missing its SHA sidecar: {path}")
    if file_sha256(path) != sidecar.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError(f"B3 recovery SHA mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema": 1,
        "stage": "B3-recovery",
        "unit": unit,
        "arm": arm.name,
        "width": arm.width,
        "field_cloud": arm.field_cloud,
        "backward_microbatch": arm.backward_microbatch,
        "preflight_sha256": preflight_sha,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"B3 recovery metadata differs at {key}")
    step = int(payload.get("step", -1))
    if not 0 <= step <= config.steps or step % config.recovery_every:
        raise RuntimeError("B3 recovery step violates the frozen cadence")
    return payload


def _load_checkpoint(
    record: dict, unit: int, arm, step: int, config, preflight_sha, device
):
    path = Path(record["path"])
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise RuntimeError("B3 checkpoint missing or changed")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    expected = {
        "stage": "B3",
        "unit": unit,
        "arm": arm.name,
        "width": arm.width,
        "field_cloud": arm.field_cloud,
        "backward_microbatch": arm.backward_microbatch,
        "step": step,
        "preflight_sha256": preflight_sha,
        "raw_non_ema": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"B3 checkpoint metadata differs at {key}")
    model = build_generator(arm, config, unit, device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def _extract_features(images, device) -> np.ndarray:
    return inception_features(images.detach().cpu(), device).double().numpy()


def _evaluate_features(images, features, reference_features, density_k) -> dict:
    return metrics_from_features(
        images.detach().cpu(),
        features,
        reference_features,
        density_k=density_k,
    )


def _b3_samples(model, unit, config, count, device, role, event=None):
    latent = sample_latent(count, config.latent_dim, b3_seed(unit, role, event), device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        result = model(latent).detach()
    model.train(was_training)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", type=int, required=True)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--external-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = B3Config()
    config.validate()
    if args.unit not in config.units:
        raise ValueError(f"unit must lie in {config.units}")
    output = args.out or HERE / f"b3_unit_{args.unit}.json"
    assert_unused(output)
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    settings.update(configure_deterministic_execution(device))
    preflight = load_preflight(args.preflight)
    preflight_sha = preflight["artifact_sha256"]
    frozen_device = preflight["device"]
    for key in (
        "device",
        "torch_version",
        "cuda_version",
        "gpu_name",
        "capability",
        "deterministic_algorithms",
        "cublas_workspace_config",
    ):
        if settings.get(key) != frozen_device.get(key):
            raise RuntimeError(f"B3 execution device differs from preflight at {key}")
    references = load_reference_artifacts()
    train_pool = cifar.cifar_pool(config.image_size, "train", args.data_root)
    in_domain_pool = cifar.cifar_pool(config.image_size, "test", args.data_root)
    shifted_pool, shifted_record = load_fresh_pool(
        args.external_data,
        preflight["external_data"]["source_id"],
        config.image_size,
        "auto",
    )
    if shifted_record["sha256"] != preflight["external_data"]["artifact_sha256"]:
        raise RuntimeError("B3 shifted data changed after preflight")

    branch, kernel, calibration = calibrate_operator(
        train_pool, args.unit, config, device
    )
    frozen_calibration = preflight["calibrations"][str(args.unit)]
    calibration_digest = hashlib.sha256(calibration.tobytes()).hexdigest()
    if calibration_digest != frozen_calibration["indices_sha256"]:
        raise RuntimeError("B3 calibration indices changed after preflight")
    if kernel_payload(kernel) != frozen_calibration["kernel"]:
        raise RuntimeError("B3 calibrated kernel changed after preflight")

    b2 = b2_config()
    selected_eval = preflight["profile"]["evaluation"]
    allocations = {
        "in_domain_development_reused": evaluation_allocation(
            len(in_domain_pool),
            "cifar10-test-adaptively-reused-b3-development",
            units=len(config.units),
            generated_samples=int(selected_eval["generated_samples"]),
            reference_samples=int(selected_eval["reference_samples"]),
            audit_replicates=b2.audit_replicates,
            audit_batch=b2.audit_batch,
        ),
        "shifted_disjoint": evaluation_allocation(
            len(shifted_pool),
            preflight["external_data"]["source_id"],
            units=len(config.units),
            generated_samples=int(selected_eval["generated_samples"]),
            reference_samples=int(selected_eval["reference_samples"]),
            audit_replicates=b2.audit_replicates,
            audit_batch=b2.audit_batch,
        ),
    }
    for name, allocation in allocations.items():
        if allocation.digests != preflight["evaluation_allocations"][name]:
            raise RuntimeError(f"B3 {name} allocation changed after preflight")
    unit_index = config.units.index(args.unit)
    source_pools = {
        "in_domain_development_reused": in_domain_pool,
        "shifted_disjoint": shifted_pool,
    }
    sources = {}
    for name, pool in source_pools.items():
        allocation = allocations[name]
        reference = pool[torch.as_tensor(allocation.reference)]
        control = pool[torch.as_tensor(allocation.controls[unit_index])]
        reference_features = inception_features(reference, device).double().numpy()
        control_features = _extract_features(control, device)
        control_metrics = _evaluate_features(
            control,
            control_features,
            reference_features,
            config.density_coverage_neighbors,
        )
        sources[name] = {
            "pool": pool,
            "allocation": allocation,
            "reference_features": reference_features,
            "control_metrics": control_metrics,
            "scope": (
                "adaptively reused in-domain development instrument"
                if name == "in_domain_development_reused"
                else "disjoint class-aligned shifted development instrument"
            ),
        }

    started = time.time()
    cells: dict[str, dict] = {}
    feature_cache: dict[tuple[str, int, str], np.ndarray] = {}
    for arm in B3_ARMS:
        print(f"\n=== B3 unit {args.unit} arm {arm.name} ===", flush=True)
        recovery_path = _recovery_path(
            args.checkpoint_dir, args.unit, arm.name
        ).resolve()
        resume = _load_recovery(recovery_path, args.unit, arm, config, preflight_sha)
        resumed_step = int(resume["step"]) if resume is not None else 0
        checkpoints: dict[str, dict] = {}
        for step in config.checkpoint_steps:
            path = _checkpoint_path(
                args.checkpoint_dir, args.unit, arm.name, step
            ).resolve()
            sidecar = path.with_suffix(path.suffix + ".sha256")
            if step <= resumed_step:
                if not path.is_file() or not sidecar.is_file():
                    raise RuntimeError(
                        f"B3 recovery at {resumed_step} requires checkpoint {step}"
                    )
                digest = sidecar.read_text(encoding="utf-8").split()[0]
                if file_sha256(path) != digest:
                    raise RuntimeError(f"B3 checkpoint SHA mismatch: {path}")
                checkpoints[str(step)] = {"path": str(path), "sha256": digest}
            elif path.exists() or sidecar.exists():
                if resume is None:
                    raise RuntimeError(
                        f"stale B3 checkpoint exists without recovery state: {path}"
                    )
                # A crash can occur after the immutable checkpoint write but
                # before the mutable recovery write.  Replay from the older
                # recovery and atomically overwrite this checkpoint.

        def save(
            step,
            model,
            _optimizer,
            _target_rng,
            _record,
            _history,
            _wall_seconds,
            *,
            arm=arm,
            checkpoints=checkpoints,
        ):
            path = _checkpoint_path(
                args.checkpoint_dir, args.unit, arm.name, step
            ).resolve()
            checkpoints[str(step)] = {
                "path": str(path),
                "sha256": _save_checkpoint(
                    path,
                    model,
                    unit=args.unit,
                    arm=arm,
                    step=step,
                    preflight_sha=preflight_sha,
                ),
            }

        def recover(
            step,
            model,
            optimizer,
            target_rng,
            _record,
            history,
            wall_seconds,
            *,
            arm=arm,
            recovery_path=recovery_path,
        ):
            _save_recovery(
                recovery_path,
                model,
                optimizer,
                target_rng,
                history,
                wall_seconds,
                unit=args.unit,
                arm=arm,
                step=step,
                preflight_sha=preflight_sha,
                device=device,
            )

        outcome = train_b3_arm(
            train_pool,
            args.unit,
            arm,
            config,
            branch,
            kernel,
            device,
            checkpoint=save,
            recovery=recover,
            resume=resume,
        )
        evaluations = {}
        for step in config.checkpoint_steps:
            model = _load_checkpoint(
                checkpoints[str(step)],
                args.unit,
                arm,
                step,
                config,
                preflight_sha,
                device,
            )
            generated = _b3_samples(
                model,
                args.unit,
                config,
                int(selected_eval["generated_samples"]),
                device,
                "evaluation-latent",
            )
            generated_features = _extract_features(generated, device)
            audit_batches = [
                _b3_samples(
                    model,
                    args.unit,
                    config,
                    b2.audit_batch,
                    device,
                    "audit-latent",
                    replicate,
                )
                for replicate in range(b2.audit_replicates)
            ]
            source_rows = {}
            for source_name, source in sources.items():
                metrics = _evaluate_features(
                    generated,
                    generated_features,
                    source["reference_features"],
                    config.density_coverage_neighbors,
                )
                audits = drift_energy_audit_suite(
                    audit_batches,
                    source["pool"],
                    source["allocation"],
                    tau=float(preflight["b2_tau"]),
                    unit=args.unit,
                    step=step,
                    config=b2,
                    device=device,
                )
                source_rows[source_name] = {
                    "metrics": metrics,
                    "drift_audit": audits,
                    "drift_summary": summarize_drift_audits(audits),
                }
                feature_cache[(arm.name, step, source_name)] = generated_features
            evaluations[str(step)] = source_rows
            del model, generated, audit_batches
            if device.type == "cuda":
                torch.cuda.empty_cache()
        cells[arm.name] = {
            "arm": arm.name,
            "width": arm.width,
            "parameters": outcome.model.parameter_count(),
            "field_cloud": arm.field_cloud,
            "backward_microbatch": arm.backward_microbatch,
            "training": {
                "history": outcome.history,
                "wall_seconds": outcome.wall_seconds,
                "peak_memory_bytes": outcome.peak_memory_bytes,
                "peak_memory_reserved_bytes": outcome.peak_memory_reserved_bytes,
                "optimizer_updates": outcome.optimizer_updates,
                "generated_examples": outcome.generated_examples,
                "target_examples": outcome.target_examples,
                "model_batch_forward_calls": outcome.model_batch_forward_calls,
                "sample_forward_equivalents": outcome.sample_forward_equivalents,
                "resumed_from_step": outcome.resumed_from_step,
            },
            "checkpoints": checkpoints,
            "evaluations": evaluations,
        }
        del outcome
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Re-evaluate the already-frozen bridge checkpoints on these exact sources.
    bridge_models = load_reference_models(references, args.unit, device)
    bridge_generated = sample_reference_models(
        bridge_models,
        references["B0"]["profile"],
        args.unit,
        int(selected_eval["generated_samples"]),
        device,
        role="bridge-evaluation-prior",
    )
    bridge_audits = [
        sample_reference_models(
            bridge_models,
            references["B0"]["profile"],
            args.unit,
            b2.audit_batch,
            device,
            role="bridge-audit-prior",
            event=replicate,
        )
        for replicate in range(b2.audit_replicates)
    ]
    bridge_rows = {}
    bridge_feature_cache: dict[tuple[str, str], np.ndarray] = {}
    for name in ("B0", "B1", "B2"):
        source_rows = {}
        audit_batches = [row[name] for row in bridge_audits]
        generated_features = _extract_features(bridge_generated[name], device)
        for source_name, source in sources.items():
            metrics = _evaluate_features(
                bridge_generated[name],
                generated_features,
                source["reference_features"],
                config.density_coverage_neighbors,
            )
            audits = drift_energy_audit_suite(
                audit_batches,
                source["pool"],
                source["allocation"],
                tau=float(preflight["b2_tau"]),
                unit=args.unit,
                step=config.steps,
                config=b2,
                device=device,
            )
            source_rows[source_name] = {
                "metrics": metrics,
                "drift_audit": audits,
                "drift_summary": summarize_drift_audits(audits),
            }
            bridge_feature_cache[(name, source_name)] = generated_features
        bridge_rows[name] = {
            "source_checkpoint_unit": int(300 + unit_index),
            "checkpoint": references[name]["checkpoints"][str(300 + unit_index)],
            "evaluations": source_rows,
            "nfe": int(preflight["profile"]["evaluation"]["nfe_ladder"][0]),
        }

    comparisons = {"within_b3": {}, "cross_architecture": {}}
    for step in config.checkpoint_steps:
        comparisons["within_b3"][str(step)] = {}
        for source_name, source in sources.items():
            comparisons["within_b3"][str(step)][source_name] = compare_within_b3(
                feature_cache[("B3-capacity", step, source_name)],
                feature_cache[("B3-native", step, source_name)],
                source["reference_features"],
                seed=b3_seed(args.unit, "bootstrap", step),
                replicates=config.bootstrap_replicates,
            )
    for arm in B3_ARMS:
        comparisons["cross_architecture"][arm.name] = {}
        for reference_name in ("B0", "B1", "B2"):
            comparisons["cross_architecture"][arm.name][reference_name] = {}
            for source_name, source in sources.items():
                comparisons["cross_architecture"][arm.name][reference_name][
                    source_name
                ] = compare_cross_architecture(
                    feature_cache[(arm.name, config.steps, source_name)],
                    bridge_feature_cache[(reference_name, source_name)],
                    source["reference_features"],
                    seed=b3_seed(
                        args.unit,
                        f"bootstrap-{arm.name}-{reference_name}-{source_name}",
                    ),
                    replicates=config.bootstrap_replicates,
                )

    payload = {
        "status": "b3-matched-reference-unit",
        "reference_measurement_only": True,
        "unit": args.unit,
        "device": settings,
        "preflight_path": str(args.preflight.resolve()),
        "preflight_sha256": preflight_sha,
        "seed_manifest": seed_manifest(args.unit),
        "calibration": frozen_calibration,
        "evaluation_allocations": {
            name: allocation.digests for name, allocation in allocations.items()
        },
        "source_scope": {name: source["scope"] for name, source in sources.items()},
        "matched_real_controls": {
            name: source["control_metrics"] for name, source in sources.items()
        },
        "cells": cells,
        "frozen_bridge_references": bridge_rows,
        "comparisons": comparisons,
        "elapsed_seconds": time.time() - started,
        "limits": [
            "B3 has no PASS category; it is a matched descriptive reference.",
            "The in-domain source was adaptively consumed by prior stages.",
            "The shifted disjoint source is a development distribution-shift instrument.",
            "Only step 30000 is primary; earlier checkpoints diagnose trajectories.",
            "B3 is an R11-corrected raw-pixel smooth-Laplace one-step proxy, not the paper model.",
        ],
    }
    digest = write_json(output, payload)
    for arm in B3_ARMS:
        recovery_path = _recovery_path(args.checkpoint_dir, args.unit, arm.name)
        recovery_path.unlink(missing_ok=True)
        recovery_path.with_suffix(recovery_path.suffix + ".sha256").unlink(
            missing_ok=True
        )
    print(f"wrote {output} sha256={digest}")


if __name__ == "__main__":
    main()
