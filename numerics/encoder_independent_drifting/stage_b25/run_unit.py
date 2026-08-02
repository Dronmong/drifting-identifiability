"""Run one complete paired B2.5 unit (four arms, three checkpoints)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .. import cifar
from ..b1 import b1_config
from ..config import MASTER_SEED, derive_seed
from ..device import configure, resolve_device
from ..diagnostics import write_json
from ..f3b import TimeConditionedUNet, f3b_seed, sample_model
from ..f3b_freeze import profile_from_payload
from ..fid import inception_features
from ..stage_b2.core import b2_config
from ..stage_b2.fresh_data import load_fresh_pool
from .artifacts import (
    DEFAULT_DATA,
    DEFAULT_PREFLIGHT,
    HERE,
    assert_result_path_unused,
    file_sha256,
    load_frozen_inputs,
    load_preflight,
)
from .core import B25_ARMS, B25_PHASE, B25Config, paired_seed_manifest, train_b25_arm
from .evaluation import (
    drift_energy_audit_suite,
    evaluation_allocation,
    metrics_from_features,
    paired_kid_subsample_interval,
    paired_precision_recall_intervals,
    paired_values_interval,
    summarize_drift_audits,
)


def _checkpoint_path(root: Path, unit: int, arm: str, step: int) -> Path:
    return root / f"b25_u{unit}_{arm.lower()}_step{step}_ema.pt"


def _save_checkpoint(
    path: Path,
    model: TimeConditionedUNet,
    *,
    unit: int,
    arm: str,
    step: int,
    profile: dict,
    preflight_sha: str,
) -> str:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite B2.5 checkpoint {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "b25-development",
            "unit": unit,
            "arm": arm,
            "step": step,
            "profile": profile,
            "preflight_sha256": preflight_sha,
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        },
        path,
    )
    return file_sha256(path)


def _load_checkpoint(
    record: dict,
    profile: dict,
    unit: int,
    arm: str,
    step: int,
    preflight_sha: str,
    device,
) -> TimeConditionedUNet:
    path = Path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError("B2.5 checkpoint changed after training")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    expected = {
        "stage": "b25-development",
        "unit": unit,
        "arm": arm,
        "step": step,
        "profile": profile,
        "preflight_sha256": preflight_sha,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"B2.5 checkpoint differs at {key}")
    selected = profile_from_payload(profile)
    model = TimeConditionedUNet(
        selected.model, f3b_seed(B25_PHASE, unit, "model-init")
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def _evaluation_prior_seed(unit: int, step: int, role: str, replicate: int = 0) -> int:
    return derive_seed(
        MASTER_SEED + 125_000,
        "b25-evaluation",
        unit,
        step,
        role,
        replicate,
    )


def _evaluate_checkpoint(
    model: TimeConditionedUNet,
    *,
    unit: int,
    step: int,
    selected,
    device,
    tau: float,
    sources: dict,
    config: B25Config,
) -> tuple[dict, dict[str, np.ndarray]]:
    b2 = b2_config()
    main_seed = _evaluation_prior_seed(unit, step, "main-prior")
    generated = sample_model(
        model,
        selected.evaluation.generated_samples,
        selected.model,
        selected.evaluation.nfe_ladder[0],
        main_seed,
        device,
    ).cpu()
    features = inception_features(generated, device).double().numpy()
    audit_batches = [
        sample_model(
            model,
            b2.audit_batch,
            selected.model,
            selected.evaluation.nfe_ladder[0],
            _evaluation_prior_seed(unit, step, "drift-prior", replicate),
            device,
        ).cpu()
        for replicate in range(b2.audit_replicates)
    ]
    source_rows = {}
    for name, source in sources.items():
        audits = drift_energy_audit_suite(
            audit_batches,
            source["pool"],
            source["allocation"],
            tau=tau,
            unit=unit,
            step=step,
            config=b2,
            device=device,
        )
        source_rows[name] = {
            "metrics": metrics_from_features(
                generated,
                features,
                source["reference_features"],
                density_k=config.density_coverage_neighbors,
            ),
            "drift_audits": audits,
            "drift_summary": summarize_drift_audits(audits),
        }
    return {
        "step": step,
        "main_evaluation_prior_seed": main_seed,
        "audit_prior_seeds": [
            _evaluation_prior_seed(unit, step, "drift-prior", replicate)
            for replicate in range(b2.audit_replicates)
        ],
        "sources": source_rows,
    }, {name: features for name in sources}


def _paired_comparisons(
    cells: dict,
    feature_cache: dict,
    sources: dict,
    unit: int,
    config: B25Config,
) -> dict:
    pairs = (
        ("B1", "B0"),
        ("B2", "B0"),
        ("B1B2", "B0"),
        ("B1B2", "B1"),
        ("B1B2", "B2"),
    )
    result = {}
    for step in config.checkpoint_steps:
        step_rows = {}
        for source_name, source in sources.items():
            comparisons = {}
            reference = source["reference_features"]
            for candidate, baseline in pairs:
                candidate_features = feature_cache[(candidate, step, source_name)]
                baseline_features = feature_cache[(baseline, step, source_name)]
                seed = _evaluation_prior_seed(
                    unit, step, f"interval:{source_name}:{candidate}:{baseline}"
                )
                intervals = paired_precision_recall_intervals(
                    candidate_features,
                    baseline_features,
                    reference,
                    seed=seed,
                    replicates=config.bootstrap_replicates,
                )
                candidate_audits = cells[candidate][str(step)]["sources"][source_name][
                    "drift_audits"
                ]
                baseline_audits = cells[baseline][str(step)]["sources"][source_name][
                    "drift_audits"
                ]
                drift = paired_values_interval(
                    [row["raw_energy"] for row in candidate_audits],
                    [row["raw_energy"] for row in baseline_audits],
                    seed=seed + 2,
                    replicates=config.bootstrap_replicates,
                )
                item = {
                    "candidate": candidate,
                    "baseline": baseline,
                    "precision_recall": intervals,
                    "raw_drift_energy": drift,
                }
                if step == config.final_step:
                    item["kid"] = paired_kid_subsample_interval(
                        candidate_features,
                        baseline_features,
                        reference,
                        seed=seed + 3,
                        replicates=config.bootstrap_replicates,
                    )
                comparisons[f"{candidate}_vs_{baseline}"] = item
            step_rows[source_name] = comparisons
        result[str(step)] = step_rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", type=int, required=True)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--external-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = B25Config()
    config.validate()
    if args.unit not in config.units:
        raise ValueError(f"unit must lie in {config.units}")
    output = args.out or HERE / f"b25_unit_{args.unit}.json"
    assert_result_path_unused(output)
    planned = [
        _checkpoint_path(args.checkpoint_dir, args.unit, arm, step)
        for arm in config.arms
        for step in config.checkpoint_steps
    ]
    if any(path.exists() for path in planned):
        raise RuntimeError("a planned B2.5 checkpoint path already exists")

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    preflight = load_preflight(args.preflight)
    preflight_sha = preflight["artifact_sha256"]
    frozen = load_frozen_inputs(
        Path(preflight["b1_freeze_path"]), Path(preflight["b2_freeze_path"])
    )
    for prefix in ("b1", "b2"):
        if frozen[f"{prefix}_freeze_sha256"] != preflight[f"{prefix}_freeze_sha256"]:
            raise RuntimeError(f"B2.5 {prefix.upper()} freeze changed after preflight")
    if {
        name: frozen[name] for name in ("b1_scale", "lambda_b1", "tau_b2", "lambda_b2")
    } != preflight["frozen_constants"]:
        raise RuntimeError("B2.5 inherited constants differ from preflight")
    selected = profile_from_payload(preflight["profile"])
    train = cifar.cifar_pool(selected.model.image_size, "train", args.data_root)
    in_domain_pool = cifar.cifar_pool(selected.model.image_size, "test", args.data_root)
    external_pool, external_record = load_fresh_pool(
        args.external_data,
        preflight["external_data"]["source_id"],
        selected.model.image_size,
        "auto",
    )
    if external_record["sha256"] != preflight["external_data"]["artifact_sha256"]:
        raise RuntimeError("B2.5 external data changed after preflight")
    b2 = b2_config()
    allocations = {
        "in_domain_development_reused": evaluation_allocation(
            len(in_domain_pool),
            "cifar10-test-adaptively-reused-b25-development",
            units=len(config.units),
            generated_samples=selected.evaluation.generated_samples,
            reference_samples=selected.evaluation.reference_samples,
            audit_replicates=b2.audit_replicates,
            audit_batch=b2.audit_batch,
        ),
        "shifted_disjoint": evaluation_allocation(
            len(external_pool),
            preflight["external_data"]["source_id"],
            units=len(config.units),
            generated_samples=selected.evaluation.generated_samples,
            reference_samples=selected.evaluation.reference_samples,
            audit_replicates=b2.audit_replicates,
            audit_batch=b2.audit_batch,
        ),
    }
    for name, allocation in allocations.items():
        if allocation.digests != preflight["evaluation_allocations"][name]:
            raise RuntimeError(f"B2.5 {name} allocation changed after preflight")
    pools = {
        "in_domain_development_reused": in_domain_pool,
        "shifted_disjoint": external_pool,
    }
    unit_index = config.units.index(args.unit)
    sources = {}
    for name, pool in pools.items():
        allocation = allocations[name]
        reference = pool[torch.as_tensor(allocation.reference)]
        control = pool[torch.as_tensor(allocation.controls[unit_index])]
        reference_features = inception_features(reference, device).double().numpy()
        control_features = inception_features(control, device).double().numpy()
        sources[name] = {
            "pool": pool,
            "allocation": allocation,
            "reference_features": reference_features,
            "control_indices": allocation.controls[unit_index],
            "control_metrics": metrics_from_features(
                control,
                control_features,
                reference_features,
                density_k=config.density_coverage_neighbors,
            ),
            "scope": (
                "adaptively reused in-domain development instrument"
                if name == "in_domain_development_reused"
                else "disjoint class-aligned distribution-shift instrument"
            ),
        }

    cells = {}
    feature_cache: dict[tuple[str, int, str], np.ndarray] = {}
    run_started = time.time()
    for arm in B25_ARMS:
        print(f"\n=== B2.5 unit {args.unit} arm {arm} ===", flush=True)
        checkpoints = {}

        def save(step, model, _record, *, arm=arm, checkpoints=checkpoints):
            path = _checkpoint_path(args.checkpoint_dir, args.unit, arm, step)
            checkpoints[str(step)] = {
                "path": str(path.resolve()),
                "sha256": _save_checkpoint(
                    path,
                    model,
                    unit=args.unit,
                    arm=arm,
                    step=step,
                    profile=preflight["profile"],
                    preflight_sha=preflight_sha,
                ),
            }

        outcome = train_b25_arm(
            train,
            selected.model,
            selected.train,
            args.unit,
            arm,
            device,
            b1_scale=frozen["b1_scale"],
            lambda_b1=frozen["lambda_b1"],
            tau_b2=frozen["tau_b2"],
            lambda_b2=frozen["lambda_b2"],
            b1_config=b1_config(),
            b2_config=b2,
            stage_config=config,
            checkpoint=save,
        )
        evaluations = {}
        for step in config.checkpoint_steps:
            model = _load_checkpoint(
                checkpoints[str(step)],
                preflight["profile"],
                args.unit,
                arm,
                step,
                preflight_sha,
                device,
            )
            evaluation, features = _evaluate_checkpoint(
                model,
                unit=args.unit,
                step=step,
                selected=selected,
                device=device,
                tau=frozen["tau_b2"],
                sources=sources,
                config=config,
            )
            evaluations[str(step)] = evaluation
            for source_name, values in features.items():
                feature_cache[(arm, step, source_name)] = values
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        cells[arm] = {
            "unit": args.unit,
            "arm": arm,
            "seeds": paired_seed_manifest(args.unit, arm),
            "training": {
                "history": outcome.history,
                "wall_seconds": outcome.wall_seconds,
                "peak_memory_bytes": outcome.peak_memory_bytes,
                "peak_memory_reserved_bytes": outcome.peak_memory_reserved_bytes,
                "examples_seen": outcome.examples_seen,
                "optimizer_updates": outcome.optimizer_updates,
                "anchor_events": outcome.anchor_events,
                "anchor_refreshes": outcome.anchor_refreshes,
                "correction_events": outcome.correction_events,
                "anchor_model_forwards": outcome.anchor_model_forwards,
                "correction_model_forwards": outcome.correction_model_forwards,
                "lambda_b1": outcome.lambda_b1,
                "lambda_b2": outcome.lambda_b2,
                "kernel_health_summary": outcome.kernel_health_summary,
                "component_gradient_diagnostics": outcome.component_gradient_diagnostics,
            },
            "checkpoints": checkpoints,
            **evaluations,
        }
        del outcome
        if device.type == "cuda":
            torch.cuda.empty_cache()

    step_one_losses = {
        arm: cells[arm]["training"]["history"][0]["flow_loss"] for arm in B25_ARMS
    }
    if len(set(step_one_losses.values())) != 1:
        raise RuntimeError("B2.5 arms did not replay the paired flow stream")
    paired = _paired_comparisons(cells, feature_cache, sources, args.unit, config)
    payload = {
        "status": "b25-development-unit",
        "development_only": True,
        "unit": args.unit,
        "device": settings,
        "preflight_path": str(args.preflight.resolve()),
        "preflight_sha256": preflight_sha,
        "profile": preflight["profile"],
        "frozen_constants": preflight["frozen_constants"],
        "evaluation_allocations": {
            name: allocation.digests for name, allocation in allocations.items()
        },
        "evaluation_source_scope": {
            name: source["scope"] for name, source in sources.items()
        },
        "matched_real_controls": {
            name: source["control_metrics"] for name, source in sources.items()
        },
        "flow_pairing_step_one": step_one_losses,
        "cells": cells,
        "paired_comparisons": paired,
        "elapsed_seconds": time.time() - run_started,
        "limits": [
            "This is one development unit, not a confirmation.",
            "The in-domain instrument was adaptively consumed by earlier stages.",
            "The shifted instrument is disjoint but is not the CIFAR target law.",
            "Intermediate checkpoints are diagnostic; step 30000 is primary.",
        ],
    }
    digest = write_json(output, payload)
    print(f"wrote {output} sha256={digest}")


if __name__ == "__main__":
    main()
