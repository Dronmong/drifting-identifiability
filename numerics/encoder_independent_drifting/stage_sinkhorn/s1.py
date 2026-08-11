"""Run the resumable corrected S1 matched-continuation screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .. import cifar
from ..device import configure, resolve_device
from ..diagnostics import provenance, write_json
from ..f3b import sample_model
from ..f3b_freeze import profile_from_payload, verify_sidecar
from ..fid import inception_features
from ..stage_b2.artifacts import (
    load_b1_calibration,
    load_compatible_artifacts,
)
from ..stage_b2.artifacts import (
    load_freeze as load_b2_freeze,
)
from ..stage_b2.core import b2_config
from ..stage_b2.evaluation import fresh_evaluation_allocation
from ..stage_b2.fresh_data import load_fresh_pool
from ..stage_b2.metrics import (
    apply_vetoes,
    generated_metrics,
    memorization_statistics_augmented,
)
from .continuation import (
    CONTINUATION_ARMS,
    ContinuationConfig,
    continuation_flow_seed_manifest,
    train_continuation_arm,
)
from .core import SinkhornConfig
from .s1_evaluation import (
    S1AuditConfig,
    all_plans_converged,
    audit_model,
    compare_audits,
)
from .s1_freeze import DEFAULT_S1_FREEZE, load_s1_freeze

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "s1_v2_initial_two.json"
DEFAULT_SHARD_DIR = HERE / "s1_v2_shards"
DEFAULT_CHECKPOINT_DIR = HERE / "s1_v2_checkpoints"


def _save_checkpoint(
    path: Path,
    outcome,
    profile: dict,
    unit: int,
    freeze_sha: str,
) -> dict:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite corrected S1 checkpoint {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with outcome.ema.average_parameters(outcome.model):
        torch.save(
            {
                "unit": unit,
                "arm": outcome.arm,
                "profile": profile,
                "s1_v2_freeze_sha256": freeze_sha,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in outcome.model.state_dict().items()
                },
            },
            path,
        )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load_shard(path: Path, freeze_sha: str, unit: int) -> dict:
    verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "sinkhorn-s1-v2-unit":
        raise RuntimeError(f"not a corrected S1 unit shard: {path}")
    if int(payload.get("unit", -1)) != unit:
        raise RuntimeError(f"corrected S1 shard {path} has the wrong unit")
    if payload.get("s1_v2_freeze_sha256") != freeze_sha:
        raise RuntimeError(f"corrected S1 shard {path} belongs to another freeze")
    for arm in CONTINUATION_ARMS:
        checkpoint = Path(payload["arms"][arm]["checkpoint"]["path"])
        if (
            not checkpoint.exists()
            or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != payload["arms"][arm]["checkpoint"]["sha256"]
        ):
            raise RuntimeError(f"corrected S1 checkpoint changed for {unit}/{arm}")
    return payload


def _evaluate_arm(
    *,
    outcome,
    arm: str,
    unit: int,
    evaluation_seed: int,
    selected,
    fresh_pool: torch.Tensor,
    allocation,
    reference_features,
    control: torch.Tensor,
    train: torch.Tensor,
    b1_calibration: dict,
    sinkhorn_config: SinkhornConfig,
    audit_config: S1AuditConfig,
    cost_scale: float,
    device: torch.device,
) -> dict:
    with outcome.ema.average_parameters(outcome.model):
        outcome.model.eval()
        generated = sample_model(
            outcome.model,
            selected.evaluation.generated_samples,
            selected.model,
            selected.evaluation.nfe_ladder[0],
            evaluation_seed,
            device,
        )
        audit = audit_model(
            outcome.model,
            selected.model,
            fresh_pool,
            allocation,
            unit,
            device,
            cost_scale,
            sinkhorn_config,
            audit_config,
        )
    metrics = generated_metrics(
        generated,
        reference_features,
        control,
        device,
        include_fid=True,
    )
    memorization = memorization_statistics_augmented(
        generated,
        train,
        float(b1_calibration["normalizer"]),
        device,
    )
    veto = apply_vetoes(metrics, memorization, b1_calibration["thresholds"])
    return {
        "arm": arm,
        "metrics": metrics,
        "memorization": memorization,
        "veto": veto,
        "sinkhorn_audit": audit,
        "audit_solver_converged": all_plans_converged(audit),
    }


def _unit_gate(
    arms: dict[str, dict],
    comparison: dict,
    audit_config: S1AuditConfig,
) -> dict:
    control = arms["control"]
    laplace = arms["laplace"]
    sinkhorn = arms["sinkhorn"]
    recall_floor = float(control["metrics"]["recall"]) - (
        audit_config.recall_noninferiority_margin
    )
    rank_floor = audit_config.effective_rank_retention * float(
        control["metrics"]["effective_rank"]
    )
    laplace_pr_f1_floor = float(laplace["metrics"]["pr_f1"]) - 0.02
    laplace_rank_floor = 0.90 * float(laplace["metrics"]["effective_rank"])
    summary = sinkhorn["training"]["correction_summary"]
    laplace_summary = laplace["training"]["correction_summary"]
    checks = {
        "field_reduction": bool(comparison["passes"]),
        "recall_noninferior": float(sinkhorn["metrics"]["recall"]) >= recall_floor,
        "effective_rank_retained": float(sinkhorn["metrics"]["effective_rank"])
        >= rank_floor,
        "pr_f1_improves_control": float(sinkhorn["metrics"]["pr_f1"])
        > float(control["metrics"]["pr_f1"]),
        "all_safety_vetoes": all(
            bool(arms[arm]["veto"]["passes"]) for arm in CONTINUATION_ARMS
        ),
        "sinkhorn_training_solver": bool(
            summary["cap_hits"] == 0 and summary["maximum_relative_error"] <= 1e-3
        ),
        "laplace_training_normalization": bool(
            laplace_summary["positive_row_sum_error_maximum"] <= 1e-5
            and laplace_summary["negative_row_sum_error_maximum"] <= 1e-5
        ),
        "all_audit_solvers": all(
            bool(arms[arm]["audit_solver_converged"]) for arm in CONTINUATION_ARMS
        ),
        "laplace_pr_f1_competitive": float(sinkhorn["metrics"]["pr_f1"])
        >= laplace_pr_f1_floor,
        "laplace_rank_competitive": float(sinkhorn["metrics"]["effective_rank"])
        >= laplace_rank_floor,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "recall_floor": recall_floor,
        "control_rank_floor": rank_floor,
        "laplace_pr_f1_floor": laplace_pr_f1_floor,
        "laplace_rank_floor": laplace_rank_floor,
    }


def _train_evaluate_unit(
    *,
    unit: int,
    freeze: dict,
    freeze_sha: str,
    b0: dict,
    baseline_row: dict,
    historical_b2_row: dict,
    train: torch.Tensor,
    fresh_pool: torch.Tensor,
    allocation,
    reference_features,
    control: torch.Tensor,
    b1_calibration: dict,
    selected,
    sinkhorn_config: SinkhornConfig,
    audit_config: S1AuditConfig,
    continuation_config: ContinuationConfig,
    laplace_config,
    device: torch.device,
    checkpoint_dir: Path,
) -> dict:
    evaluation_seed = int(baseline_row["main_evaluation_prior_seed"])
    arms: dict[str, dict] = {}
    started = time.time()
    for arm in CONTINUATION_ARMS:
        print(f"  -- {arm} continuation", flush=True)
        outcome = train_continuation_arm(
            arm=arm,
            pool=train,
            checkpoint_record=b0["checkpoints"][str(unit)],
            frozen_profile=freeze["profile"],
            model_config=selected.model,
            base_train_config=selected.train,
            continuation_config=continuation_config,
            unit=unit,
            device=device,
            sinkhorn_cost_scale=float(freeze["sinkhorn"]["cost_scale"]),
            sinkhorn_lambda=float(freeze["sinkhorn"]["lambda_event"]),
            sinkhorn_config=sinkhorn_config,
            laplace_tau=float(freeze["laplace"]["tau"]),
            laplace_lambda=float(freeze["laplace"]["lambda_event"]),
            laplace_config=laplace_config,
        )
        if arm == "control":
            expected_events = 0
        elif arm == "laplace":
            expected_events = (
                continuation_config.steps // laplace_config.correction_every
            )
        else:
            expected_events = (
                continuation_config.steps // sinkhorn_config.correction_every
            )
        if outcome.correction_events != expected_events:
            raise RuntimeError(
                f"{arm} executed {outcome.correction_events} correction events; "
                f"expected {expected_events}"
            )
        checkpoint = _save_checkpoint(
            checkpoint_dir / f"s1_v2_unit_{unit}_{arm}_ema.pt",
            outcome,
            freeze["profile"],
            unit,
            freeze_sha,
        )
        evaluated = _evaluate_arm(
            outcome=outcome,
            arm=arm,
            unit=unit,
            evaluation_seed=evaluation_seed,
            selected=selected,
            fresh_pool=fresh_pool,
            allocation=allocation,
            reference_features=reference_features,
            control=control,
            train=train,
            b1_calibration=b1_calibration,
            sinkhorn_config=sinkhorn_config,
            audit_config=audit_config,
            cost_scale=float(freeze["sinkhorn"]["cost_scale"]),
            device=device,
        )
        evaluated.update(
            {
                "checkpoint": checkpoint,
                "training": {
                    "history": outcome.history,
                    "wall_seconds": outcome.wall_seconds,
                    "peak_memory_bytes": outcome.peak_memory_bytes,
                    "optimizer_updates": outcome.optimizer_updates,
                    "examples_seen": outcome.examples_seen,
                    "correction_events": outcome.correction_events,
                    "correction_model_forwards": outcome.correction_model_forwards,
                    "start_state_sha256": outcome.start_state_sha256,
                    "first_flow_batch_sha256": outcome.first_flow_batch_sha256,
                    "correction_summary": outcome.correction_summary,
                },
            }
        )
        arms[arm] = evaluated
        del outcome
        if device.type == "cuda":
            torch.cuda.empty_cache()

    starting_states = {
        arms[arm]["training"]["start_state_sha256"] for arm in CONTINUATION_ARMS
    }
    first_batches = {
        arms[arm]["training"]["first_flow_batch_sha256"] for arm in CONTINUATION_ARMS
    }
    if len(starting_states) != 1:
        raise RuntimeError("corrected S1 arms did not clone the same B0 state")
    if len(first_batches) != 1:
        raise RuntimeError("corrected S1 arms did not receive the same flow batch")
    comparison = compare_audits(
        arms["sinkhorn"]["sinkhorn_audit"],
        arms["control"]["sinkhorn_audit"],
        audit_config,
    )
    gate = _unit_gate(arms, comparison, audit_config)
    return {
        "status": "sinkhorn-s1-v2-unit",
        "unit": unit,
        "s1_v2_freeze_sha256": freeze_sha,
        "evaluation_prior_seed": evaluation_seed,
        "assigned_control_group": int(baseline_row["assigned_control_group"]),
        "arms": arms,
        "sinkhorn_vs_control": comparison,
        "gate": gate,
        "matched_start_state_sha256": next(iter(starting_states)),
        "matched_first_flow_batch_sha256": next(iter(first_batches)),
        "flow_seeds": continuation_flow_seed_manifest(unit),
        "historical_b0_metrics_report_only": baseline_row["metrics"],
        "historical_b2_metrics_report_only": historical_b2_row["metrics"],
        "wall_seconds_total": time.time() - started,
    }


def staged_verdict(rows: list[dict], freeze: dict) -> dict:
    expected = [int(value) for value in freeze["initial_units"]]
    if [int(row["unit"]) for row in rows] != expected:
        raise RuntimeError("corrected S1 aggregation order differs from freeze")
    passed = sum(bool(row["gate"]["passes"]) for row in rows)
    if passed == len(expected):
        decision = "PROVISIONAL-GO"
        reading = "Both corrected expedited continuation units passed."
    elif passed == 0:
        decision = "STOP"
        reading = "Neither corrected continuation unit passed."
    else:
        decision = f"RUN-TIEBREAKER-{freeze['tiebreaker_unit']}"
        reading = "Corrected continuation units conflict; run only the reserve."
    return {
        "decision": decision,
        "units_passing": passed,
        "units_total": len(expected),
        "reading": reading,
        "provisional": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=DEFAULT_S1_FREEZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(args.out.suffix + ".sha256").exists():
        raise RuntimeError("corrected S1 initial-two output already exists")

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    freeze = load_s1_freeze(args.freeze)
    freeze_sha = verify_sidecar(args.freeze)
    b2_freeze = load_b2_freeze(Path(freeze["b2_freeze"]))
    b0, b1, _, baseline = load_compatible_artifacts(
        Path(freeze["b2_freeze"]), b2_freeze
    )
    historical_b2 = json.loads(Path(freeze["b2_result"]).read_text(encoding="utf-8"))
    selected = profile_from_payload(freeze["profile"])
    sinkhorn_config = SinkhornConfig(**freeze["sinkhorn"]["config"])
    audit_config = S1AuditConfig(**freeze["audit_config"])
    continuation_config = ContinuationConfig(**freeze["continuation_config"])
    laplace_config = b2_config()
    sinkhorn_config.validate()
    audit_config.validate()
    continuation_config.validate()
    train = cifar.cifar_pool(selected.model.image_size, "train", args.data_root)
    fresh_record = freeze["fresh_data"]
    fresh_pool, loaded_fresh = load_fresh_pool(
        Path(fresh_record["path"]),
        fresh_record["source_id"],
        selected.model.image_size,
        fresh_record["requested_float_encoding"],
    )
    if loaded_fresh["sha256"] != fresh_record["sha256"]:
        raise RuntimeError("corrected S1 developmental source bytes changed")
    allocation = fresh_evaluation_allocation(
        len(fresh_pool),
        fresh_record["source_id"],
        laplace_config,
        generated_samples=selected.evaluation.generated_samples,
        reference_samples=selected.evaluation.reference_samples,
        control_groups=3,
    )
    if allocation.digests != freeze["allocation_digests"]:
        raise RuntimeError("corrected S1/B2 allocations differ")
    reference = fresh_pool[torch.as_tensor(allocation.reference)]
    controls = tuple(
        fresh_pool[torch.as_tensor(indices)] for indices in allocation.controls
    )
    reference_features = inception_features(reference, device).double().numpy()
    b1_calibration = load_b1_calibration(b1)
    baseline_by_unit = {int(row["unit"]): row for row in baseline["rows"]}
    b2_by_unit = {int(row["unit"]): row for row in historical_b2["rows"]}

    rows: list[dict] = []
    started = time.time()
    for unit in (int(value) for value in freeze["initial_units"]):
        shard_path = args.shard_dir / f"s1_v2_unit_{unit}.json"
        if shard_path.exists():
            print(f"Reusing corrected S1 unit {unit}", flush=True)
            row = _load_shard(shard_path, freeze_sha, unit)
        else:
            orphaned = [
                args.checkpoint_dir / f"s1_v2_unit_{unit}_{arm}_ema.pt"
                for arm in CONTINUATION_ARMS
                if (args.checkpoint_dir / f"s1_v2_unit_{unit}_{arm}_ema.pt").exists()
            ]
            if orphaned:
                raise RuntimeError(f"orphan corrected S1 checkpoints: {orphaned}")
            baseline_row = baseline_by_unit[unit]
            control_group = int(baseline_row["assigned_control_group"])
            print(f"\n=== CORRECTED S1 unit {unit} ===", flush=True)
            row = _train_evaluate_unit(
                unit=unit,
                freeze=freeze,
                freeze_sha=freeze_sha,
                b0=b0,
                baseline_row=baseline_row,
                historical_b2_row=b2_by_unit[unit],
                train=train,
                fresh_pool=fresh_pool,
                allocation=allocation,
                reference_features=reference_features,
                control=controls[control_group],
                b1_calibration=b1_calibration,
                selected=selected,
                sinkhorn_config=sinkhorn_config,
                audit_config=audit_config,
                continuation_config=continuation_config,
                laplace_config=laplace_config,
                device=device,
                checkpoint_dir=args.checkpoint_dir,
            )
            write_json(shard_path, row)
            print(
                f"unit {unit}: pass={row['gate']['passes']}; "
                f"field wins={row['sinkhorn_vs_control']['paired_wins']}/"
                f"{audit_config.replicates}",
                flush=True,
            )
        rows.append(row)

    verdict = staged_verdict(rows, freeze)
    payload = {
        "status": "sinkhorn-s1-v2-initial-two",
        "confirmatory": False,
        "verdict": verdict,
        "rows": rows,
        "profile": freeze["profile"],
        "continuation_config": freeze["continuation_config"],
        "sinkhorn": freeze["sinkhorn"],
        "laplace": freeze["laplace"],
        "audit_config": freeze["audit_config"],
        "s1_v2_freeze_sha256": freeze_sha,
        "b0_sha256": freeze["b0_sha256"],
        "b2_result_sha256": freeze["b2_result_sha256"],
        "fresh_data": loaded_fresh
        | {
            "reuse_status": (
                "previously consumed by B2; reused for a developmental screen"
            )
        },
        "allocation_digests": allocation.digests,
        "elapsed_seconds": time.time() - started,
        "claim_scope": freeze["claim_scope"],
        "provenance": provenance(),
        "device": settings,
    }
    write_json(args.out, payload)
    print(f"\nS1 v2 verdict: {verdict['decision']} -- {verdict['reading']}")


if __name__ == "__main__":
    main()
