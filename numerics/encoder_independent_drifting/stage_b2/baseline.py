"""Measure the paired B0 baseline on B2's genuinely fresh confirmation pool."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .. import cifar
from ..device import configure, resolve_device
from ..diagnostics import provenance, write_json
from ..f3b import sample_model
from ..f3b_freeze import profile_from_payload
from ..fid import inception_features
from .artifacts import (
    DEFAULT_B0_RESULT,
    DEFAULT_B1_RESULT,
    DEFAULT_PREFLIGHT,
    PROTOCOL,
    file_sha256,
    load_b1_calibration,
    load_checkpoint_model,
    load_passing_prerequisites,
    load_preflight,
    source_manifest,
)
from .core import (
    B2_CONFIRMATION_UNITS,
    b2_config,
    b2_seed,
    config_payload,
    evaluation_prior_seed,
)
from .evaluation import (
    drift_energy_audit_suite,
    fresh_evaluation_allocation,
    summarize_drift_audits,
)
from .fresh_data import load_fresh_pool
from .metrics import (
    apply_vetoes,
    generated_metrics,
    matched_real_metrics,
    memorization_statistics_augmented,
)

DEFAULT_OUT = Path(__file__).resolve().parent / "b2_baseline.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--b0", type=Path, default=DEFAULT_B0_RESULT)
    parser.add_argument("--b1", type=Path, default=DEFAULT_B1_RESULT)
    parser.add_argument("--fresh-data", type=Path, required=True)
    parser.add_argument("--fresh-source-id", required=True)
    parser.add_argument(
        "--fresh-float-encoding",
        choices=("auto", "minus-one-one", "zero-one"),
        default="auto",
    )
    parser.add_argument("--attest-unused", action="store_true")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.attest_unused:
        raise RuntimeError(
            "B2 baseline requires --attest-unused: the fresh source was not used "
            "for B0/B1 selection, B2 design, training, or calibration"
        )

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    preflight = load_preflight(args.preflight)
    b0, b1 = load_passing_prerequisites(args.b0, args.b1)
    if preflight["b0_artifact_sha256"] != b0["artifact_sha256"]:
        raise RuntimeError("B2 preflight and baseline adopted different B0 results")
    if preflight["b1_artifact_sha256"] != b1["artifact_sha256"]:
        raise RuntimeError("B2 preflight and baseline adopted different B1 results")
    b1_calibration = load_b1_calibration(b1)
    selected = profile_from_payload(b0["profile"])
    config = b2_config()
    fresh_pool, fresh_record = load_fresh_pool(
        args.fresh_data,
        args.fresh_source_id,
        selected.model.image_size,
        args.fresh_float_encoding,
    )
    allocation = fresh_evaluation_allocation(
        len(fresh_pool),
        args.fresh_source_id,
        config,
        generated_samples=selected.evaluation.generated_samples,
        reference_samples=selected.evaluation.reference_samples,
        control_groups=len(B2_CONFIRMATION_UNITS),
    )
    reference = fresh_pool[torch.as_tensor(allocation.reference)]
    controls = tuple(
        fresh_pool[torch.as_tensor(indices)] for indices in allocation.controls
    )
    reference_features = inception_features(reference, device).double().numpy()
    control_rows = [
        {"group": index} | matched_real_metrics(control, reference_features, device)
        for index, control in enumerate(controls)
    ]
    train = cifar.cifar_pool(selected.model.image_size, "train", args.data_root)
    rows = []
    tau = float(preflight["tau"])
    for index, unit in enumerate(B2_CONFIRMATION_UNITS):
        print(f"\n=== B2 PAIRED B0 BASELINE unit {unit} ===", flush=True)
        model = load_checkpoint_model(
            b0["checkpoints"][str(unit)], b0["profile"], unit, device
        )
        main_seed = b2_seed("confirmation-evaluation", unit, "metric-prior")
        generated = sample_model(
            model,
            selected.evaluation.generated_samples,
            selected.model,
            selected.evaluation.nfe_ladder[0],
            main_seed,
            device,
        )
        audit_batches = [
            sample_model(
                model,
                config.audit_batch,
                selected.model,
                selected.evaluation.nfe_ladder[0],
                evaluation_prior_seed(unit, replicate),
                device,
            )
            for replicate in range(config.audit_replicates)
        ]
        audits = drift_energy_audit_suite(
            audit_batches,
            fresh_pool,
            allocation,
            tau,
            unit,
            config,
            device,
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

        # B1 is not part of the B2 gate, but its fresh-source measurement is
        # needed for a scientifically coherent three-way report.  Reusing the
        # old official-test B1 metrics here would compare different datasets.
        b1_model = load_checkpoint_model(
            b1["checkpoints"][str(unit)], b1["profile"], unit, device
        )
        b1_generated = sample_model(
            b1_model,
            selected.evaluation.generated_samples,
            selected.model,
            selected.evaluation.nfe_ladder[0],
            main_seed,
            device,
        )
        b1_audit_batches = [
            sample_model(
                b1_model,
                config.audit_batch,
                selected.model,
                selected.evaluation.nfe_ladder[0],
                evaluation_prior_seed(unit, replicate),
                device,
            )
            for replicate in range(config.audit_replicates)
        ]
        b1_audits = drift_energy_audit_suite(
            b1_audit_batches,
            fresh_pool,
            allocation,
            tau,
            unit,
            config,
            device,
        )
        b1_metrics = generated_metrics(
            b1_generated,
            reference_features,
            controls[index],
            device,
            include_fid=True,
        )
        rows.append(
            {
                "unit": unit,
                "main_evaluation_prior_seed": main_seed,
                "metrics": metrics,
                "memorization": memory,
                "veto": veto,
                "drift_audit": audits,
                "drift_summary": summarize_drift_audits(audits),
                "b1_report_only": {
                    "metrics": b1_metrics,
                    "drift_audit": b1_audits,
                    "drift_summary": summarize_drift_audits(b1_audits),
                    "same_fresh_source": True,
                    "enters_b2_gate": False,
                },
                "assigned_control_group": index,
            }
        )
        print(
            f"recall={metrics['recall']:.4f}; "
            f"drift excess={rows[-1]['drift_summary']['median_excess_over_real']:.6g}",
            flush=True,
        )
        del (
            model,
            generated,
            audit_batches,
            b1_model,
            b1_generated,
            b1_audit_batches,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    controls_pass = all(
        float(row["recall"]) > config.metric_control_floor for row in control_rows
    )
    recalls_pass = all(
        float(row["metrics"]["recall"]) > config.baseline_recall_floor for row in rows
    )
    drift_resolvable = all(
        float(row["drift_summary"]["median_excess_over_real"]) > 0 for row in rows
    )
    vetoes_pass = all(bool(row["veto"]["passes"]) for row in rows)
    decision = (
        "GO"
        if controls_pass and recalls_pass and drift_resolvable and vetoes_pass
        else "NO-GO"
    )
    payload = {
        "status": "b2-b0-paired-baseline",
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "preflight_sha256": preflight["artifact_sha256"],
        "b0_artifact_sha256": b0["artifact_sha256"],
        "b1_artifact_sha256": b1["artifact_sha256"],
        "profile": b0["profile"],
        "b2_config": config_payload(config),
        "tau": tau,
        "fresh_data": fresh_record | {"operator_attested_unused": True},
        "allocation_digests": allocation.digests,
        "rows": rows,
        "matched_real_controls": control_rows,
        "verdict": {
            "decision": decision,
            "matched_real_controls_pass": controls_pass,
            "baseline_recall_pass": recalls_pass,
            "drift_excess_resolvable": drift_resolvable,
            "baseline_vetoes_pass": vetoes_pass,
            "reading": (
                "GO: the fresh instrument resolves B0 coverage and positive "
                "drift-energy excess in every paired unit"
                if decision == "GO"
                else "NO-GO: B2 cannot test its frozen reduction gate on this source"
            ),
        },
        "claim_scope": (
            "fresh-source paired B0 measurement; no B2 candidate has been trained"
        ),
        "provenance": provenance(),
        "device": settings,
    }
    write_json(args.out, payload)
    print(f"B2 baseline: {decision}", flush=True)


if __name__ == "__main__":
    main()
