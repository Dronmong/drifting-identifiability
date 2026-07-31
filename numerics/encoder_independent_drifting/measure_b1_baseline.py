"""Measure the paired B0 baseline under B1's fresh, frozen instruments.

No B1 model is trained here.  This stage establishes the exact official-test
metric controls, inference priors, audit banks, target batches, and B0 audit
denominators that the later confirmation must reuse.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from . import cifar
from .b1 import (
    B1_CONFIRMATION_UNITS,
    b1_config,
    config_payload,
    evaluation_prior_seed,
    load_b0_checkpoint_model,
)
from .b1_evaluation import (
    anchor_audit_suite,
    apply_b1_vetoes,
    confirmation_images,
    evaluation_allocation,
    memorization_statistics_augmented,
    summarize_anchor_audit,
)
from .b1_freeze import (
    DEFAULT_B0_RESULT,
    DEFAULT_CALIBRATION,
    HERE,
    PROTOCOL,
    file_sha256,
    load_calibration,
    source_manifest,
)
from .device import configure, resolve_device
from .diagnostics import provenance, write_json
from .f3b import sample_model
from .f3b_evaluation import generated_metrics, matched_real_metrics
from .f3b_freeze import profile_from_payload
from .fid import inception_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", type=Path, default=DEFAULT_B0_RESULT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", type=Path, default=HERE / "b1_baseline.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    b0, calibration = load_calibration(args.calibration, args.b0)
    selected = profile_from_payload(b0["profile"])
    config = b1_config()
    allocation = evaluation_allocation(
        config,
        selected.evaluation.generated_samples,
        selected.evaluation.reference_samples,
        len(B1_CONFIRMATION_UNITS),
    )
    if allocation.digests != calibration["allocation_digests"]:
        raise RuntimeError("B1 allocation changed after calibration")
    train = cifar.cifar_pool(selected.model.image_size, "train", args.data_root)
    reference, controls, target_pairs = confirmation_images(
        allocation, selected.model.image_size, args.data_root
    )
    reference_features = inception_features(reference, device).double().numpy()
    control_rows = [
        {"group": index, **matched_real_metrics(control, reference_features, device)}
        for index, control in enumerate(controls)
    ]

    started = time.time()
    historical = {int(row["unit"]): row for row in b0.get("rows", [])}
    rows = []
    for index, unit in enumerate(B1_CONFIRMATION_UNITS):
        print(f"\n=== B1 PAIRED B0 BASELINE unit {unit} ===", flush=True)
        model = load_b0_checkpoint_model(
            b0["checkpoints"][str(unit)],
            b0["profile"],
            selected.model,
            unit,
            device,
        )
        prior_seed = evaluation_prior_seed(unit)
        generated = sample_model(
            model,
            selected.evaluation.generated_samples,
            selected.model,
            selected.evaluation.nfe_ladder[0],
            prior_seed,
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
            float(calibration["normalizer"]),
            device,
        )
        veto = apply_b1_vetoes(metrics, memory, calibration["thresholds"])
        audits = anchor_audit_suite(
            model,
            selected.model,
            selected.evaluation.nfe_ladder[0],
            unit,
            float(calibration["scale"]),
            target_pairs,
            device,
            config,
        )
        summary = summarize_anchor_audit(audits)
        row = {
            "unit": unit,
            "checkpoint": b0["checkpoints"][str(unit)],
            "main_evaluation_prior_seed": prior_seed,
            "metrics": metrics,
            "historical_b0_metrics": historical.get(unit, {}).get("metrics"),
            "memorization": memory,
            "veto": veto,
            "anchor_audit": audits,
            "anchor_summary": summary,
            "assigned_control_group": index,
        }
        rows.append(row)
        print(
            f"recall={metrics['recall']:.4f}; "
            f"median anchor excess="
            f"{summary['median_biased_excess_over_real']:.6g}; "
            f"veto={veto['passes']}",
            flush=True,
        )
        del model, generated
        if device.type == "cuda":
            torch.cuda.empty_cache()

    controls_valid = all(
        float(row["recall"]) > config.metric_control_floor for row in control_rows
    )
    vetoes_valid = all(row["veto"]["passes"] for row in rows)
    audit_resolvable = all(
        row["anchor_summary"]["median_biased_excess_over_real"] > 0
        and row["anchor_summary"]["replicates_above_real_floor"]
        >= config.anchor_paired_wins_required
        for row in rows
    )
    decision = "GO" if controls_valid and vetoes_valid and audit_resolvable else "NO-GO"
    verdict = {
        "decision": decision,
        "controls_valid": controls_valid,
        "b0_vetoes_valid": vetoes_valid,
        "audit_resolvable": audit_resolvable,
        "reading": (
            f"{decision}: controls={controls_valid}; B0 vetoes={vetoes_valid}; "
            f"positive finite-sample audit denominator={audit_resolvable}"
        ),
    }
    payload = {
        "status": "b1-b0-paired-baseline",
        "protocol": "numerics/EncoderIndependentB1Protocol.md",
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "b0_artifact": str(args.b0),
        "b0_artifact_sha256": b0["artifact_sha256"],
        "calibration_artifact": str(args.calibration),
        "calibration_sha256": calibration["artifact_sha256"],
        "provenance": provenance(),
        "device": settings,
        "profile": b0["profile"],
        "b1_config": config_payload(config),
        "allocation_digests": allocation.digests,
        "matched_real_controls": control_rows,
        "rows": rows,
        "verdict": verdict,
        "elapsed_seconds": time.time() - started,
        "claim_scope": (
            "fresh paired B0 measurement under B1's frozen official-test "
            "metrics and common-random spectral audit; no B1 outcome"
        ),
    }
    digest = write_json(args.out, payload)
    print("\n=== B1 PAIRED B0 BASELINE ===")
    print(verdict["reading"])
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
