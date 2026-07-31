"""Outcome-blind calibration and resource preflight for paired Stage B1.

This stage may inspect target data, immutable B0 checkpoints, gradient scales,
and hardware use.  It does not train or evaluate a B1 candidate.  Its output
freezes the anchor weight and the collapse/memorization vetoes before any B1
outcome exists.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import cifar
from .b1 import (
    B1_CALIBRATION_UNITS,
    anchor_target_batch,
    anchor_term,
    b1_config,
    b1_seed,
    b1_streams,
    build_bank_for_dimension,
    build_training_bank,
    calibrate_projected_scale,
    calibrated_event_lambda,
    config_payload,
    load_b0_checkpoint_model,
    parameter_gradient_norm,
)
from .b1_evaluation import (
    augmented_real_health,
    calibration_images,
    evaluation_allocation,
    memorization_statistics_augmented,
)
from .b1_freeze import (
    DEFAULT_B0_RESULT,
    HERE,
    PROTOCOL,
    file_sha256,
    load_adopted_b0,
    source_manifest,
)
from .device import configure, resolve_device
from .diagnose_phase15 import gaussian_moment_match
from .diagnostics import provenance, write_json
from .f3b import (
    TimeConditionedUNet,
    bridge_streams,
    independent_bridge_batch,
    sample_model,
)
from .f3b_evaluation import matched_real_metrics
from .f3b_freeze import profile_from_payload
from .fid import inception_features
from .spectral_anchor import anchor_loss


def _gradient_calibration(
    train: torch.Tensor,
    selected,
    scale: float,
    device: torch.device,
    config,
) -> tuple[list[dict], float, dict]:
    """Measure λ from independent, outcome-blind model/batch draws."""
    rows: list[dict] = []
    dimension = selected.model.channels * selected.model.image_size**2
    global_started = time.perf_counter()
    maximum_peak = 0
    for unit in B1_CALIBRATION_UNITS:
        model = TimeConditionedUNet(
            selected.model, b1_seed("calibration", unit, "model-init")
        ).to(device)
        streams = bridge_streams("b1-calibration", unit)
        mixed, target, _, _, time_value = independent_bridge_batch(
            train,
            selected.train.batch,
            streams,
            device,
            selected.train.horizontal_flip,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = model(mixed, time_value)
        flow = F.mse_loss(prediction, target)
        flow_norm = parameter_gradient_norm(flow, model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        flow_probe_seconds = time.perf_counter() - started

        anchor_streams = b1_streams("calibration", unit)
        real = anchor_target_batch(
            train,
            anchor_streams,
            config.anchor_batch,
            selected.train.horizontal_flip,
        )
        bank = build_training_bank(scale, dimension, "calibration", unit, config)
        anchor, anchor_record = anchor_term(
            model,
            bank,
            real,
            selected.model,
            anchor_streams.anchor_prior,
            device,
            progress=0.5,
            config=config,
        )
        anchor_norm = parameter_gradient_norm(anchor, model)
        lambda_event = calibrated_event_lambda(flow_norm, anchor_norm, config)
        combined = flow + lambda_event * anchor
        model.zero_grad(set_to_none=True)
        combined.backward()
        combined_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), selected.train.gradient_clip
            )
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
        else:
            peak = 0
        maximum_peak = max(maximum_peak, peak)
        row = {
            "unit": unit,
            "flow_loss": float(flow.detach()),
            "anchor_loss": float(anchor.detach()),
            "flow_gradient_norm": flow_norm,
            "anchor_gradient_norm": anchor_norm,
            "lambda_event": lambda_event,
            "weighted_event_gradient_ratio": (lambda_event * anchor_norm / flow_norm),
            "combined_gradient_norm_before_clip": combined_norm,
            "peak_memory_bytes": peak,
            "elapsed_seconds": time.perf_counter() - started,
            "flow_probe_seconds": flow_probe_seconds,
            "anchor": anchor_record,
        }
        if not all(
            math.isfinite(float(row[name])) and float(row[name]) > 0
            for name in (
                "flow_loss",
                "anchor_loss",
                "flow_gradient_norm",
                "anchor_gradient_norm",
                "lambda_event",
                "combined_gradient_norm_before_clip",
            )
        ):
            raise FloatingPointError(f"invalid B1 gradient calibration row: {row}")
        rows.append(row)
        del combined, anchor, flow, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    chosen = float(np.median([row["lambda_event"] for row in rows]))
    for row in rows:
        row["frozen_lambda_weighted_event_gradient_ratio"] = (
            chosen
            * float(row["anchor_gradient_norm"])
            / float(row["flow_gradient_norm"])
        )
    resource = {
        "maximum_peak_memory_bytes": maximum_peak,
        "gradient_calibration_seconds": time.perf_counter() - global_started,
        "mean_calibration_seconds_per_unit": float(
            np.mean([row["elapsed_seconds"] for row in rows])
        ),
        "mean_flow_probe_seconds": float(
            np.mean([row["flow_probe_seconds"] for row in rows])
        ),
        "mean_incremental_anchor_probe_seconds": float(
            np.mean(
                [row["elapsed_seconds"] - row["flow_probe_seconds"] for row in rows]
            )
        ),
        "timing_caveat": (
            "Calibration computes separate gradient norms and then a combined "
            "backward pass, so its incremental timing is a conservative "
            "diagnostic, not a prediction of confirmation wall time."
        ),
    }
    return rows, chosen, resource


def _instrument_sensitivity(
    reference: torch.Tensor,
    controls: tuple[torch.Tensor, ...],
    b0: dict,
    selected,
    scale: float,
    device: torch.device,
    config,
) -> dict:
    """Check the frozen audit instrument against two obvious alternatives."""
    dimension = selected.model.channels * selected.model.image_size**2
    rows = []
    for index, (unit, real_control) in enumerate(
        zip((300, 301, 302), controls, strict=True)
    ):
        bank = build_bank_for_dimension(
            scale,
            dimension,
            "calibration-instrument",
            index,
            "audit-bank",
            config.audit_features,
            config,
        )
        gaussian = gaussian_moment_match(
            reference,
            len(real_control),
            np.random.default_rng(b1_seed("calibration-instrument", index, "gaussian")),
        )
        model = load_b0_checkpoint_model(
            b0["checkpoints"][str(unit)],
            b0["profile"],
            selected.model,
            unit,
            device,
        )
        b0_samples = sample_model(
            model,
            len(real_control),
            selected.model,
            selected.evaluation.nfe_ladder[0],
            b1_seed("calibration-instrument", unit, "b0-prior"),
            device,
        )
        with torch.no_grad():
            floor = float(anchor_loss(bank, real_control, reference, "biased"))
            gaussian_value = float(anchor_loss(bank, gaussian, reference, "biased"))
            b0_value = float(anchor_loss(bank, b0_samples, reference, "biased"))
        rows.append(
            {
                "replicate": index,
                "real_real_biased": floor,
                "gaussian_biased": gaussian_value,
                "b0_biased": b0_value,
                "gaussian_excess_over_real": gaussian_value - floor,
                "b0_excess_over_real": b0_value - floor,
            }
        )
        del model, b0_samples
    gaussian_positive = sum(row["gaussian_excess_over_real"] > 0 for row in rows)
    b0_positive = sum(row["b0_excess_over_real"] > 0 for row in rows)
    valid = bool(
        gaussian_positive >= 2
        and b0_positive >= 2
        and np.median([row["gaussian_excess_over_real"] for row in rows]) > 0
        and np.median([row["b0_excess_over_real"] for row in rows]) > 0
    )
    return {
        "rows": rows,
        "gaussian_positive_replicates": gaussian_positive,
        "b0_positive_replicates": b0_positive,
        "passes": valid,
        "interpretation": (
            "The independent finite audit bank resolves both a moment-matched "
            "Gaussian and the adopted B0 outputs above the matched-real floor."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", type=Path, default=DEFAULT_B0_RESULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", type=Path, default=HERE / "b1_calibration.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    b0 = load_adopted_b0(args.b0)
    selected = profile_from_payload(b0["profile"])
    config = b1_config()
    train = cifar.cifar_pool(selected.model.image_size, "train", args.data_root)
    scale, scale_indices = calibrate_projected_scale(train, config)
    allocation = evaluation_allocation(
        config,
        selected.evaluation.generated_samples,
        selected.evaluation.reference_samples,
        len(B1_CALIBRATION_UNITS),
    )
    reference, controls = calibration_images(
        allocation, selected.model.image_size, args.data_root
    )
    reference_features = inception_features(reference, device).double().numpy()
    control_rows = [
        {"group": index, **matched_real_metrics(control, reference_features, device)}
        for index, control in enumerate(controls)
    ]
    health_rows = [
        {
            "group": index,
            **augmented_real_health(control, train, device),
        }
        for index, control in enumerate(controls)
    ]
    normalizer = float(np.median([row["real_nn_scale"] for row in health_rows]))
    memorization_rows = [
        {
            "group": index,
            **memorization_statistics_augmented(control, train, normalizer, device),
        }
        for index, control in enumerate(controls)
    ]
    thresholds = {
        "effective_rank": 0.5 * min(row["effective_rank"] for row in health_rows),
        "one_minus_duplicate_rate": 0.5
        * min(row["one_minus_duplicate_rate"] for row in health_rows),
        "nn_diversity": 0.5 * min(row["nn_diversity"] for row in health_rows),
        "nearest_train_or_flip_normalized": 0.5
        * min(row["nearest_train_or_flip_normalized"] for row in memorization_rows),
    }
    gradient_rows, lambda_event, resource = _gradient_calibration(
        train, selected, scale, device, config
    )
    expected_anchor_events = selected.train.steps // config.anchor_every
    historical_b0_median_seconds = float(
        np.median([row["training"]["wall_seconds"] for row in b0["rows"]])
    )
    resource["expected_anchor_events"] = expected_anchor_events
    resource["historical_b0_median_wall_seconds"] = historical_b0_median_seconds
    resource["conservative_projected_added_seconds"] = (
        expected_anchor_events * resource["mean_incremental_anchor_probe_seconds"]
    )
    sensitivity = _instrument_sensitivity(
        reference, controls, b0, selected, scale, device, config
    )
    total_memory = (
        int(torch.cuda.get_device_properties(device).total_memory)
        if device.type == "cuda"
        else None
    )
    resource["device_total_memory_bytes"] = total_memory
    resource["peak_memory_fraction"] = (
        resource["maximum_peak_memory_bytes"] / total_memory if total_memory else None
    )
    resource["passes"] = bool(
        total_memory is None or resource["peak_memory_fraction"] < 0.90
    )
    controls_valid = all(
        float(row["recall"]) > config.metric_control_floor for row in control_rows
    )
    thresholds_valid = all(
        math.isfinite(float(value)) and float(value) > 0
        for value in thresholds.values()
    )
    ratio_low = config.event_gradient_ratio / config.calibration_ratio_tolerance_factor
    ratio_high = config.event_gradient_ratio * config.calibration_ratio_tolerance_factor
    gradients_valid = all(
        ratio_low
        <= float(row["frozen_lambda_weighted_event_gradient_ratio"])
        <= ratio_high
        for row in gradient_rows
    )
    decision = (
        "GO"
        if controls_valid
        and thresholds_valid
        and gradients_valid
        and resource["passes"]
        and sensitivity["passes"]
        else "NO-GO"
    )
    verdict = {
        "decision": decision,
        "controls_valid": controls_valid,
        "thresholds_valid": thresholds_valid,
        "gradients_valid": gradients_valid,
        "frozen_lambda_gradient_ratio_interval": [ratio_low, ratio_high],
        "resource_valid": resource["passes"],
        "instrument_valid": sensitivity["passes"],
        "reading": (
            f"{decision}: controls={controls_valid}; thresholds="
            f"{thresholds_valid}; gradients={gradients_valid}; resource="
            f"{resource['passes']}; instrument={sensitivity['passes']}"
        ),
    }
    payload = {
        "status": "b1-calibration",
        "protocol": "numerics/EncoderIndependentB1Protocol.md",
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "b0_artifact": str(args.b0),
        "b0_artifact_sha256": b0["artifact_sha256"],
        "provenance": provenance(),
        "device": settings,
        "profile": b0["profile"],
        "b1_config": config_payload(config),
        "allocation_digests": allocation.digests,
        "scale": scale,
        "scale_indices_digest": hashlib.sha256(
            np.asarray(scale_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "scale_indices_count": len(scale_indices),
        "lambda_event": lambda_event,
        "effective_average_gradient_ratio": config.effective_gradient_ratio,
        "gradient_calibration": gradient_rows,
        "resource_preflight": resource,
        "health_controls": health_rows,
        "memorization_controls": memorization_rows,
        "matched_real_controls": control_rows,
        "normalizer": normalizer,
        "thresholds": thresholds,
        "instrument_sensitivity": sensitivity,
        "verdict": verdict,
        "claim_scope": (
            "outcome-blind resource, gradient, metric, and audit-instrument "
            "calibration; no B1 candidate was trained or inspected"
        ),
    }
    digest = write_json(args.out, payload)
    print("\n=== B1 PREFLIGHT + CALIBRATION ===")
    print(verdict["reading"])
    print(
        f"lambda_event={lambda_event:.6g}; "
        f"average-event ratio={config.effective_gradient_ratio:.4f}"
    )
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
