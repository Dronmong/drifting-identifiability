"""Mechanical, provenance, memory, and throughput gate for Stage B3."""

from __future__ import annotations

import argparse
import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import torch

from .. import cifar
from .. import kernel_gradient as KG
from ..device import configure, resolve_device
from ..diagnostics import write_json
from ..models import OneStepGenerator
from ..objectives import corrected_teacher
from ..stage_b2.core import b2_config
from ..stage_b2.fresh_data import load_fresh_pool
from .artifacts import DEFAULT_DATA, DEFAULT_PREFLIGHT, assert_unused, preflight_header
from .core import (
    B3_ARMS,
    B3Config,
    assert_samplewise_generator,
    build_generator,
    calibrate_operator,
    configure_deterministic_execution,
    construct_full_teacher,
    kernel_payload,
    regression_backward,
    train_b3_arm,
)
from .evaluation import evaluation_allocation


def _gradient_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros_like(parameter).flatten()
            if parameter.grad is None
            else parameter.grad.detach().flatten()
            for parameter in model.parameters()
        ]
    )


def equation_compatibility_probe(branch, kernel, device) -> dict:
    """Legacy inline Phase-30 equation vs full and chunked B3 paths."""
    latent_generator = torch.Generator(device="cpu").manual_seed(771)
    latent = torch.randn(8, 8, generator=latent_generator).to(device)
    positive = torch.randn(6, 3, 8, 8, generator=latent_generator).to(device)
    base = OneStepGenerator(8, 3, 8, 16, 991).to(device)
    assert_samplewise_generator(base)

    with torch.no_grad():
        legacy_cloud = base(latent).detach()
        legacy_drift, _ = KG.field(
            legacy_cloud,
            positive,
            legacy_cloud,
            branch,
            kernel,
            direction_mode="paper",
            normalization="rms",
            diagnostics=False,
        )
        legacy_teacher = corrected_teacher(
            legacy_cloud + 0.5 * legacy_drift, positive, mode="scalar"
        ).detach()
    helper_teacher, helper_cloud, _ = construct_full_teacher(
        base, latent, positive, branch, kernel, 0.5
    )

    full = copy.deepcopy(base)
    chunked = copy.deepcopy(base)
    with torch.no_grad():
        chunk_forward = torch.cat(
            [chunked(latent[start : start + 2]) for start in range(0, len(latent), 2)]
        )
    full.zero_grad(set_to_none=True)
    output = full(latent)
    full_loss = (output - helper_teacher).square().flatten(1).sum(1).mean()
    full_loss.backward()
    chunked.zero_grad(set_to_none=True)
    chunk_loss = regression_backward(chunked, latent, helper_teacher, 2)
    full_gradient = _gradient_vector(full)
    chunk_gradient = _gradient_vector(chunked)
    gradient_scale = float(full_gradient.norm().clamp_min(1e-30))
    gradient_relative = float((full_gradient - chunk_gradient).norm()) / gradient_scale

    full_optimizer = torch.optim.Adam(full.parameters(), lr=2e-3)
    chunk_optimizer = torch.optim.Adam(chunked.parameters(), lr=2e-3)
    full_optimizer.step()
    chunk_optimizer.step()
    full_parameters = torch.cat(
        [parameter.detach().flatten() for parameter in full.parameters()]
    )
    chunk_parameters = torch.cat(
        [parameter.detach().flatten() for parameter in chunked.parameters()]
    )
    update_scale = float(full_parameters.norm().clamp_min(1e-30))
    update_relative = float((full_parameters - chunk_parameters).norm()) / update_scale
    values = {
        "legacy_vs_helper_teacher_max_abs": float(
            (legacy_teacher - helper_teacher).abs().max()
        ),
        "legacy_vs_helper_cloud_max_abs": float(
            (legacy_cloud - helper_cloud).abs().max()
        ),
        "full_vs_chunk_forward_max_abs": float(
            (helper_cloud - chunk_forward).abs().max()
        ),
        "full_loss": float(full_loss.detach()),
        "chunk_loss": float(chunk_loss),
        "loss_abs_difference": abs(float(full_loss.detach()) - chunk_loss),
        "gradient_relative_difference": gradient_relative,
        "gradient_max_abs_difference": float(
            (full_gradient - chunk_gradient).abs().max()
        ),
        "adam_update_relative_difference": update_relative,
        "adam_update_max_abs_difference": float(
            (full_parameters - chunk_parameters).abs().max()
        ),
    }
    values["passes"] = bool(
        values["legacy_vs_helper_teacher_max_abs"] <= 2e-6
        and values["full_vs_chunk_forward_max_abs"] <= 2e-6
        and values["loss_abs_difference"] <= 2e-5
        and gradient_relative <= 5e-5
        and update_relative <= 5e-6
    )
    return values


def _calibrations(pool, config, device) -> tuple[dict, object, object]:
    records = {}
    first_branch = first_kernel = None
    for unit in config.units:
        branch, kernel, indices = calibrate_operator(pool, unit, config, device)
        records[str(unit)] = {
            "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "sample_count": len(indices),
            "with_replacement": True,
            "kernel": kernel_payload(kernel),
        }
        if first_branch is None:
            first_branch, first_kernel = branch, kernel
    return records, first_branch, first_kernel


def _resource_probe(pool, branch, kernel, config, device, steps) -> dict:
    if steps <= 0:
        raise ValueError("B3 throughput steps must be positive")
    probe_config = replace(
        config,
        steps=steps,
        checkpoint_steps=(steps,),
        log_every=max(1, steps),
        recovery_every=steps,
    )
    rows = {}
    for arm in B3_ARMS:
        outcome = train_b3_arm(
            pool,
            config.units[0],
            arm,
            probe_config,
            branch,
            kernel,
            device,
        )
        rows[arm.name] = {
            "steps": steps,
            "elapsed_seconds": outcome.wall_seconds,
            "seconds_per_step": outcome.wall_seconds / steps,
            "projected_hours_per_30000_step_unit": (
                outcome.wall_seconds / steps * 30_000 / 3600
            ),
            "peak_memory_bytes": outcome.peak_memory_bytes,
            "peak_memory_reserved_bytes": outcome.peak_memory_reserved_bytes,
            "parameters": outcome.model.parameter_count(),
        }
        del outcome
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--external-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--throughput-steps", type=int, default=100)
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_PREFLIGHT)
    args = parser.parse_args()
    assert_unused(args.out)
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    settings.update(configure_deterministic_execution(device))
    config = B3Config()
    header = preflight_header(config)

    train_pool = cifar.cifar_pool(config.image_size, "train", args.data_root)
    in_domain = cifar.cifar_pool(config.image_size, "test", args.data_root)
    shifted, shifted_record = load_fresh_pool(
        args.external_data,
        header["external_data"]["source_id"],
        config.image_size,
        "auto",
    )
    if shifted_record["sha256"] != header["external_data"]["artifact_sha256"]:
        raise RuntimeError("B3 external loader observed different bytes")
    b2 = b2_config()
    evaluation = header["profile"]["evaluation"]
    allocations = {
        "in_domain_development_reused": evaluation_allocation(
            len(in_domain),
            "cifar10-test-adaptively-reused-b3-development",
            units=len(config.units),
            generated_samples=int(evaluation["generated_samples"]),
            reference_samples=int(evaluation["reference_samples"]),
            audit_replicates=b2.audit_replicates,
            audit_batch=b2.audit_batch,
        ),
        "shifted_disjoint": evaluation_allocation(
            len(shifted),
            header["external_data"]["source_id"],
            units=len(config.units),
            generated_samples=int(evaluation["generated_samples"]),
            reference_samples=int(evaluation["reference_samples"]),
            audit_replicates=b2.audit_replicates,
            audit_batch=b2.audit_batch,
        ),
    }
    calibrations, branch, kernel = _calibrations(train_pool, config, device)
    compatibility = equation_compatibility_probe(branch, kernel, device)

    parameter_rows = {}
    for arm in B3_ARMS:
        model = build_generator(arm, config, config.units[0], device)
        assert_samplewise_generator(model)
        parameter_rows[arm.name] = model.parameter_count()
        del model
    capacity_gap = (
        abs(parameter_rows["B3-capacity"] - config.bridge_parameter_count)
        / config.bridge_parameter_count
    )
    capacity_pass = bool(
        parameter_rows == {"B3-native": 146_691, "B3-capacity": 3_864_003}
        and capacity_gap <= config.capacity_parameter_tolerance_fraction
        and all(arm.field_cloud == 256 for arm in B3_ARMS)
    )

    resources = None
    if not args.skip_throughput:
        resources = _resource_probe(
            train_pool,
            branch,
            kernel,
            config,
            device,
            args.throughput_steps,
        )
    resource_measured = bool(resources is not None and device.type == "cuda")
    memory_pass = False
    projected_total = None
    if resource_measured:
        total_memory = torch.cuda.get_device_properties(device).total_memory
        memory_pass = all(
            int(row["peak_memory_reserved_bytes"])
            <= int(total_memory * config.maximum_preflight_memory_fraction)
            for row in resources.values()
        )
        projected_total = 3 * sum(
            row["projected_hours_per_30000_step_unit"] for row in resources.values()
        )

    decision = (
        "GO"
        if all((compatibility["passes"], capacity_pass, resource_measured, memory_pass))
        else "NO-GO"
    )
    payload = {
        **header,
        "device": settings,
        "evaluation_allocations": {
            name: allocation.digests for name, allocation in allocations.items()
        },
        "calibrations": calibrations,
        "equation_compatibility": compatibility,
        "capacity_match": {
            "parameters": parameter_rows,
            "bridge_parameters": config.bridge_parameter_count,
            "relative_gap": capacity_gap,
            "same_field_cloud": True,
            "passes": capacity_pass,
        },
        "resource_probe": resources,
        "projected_training_hours_three_units_both_arms": projected_total,
        "verdict": {
            "decision": decision,
            "equation_compatibility_passes": compatibility["passes"],
            "capacity_match_passes": capacity_pass,
            "gpu_resource_probe_completed": resource_measured,
            "memory_passes": memory_pass,
            "reading": (
                "GO: exact microbatch mechanics, matched capacity, data boundaries, and GPU headroom pass"
                if decision == "GO"
                else "NO-GO: do not train B3 until every preflight item passes on the execution GPU"
            ),
        },
        "scope": "mechanical and resource readiness only; no B3 training unit was consumed",
    }
    digest = write_json(args.out, payload)
    print(payload["verdict"]["reading"])
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
