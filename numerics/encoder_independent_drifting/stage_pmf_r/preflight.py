"""Mathematical, gradient, inference, and cost checks for S3R."""

from __future__ import annotations

import argparse
import gc
import os
import time
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path

import torch

from ..device import configure, resolve_device
from ..diagnostics import provenance, write_json
from ..stage_pmf.objective import TriangleSample, sample_time_triangle
from .audit import HERE, source_digest, source_manifest
from .config import S3R_ARMS, profile
from .diagnostics import endpoint_health, haar_subbands
from .model import RepairedPixelMeanFlowTransformer
from .objectives import (
    alpha_flow_loss,
    alpha_schedule,
    emf_local_difference,
    emf_x1_loss,
    one_step_sample,
    pmf_loss,
)
from .training import s3r_seed


def _activate(model: RepairedPixelMeanFlowTransformer, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.pixel_head.weight.copy_(
            torch.randn(model.pixel_head.weight.shape, generator=generator) * 1e-3
        )
        model.auxiliary_pixel_head.weight.copy_(
            torch.randn(model.auxiliary_pixel_head.weight.shape, generator=generator)
            * 1e-3
        )
        for block in (*model.encoder, *model.decoder, *model.auxiliary_blocks):
            block.attention_scale.fill_(0.1)
            block.mlp_scale.fill_(0.1)


def mathematical_checks(device: torch.device) -> dict:
    selected = profile("smoke")
    model = (
        RepairedPixelMeanFlowTransformer(selected.model, seed=91).double().to(device)
    )
    _activate(model, 92)
    generator = torch.Generator().manual_seed(93)
    clean = torch.randn(4, 3, 8, 8, generator=generator, dtype=torch.float64).to(device)
    noise = torch.randn(4, 3, 8, 8, generator=generator, dtype=torch.float64).to(device)
    triangle = TriangleSample(
        t=torch.tensor([0.8, 0.7, 0.9, 0.6], dtype=torch.float64),
        r=torch.tensor([0.2, 0.7, 0.4, 0.1], dtype=torch.float64),
        diagonal=torch.tensor([False, True, False, False]),
    )

    alpha_one = alpha_flow_loss(model, clean, noise, triangle, selected.objective, 1.0)
    diagonal_triangle = TriangleSample(
        t=triangle.t,
        r=triangle.t,
        diagonal=torch.ones_like(triangle.diagonal),
    )
    alpha_diagonal_one = alpha_flow_loss(
        model, clean, noise, diagonal_triangle, selected.objective, 1.0
    )
    alpha_diagonal_small = alpha_flow_loss(
        model, clean, noise, diagonal_triangle, selected.objective, 0.125
    )
    schedule = [
        alpha_schedule(step, 10_000, selected.objective)
        for step in (0, 1_000, 5_000, 9_000, 10_000)
    ]

    # Equation-18 local quotient versus the exact directional JVP, holding the
    # stopped boundary field fixed as the EMF construction requires.
    t = triangle.t.to(device)
    r = triangle.r.to(device)
    state = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    with torch.no_grad():
        boundary = model(state, t, torch.zeros_like(t))
    direction_state = (boundary - state) / t[:, None, None, None]

    def field(z_value, t_value, h_value):
        return model(z_value, t_value, h_value)

    _, exact = torch.func.jvp(
        field,
        (state, t, t - r),
        (direction_state, -torch.ones_like(t), -torch.ones_like(t)),
    )
    _, _, finite = emf_local_difference(model, state, t, r, 1e-6)
    exact = exact * ((t - r) > 1e-6)[:, None, None, None]
    emf_relative = float(
        (finite - exact).norm().double() / exact.norm().double().clamp_min(1e-30)
    )

    bands = haar_subbands(clean)
    input_energy = clean.square().sum()
    band_energy = sum(value.square().sum() for value in bands.values())
    haar_relative = float((input_energy - band_energy).abs() / input_energy)

    model.zero_grad(set_to_none=True)
    pmf = pmf_loss(model, clean, noise, triangle, selected.objective)
    pmf.loss.backward()
    auxiliary_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("auxiliary_blocks")
    ]
    deep_aux_grad = any(
        gradient is not None and bool((gradient != 0).any())
        for gradient in auxiliary_gradients
    )

    calls = 0

    def counted(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.auxiliary_pixel_head.register_forward_hook(counted)
    try:
        with torch.no_grad():
            sampled = one_step_sample(model, noise)
    finally:
        handle.remove()

    target = clean
    health = endpoint_health(sampled, target)
    constant_health = endpoint_health(torch.full_like(target, 0.75), target)
    return {
        "alpha_one_is_tfm_raw_mse": float(alpha_one.raw_mse),
        "alpha_diagonal_target_invariant": bool(
            torch.allclose(
                alpha_diagonal_one.raw_mse,
                alpha_diagonal_small.raw_mse,
                atol=1e-12,
                rtol=1e-12,
            )
        ),
        "alpha_adaptive_epsilon": selected.objective.alpha_adaptive_epsilon,
        "alpha_schedule": schedule,
        "alpha_schedule_monotone": all(
            left >= right for left, right in pairwise(schedule)
        ),
        "alpha_schedule_floor": min(schedule),
        "emf_difference_jvp_relative_error": emf_relative,
        "haar_energy_relative_error": haar_relative,
        "deep_auxiliary_gradient": deep_aux_grad,
        "auxiliary_calls_at_inference": calls,
        "one_step_shape_ok": sampled.shape == noise.shape,
        "endpoint_health_fields_present": bool(
            health["rank_interpretable"] is not None and health["haar"]
        ),
        "constant_output_rejected_as_rank_evidence": bool(
            not constant_health["rank_interpretable"]
        ),
    }


def arm_gradient_checks(device: torch.device) -> dict[str, dict]:
    selected = profile("smoke")
    generator = torch.Generator().manual_seed(104)
    pool = torch.rand(16, 3, 8, 8, generator=generator) * 2 - 1
    clean = pool[:4].to(device)
    noise = torch.randn(clean.shape, generator=generator).to(device)
    triangle = sample_time_triangle(
        4,
        selected.objective,
        torch.Generator().manual_seed(105),
        torch.Generator().manual_seed(106),
    )
    reports = {}
    for arm in S3R_ARMS:
        config = replace(selected.model, condition_on_absolute_time=(arm != "pmf"))
        model = RepairedPixelMeanFlowTransformer(config, seed=107).to(device)
        started = time.time()
        if arm == "pmf":
            outcome = pmf_loss(model, clean, noise, triangle, selected.objective)
        elif arm == "alpha":
            outcome = alpha_flow_loss(
                model, clean, noise, triangle, selected.objective, 0.25
            )
        else:
            outcome = emf_x1_loss(model, clean, noise, triangle, selected.objective)
        outcome.loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        reports[arm] = {
            "finite_loss": bool(torch.isfinite(outcome.loss)),
            "finite_gradients": all(torch.isfinite(value).all() for value in gradients),
            "nonzero_gradient": any(bool((value != 0).any()) for value in gradients),
            "seconds": time.time() - started,
        }
    return reports


def full_shape_cost_checks(selected, device: torch.device) -> dict[str, dict]:
    """Measure one production-shaped microbatch for each arm, sequentially."""
    generator = torch.Generator().manual_seed(204)
    batch = selected.train.micro_batch
    clean_cpu = (
        torch.rand(
            batch,
            selected.model.channels,
            selected.model.image_size,
            selected.model.image_size,
            generator=generator,
        )
        * 2
        - 1
    )
    noise_cpu = torch.randn(clean_cpu.shape, generator=generator)
    triangle = sample_time_triangle(
        batch,
        selected.objective,
        torch.Generator().manual_seed(205),
        torch.Generator().manual_seed(206),
    )
    reports = {}
    for arm in S3R_ARMS:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        config = replace(selected.model, condition_on_absolute_time=(arm != "pmf"))
        model = RepairedPixelMeanFlowTransformer(
            config, seed=s3r_seed("preflight", 0, "model-init")
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=selected.train.learning_rate,
            betas=(selected.train.beta1, selected.train.beta2),
            weight_decay=selected.train.weight_decay,
        )
        clean, noise = clean_cpu.to(device), noise_cpu.to(device)
        started = time.time()
        if arm == "pmf":
            outcome = pmf_loss(model, clean, noise, triangle, selected.objective)
        elif arm == "alpha":
            outcome = alpha_flow_loss(
                model, clean, noise, triangle, selected.objective, 0.25
            )
        else:
            outcome = emf_x1_loss(model, clean, noise, triangle, selected.objective)
        outcome.loss.backward()
        microbatch_seconds = time.time() - started
        optimizer_started = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), selected.train.gradient_clip)
        optimizer.step()
        optimizer_seconds = time.time() - optimizer_started
        projected_update_seconds = (
            microbatch_seconds * selected.train.accumulation_steps + optimizer_seconds
        )
        reports[arm] = {
            "micro_batch": batch,
            "seconds_per_microbatch": microbatch_seconds,
            "optimizer_seconds": optimizer_seconds,
            "projected_seconds_per_optimizer_update": projected_update_seconds,
            "projected_hours_for_arm": (
                projected_update_seconds * selected.train.updates / 3600
            ),
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "parameters": model.parameter_count(),
            "inference_parameters": model.inference_parameter_count(),
            "finite": bool(torch.isfinite(outcome.loss)),
        }
        del outcome, optimizer, model, clean, noise
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("smoke", "developmental"), default="smoke"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "s3r_preflight.json")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    selected = profile(args.profile)
    math = mathematical_checks(device)
    arms = arm_gradient_checks(device)
    cost = full_shape_cost_checks(selected, device)
    passes = bool(
        math["alpha_schedule_monotone"]
        and math["alpha_diagonal_target_invariant"]
        and math["alpha_adaptive_epsilon"] == 0.001
        and math["alpha_schedule_floor"] >= selected.objective.alpha_floor
        and math["emf_difference_jvp_relative_error"] < 1e-3
        and math["haar_energy_relative_error"] < 1e-12
        and math["deep_auxiliary_gradient"]
        and math["auxiliary_calls_at_inference"] == 0
        and math["one_step_shape_ok"]
        and math["constant_output_rejected_as_rank_evidence"]
        and all(
            report["finite_loss"]
            and report["finite_gradients"]
            and report["nonzero_gradient"]
            for report in arms.values()
        )
        and all(report["finite"] for report in cost.values())
    )
    payload = {
        "status": "s3r-preflight-passed" if passes else "s3r-preflight-failed",
        "launch_authorized": False,
        "scope": "mechanics and train-only developmental cost; never final-run authorization",
        "research_plan": "numerics/EncoderIndependentS3FailureResearch.md",
        "profile": asdict(selected),
        "source_manifest": source_manifest(),
        "source_sha256": source_digest(),
        "provenance": provenance(),
        "device": settings,
        "mathematical_checks": math,
        "arm_gradient_checks": arms,
        "full_shape_cost_checks": cost,
        "passes": passes,
    }
    digest = write_json(args.out, payload)
    print(f"S3R preflight passes={passes}; wrote {args.out} sha256={digest}")
    if not passes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
