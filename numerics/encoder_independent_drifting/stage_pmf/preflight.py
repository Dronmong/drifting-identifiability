"""Outcome-blind mechanics and cost preflight for local pMF S3."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from ..device import configure, resolve_device
from ..diagnostics import provenance, write_json
from .audit import HERE, profile_digest, profile_payload, source_digest, source_manifest
from .config import profile
from .data import automobile_data
from .data import manifest as data_manifest
from .model import PixelMeanFlowTransformer
from .objective import average_velocity, one_step_sample
from .training import pmf_seed, train_pmf


class ConstantVelocityPixels(nn.Module):
    def __init__(self, velocity: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(self, state, t, interval):
        del interval
        return state - t[:, None, None, None] * self.velocity


class CountingModel(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, *args):
        self.calls += 1
        return self.inner(*args)


def analytic_checks(device: torch.device) -> dict:
    state = torch.tensor([[[[0.2, -0.3], [0.4, 0.1]]]], device=device)
    velocity = torch.tensor([[[[0.7, -0.4], [0.2, 0.5]]]], device=device)
    model = ConstantVelocityPixels(velocity)
    t = torch.tensor([0.8], device=device)
    r = torch.tensor([0.2], device=device)

    def fn(z, tv, rv):
        return average_velocity(model, z, tv, rv, 0.05)

    average, derivative = torch.func.jvp(
        fn, (state, t, r), (velocity, torch.ones_like(t), torch.zeros_like(r))
    )
    counting = CountingModel(model)
    sampled = one_step_sample(counting, state)
    return {
        "constant_velocity_max_error": float((average - velocity).abs().max()),
        "constant_velocity_jvp_max": float(derivative.abs().max()),
        "one_step_calls": counting.calls,
        "one_step_expected_max_error": float(
            (sampled - (state - velocity)).abs().max()
        ),
    }


def finite_difference_check(device: torch.device) -> dict:
    selected = profile("smoke")
    model = PixelMeanFlowTransformer(
        selected.model, pmf_seed("preflight", 0, "finite-difference-model")
    ).to(device=device, dtype=torch.float64)
    with torch.no_grad():
        # The production zero head is correct initialization; make this test
        # nontrivial so it also exercises time and interval conditioning.
        generator = torch.Generator(device="cpu").manual_seed(93)
        model.pixel_head.weight.copy_(
            torch.randn(
                model.pixel_head.weight.shape,
                generator=generator,
                dtype=torch.float64,
            ).to(device)
            * 1e-3
        )
        for block in (*model.encoder, *model.decoder):
            block.attention_scale.fill_(0.1)
            block.mlp_scale.fill_(0.1)
    z = torch.randn(
        2,
        3,
        8,
        8,
        generator=torch.Generator().manual_seed(81),
        dtype=torch.float64,
    ).to(device)
    t = torch.full((2,), 0.7, device=device, dtype=torch.float64)
    r = torch.full((2,), 0.25, device=device, dtype=torch.float64)

    def fn(zv, tv, rv):
        return average_velocity(model, zv, tv, rv, 0.05)

    with torch.no_grad():
        tangent = fn(z, t, t)
    _, actual = torch.func.jvp(
        fn, (z, t, r), (tangent, torch.ones_like(t), torch.zeros_like(r))
    )
    # Float64 separates truncation error from float32 cancellation while the
    # same function is still trained in full float32.
    epsilon = 1e-5
    plus = fn(z + epsilon * tangent, t + epsilon, r)
    minus = fn(z - epsilon * tangent, t - epsilon, r)
    finite = (plus - minus) / (2 * epsilon)
    difference = (actual - finite).double()
    relative = difference.norm() / finite.double().norm().clamp_min(1e-12)
    return {
        "jvp_fd_relative_error": float(relative),
        "jvp_fd_max_absolute_error": float(difference.abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "local_s3"), default="smoke")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "s3_preflight.json")
    args = parser.parse_args()
    if args.updates <= 0:
        raise ValueError("updates must be positive")
    # Required by deterministic CUDA matmul/einsum on CUDA >= 10.2.  This is
    # set before device resolution performs any CUDA work.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    selected = profile(args.profile)
    short = replace(
        selected,
        train=replace(
            selected.train,
            updates=args.updates,
            log_every=1,
            checkpoint_updates=(args.updates,),
        ),
    )
    synthetic = (
        torch.rand(
            max(16, 2 * short.train.micro_batch),
            short.model.channels,
            short.model.image_size,
            short.model.image_size,
            generator=torch.Generator().manual_seed(55),
        )
        * 2
        - 1
    )
    outcome = train_pmf(synthetic, short, "preflight", 0, device)
    analytic = analytic_checks(device)
    finite = finite_difference_check(device)
    data = None if args.skip_data else data_manifest(automobile_data(args.data_root))
    seconds_per_update = outcome.wall_seconds / args.updates
    mechanics_pass = bool(
        analytic["constant_velocity_max_error"] < 1e-5
        and analytic["constant_velocity_jvp_max"] < 1e-5
        and analytic["one_step_calls"] == 1
        and analytic["one_step_expected_max_error"] < 1e-5
        and finite["jvp_fd_relative_error"] < 2e-2
        and all(
            torch.isfinite(value).all() for value in outcome.model.state_dict().values()
        )
        and (data is None or (data["train_count"], data["test_count"]) == (5000, 1000))
    )
    payload = {
        "status": "pmf-s3-ready-for-audit"
        if mechanics_pass
        else "pmf-s3-preflight-failed",
        "launch_authorized": False,
        "scope": "mechanics, data provenance, memory, and timing only; no sample-quality inspection",
        "protocol": "numerics/EncoderIndependentPMFS3Protocol.md",
        "provenance": provenance(),
        "device": settings,
        "profile": profile_payload(selected),
        "profile_sha256": profile_digest(selected),
        "source_manifest": source_manifest(),
        "source_sha256": source_digest(),
        "preflight_updates": args.updates,
        "model_parameters": outcome.model.parameter_count(),
        "inference_parameters": outcome.model.inference_parameter_count(),
        "seconds_per_update": seconds_per_update,
        "projected_hours_per_unit": seconds_per_update * selected.train.updates / 3600,
        "projected_hours_two_units_sequential": (
            seconds_per_update * selected.train.updates * 2 / 3600
        ),
        "peak_memory_bytes": outcome.peak_memory_bytes,
        "analytic": analytic,
        "finite_difference": finite,
        "data": data,
        "training_history": outcome.history,
        "verdict": {
            "passes": mechanics_pass,
            "next_action": "independent code/method audit; do not launch full run",
        },
    }
    digest = write_json(args.out, payload)
    print("=== LOCAL pMF S3 PREFLIGHT ===")
    print(f"parameters={payload['model_parameters']:,}")
    print(f"seconds/update={seconds_per_update:.4f}")
    print(
        f"projected sequential hours={payload['projected_hours_two_units_sequential']:.2f}"
    )
    print(f"mechanics_pass={mechanics_pass}; launch_authorized=False")
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
