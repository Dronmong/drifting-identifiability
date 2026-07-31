"""Cheap F3B architecture, bridge-direction, and measured-cost preflight."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from .device import configure, resolve_device
from .diagnostics import provenance, write_json
from .f3b import (
    TimeConditionedUNet,
    f3b_seed,
    oracle_endpoint,
    profile,
    train_bridge,
)
from .f3b_freeze import HERE, profile_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("smoke", "compact", "reference_scale"), default="compact"
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "f3b_preflight.json")
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    selected = profile(args.profile)

    generator = torch.Generator(device="cpu").manual_seed(
        f3b_seed("preflight", 0, "synthetic-pool")
    )
    pool = (
        torch.rand(
            max(2 * selected.train.batch, 128),
            selected.model.channels,
            selected.model.image_size,
            selected.model.image_size,
            generator=generator,
        )
        * 2.0
        - 1.0
    )
    short_train = replace(
        selected.train,
        steps=args.iterations,
        log_every=1,
        checkpoint_steps=(args.iterations,),
    )
    outcome = train_bridge(pool, selected.model, short_train, "preflight", 0, device)

    model = TimeConditionedUNet(
        selected.model, f3b_seed("preflight", 0, "dependence-model")
    ).to(device)
    probe = pool[:2].to(device)
    with torch.no_grad():
        early = model(probe, torch.zeros(2, device=device))
        late = model(probe, torch.ones(2, device=device))
        moved = model(probe + 0.01, torch.zeros(2, device=device))
    noise = torch.randn(
        4,
        selected.model.channels,
        selected.model.image_size,
        selected.model.image_size,
        generator=generator,
    )
    endpoint = pool[:4]
    oracle_errors = {
        str(nfe): float((oracle_endpoint(noise, endpoint, nfe) - endpoint).abs().max())
        for nfe in selected.evaluation.nfe_ladder
    }

    seconds_per_step = outcome.wall_seconds / args.iterations
    payload = {
        "status": "f3b-preflight",
        "confirmatory": False,
        "protocol": "numerics/EncoderIndependentF3BProtocol.md",
        "provenance": provenance(),
        "device": settings,
        "profile": profile_payload(selected),
        "iterations": args.iterations,
        "model_parameters": outcome.model.parameter_count(),
        "seconds_per_step": seconds_per_step,
        "projected_training_seconds": seconds_per_step * selected.train.steps,
        "peak_memory_bytes": outcome.peak_memory_bytes,
        "finite_loss_and_gradient": all(
            torch.isfinite(value).all() for value in outcome.model.state_dict().values()
        ),
        "image_dependence_l2": float((early - moved).norm()),
        "time_dependence_l2": float((early - late).norm()),
        "oracle_max_errors": oracle_errors,
        "verdict": {
            "passes": bool(
                float((early - moved).norm()) > 0
                and float((early - late).norm()) > 0
                and max(oracle_errors.values()) < 1e-5
            ),
            "scope": "mechanics and measured local cost only",
        },
    }
    digest = write_json(args.out, payload)
    print("=== F3B PREFLIGHT ===")
    print(f"parameters={payload['model_parameters']:,}")
    print(f"seconds/step={seconds_per_step:.4f}")
    print(f"projected hours={payload['projected_training_seconds'] / 3600:.2f}")
    print(f"peak bytes={payload['peak_memory_bytes']}")
    print(f"verdict={payload['verdict']}")
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
