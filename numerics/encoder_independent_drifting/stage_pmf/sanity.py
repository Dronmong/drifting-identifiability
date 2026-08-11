"""Outcome-blind real-training-data learning check before the full S3 run."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import torch

from ..device import configure, resolve_device
from ..diagnostics import provenance, write_json
from .audit import HERE, profile_digest, profile_payload, source_digest, source_manifest
from .config import profile
from .data import automobile_data
from .data import manifest as data_manifest
from .model import PixelMeanFlowTransformer
from .objective import meanflow_loss, one_step_sample, sample_time_triangle
from .training import pmf_seed, train_pmf


def _diagnostic_batches(pool: torch.Tensor, count: int = 4, batch: int = 16):
    data_rng = torch.Generator().manual_seed(9201)
    noise_rng = torch.Generator().manual_seed(9202)
    time_rng = torch.Generator().manual_seed(9203)
    mask_rng = torch.Generator().manual_seed(9204)
    objective = profile("local_s3").objective
    result = []
    for _ in range(count):
        indices = torch.randint(len(pool), (batch,), generator=data_rng)
        clean = pool[indices].clone()
        noise = torch.randn(clean.shape, generator=noise_rng)
        triangle = sample_time_triangle(batch, objective, time_rng, mask_rng)
        result.append((clean, noise, triangle))
    return result


def _raw_diagnostic(model, batches, objective, device: torch.device) -> dict:
    model.eval()
    values = {"all": [], "diagonal": [], "interior": []}
    with torch.no_grad():
        for clean, noise, triangle in batches:
            result = meanflow_loss(
                model, clean.to(device), noise.to(device), triangle, objective
            )
            per_sample = (
                (result.compound - result.target_velocity)
                .square()
                .flatten(1)
                .mean(dim=1)
                .detach()
                .double()
                .cpu()
            )
            diagonal = triangle.diagonal.cpu()
            values["all"].append(per_sample)
            values["diagonal"].append(per_sample[diagonal])
            values["interior"].append(per_sample[~diagonal])
    return {name: float(torch.cat(parts).mean()) for name, parts in values.items()}


def _endpoint_diagnostic(
    model, selected, device: torch.device, count: int = 64
) -> dict:
    generator = torch.Generator().manual_seed(
        pmf_seed("local-s3-sanity", 0, "sealed-endpoints")
    )
    noise = torch.randn(
        count,
        selected.model.channels,
        selected.model.image_size,
        selected.model.image_size,
        generator=generator,
    ).to(device)
    model.eval()
    with torch.no_grad():
        generated = one_step_sample(model, noise).detach().cpu().double()
    flat = generated.flatten(1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    singular_squared = torch.linalg.svdvals(centered).square()
    total = singular_squared.sum()
    effective_rank = (
        0.0
        if float(total) <= 0
        else float(total.square() / singular_squared.square().sum().clamp_min(1e-30))
    )
    return {
        "samples": count,
        "mean_pixel_variance": float(flat.var(dim=0).mean()),
        "effective_rank": effective_rank,
        "rms": float(generated.square().mean().sqrt()),
        "minimum": float(generated.min()),
        "maximum": float(generated.max()),
        "finite": bool(torch.isfinite(generated).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", type=Path, default=HERE / "s3_learning_sanity.json")
    args = parser.parse_args()
    if args.updates < 100:
        raise ValueError("the learning sanity requires at least 100 updates")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    selected = profile("local_s3")
    short = replace(
        selected,
        train=replace(
            selected.train,
            updates=args.updates,
            log_every=max(1, args.updates // 10),
            checkpoint_updates=(args.updates,),
        ),
    )
    data = automobile_data(args.data_root)
    batches = _diagnostic_batches(data.train)
    initial = PixelMeanFlowTransformer(
        short.model, pmf_seed("local-s3-sanity", 0, "model-init")
    ).to(device)
    raw_before = _raw_diagnostic(initial, batches, short.objective, device)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in initial.state_dict().items()
    }
    del initial
    if device.type == "cuda":
        torch.cuda.empty_cache()

    outcome = train_pmf(data.train, short, "local-s3-sanity", 0, device)
    raw_after = _raw_diagnostic(outcome.model, batches, short.objective, device)
    changed = any(
        not torch.equal(value.detach().cpu(), initial_state[name])
        for name, value in outcome.model.state_dict().items()
    )
    with outcome.ema.average_parameters(outcome.model):
        endpoints = _endpoint_diagnostic(outcome.model, short, device)
    passes = bool(
        raw_after["all"] < raw_before["all"]
        and changed
        and endpoints["finite"]
        and endpoints["mean_pixel_variance"] > 1e-5
        and endpoints["effective_rank"] > 2.0
        and all(
            torch.isfinite(value).all() for value in outcome.model.state_dict().values()
        )
    )
    payload = {
        "status": "pmf-s3-learning-sanity-passed"
        if passes
        else "pmf-s3-learning-sanity-failed",
        "scope": (
            "outcome-blind optimization check on training automobiles only; "
            "no test data, Inception features, grids, or human sample inspection"
        ),
        "provenance": provenance(),
        "device": settings,
        "profile": profile_payload(selected),
        "profile_sha256": profile_digest(selected),
        "source_manifest": source_manifest(),
        "source_sha256": source_digest(),
        "data": data_manifest(data),
        "updates": args.updates,
        "raw_velocity_mse_before": raw_before,
        "raw_velocity_mse_after": raw_after,
        "raw_velocity_mse_ratio": {
            name: raw_after[name] / max(raw_before[name], 1e-30) for name in raw_before
        },
        "parameters_changed": changed,
        "endpoint_mechanics": endpoints,
        "training": {
            "wall_seconds": outcome.wall_seconds,
            "peak_memory_bytes": outcome.peak_memory_bytes,
            "examples_seen": outcome.examples_seen,
            "history": outcome.history,
        },
        "passes": passes,
    }
    digest = write_json(args.out, payload)
    print("=== LOCAL pMF S3 LEARNING SANITY ===")
    print(
        f"raw MSE: {raw_before['all']:.6f} -> {raw_after['all']:.6f} "
        f"(diagonal {raw_before['diagonal']:.6f} -> "
        f"{raw_after['diagonal']:.6f}; interior "
        f"{raw_before['interior']:.6f} -> {raw_after['interior']:.6f})"
    )
    print(f"endpoint effective rank: {endpoints['effective_rank']:.3f}")
    print(f"passes={passes}; wrote {args.out} sha256={digest}")
    if not passes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
