"""Read-only CAP checkpoint mechanism audit on fixed train-only latents.

This records the base-head/refiner/final decomposition, raw-versus-clipped
health, patch-phase energy, residual spectra, and a two-dimensional ``(t,h)``
response grid.  It never trains, selects a checkpoint, or opens CIFAR-10 test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ..appearance import spectrum_slope
from ..device import configure, resolve_device
from ..stage_cap.data import cifar10_train_pool
from ..stage_cap.diagnostics import (
    component_health,
    endpoint_reference,
    patch_phase_report,
)
from ..stage_cap.model import CAPPixelTransformer
from .artifacts import assert_unused, file_sha256, source_manifest, write_json_atomic
from .hardware import hardware_binding

T_GRID = (0.20, 0.50, 0.80, 0.95, 0.98, 1.00)
H_FRACTIONS = (0.00, 0.25, 0.50, 0.90, 1.00)


def _load_model(path: Path, device: torch.device):
    from ..stage_cap.config import CAPModelConfig

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old torch
        payload = torch.load(path, map_location="cpu")
    config = CAPModelConfig(**payload["profile"]["model"])
    model = CAPPixelTransformer(config, seed=1).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model, config


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _component_bundle(
    components: dict[str, torch.Tensor], target: torch.Tensor
) -> dict[str, object]:
    clipped_base = components["base"].clamp(-1, 1)
    clipped_final = components["final"].clamp(-1, 1)
    clipped = {
        "base": clipped_base,
        "refiner_residual": clipped_final - clipped_base,
        "final": clipped_final,
    }
    reference = endpoint_reference(target)
    return {
        "raw_health": component_health(components, target, reference),
        "clipped_health": component_health(clipped, target, reference),
        "phase": {
            name: patch_phase_report(value) for name, value in components.items()
        },
        "clipped_phase": {
            name: patch_phase_report(value) for name, value in clipped.items()
        },
        "spectrum": {
            "target": _safe_spectrum(target),
            **{name: _safe_spectrum(value) for name, value in components.items()},
        },
        "clipped_spectrum": {
            "target": _safe_spectrum(target),
            **{name: _safe_spectrum(value) for name, value in clipped.items()},
        },
    }


def _safe_spectrum(images: torch.Tensor) -> dict[str, object]:
    try:
        return {"available": True, **spectrum_slope(images)}
    except ValueError as error:
        # The production 32x32 audit always has a valid fit band.  Keeping the
        # small-image path explicit makes mechanical smoke tests honest.
        return {"available": False, "reason": str(error)}


def _response_record(
    components: dict[str, torch.Tensor], target: torch.Tensor, t: float, h: float
) -> dict[str, float]:
    base = components["base"].double()
    residual = components["refiner_residual"].double()
    final = components["final"].double()
    denominator = base.flatten(1).norm(dim=1) * residual.flatten(1).norm(dim=1)
    cosine = torch.where(
        denominator > 0,
        (base.flatten(1) * residual.flatten(1)).sum(1) / denominator,
        torch.zeros_like(denominator),
    )
    return {
        "t": t,
        "h": h,
        "h_over_t": 0.0 if t == 0 else h / t,
        "mse_to_clean": float((final - target.double()).square().mean()),
        "base_rms": float(base.square().mean().sqrt()),
        "residual_rms": float(residual.square().mean().sqrt()),
        "final_rms": float(final.square().mean().sqrt()),
        "base_residual_cosine": float(cosine.mean()),
        "raw_saturation_fraction": float((final.abs() > 1).double().mean()),
    }


def run_forensics(
    checkpoint: Path,
    pool: torch.Tensor,
    *,
    device: torch.device,
    samples: int,
    grid_samples: int,
    batch: int,
    seed: int,
    expected_gpu_name: str | None,
) -> dict:
    if samples < 64 or samples > len(pool):
        raise ValueError("forensics samples must be between 64 and the train-pool size")
    if batch <= 0:
        raise ValueError("batch must be positive")
    if grid_samples < 32 or grid_samples > samples:
        raise ValueError("grid samples must lie between 32 and total samples")
    # Match the production inference arithmetic.  The separate numerical
    # admission artifact owns the strict-FP32 comparison.
    settings = configure(device, allow_tf32=device.type == "cuda")
    payload, model, config = _load_model(checkpoint, device)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(pool), generator=generator)[:samples]
    clean = pool[indices].contiguous()
    noise = torch.randn(clean.shape, generator=generator)

    exact_parts: dict[str, list[torch.Tensor]] = {
        "base": [],
        "refiner_residual": [],
        "final": [],
    }
    grid_parts: dict[tuple[float, float], dict[str, list[torch.Tensor]]] = {}
    calls = 0
    with torch.no_grad():
        for start in range(0, samples, batch):
            stop = min(start + batch, samples)
            clean_chunk = clean[start:stop].to(device)
            noise_chunk = noise[start:stop].to(device)
            ones = torch.ones(len(noise_chunk), device=device)
            exact = model.forward_components(noise_chunk, ones, ones)
            calls += 1
            for name, value in exact.items():
                exact_parts[name].append(value.cpu())

            if start < grid_samples:
                grid_stop = min(stop, grid_samples)
                clean_grid = clean_chunk[: grid_stop - start]
                noise_grid = noise_chunk[: grid_stop - start]
                for t_value in T_GRID:
                    state = (1 - t_value) * clean_grid + t_value * noise_grid
                    t = torch.full((len(state),), t_value, device=device)
                    for fraction in H_FRACTIONS:
                        h_value = t_value * fraction
                        h = torch.full((len(state),), h_value, device=device)
                        components = model.forward_components(state, t, h)
                        calls += 1
                        destination = grid_parts.setdefault(
                            (t_value, h_value),
                            {"base": [], "refiner_residual": [], "final": []},
                        )
                        for name, value in components.items():
                            destination[name].append(value.cpu())

    exact = {name: torch.cat(parts) for name, parts in exact_parts.items()}
    response_grid = []
    for (t_value, h_value), parts in sorted(grid_parts.items()):
        components = {name: torch.cat(chunks) for name, chunks in parts.items()}
        response_grid.append(
            _response_record(components, clean[:grid_samples], t_value, h_value)
        )

    hardware = hardware_binding(device, expected_gpu_name)
    expected_calls = (samples + batch - 1) // batch
    expected_calls += (
        len(T_GRID) * len(H_FRACTIONS) * ((grid_samples + batch - 1) // batch)
    )
    finite = all(torch.isfinite(value).all() for value in exact.values()) and all(
        torch.isfinite(value).all()
        for parts in grid_parts.values()
        for chunks in parts.values()
        for value in chunks
    )
    checks = {
        "checkpoint_stage": payload.get("stage") in {"cap-emf-1", "cap-emf-2-screen"},
        "checkpoint_has_profile": isinstance(payload.get("profile"), dict),
        "checkpoint_kind_ema": payload.get("kind") == "ema",
        "hardware_bound": hardware["matches"],
        "all_outputs_finite": bool(finite),
        "call_count_exact": calls == expected_calls,
        "response_grid_complete": len(response_grid) == len(T_GRID) * len(H_FRACTIONS),
    }
    return {
        "status": "cap-emf2-checkpoint-forensics",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": payload.get("step"),
        "checkpoint_kind": payload.get("kind"),
        "device": settings,
        "hardware": hardware,
        "samples": samples,
        "grid_samples": grid_samples,
        "batch": batch,
        "seed": seed,
        "train_indices": indices.tolist(),
        "clean_sha256": tensor_sha256(clean),
        "noise_sha256": tensor_sha256(noise),
        "model_calls": calls,
        "expected_model_calls": expected_calls,
        "inference_condition": _component_bundle(exact, clean),
        "response_grid": response_grid,
        "checks": checks,
        "decision": "COMPLETE" if all(checks.values()) else "INCOMPLETE",
        "limits": [
            "Read-only mechanism audit; it is not a quality or promotion metric.",
            "Only CIFAR-10 train images and fixed synthetic noise are used.",
            "Clipped diagnostics are reported beside, never instead of, raw diagnostics.",
        ],
        "model": {
            "image_size": config.image_size,
            "scalar_embedding_scale": config.scalar_embedding_scale,
        },
        "source_sha256": source_manifest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-gpu-name", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--samples", type=int, default=2_048)
    parser.add_argument("--grid-samples", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_805)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("checkpoint_forensics.json"),
    )
    args = parser.parse_args()
    assert_unused(args.out)
    device = resolve_device(args.device)
    result = run_forensics(
        args.checkpoint,
        cifar10_train_pool(args.data_root),
        device=device,
        samples=args.samples,
        grid_samples=args.grid_samples,
        batch=args.batch,
        seed=args.seed,
        expected_gpu_name=args.expected_gpu_name,
    )
    digest = write_json_atomic(args.out, result)
    print(
        json.dumps({k: result[k] for k in ("status", "decision", "samples")}, indent=2)
    )
    print(f"wrote {args.out} sha256={digest}")
    return 0 if result["decision"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
