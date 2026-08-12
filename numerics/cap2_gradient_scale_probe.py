"""What clip threshold does the CAP objective actually need?

The shipped configuration pairs a ``1/t^2`` regression weight clamped at 0.02
with a global gradient clip of 10.0. Measured consequences: CAP-EMF-1 clipped
15.3% of its updates and the scale-100 candidate clipped 100% of 50,000. H7
permits at most 5%, so the configuration contradicts its own gate.

Rather than guess a repair, this measures it. For each candidate
``loss_weight_floor`` it reproduces a production optimizer update exactly --
``accumulation_steps`` micro-batches of ``micro_batch`` rows, drawn from the
real sampler, each scaled by ``1/accumulation_steps`` and accumulated -- then
records the pre-clip global gradient norm that ``clip_grad_norm_`` would see.

The output is the distribution the clip threshold has to be set against. A clip
at the 95th percentile is what H7's 5% allowance actually means.

This is a measurement, not a gate: it lives outside the sealed package, writes
no promoting artifact, and its numbers select nothing on a quality metric.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from numerics.encoder_independent_drifting.stage_cap.config import (  # noqa: E402
    CAPModelConfig,
    CAPObjectiveConfig,
)
from numerics.encoder_independent_drifting.stage_cap.data import (  # noqa: E402
    cifar10_train_pool,
    flip_batch,
)
from numerics.encoder_independent_drifting.stage_cap.model import (  # noqa: E402
    CAPPixelTransformer,
)
from numerics.encoder_independent_drifting.stage_cap.objective import (  # noqa: E402
    emf_loss,
    sample_time_triangle,
)
from numerics.encoder_independent_drifting.stage_cap2.config import (  # noqa: E402
    SAMPLER_ARMS,
)

def load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model_config = CAPModelConfig(**payload["profile"]["model"])
    objective = CAPObjectiveConfig(**payload["profile"]["objective"])
    model = CAPPixelTransformer(model_config, seed=1).to(device)
    model.load_state_dict(payload["state_dict"])
    model.train()
    return model, objective


def update_gradient_norm(model, pool, objective, *, micro_batch, accumulation,
                         device, generators, horizontal_flip=True) -> float:
    """The pre-clip norm one production optimizer update would produce."""
    data_generator, flip_generator, noise_generator, time_generator, diag = generators
    model.zero_grad(set_to_none=True)
    for _ in range(accumulation):
        order = torch.randint(0, len(pool), (micro_batch,), generator=data_generator)
        clean = pool[order].to(device)
        if horizontal_flip:
            flips = torch.rand(micro_batch, generator=flip_generator) < 0.5
            clean = flip_batch(clean, flips)
        noise = torch.randn(
            clean.shape, generator=noise_generator, dtype=clean.dtype
        ).to(device)
        triangle = sample_time_triangle(
            micro_batch, objective, time_generator, device, diagonal_generator=diag
        )
        result = emf_loss(model, clean, noise, triangle, objective)
        (result.loss / accumulation).backward()
    total = torch.zeros((), device=device, dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum()
    model.zero_grad(set_to_none=True)
    return float(total.sqrt())


def summarize(values: list[float], clip: float) -> dict:
    ordered = sorted(values)

    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "clip_threshold": clip,
        "clipped_fraction": sum(v > clip for v in ordered) / len(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--floors",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10, 0.20, 0.30, 0.50],
        help="loss_weight_floor values to probe; 0.02 is the shipped control",
    )
    parser.add_argument(
        "--arm",
        default=None,
        help=(
            "override the sampler arm (legacy / ordered_logitnormal / "
            "ordered_uniform). The weights are unchanged, so this isolates what "
            "the (t, r) draw alone contributes to the gradient scale."
        ),
    )
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seed", type=int, default=20_260_912)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    pool = cifar10_train_pool(args.data_root)
    model, base_objective = load_model(args.checkpoint, device)
    print(f"checkpoint objective floor={base_objective.resolved_loss_weight_floor} "
          f"scale={base_objective.emf_delta}", flush=True)

    if args.arm is not None:
        mode, r_floor, coefficient_floor = SAMPLER_ARMS[args.arm]
        base_objective = replace(
            base_objective,
            sampler_mode=mode,
            sampled_r_floor=r_floor,
            coefficient_denominator_floor=coefficient_floor,
            diagonal_sampling="fixed_count_first_draw",
        )
        print(f"sampler overridden to {args.arm} ({mode})", flush=True)

    results = {}
    for floor in args.floors:
        objective = replace(base_objective, loss_weight_floor=floor)
        # Identical stream per floor: the only thing that differs between arms
        # is the weight, so the same batches and the same times are drawn.
        generators = tuple(
            torch.Generator().manual_seed(args.seed + offset)
            for offset in range(5)
        )
        norms = [
            update_gradient_norm(
                model, pool, objective,
                micro_batch=args.micro_batch, accumulation=args.accumulation,
                device=device, generators=generators,
            )
            for _ in range(args.updates)
        ]
        stats = summarize(norms, args.clip)
        results[str(floor)] = {"objective": asdict(objective), "stats": stats}
        print(
            f"  floor={floor:<5} p50={stats['p50']:10.2f} p95={stats['p95']:12.2f} "
            f"max={stats['max']:14.2f}  clipped@{args.clip}={stats['clipped_fraction']:6.1%}",
            flush=True,
        )

    payload = {
        "kind": "cap2_gradient_scale_probe",
        "promoting": False,
        "checkpoint": str(args.checkpoint),
        "updates_probed": args.updates,
        "micro_batch": args.micro_batch,
        "accumulation_steps": args.accumulation,
        "effective_batch": args.micro_batch * args.accumulation,
        "clip": args.clip,
        "h7_clip_fraction_max": 0.05,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "floors": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    print("\n=== what clip would H7's 5% allowance imply, per floor? ===")
    for floor, record in results.items():
        stats = record["stats"]
        print(
            f"  floor={floor:<5} implied clip (p95) = {stats['p95']:12.2f}   "
            f"at the shipped 10.0 this floor clips {stats['clipped_fraction']:.1%}"
        )


if __name__ == "__main__":
    main()
