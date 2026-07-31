"""Run the explicitly exploratory F3B B0 budget/NFE ladder.

This runner is allowed to inform model selection.  Its artifact can be consumed
by ``freeze_f3b``; the later confirmation uses new units and one selected
checkpoint/NFE only.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from . import cifar
from .device import configure, resolve_device
from .diagnostics import provenance, write_json
from .f3b import (
    DEVELOPMENT_UNITS,
    f3b_seed,
    profile,
    sample_model,
    train_bridge,
)
from .f3b_evaluation import (
    allocated_images,
    evaluation_allocation,
    generated_metrics,
    matched_real_metrics,
)
from .f3b_freeze import HERE, profile_payload
from .fid import inception_features


def _unit_list(raw: str) -> tuple[int, ...]:
    result = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not result or any(unit not in DEVELOPMENT_UNITS for unit in result):
        raise ValueError(f"development units must be drawn from {DEVELOPMENT_UNITS}")
    if len(set(result)) != len(result):
        raise ValueError("development units must be distinct")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("smoke", "compact", "reference_scale"), default="compact"
    )
    parser.add_argument("--units", default=",".join(map(str, DEVELOPMENT_UNITS)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--include-fid", action="store_true")
    parser.add_argument(
        "--mechanics-only",
        action="store_true",
        help="train without Inception evaluation; cannot freeze",
    )
    parser.add_argument("--out", type=Path, default=HERE / "f3b_development.json")
    args = parser.parse_args()
    units = _unit_list(args.units)
    selected = profile(args.profile)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    device_settings = configure(device)

    train_pool = cifar.cifar_pool(selected.model.image_size, "train", args.data_root)
    allocation = evaluation_allocation(
        len(cifar.cifar_pool(selected.model.image_size, "eval", args.data_root)),
        selected.evaluation.reference_samples,
        selected.evaluation.generated_samples,
        len(DEVELOPMENT_UNITS),
    )
    reference = None
    controls = None
    reference_features = None
    control_rows = []
    if not args.mechanics_only:
        reference, controls = allocated_images(
            allocation, "development", selected.model.image_size, args.data_root
        )
        reference_features = inception_features(reference, device).double().numpy()
        for index, control in enumerate(controls):
            control_rows.append(
                {
                    "control_group": index,
                    **matched_real_metrics(control, reference_features, device),
                }
            )

    started = time.time()
    evaluations: list[dict] = []
    training_rows: list[dict] = []
    for unit in units:
        print(f"\n=== F3B DEVELOPMENT {args.profile} unit {unit} ===", flush=True)

        def evaluate_checkpoint(step, model, training_record, _unit=unit):
            if args.mechanics_only:
                return
            control = controls[DEVELOPMENT_UNITS.index(_unit)]
            prior_seed = f3b_seed("development", _unit, f"evaluation-prior-step-{step}")
            for nfe in selected.evaluation.nfe_ladder:
                generated = sample_model(
                    model,
                    selected.evaluation.generated_samples,
                    selected.model,
                    nfe,
                    prior_seed,
                    device,
                )
                metrics = generated_metrics(
                    generated,
                    reference_features,
                    control,
                    device,
                    include_fid=args.include_fid,
                )
                row = {
                    "unit": _unit,
                    "step": step,
                    "nfe": nfe,
                    "prior_seed": prior_seed,
                    "training_loss": training_record["loss"],
                    "metrics": metrics,
                }
                evaluations.append(row)
                print(
                    f"step={step:7d} nfe={nfe:3d} "
                    f"recall={metrics['recall']:.4f} "
                    f"precision={metrics['precision']:.4f} "
                    f"KID={metrics['kid']:+.5f} "
                    f"rank={metrics['effective_rank']:.2f}",
                    flush=True,
                )

        outcome = train_bridge(
            train_pool,
            selected.model,
            selected.train,
            "development",
            unit,
            device,
            checkpoint=evaluate_checkpoint,
        )
        training_rows.append(
            {
                "unit": unit,
                "history": outcome.history,
                "wall_seconds": outcome.wall_seconds,
                "peak_memory_bytes": outcome.peak_memory_bytes,
                "examples_seen": outcome.examples_seen,
                "optimizer_updates": outcome.optimizer_updates,
                "model_parameters": outcome.model.parameter_count(),
                "seed_roles": {
                    role: f3b_seed("development", unit, role)
                    for role in (
                        "model-init",
                        "data-order",
                        "endpoint-noise",
                        "bridge-time",
                        "augmentation",
                    )
                },
            }
        )

    payload = {
        "status": "f3b-b0-development",
        "confirmatory": False,
        "protocol": "numerics/EncoderIndependentF3BProtocol.md",
        "provenance": provenance(),
        "device": device_settings,
        "profile": profile_payload(selected),
        "units": list(units),
        "allocation_digests": allocation.digests,
        "include_fid": bool(args.include_fid),
        "mechanics_only": bool(args.mechanics_only),
        "matched_real_controls": control_rows,
        "training": training_rows,
        "evaluations": evaluations,
        "elapsed_seconds": time.time() - started,
        "claim_scope": (
            "exploratory development; may select a later frozen budget/NFE; "
            "not a confirmation"
        ),
    }
    digest = write_json(args.out, payload)
    print("\n=== F3B DEVELOPMENT COMPLETE ===")
    print(f"wrote {args.out} sha256={digest}")
    if args.mechanics_only:
        print("mechanics-only artifact has no outcomes and cannot be frozen")


if __name__ == "__main__":
    main()
