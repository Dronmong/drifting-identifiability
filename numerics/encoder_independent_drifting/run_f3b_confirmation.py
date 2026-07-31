"""Run the source-frozen, one-configuration F3B B0 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from .device import configure, resolve_device
from .diagnose_phase20 import save_grid
from .diagnostics import provenance, write_json
from .f3b import (
    CONFIRMATION_UNITS,
    adjudicate_b0,
    f3b_seed,
    sample_model,
    train_bridge,
)
from .f3b_evaluation import (
    allocated_images,
    apply_vetoes,
    evaluation_allocation,
    generated_metrics,
    matched_real_metrics,
    memorization_statistics,
)
from .f3b_freeze import (
    HERE,
    load_compatible_calibration,
    load_freeze,
    profile_from_payload,
    verify_sidecar,
)
from .fid import inception_features


def _save_ema_checkpoint(path: Path, outcome, selected, unit: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with outcome.ema.average_parameters(outcome.model):
        torch.save(
            {
                "unit": unit,
                "profile": selected,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in outcome.model.state_dict().items()
                },
            },
            path,
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=HERE / "f3b_freeze.json")
    parser.add_argument(
        "--calibration", type=Path, default=HERE / "f3b_calibration.json"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE / "f3b_checkpoints")
    parser.add_argument("--out", type=Path, default=HERE / "f3b_confirmatory.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)

    freeze = load_freeze(args.freeze)
    calibration = load_compatible_calibration(args.calibration, args.freeze, freeze)
    selected = profile_from_payload(freeze["profile"])
    if (
        len(selected.train.checkpoint_steps) != 1
        or len(selected.evaluation.nfe_ladder) != 1
    ):
        raise RuntimeError("confirmation freeze contains a development ladder")
    nfe = selected.evaluation.nfe_ladder[0]

    train = cifar.cifar_pool(selected.model.image_size, "train", args.data_root)
    eval_pool = cifar.cifar_pool(selected.model.image_size, "eval", args.data_root)
    allocation = evaluation_allocation(
        len(eval_pool),
        selected.evaluation.reference_samples,
        selected.evaluation.generated_samples,
        len(CONFIRMATION_UNITS),
    )
    if allocation.digests != calibration["allocation_digests"]:
        raise RuntimeError("evaluation allocation differs from calibration")
    reference, controls = allocated_images(
        allocation, "confirmation", selected.model.image_size, args.data_root
    )
    reference_features = inception_features(reference, device).double().numpy()
    control_rows = [
        {"group": index, **matched_real_metrics(control, reference_features, device)}
        for index, control in enumerate(controls)
    ]
    matched_real_recall = float(np.median([row["recall"] for row in control_rows]))

    started = time.time()
    rows = []
    checkpoints = {}
    for unit in CONFIRMATION_UNITS:
        print(f"\n=== F3B CONFIRMATION unit {unit} ===", flush=True)
        outcome = train_bridge(
            train, selected.model, selected.train, "confirmation", unit, device
        )
        checkpoint_path = args.checkpoint_dir / f"f3b_unit_{unit}_ema.pt"
        checkpoints[str(unit)] = {
            "path": str(checkpoint_path),
            "sha256": _save_ema_checkpoint(
                checkpoint_path, outcome, freeze["profile"], unit
            ),
        }
        with outcome.ema.average_parameters(outcome.model):
            prior_seed = f3b_seed("confirmation", unit, "evaluation-prior")
            generated = sample_model(
                outcome.model,
                selected.evaluation.generated_samples,
                selected.model,
                nfe,
                prior_seed,
                device,
            )
        control = controls[CONFIRMATION_UNITS.index(unit)]
        metrics = generated_metrics(
            generated, reference_features, control, device, include_fid=True
        )
        memory = memorization_statistics(
            generated, train, float(calibration["normalizer"]), device
        )
        veto = apply_vetoes(metrics, memory, calibration["thresholds"])
        save_grid(
            generated[:64].cpu(),
            HERE / f"f3b_u{unit}_step{selected.train.steps}_nfe{nfe}.png",
        )
        row = {
            "unit": unit,
            "training": {
                "history": outcome.history,
                "wall_seconds": outcome.wall_seconds,
                "peak_memory_bytes": outcome.peak_memory_bytes,
                "examples_seen": outcome.examples_seen,
                "optimizer_updates": outcome.optimizer_updates,
                "model_parameters": outcome.model.parameter_count(),
            },
            "nfe": nfe,
            "prior_seed": prior_seed,
            "metrics": metrics,
            "recall_fraction_of_matched_real": (
                float(metrics["recall"]) / max(matched_real_recall, 1e-12)
            ),
            "memorization": memory,
            "veto": veto,
            "seed_roles": {
                role: f3b_seed("confirmation", unit, role)
                for role in (
                    "model-init",
                    "data-order",
                    "endpoint-noise",
                    "bridge-time",
                    "augmentation",
                    "evaluation-prior",
                )
            },
        }
        rows.append(row)
        print(
            f"recall={metrics['recall']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"KID={metrics['kid']:+.5f} FID={metrics['fid']:.2f} "
            f"rank={metrics['effective_rank']:.2f} "
            f"veto={veto['passes']}",
            flush=True,
        )

    verdict = adjudicate_b0(rows, matched_real_recall)
    payload = {
        "status": "f3b-b0-confirmatory",
        "confirmatory": True,
        "protocol": "numerics/EncoderIndependentF3BProtocol.md",
        "provenance": provenance(),
        "device": settings,
        "freeze_sha256": verify_sidecar(args.freeze),
        "calibration_sha256": verify_sidecar(args.calibration),
        "profile": freeze["profile"],
        "allocation_digests": allocation.digests,
        "matched_real_controls": control_rows,
        "matched_real_recall": matched_real_recall,
        "checkpoints": checkpoints,
        "rows": rows,
        "verdict": verdict,
        "elapsed_seconds": time.time() - started,
        "claim_scope": freeze["claim_scope"],
    }
    digest = write_json(args.out, payload)
    print("\n=== F3B B0 CONFIRMATION ===")
    print(verdict["reading"])
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
