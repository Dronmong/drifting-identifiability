"""Explicitly interlocked runner for one S3R developmental arm."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch

from ..device import configure, resolve_device
from ..diagnostics import provenance, write_json
from .audit import HERE, require_developmental_preflight, source_digest
from .config import S3R_ARMS, profile
from .data import automobile_train_pool
from .diagnostics import developmental_series_gate
from .training import train_arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=S3R_ARMS, required=True)
    parser.add_argument("--unit", type=int, default=800)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--preflight", type=Path, default=HERE / "s3r_preflight.json")
    parser.add_argument("--execute-developmental", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=HERE / "runs")
    args = parser.parse_args()
    if not args.execute_developmental:
        raise RuntimeError(
            "S3R RUN BLOCKED: pass --execute-developmental after reviewing preflight"
        )
    preflight = require_developmental_preflight(args.preflight)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    selected = profile("developmental")
    train_pool = automobile_train_pool(args.data_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    resume_payload = None
    if args.resume is not None:
        resume_payload = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        if resume_payload.get("source_sha256") != source_digest():
            raise RuntimeError("S3R RUN BLOCKED: resume source identity mismatch")

    def save_checkpoint(payload: dict) -> None:
        path = args.out_dir / (
            f"s3r_{args.arm}_unit_{args.unit}_update_{payload['update']}.pt"
        )
        torch.save(payload, path)

    result = train_arm(
        train_pool,
        selected,
        args.arm,
        "developmental-screen",
        args.unit,
        device,
        checkpoint=save_checkpoint,
        resume_payload=resume_payload,
        source_sha256=source_digest(),
    )
    stem = f"s3r_{args.arm}_unit_{args.unit}"
    checkpoint = args.out_dir / f"{stem}.pt"
    torch.save(
        {
            "status": "s3r-developmental-train-only",
            "arm": args.arm,
            "unit": args.unit,
            "profile": asdict(selected),
            "source_sha256": source_digest(),
            "model": {
                name: value.detach().cpu()
                for name, value in result.model.state_dict().items()
            },
            "ema": {
                name: value.detach().cpu() for name, value in result.ema.shadow.items()
            },
            "optimizer": result.optimizer.state_dict(),
            "history": result.history,
            "endpoint_history": result.endpoint_history,
        },
        checkpoint,
    )
    payload = {
        "status": "s3r-developmental-train-only",
        "claim_boundary": "no official test access and no image-quality promotion",
        "arm": args.arm,
        "unit": args.unit,
        "profile": asdict(selected),
        "source_sha256": source_digest(),
        "preflight_sha256": preflight["source_sha256"],
        "provenance": provenance(),
        "device": settings,
        "optimizer_updates": result.optimizer_updates,
        "examples_seen": result.examples_seen,
        "clipping_fraction": result.clipping_fraction,
        "wall_seconds": result.wall_seconds,
        "peak_memory_bytes": result.peak_memory_bytes,
        "history": result.history,
        "endpoint_history": result.endpoint_history,
        "developmental_gate": developmental_series_gate(
            result.endpoint_history, result.clipping_fraction
        ),
        "checkpoint": str(checkpoint),
    }
    digest = write_json(args.out_dir / f"{stem}.json", payload)
    print(f"completed developmental {args.arm}; artifact sha256={digest}")


if __name__ == "__main__":
    main()
