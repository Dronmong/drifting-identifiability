"""Guarded two-unit S3 runner. Dry-run is safe; full launch needs audit token."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch import nn

from ..device import configure, resolve_device
from ..diagnose_phase20 import save_grid
from ..diagnostics import provenance, write_json
from ..fid import inception_features
from ..stage_b2.metrics import (
    generated_metrics,
    memorization_statistics_augmented,
    nearest_reference_distances,
)
from .audit import (
    AUTHORIZATION,
    HERE,
    profile_digest,
    profile_payload,
    require_launch_authorization,
    source_digest,
)
from .config import INITIAL_UNITS, profile
from .data import automobile_data
from .data import manifest as data_manifest
from .model import PixelMeanFlowTransformer
from .objective import one_step_sample
from .training import checkpoint_payload, pmf_evaluation_seed, train_pmf


class ForwardCounter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.calls = 0

    def forward(self, *args):
        self.calls += 1
        return self.model(*args)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_source_frozen(expected: str) -> None:
    actual = source_digest()
    if actual != expected:
        raise RuntimeError(
            "S3 sources changed after launch authorization: "
            f"expected {expected}, found {actual}"
        )


def _validated_completed_shard(
    path: Path, unit: int, selected, source_sha256: str
) -> dict:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists() or sidecar.read_text(encoding="utf-8").split()[0] != _sha(
        path
    ):
        raise RuntimeError(f"missing or invalid completed-shard hash for {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "pmf-s3-unit-complete",
        "unit": unit,
        "profile_sha256": profile_digest(selected),
        "source_sha256": source_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"incompatible completed shard {path}: field {key}")
    final = payload.get("checkpoints", {}).get(str(selected.train.updates))
    if not final:
        raise RuntimeError(f"completed shard {path} lacks its final checkpoint")
    final_path = Path(final["path"])
    if not final_path.exists() or _sha(final_path) != final.get("sha256"):
        raise RuntimeError(f"completed shard {path} has an invalid final checkpoint")
    return payload


def generate(
    model: nn.Module,
    count: int,
    batch: int,
    image_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    counter = ForwardCounter(model)
    output = []
    counter.eval()
    with torch.no_grad():
        for start in range(0, count, batch):
            size = min(batch, count - start)
            noise = torch.randn(
                size, 3, image_size, image_size, generator=generator
            ).to(device)
            output.append(one_step_sample(counter, noise).detach().cpu())
    expected_calls = math.ceil(count / batch)
    if counter.calls != expected_calls:
        raise RuntimeError("one-step sampler made an unexpected model call")
    return torch.cat(output), {
        "nfe_per_sample": 1,
        "model_calls": counter.calls,
        "batches": expected_calls,
        "one_call_per_batch": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--out-dir", type=Path, default=HERE / "runs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume each incomplete unit from its latest compatible checkpoint",
    )
    args = parser.parse_args()
    selected = profile("local_s3")
    if args.dry_run:
        print("=== LOCAL pMF S3 DRY RUN ===")
        print(f"units={INITIAL_UNITS}")
        print(f"profile_sha256={profile_digest(selected)}")
        print(f"source_sha256={source_digest()}")
        print("launch_authorized=False (dry-run never trains)")
        return

    authorization = require_launch_authorization(selected, args.authorization)
    launch_source_sha256 = authorization["source_sha256"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    dataset = automobile_data(args.data_root)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1 is training-only.  No test feature, test-derived statistic, or
    # sample grid is computed until every fixed unit has finished optimization.
    pending_units: list[int] = []
    final_checkpoints: dict[int, Path] = {}
    training_summaries: dict[int, dict] = {}
    for unit in INITIAL_UNITS:
        shard_path = args.out_dir / f"pmf_s3_unit_{unit}.json"
        if shard_path.exists():
            if not args.resume:
                raise RuntimeError(
                    f"completed shard exists for unit {unit}; use --resume to "
                    "validate and skip completed work"
                )
            _validated_completed_shard(shard_path, unit, selected, launch_source_sha256)
            print(f"validated and skipped completed unit {unit}", flush=True)
            continue
        pending_units.append(unit)
        checkpoint_records = {}
        resume_payload = None
        existing = sorted(
            args.checkpoint_dir.glob(f"pmf_s3_u{unit}_step*.pt"),
            key=lambda path: int(path.stem.rsplit("step", 1)[1]),
        )
        if existing and not args.resume:
            raise RuntimeError(
                f"checkpoints already exist for unit {unit}; use --resume only "
                "after auditing the preserved state"
            )
        if existing:
            latest = existing[-1]
            resume_payload = torch.load(latest, map_location="cpu", weights_only=False)
            if (
                resume_payload.get("status") != "pmf-s3-checkpoint"
                or int(resume_payload.get("unit", -1)) != unit
                or resume_payload.get("profile") != profile_payload(selected)
                or resume_payload.get("source_sha256") != launch_source_sha256
            ):
                raise RuntimeError(f"incompatible S3 resume checkpoint {latest}")
            for path in existing:
                checkpoint_records[path.stem.rsplit("step", 1)[1]] = {
                    "path": str(path),
                    "sha256": _sha(path),
                }

        def save_checkpoint(
            update,
            model,
            ema,
            optimizer,
            streams,
            row,
            history,
            unit=unit,
            checkpoint_records=checkpoint_records,
        ):
            _assert_source_frozen(launch_source_sha256)
            path = args.checkpoint_dir / f"pmf_s3_u{unit}_step{update}.pt"
            if path.exists():
                raise RuntimeError(f"refusing to overwrite checkpoint {path}")
            torch.save(
                checkpoint_payload(
                    update,
                    model,
                    ema,
                    optimizer,
                    streams,
                    profile_payload(selected),
                    unit,
                    history,
                    launch_source_sha256,
                    last_row=row,
                    peak_memory_bytes=(
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else None
                    ),
                ),
                path,
            )
            checkpoint_records[str(update)] = {
                "path": str(path),
                "sha256": _sha(path),
            }

        outcome = train_pmf(
            dataset.train,
            selected,
            "local-s3-full",
            unit,
            device,
            checkpoint=save_checkpoint,
            resume_payload=resume_payload,
        )
        _assert_source_frozen(launch_source_sha256)
        final_path = args.checkpoint_dir / (
            f"pmf_s3_u{unit}_step{selected.train.updates}.pt"
        )
        if not final_path.exists():
            raise RuntimeError(f"unit {unit} finished without its final checkpoint")
        final_checkpoints[unit] = final_path
        final_payload = torch.load(final_path, map_location="cpu", weights_only=False)
        prior_peak = final_payload.get("peak_memory_bytes_so_far")
        peaks = [value for value in (prior_peak, outcome.peak_memory_bytes) if value]
        training_summaries[unit] = {
            "history": outcome.history,
            "wall_seconds": outcome.wall_seconds,
            "peak_memory_bytes": max(peaks) if peaks else None,
            "optimizer_updates": outcome.optimizer_updates,
            "examples_seen": outcome.examples_seen,
            "parameters": outcome.model.parameter_count(),
            "inference_parameters": outcome.model.inference_parameter_count(),
            "resumed": resume_payload is not None,
            "test_access_during_training": False,
            "checkpoints": checkpoint_records,
        }
        del final_payload, outcome
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Phase 2 is report-only and begins only after all pending units have
    # completed training.  The same fixed noise is shared by both units so
    # their replication difference is not confounded by different priors.
    if not pending_units:
        print("all fixed S3 units were already complete", flush=True)
        return
    _assert_source_frozen(launch_source_sha256)
    reference_features = inception_features(dataset.test, device).double().numpy()
    test_to_train, _ = nearest_reference_distances(dataset.test, dataset.train, device)
    memorization_normalizer = float(test_to_train.median())
    if memorization_normalizer <= 0:
        raise RuntimeError("nonpositive test-to-train memorization normalizer")
    evaluation_seed = pmf_evaluation_seed(
        "local-s3-full", selected.evaluation.fixed_noise_seed_label
    )

    for unit in pending_units:
        _assert_source_frozen(launch_source_sha256)
        shard_path = args.out_dir / f"pmf_s3_unit_{unit}.json"
        final_path = final_checkpoints[unit]
        final_payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if final_payload.get("source_sha256") != launch_source_sha256:
            raise RuntimeError(f"final checkpoint source mismatch for unit {unit}")
        model = PixelMeanFlowTransformer(selected.model, seed=0).to(device)
        model.load_state_dict(final_payload["ema"], strict=True)
        generated, call_audit = generate(
            model,
            selected.evaluation.generated_samples,
            selected.evaluation.inference_batch,
            selected.model.image_size,
            evaluation_seed,
            device,
        )
        metrics = generated_metrics(
            generated,
            reference_features,
            dataset.test,
            device,
            include_fid=True,
        )
        memory = memorization_statistics_augmented(
            generated,
            dataset.train,
            memorization_normalizer,
            device,
        )
        grid_path = args.out_dir / f"pmf_s3_unit_{unit}_uncurated.png"
        save_grid(generated[: selected.evaluation.grid_samples], grid_path)
        payload = {
            "status": "pmf-s3-unit-complete",
            "confirmatory": False,
            "unit": unit,
            "protocol": "numerics/EncoderIndependentPMFS3Protocol.md",
            "authorization": authorization,
            "provenance": provenance(),
            "device": settings,
            "profile": profile_payload(selected),
            "profile_sha256": profile_digest(selected),
            "source_sha256": launch_source_sha256,
            "data": data_manifest(dataset),
            "training": {
                key: value
                for key, value in training_summaries[unit].items()
                if key != "checkpoints"
            },
            "checkpoints": training_summaries[unit]["checkpoints"],
            "evaluation_seed": evaluation_seed,
            "evaluation_noise_shared_across_units": True,
            "one_step_call_audit": call_audit,
            "metrics_report_only": metrics,
            "memorization": memory,
            "uncurated_grid": str(grid_path),
            "claim_scope": (
                "one-class developmental pMF foundation; no drifting correction "
                "and no paper-superiority claim"
            ),
        }
        digest = write_json(shard_path, payload)
        print(f"completed unit {unit}: {shard_path} sha256={digest}", flush=True)
        del final_payload, model, generated
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
