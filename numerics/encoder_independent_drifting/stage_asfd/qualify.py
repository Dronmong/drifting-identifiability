"""Target-only ASFD feature qualification for a gated 750k foundation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from ..device import configure, resolve_device
from ..stage_cap.data import cifar10_train_pool
from ..stage_cap2.artifacts import verify_json as verify_cap2_json
from ..stage_cap2.artifacts import write_json_atomic
from ..stage_cap2.standard_metrics import _load_model
from .artifacts import (
    assert_no_inherited_freeze,
    assert_result_path_unused,
    file_sha256,
    source_manifest,
)
from .calibration import (
    calibrate_level_bandwidths,
    calibrate_normalization,
    taus_from_records,
)
from .config import asfd_config
from .features import LevelNormalization, encode, freeze_trunk, to_locations
from .field import laplace_field
from .qualification import (
    gate_induced_gradient_distinctness,
    gate_inter_level_field_cosine,
    run_gate,
)

STATUS = "asfd-target-only-qualification"
QUALIFICATION_SEED = 20_260_841
QUALIFICATION_SAMPLES = 512
AUDIT_SAMPLES = 128


def _resolve(reference: object, anchor: Path) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise RuntimeError("foundation gate has an empty checkpoint reference")
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (anchor / path).resolve()


def _portable(path: Path, anchor: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def _encode_chunks(
    trunk,
    images: torch.Tensor,
    *,
    t_f: float,
    features,
    normalization: dict[str, LevelNormalization] | None,
    seed: int,
    device: torch.device,
    batch: int,
    require_graph: bool = False,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    chunks: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(images), batch):
        block = images[start : start + batch].to(device)
        encoded = encode(
            trunk,
            block,
            t_f,
            features,
            normalization,
            generator=generator,
        )
        for name, values in encoded.items():
            chunks.setdefault(name, []).append(
                values if require_graph else values.detach().cpu()
            )
    return {name: torch.cat(values, dim=0) for name, values in chunks.items()}


def _normalization_payload(values: dict[str, LevelNormalization]) -> dict:
    return {
        name: {
            "channel_scale": (
                None
                if record.channel_scale is None
                else record.channel_scale.detach().cpu().tolist()
            ),
            "level_scale": record.level_scale,
            "pc1_share": record.pc1_share,
            "per_channel_applied": record.per_channel_applied,
        }
        for name, record in values.items()
    }


def _candidate(
    trunk,
    images: torch.Tensor,
    *,
    t_f: float,
    config,
    device: torch.device,
    batch: int,
) -> dict:
    unnormalized = _encode_chunks(
        trunk,
        images,
        t_f=t_f,
        features=config.features,
        normalization=None,
        seed=QUALIFICATION_SEED + round(t_f * 10_000),
        device=device,
        batch=batch,
    )
    normalization = calibrate_normalization(unnormalized, config.features)
    normalized = {
        name: normalization[name].apply(values) for name, values in unnormalized.items()
    }
    audit = images[:AUDIT_SAMPLES].to(device)
    gates = run_gate(
        trunk,
        audit,
        t_f,
        config.features,
        normalization,
        config.qualification,
    )
    if not gates["passed"]:
        return {
            **gates,
            "normalization": _normalization_payload(normalization),
            "bandwidths": None,
            "geometry_gates": [],
        }

    feature_bandwidth_records = {
        name: calibrate_level_bandwidths(values, config.field_config)
        for name, values in normalized.items()
    }
    raw_bandwidth_records = calibrate_level_bandwidths(
        images.flatten(1), config.field_config
    )
    feature_taus = taus_from_records(feature_bandwidth_records)
    raw_taus = {
        radius: record["tau"] for radius, record in raw_bandwidth_records.items()
    }

    probes_count = config.field_config.probes
    positives_count = config.field_config.positives
    negatives_count = config.field_config.negatives
    required = probes_count + positives_count + negatives_count
    if len(images) < required:
        raise RuntimeError("qualification sample allocation is too small")
    probes_slice = slice(0, probes_count)
    positives_slice = slice(probes_count, probes_count + positives_count)
    negatives_slice = slice(
        probes_count + positives_count,
        probes_count + positives_count + negatives_count,
    )
    negative_images = (
        torch.roll(images[negatives_slice].to(device), shifts=4, dims=-1)
        .detach()
        .requires_grad_(True)
    )
    feature_negative = _encode_chunks(
        trunk,
        negative_images,
        t_f=t_f,
        features=config.features,
        normalization=normalization,
        seed=QUALIFICATION_SEED + 70_000 + round(t_f * 10_000),
        device=device,
        batch=batch,
        require_graph=True,
    )
    feature_roles = (
        to_locations(
            {
                name: values[probes_slice].to(device)
                for name, values in normalized.items()
            }
        ),
        to_locations(
            {
                name: values[positives_slice].to(device)
                for name, values in normalized.items()
            }
        ),
        to_locations(feature_negative),
    )
    raw_roles = (
        images[probes_slice].to(device).flatten(1),
        images[positives_slice].to(device).flatten(1),
        negative_images,
    )
    distinctness = gate_induced_gradient_distinctness(
        raw_roles,
        feature_roles,
        raw_taus,
        feature_taus,
        config.qualification,
    )
    middle_radius = sorted(config.field_config.radii)[
        len(config.field_config.radii) // 2
    ]
    fields = {}
    for name in sorted(feature_roles[0]):
        fields[name], _ = laplace_field(
            feature_roles[0][name],
            feature_roles[1][name],
            feature_roles[2][name],
            feature_taus[name][middle_radius],
            diagnostics=False,
        )
    inter_level = gate_inter_level_field_cosine(fields, config.qualification)
    geometry = [distinctness, inter_level]
    gates["gates"].extend(
        {"name": name, "passed": ok, "detail": detail, "data": data}
        for name, ok, detail, data in geometry
    )
    gates["failed"] = [
        record["name"] for record in gates["gates"] if not record["passed"]
    ]
    gates["passed"] = not gates["failed"]
    return {
        **gates,
        "normalization": _normalization_payload(normalization),
        "bandwidths": {
            "feature": feature_bandwidth_records,
            "raw": raw_bandwidth_records,
        },
        "geometry_gates": [record["name"] for record in gates["gates"][-2:]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-gate", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_result_path_unused(args.out)
    assert_no_inherited_freeze()
    foundation = verify_cap2_json(args.foundation_gate, "cap-emf2-750k-foundation-gate")
    if foundation.get("decision") != "GO":
        raise RuntimeError("ASFD qualification requires a GO foundation gate")
    checkpoint = _resolve(
        foundation.get("foundation", {}).get("ema_checkpoint"),
        args.foundation_gate.parent,
    )
    if file_sha256(checkpoint) != foundation["foundation"].get("ema_checkpoint_sha256"):
        raise RuntimeError("foundation teacher checkpoint changed")
    device = resolve_device(args.device)
    settings = configure(device, allow_tf32=False)
    torch.use_deterministic_algorithms(True)
    _payload, model, _model_config = _load_model(checkpoint, device)
    trunk = freeze_trunk(model)
    pool = cifar10_train_pool(args.data_root)
    generator = torch.Generator().manual_seed(QUALIFICATION_SEED)
    indices = torch.randperm(len(pool), generator=generator)[:QUALIFICATION_SAMPLES]
    images = pool[indices]
    config = asfd_config()
    candidates = [
        _candidate(
            trunk,
            images,
            t_f=t_f,
            config=config,
            device=device,
            batch=args.batch,
        )
        for t_f in config.features.t_f_grid
    ]
    passing = [record for record in candidates if record["passed"]]
    selected = (
        min(
            passing,
            key=lambda record: (
                abs(float(record["t_f"]) - config.features.t_f),
                float(record["t_f"]),
            ),
        )
        if passing
        else None
    )
    decision = "GO" if selected is not None else "NO_GO"
    result = {
        "status": STATUS,
        "decision": decision,
        "foundation_gate": {
            "path": _portable(args.foundation_gate, args.out.parent),
            "sha256": foundation["artifact_sha256"],
        },
        "teacher_checkpoint": {
            "path": _portable(checkpoint, args.out.parent),
            "sha256": file_sha256(checkpoint),
        },
        "dataset": "CIFAR-10 official training split, all classes",
        "sample_count": QUALIFICATION_SAMPLES,
        "sample_indices": indices.tolist(),
        "sample_indices_sha256": __import__("hashlib")
        .sha256(indices.numpy().tobytes())
        .hexdigest(),
        "config": json.loads(json.dumps(asdict(config), sort_keys=True)),
        "candidates": candidates,
        "selected": selected,
        "selection_rule": (
            "among candidates passing every target-only gate, choose closest to "
            "the preregistered t_f=0.10; ties choose lower t_f"
        ),
        "device": settings,
        "source_sha256": source_manifest(),
        "limits": [
            "No test image or held-out quality score is used.",
            "Qualification rejects an unhealthy geometry; it does not establish semantic correctness.",
            "G6 compares induced negative-image gradients because raw and feature fields live in different vector spaces.",
        ],
    }
    digest = write_json_atomic(args.out, result)
    print(f"wrote {args.out} sha256={digest} decision={decision}")
    return 0 if decision == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
