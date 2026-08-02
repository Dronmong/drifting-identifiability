"""Mechanical, data-integrity, memory, and throughput preflight for B2.5."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch

from .. import cifar
from ..b1 import B1Config, b1_config
from ..device import configure, resolve_device
from ..diagnostics import write_json
from ..f3b import profile
from ..f3b_freeze import profile_from_payload
from ..stage_b2.core import B2Config, b2_config
from ..stage_b2.fresh_data import load_fresh_pool
from .artifacts import (
    DEFAULT_DATA,
    DEFAULT_PREFLIGHT,
    development_profile,
    load_frozen_inputs,
    preflight_header,
)
from .core import B25_ARMS, B25Config, train_b25_arm
from .evaluation import evaluation_allocation


def _small_b1() -> B1Config:
    base = b1_config()
    anchor = replace(
        base.anchor,
        features=8,
        audit_features=8,
        refresh_every=2,
        refresh_fraction=0.5,
    )
    result = replace(
        base,
        anchor_every=1,
        anchor_batch=4,
        anchor_nfe=1,
        audit_features=8,
        audit_replicates=2,
        refresh_every_events=2,
        refresh_fraction=0.5,
        anchor_paired_wins_required=1,
        anchor=anchor,
    )
    result.validate()
    return result


def _small_b2() -> B2Config:
    result = replace(
        b2_config(),
        correction_every=1,
        probe_batch=4,
        positive_batch=4,
        negative_batch=4,
        correction_nfe=1,
        ess_samples=8,
        ess_iterations=8,
        audit_batch=4,
        audit_replicates=2,
        drift_paired_wins_required=1,
    )
    result.validate()
    return result


def _smoke(frozen: dict, device: torch.device) -> dict:
    selected = profile("smoke")
    stage = B25Config(
        units=(500,),
        checkpoint_steps=(2, 4),
        diagnostic_steps=(2, 4),
        final_step=4,
        bootstrap_replicates=100,
        unit_wins_required=1,
    )
    generator = torch.Generator().manual_seed(20260801)
    pool = torch.rand(24, 3, 8, 8, generator=generator) * 2 - 1
    rows = {}
    for arm in B25_ARMS:
        outcome = train_b25_arm(
            pool,
            selected.model,
            selected.train,
            500,
            arm,
            device,
            b1_scale=frozen["b1_scale"],
            lambda_b1=frozen["lambda_b1"],
            tau_b2=frozen["tau_b2"],
            lambda_b2=frozen["lambda_b2"],
            b1_config=_small_b1(),
            b2_config=_small_b2(),
            stage_config=stage,
        )
        rows[arm] = {
            "flow_loss_step_1": outcome.history[0]["flow_loss"],
            "anchor_events": outcome.anchor_events,
            "correction_events": outcome.correction_events,
            "gradient_diagnostics": outcome.component_gradient_diagnostics,
            "peak_memory_bytes": outcome.peak_memory_bytes,
            "peak_memory_reserved_bytes": outcome.peak_memory_reserved_bytes,
        }
        del outcome
        if device.type == "cuda":
            torch.cuda.empty_cache()
    paired = len({row["flow_loss_step_1"] for row in rows.values()}) == 1
    full = bool(
        rows["B1B2"]["anchor_events"] == 4
        and rows["B1B2"]["correction_events"] == 4
        and rows["B1"]["anchor_events"] == 4
        and rows["B2"]["correction_events"] == 4
    )
    diagnostic = rows["B1B2"]["gradient_diagnostics"][-1]
    components = diagnostic["weighted_component_norms_pre_clip"]
    component_complete = set(components) == {"flow", "b1_weighted", "b2_weighted"}
    return {
        "rows": rows,
        "flow_streams_exactly_paired": paired,
        "combined_cell_is_full_dose": full,
        "combined_diagnostics_have_all_components": component_complete,
        "passes": bool(paired and full and component_complete),
    }


def _throughput_probe(
    frozen: dict,
    device: torch.device,
    data_root: str | None,
    steps: int,
) -> dict:
    if steps < 10 or steps % 10:
        raise ValueError("throughput steps must be a positive multiple of ten")
    base_stage = B25Config(
        units=(500,),
        checkpoint_steps=(steps,),
        diagnostic_steps=(steps,),
        final_step=steps,
        bootstrap_replicates=100,
        unit_wins_required=1,
    )
    selected = development_profile(frozen["b2"], base_stage)
    train = cifar.cifar_pool(selected.model.image_size, "train", data_root)
    started = time.time()
    outcome = train_b25_arm(
        train,
        selected.model,
        selected.train,
        500,
        "B1B2",
        device,
        b1_scale=frozen["b1_scale"],
        lambda_b1=frozen["lambda_b1"],
        tau_b2=frozen["tau_b2"],
        lambda_b2=frozen["lambda_b2"],
        b1_config=b1_config(),
        b2_config=b2_config(),
        stage_config=base_stage,
    )
    elapsed = time.time() - started
    projected = elapsed / steps * 30_000 / 3600
    return {
        "arm": "B1B2",
        "steps": steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "projected_hours_per_full_combined_unit": projected,
        "peak_memory_bytes": outcome.peak_memory_bytes,
        "peak_memory_reserved_bytes": outcome.peak_memory_reserved_bytes,
        "anchor_events": outcome.anchor_events,
        "correction_events": outcome.correction_events,
        "component_gradient_diagnostics": outcome.component_gradient_diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--external-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--throughput-steps", type=int, default=100)
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_PREFLIGHT)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(args.out.suffix + ".sha256").exists():
        raise RuntimeError("refusing to overwrite an existing B2.5 preflight")

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    frozen = load_frozen_inputs()
    header = preflight_header()
    selected = profile_from_payload(header["profile"])
    external, external_record = load_fresh_pool(
        args.external_data,
        header["external_data"]["source_id"],
        32,
        "auto",
    )
    if external_record["sha256"] != header["external_data"]["artifact_sha256"]:
        raise RuntimeError("B2.5 external loader observed different bytes")
    config = B25Config()
    in_domain = evaluation_allocation(
        10_000,
        "cifar10-test-adaptively-reused-b25-development",
        units=len(config.units),
        generated_samples=selected.evaluation.generated_samples,
        reference_samples=selected.evaluation.reference_samples,
        audit_replicates=b2_config().audit_replicates,
        audit_batch=b2_config().audit_batch,
    )
    shifted = evaluation_allocation(
        len(external),
        header["external_data"]["source_id"],
        units=len(config.units),
        generated_samples=selected.evaluation.generated_samples,
        reference_samples=selected.evaluation.reference_samples,
        audit_replicates=b2_config().audit_replicates,
        audit_batch=b2_config().audit_batch,
    )
    smoke = _smoke(frozen, device)
    throughput = None
    if not args.skip_throughput:
        throughput = _throughput_probe(
            frozen, device, args.data_root, args.throughput_steps
        )
    memory_pass = bool(
        throughput is None
        or device.type != "cuda"
        or int(throughput["peak_memory_reserved_bytes"])
        <= int(torch.cuda.get_device_properties(device).total_memory * 0.95)
    )
    decision = "GO" if smoke["passes"] and memory_pass else "NO-GO"
    payload = {
        **header,
        "device": settings,
        "evaluation_allocations": {
            "in_domain_development_reused": in_domain.digests,
            "shifted_disjoint": shifted.digests,
        },
        "smoke": smoke,
        "throughput": throughput,
        "verdict": {
            "decision": decision,
            "smoke_passes": smoke["passes"],
            "memory_passes": memory_pass,
            "reading": (
                "GO: full-dose factorial mechanics, pairing, data boundaries, "
                "and device headroom pass"
                if decision == "GO"
                else "NO-GO: repair the failed preflight check before training"
            ),
        },
        "scope": (
            "mechanical and resource readiness only; this does not consume the "
            "three-unit B2.5 development experiment"
        ),
    }
    digest = write_json(args.out, payload)
    print(payload["verdict"]["reading"])
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
