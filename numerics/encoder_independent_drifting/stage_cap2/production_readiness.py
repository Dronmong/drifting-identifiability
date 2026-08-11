"""Run CAP-EMF-2 production-GPU gates, stopping before all training.

This command deliberately has no route to ``run_screen``.  It performs the
checkpoint numerical admission, same-environment mechanism forensics, the
2,000-step full-loop benchmark, and finally the fail-closed CAP2 preflight.
Any failed command terminates the sequence before the next expense.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .artifacts import assert_unused, file_sha256, source_manifest, verify_json
from .budget import CAMPAIGNS
from .durable_mirror import load_root_attestation, probe_root
from .metric_calibration import revalidate_metric_calibration_evidence
from .standard_metrics import revalidate_clean_evaluation_evidence


def production_commands(
    *,
    python: str,
    checkpoint: Path,
    expected_gpu_name: str,
    hourly_rate: float,
    candidate: str,
    micro_batch: int,
    data_root: str | None,
    output_dir: Path,
    sampler_audit: Path,
    gate_calibration: Path,
    baseline_standard: Path,
    positive_control_standard: Path,
    metric_calibration: Path,
    max_total_cost: float,
    nontraining_reserve: float,
    contingency_fraction: float,
    durable_mirror_dir: Path,
    durable_storage_root: Path,
    artifact_storage_reserve_gib: float,
    storage_contingency_fraction: float,
    post_foundation_training_reserve: float = 10.0,
    campaign: str = "matched_screen",
) -> list[list[str]]:
    """Return the exact ordered gate commands; no training command is possible."""

    if not expected_gpu_name.strip():
        raise ValueError("expected GPU name must be explicit")
    if hourly_rate <= 0:
        raise ValueError("declared hourly price must be positive")
    if micro_batch <= 0:
        raise ValueError("microbatch must be positive")
    if (
        max_total_cost <= 0
        or nontraining_reserve < 0
        or post_foundation_training_reserve < 0
    ):
        raise ValueError("budget ceiling must be positive and reserve nonnegative")
    if not 0 <= contingency_fraction <= 1:
        raise ValueError("contingency fraction must lie in [0,1]")
    if artifact_storage_reserve_gib < 0:
        raise ValueError("artifact storage reserve must be nonnegative")
    if not 0 <= storage_contingency_fraction <= 1:
        raise ValueError("storage contingency fraction must lie in [0,1]")
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown CAP2 campaign {campaign!r}")
    numerical = output_dir / "numerical_admission.json"
    forensics = output_dir / "checkpoint_forensics.json"
    benchmark = output_dir / "benchmark.json"
    preflight = output_dir / "cap2_preflight.json"
    common_device = ["--device", "cuda", "--expected-gpu-name", expected_gpu_name]
    common_data = ["--data-root", data_root] if data_root is not None else []
    return [
        [
            python,
            "-m",
            "numerics.encoder_independent_drifting.stage_cap2.numerical_admission",
            "--checkpoint",
            str(checkpoint),
            "--candidate",
            candidate,
            *common_device,
            *common_data,
            "--batch",
            "4",
            "--repeats",
            "3",
            "--include-gradient",
            "--out",
            str(numerical),
        ],
        [
            python,
            "-m",
            "numerics.encoder_independent_drifting.stage_cap2.checkpoint_forensics",
            "--checkpoint",
            str(checkpoint),
            *common_device,
            *common_data,
            "--samples",
            "2048",
            "--grid-samples",
            "256",
            "--batch",
            str(micro_batch),
            "--out",
            str(forensics),
        ],
        [
            python,
            "-m",
            "numerics.encoder_independent_drifting.stage_cap2.benchmark",
            "--arm",
            "ordered_uniform",
            "--numerical",
            candidate,
            *common_device,
            *common_data,
            "--steps",
            "2000",
            "--micro-batch",
            str(micro_batch),
            "--hourly-rate",
            str(hourly_rate),
            "--durable-mirror-dir",
            str(durable_mirror_dir),
            "--i-confirm-durable-mirror",
            "--out",
            str(benchmark),
        ],
        [
            python,
            "-m",
            "numerics.encoder_independent_drifting.stage_cap2.preflight",
            "--numerical-admission",
            str(numerical),
            "--sampler-audit",
            str(sampler_audit),
            "--gate-calibration",
            str(gate_calibration),
            "--benchmark",
            str(benchmark),
            "--baseline-standard",
            str(baseline_standard),
            "--positive-control-standard",
            str(positive_control_standard),
            "--metric-calibration",
            str(metric_calibration),
            "--checkpoint-forensics",
            str(forensics),
            "--max-total-cost",
            str(max_total_cost),
            "--nontraining-reserve",
            str(nontraining_reserve),
            "--contingency-fraction",
            str(contingency_fraction),
            "--campaign",
            campaign,
            "--post-foundation-training-reserve",
            str(post_foundation_training_reserve),
            "--durable-storage-root",
            str(durable_storage_root),
            "--artifact-storage-reserve-gib",
            str(artifact_storage_reserve_gib),
            "--storage-contingency-fraction",
            str(storage_contingency_fraction),
            "--out",
            str(preflight),
        ],
    ]


def _require_local_evidence(paths: dict[str, Path]) -> None:
    statuses = {
        "sampler_audit": "cap-emf2-sampler-audit",
        "gate_calibration": "cap-emf2-gate-calibration",
        "baseline_standard": "cap-emf-standard-evaluation",
        "positive_control_standard": "cap-emf-standard-evaluation",
        "metric_calibration": "cap-emf2-real-real-calibration",
    }
    live_sources = source_manifest()
    for name, path in paths.items():
        payload = verify_json(path, statuses[name])
        if payload.get("source_sha256") != live_sources:
            recorded = payload.get("source_sha256")
            changed = sorted(
                key
                for key in set(live_sources)
                & (set(recorded) if isinstance(recorded, dict) else set())
                if recorded[key] != live_sources[key]
            )
            raise RuntimeError(
                f"{name} is stale relative to live CAP2 sources: {changed}"
            )
        if name in {"baseline_standard", "positive_control_standard"}:
            evidence = revalidate_clean_evaluation_evidence(payload, anchor=path.parent)
            if evidence.get("valid") is not True:
                failed = sorted(
                    check
                    for check, passed in evidence.get("checks", {}).items()
                    if not passed
                )
                raise RuntimeError(
                    f"{name} retained CleanFID/KID evidence failed: {failed}"
                )
        if name == "metric_calibration":
            evidence = revalidate_metric_calibration_evidence(
                payload, anchor=path.parent
            )
            if evidence.get("valid") is not True:
                failed = sorted(
                    check
                    for check, passed in evidence.get("checks", {}).items()
                    if not passed
                )
                raise RuntimeError(
                    f"metric calibration retained evidence failed: {failed}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--candidate", default="local_1000_d0002_fp32")
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampler-audit", type=Path, required=True)
    parser.add_argument("--gate-calibration", type=Path, required=True)
    parser.add_argument("--baseline-standard", type=Path, required=True)
    parser.add_argument("--positive-control-standard", type=Path, required=True)
    parser.add_argument("--metric-calibration", type=Path, required=True)
    parser.add_argument("--max-total-cost", type=float, required=True)
    parser.add_argument("--nontraining-reserve", type=float, required=True)
    parser.add_argument("--contingency-fraction", type=float, default=0.15)
    parser.add_argument("--campaign", choices=CAMPAIGNS, default="matched_screen")
    parser.add_argument("--post-foundation-training-reserve", type=float, default=10.0)
    parser.add_argument("--durable-mirror-dir", type=Path, required=True)
    parser.add_argument("--durable-storage-root", type=Path, required=True)
    parser.add_argument("--artifact-storage-reserve-gib", type=float, default=20.0)
    parser.add_argument("--storage-contingency-fraction", type=float, default=0.20)
    parser.add_argument("--i-confirm-durable-mirror", action="store_true")
    parser.add_argument("--i-have-authorized-production-gates", action="store_true")
    args = parser.parse_args()
    if not args.i_have_authorized_production_gates:
        raise RuntimeError(
            "refusing production GPU work without --i-have-authorized-production-gates"
        )
    if not args.i_confirm_durable_mirror:
        raise RuntimeError(
            "production gates require an attested off-instance durable mirror"
        )
    if not args.checkpoint.is_file():
        raise RuntimeError(f"preserved checkpoint is missing: {args.checkpoint}")
    if file_sha256(args.checkpoint) != args.checkpoint_sha256.lower():
        raise RuntimeError("preserved admission checkpoint SHA-256 differs")
    if not args.durable_storage_root.is_dir():
        raise RuntimeError(
            f"durable storage root is not mounted: {args.durable_storage_root}"
        )
    # Fail before numerical admission or forensics spend.  The benchmark also
    # probes this root, but discovering an absent/stale mount at gate 3 is too
    # late for a budget-conscious production sequence.
    load_root_attestation(args.durable_mirror_dir)
    durable_probe = probe_root(args.durable_mirror_dir)
    if durable_probe.get("roundtrip_verified") is not True:
        raise RuntimeError("durable mirror failed its pre-gate live roundtrip")
    evidence = {
        "sampler_audit": args.sampler_audit,
        "gate_calibration": args.gate_calibration,
        "baseline_standard": args.baseline_standard,
        "positive_control_standard": args.positive_control_standard,
        "metric_calibration": args.metric_calibration,
    }
    _require_local_evidence(evidence)
    outputs = [
        args.output_dir / name
        for name in (
            "numerical_admission.json",
            "checkpoint_forensics.json",
            "benchmark.json",
            "cap2_preflight.json",
        )
    ]
    for output in outputs:
        assert_unused(output)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = production_commands(
        python=sys.executable,
        checkpoint=args.checkpoint.resolve(),
        expected_gpu_name=args.expected_gpu_name,
        hourly_rate=args.hourly_rate,
        candidate=args.candidate,
        micro_batch=args.micro_batch,
        data_root=args.data_root,
        output_dir=args.output_dir.resolve(),
        max_total_cost=args.max_total_cost,
        nontraining_reserve=args.nontraining_reserve,
        contingency_fraction=args.contingency_fraction,
        durable_mirror_dir=args.durable_mirror_dir.resolve(),
        durable_storage_root=args.durable_storage_root.resolve(),
        artifact_storage_reserve_gib=args.artifact_storage_reserve_gib,
        storage_contingency_fraction=args.storage_contingency_fraction,
        post_foundation_training_reserve=args.post_foundation_training_reserve,
        campaign=args.campaign,
        **{name: path.resolve() for name, path in evidence.items()},
    )
    for index, command in enumerate(commands, start=1):
        print(f"CAP2 production gate {index}/{len(commands)}: {' '.join(command)}")
        subprocess.run(command, check=True)
    preflight = verify_json(outputs[-1], "cap-emf2-preflight")
    if preflight.get("decision") != "GO":
        raise RuntimeError("CAP2 production preflight did not return GO")
    print("CAP2 PRE-BUDGET READINESS: GO. No training was launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
