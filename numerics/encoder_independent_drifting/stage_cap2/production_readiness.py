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

from .artifacts import assert_unused, verify_json


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
) -> list[list[str]]:
    """Return the exact ordered gate commands; no training command is possible."""

    if not expected_gpu_name.strip():
        raise ValueError("expected GPU name must be explicit")
    if hourly_rate <= 0:
        raise ValueError("declared hourly price must be positive")
    if micro_batch <= 0:
        raise ValueError("microbatch must be positive")
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
    for name, path in paths.items():
        verify_json(path, statuses[name])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--i-have-authorized-production-gates", action="store_true")
    args = parser.parse_args()
    if not args.i_have_authorized_production_gates:
        raise RuntimeError(
            "refusing production GPU work without --i-have-authorized-production-gates"
        )
    if not args.checkpoint.is_file():
        raise RuntimeError(f"preserved checkpoint is missing: {args.checkpoint}")
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
