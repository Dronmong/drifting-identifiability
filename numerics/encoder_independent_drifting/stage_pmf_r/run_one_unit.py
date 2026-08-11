"""Sequentially run one matched S3R unit across all repaired mechanisms."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .audit import HERE, require_developmental_preflight, source_digest

ARM_ORDER = ("alpha", "emf", "pmf")


def _completed(path: Path, arm: str, unit: int, digest: str) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        payload.get("status") == "s3r-developmental-train-only"
        and payload.get("arm") == arm
        and int(payload.get("unit", -1)) == unit
        and payload.get("source_sha256") == digest
    )


def _latest_checkpoint(out_dir: Path, arm: str, unit: int) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    prefix = f"s3r_{arm}_unit_{unit}_update_"
    for path in out_dir.glob(f"{prefix}*.pt"):
        try:
            update = int(path.stem.removeprefix(prefix))
        except ValueError:
            continue
        candidates.append((update, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", type=int, default=800)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--preflight", type=Path, default=HERE / "s3r_preflight.json")
    parser.add_argument("--out-dir", type=Path, default=HERE / "runs")
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--execute-developmental", action="store_true")
    args = parser.parse_args()
    if not args.execute_developmental:
        raise RuntimeError("S3R ONE-UNIT RUN BLOCKED: explicit opt-in is required")
    require_developmental_preflight(args.preflight)
    digest = source_digest()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for arm in ARM_ORDER:
        result_path = args.out_dir / f"s3r_{arm}_unit_{args.unit}.json"
        if _completed(result_path, arm, args.unit, digest):
            print(f"S3R one-unit: verified {arm} result already complete; skipping")
            continue
        command = [
            sys.executable,
            "-m",
            "numerics.encoder_independent_drifting.stage_pmf_r.run_screen",
            "--arm",
            arm,
            "--unit",
            str(args.unit),
            "--device",
            args.device,
            "--threads",
            str(args.threads),
            "--preflight",
            str(args.preflight),
            "--out-dir",
            str(args.out_dir),
            "--execute-developmental",
        ]
        if args.data_root is not None:
            command.extend(("--data-root", args.data_root))
        if args.resume_latest:
            checkpoint = _latest_checkpoint(args.out_dir, arm, args.unit)
            if checkpoint is not None:
                command.extend(("--resume", str(checkpoint)))
                print(f"S3R one-unit: resuming {arm} from {checkpoint}")
        log_path = args.out_dir / f"s3r_{arm}_unit_{args.unit}.log"
        print(f"S3R one-unit: starting {arm}; log={log_path}", flush=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"S3R one-unit stopped: {arm} exited {completed.returncode}; "
                f"inspect {log_path}"
            )
        if not _completed(result_path, arm, args.unit, digest):
            raise RuntimeError(f"S3R one-unit: {arm} exited without a valid artifact")
        print(f"S3R one-unit: completed {arm}", flush=True)

    print("S3R one-unit: all three matched arms completed", flush=True)


if __name__ == "__main__":
    main()
