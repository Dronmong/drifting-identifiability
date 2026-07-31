"""Run the frozen, corrected K=200 F1 confirmation.

Unlike the exploratory long-horizon runner, this program validates exact source
compatibility, uses new globally disjoint units and one shared kernel, applies
every veto at the gate checkpoint, and checks the paired control in each regime.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import cifar
from .device import configure, resolve_device
from .diagnose_phase20 import save_grid
from .diagnostics import provenance, write_json
from .f1 import (
    CONTROL_RECALL_FLOOR,
    RECALL_GATE,
    _self_distances,
    bank_statistics,
)
from .f1_calibration import eval_references
from .f1_k200 import (
    ARMS,
    CHECKPOINTS,
    GRID_CHECKPOINTS,
    HERE,
    REGIMES,
    UNIT_IDS,
    build_shared_kernel,
    confirmation_allocation,
    confirmation_rollout,
    confirmation_source,
    decide,
    evaluate_veto,
    frozen_config,
    lightweight_score,
    load_compatible_artifact,
    load_freeze,
    stochastic_memorization_statistics,
)
from .fid import inception_features


def validate_preflight(args, freeze: dict) -> dict:
    calibration = load_compatible_artifact(args.calibration, freeze)
    checks = load_compatible_artifact(args.checks, freeze)
    replay_veto = load_compatible_artifact(args.vetoes, freeze)
    stochastic_veto = load_compatible_artifact(args.stochastic_veto, freeze)

    cal_verdict = calibration.get("verdict", {})
    if cal_verdict.get("decision") != "GO":
        raise RuntimeError("null-recall calibration did not return GO")
    if float(cal_verdict.get("recall_gate", -1)) != RECALL_GATE:
        raise RuntimeError("calibration used a different recall gate")
    if not checks.get("verdict", {}).get("all_passed"):
        raise RuntimeError("F1 validation checks did not all pass")
    veto_verdict = replay_veto.get("verdict", {})
    if not veto_verdict.get("any_valid_veto"):
        raise RuntimeError("replay/collapse veto calibration returned NO-GO")
    required = {
        "nearest_bank_normalized", "distinct_bank", "effective_rank",
        "one_minus_duplicate_rate", "nn_diversity",
    }
    thresholds = replay_veto.get("thresholds", {})
    if set(thresholds) != required:
        raise RuntimeError(
            f"expected all five calibrated replay vetoes, got {set(thresholds)}")
    if stochastic_veto.get("verdict", {}).get("decision") != "GO":
        raise RuntimeError("stochastic memorization veto calibration returned NO-GO")
    if not float(stochastic_veto.get("threshold", 0.0)) > 0:
        raise RuntimeError("stochastic memorization threshold is not positive")
    if not float(stochastic_veto.get("normalizer", 0.0)) > 0:
        raise RuntimeError("stochastic memorization normalizer is not positive")
    return {
        "calibration": calibration,
        "checks": checks,
        "replay_veto": replay_veto,
        "stochastic_veto": stochastic_veto,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--freeze", type=Path,
                        default=HERE / "f1_k200_freeze.json")
    parser.add_argument("--calibration", type=Path,
                        default=HERE / "f1_k200_calibration.json")
    parser.add_argument("--checks", type=Path,
                        default=HERE / "f1_k200_checks.json")
    parser.add_argument("--vetoes", type=Path,
                        default=HERE / "f1_k200_vetoes.json")
    parser.add_argument("--stochastic-veto", type=Path,
                        default=HERE / "f1_k200_stochastic_veto.json")
    parser.add_argument("--out", type=Path,
                        default=HERE / "f1_k200_confirmatory.json")
    args = parser.parse_args()
    if args.resolution != frozen_config()["resolution"]:
        raise RuntimeError("resolution differs from the frozen K=200 design")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)

    freeze = load_freeze(args.freeze)
    preflight = validate_preflight(args, freeze)
    print("=== K=200 CONFIRMATION PREFLIGHT: ALL COMPATIBLE ===", flush=True)
    print(f"null p_upper="
          f"{preflight['calibration']['verdict']['p_null_upper']:.6f}")
    print(preflight["checks"]["verdict"]["reading"])
    print(preflight["replay_veto"]["verdict"]["reading"])
    print(preflight["stochastic_veto"]["verdict"]["reading"], flush=True)

    started = time.time()
    allocation = confirmation_allocation(args.resolution, args.data_root)
    train = cifar.cifar_pool(args.resolution, "train", args.data_root)
    stochastic_reference = train[
        torch.as_tensor(allocation.stochastic_pool)]
    branch, kernel = build_shared_kernel(
        allocation, args.resolution, args.data_root, device)

    primary_reference, secondary_reference = eval_references(
        args.resolution, args.data_root)
    reference_features = inception_features(
        primary_reference, device).double().numpy()
    real_for_scale = primary_reference[:512]
    matched_real_baseline = lightweight_score(
        secondary_reference[:512].to(device), reference_features,
        real_for_scale, device)

    replay_thresholds = preflight["replay_veto"]["thresholds"]
    stochastic_threshold = float(preflight["stochastic_veto"]["threshold"])
    stochastic_normalizer = float(preflight["stochastic_veto"]["normalizer"])

    rows = []
    for unit in UNIT_IDS:
        print(f"\n=== new confirmation unit {unit} ===", flush=True)
        starts = {
            arm: confirmation_source(
                arm, unit, allocation, args.resolution, args.data_root, device)
            for arm in ARMS
        }
        replay_bank = train[torch.as_tensor(allocation.banks[unit])]
        replay_scale = float(_self_distances(
            starts["real_data"].cpu()).min(dim=1).values.median())
        for arm in ARMS:
            for regime in REGIMES:
                def observe(state, step, _arm=arm, _regime=regime):
                    if step in GRID_CHECKPOINTS:
                        save_grid(
                            state[:64].cpu(),
                            HERE / f"f1_k200_u{unit}_{_arm}_{_regime}_K{step}.png")
                    return lightweight_score(
                        state, reference_features, real_for_scale, device)

                outcome = confirmation_rollout(
                    starts[arm], unit, regime, allocation, branch, kernel,
                    args.resolution, args.data_root, device,
                    on_checkpoint=observe)
                terminal = dict(outcome["history"][-1])
                terminal.pop("step", None)

                memorization = None
                veto = None
                if arm == "random_generator":
                    if regime == "replay":
                        measured = bank_statistics(
                            outcome["final"].cpu(), replay_bank.cpu(),
                            replay_scale)
                        memorization = {
                            key: value for key, value in measured.items()
                            if not key.endswith("distribution")
                            and key != "claimed_bank_indices"
                        }
                        memorization["nearest_bank_distribution"] = measured[
                            "nearest_bank_distribution"]
                        memorization["claimed_bank_indices"] = measured[
                            "claimed_bank_indices"]
                    else:
                        memorization = stochastic_memorization_statistics(
                            outcome["final"], stochastic_reference,
                            stochastic_normalizer, device)
                    veto = evaluate_veto(
                        regime, terminal, memorization, replay_thresholds,
                        stochastic_threshold)

                row = {
                    "unit": unit,
                    "arm": arm,
                    "regime": regime,
                    "terminal": terminal,
                    "history": outcome["history"],
                    "schedule_digest": outcome["schedule_digest"],
                    "memorization": memorization,
                    "veto": veto,
                }
                rows.append(row)
                print(
                    f"u{unit} {arm:18} {regime:10} "
                    f"recall={terminal['recall']:.4f} "
                    f"rank={terminal['effective_rank']:.2f} "
                    f"control={'--' if arm != 'real_data' else terminal['recall'] > CONTROL_RECALL_FLOOR} "
                    f"veto={'--' if veto is None else veto['passes']}",
                    flush=True)

    verdict = decide(rows)
    payload = {
        "status": "f1-k200-confirmatory",
        "confirmatory": True,
        "protocol": "numerics/EncoderIndependentF1K200ConfirmationProtocol.md",
        "provenance": provenance(),
        "device": settings,
        "config": frozen_config() | {
            "threads": args.threads,
            "data_root": args.data_root,
        },
        "freeze_sha256": args.freeze.with_suffix(
            args.freeze.suffix + ".sha256").read_text(
                encoding="utf-8").split()[0],
        "preflight_artifact_sha256": {
            name: path.with_suffix(path.suffix + ".sha256").read_text(
                encoding="utf-8").split()[0]
            for name, path in {
                "calibration": args.calibration,
                "checks": args.checks,
                "replay_veto": args.vetoes,
                "stochastic_veto": args.stochastic_veto,
            }.items()
        },
        "allocation_digests": allocation.digests,
        "matched_real_baseline": matched_real_baseline,
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "verdict": verdict,
    }
    digest = write_json(args.out, payload)
    print("\n=== CORRECTED K=200 F1 VERDICT ===")
    print(verdict["reading"])
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()

