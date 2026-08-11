"""Calibrate two-sided CAP2 health gates from disjoint CIFAR-10 train subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ..stage_cap.config import CAPGateConfig
from ..stage_cap.data import cifar10_train_pool
from ..stage_cap.diagnostics import endpoint_health
from .artifacts import assert_unused, source_manifest, write_json_atomic

GATE_RATIO_FIELDS = (
    "second_moment_ratio",
    "centered_variance_ratio",
    "effective_rank_ratio",
    "haar_LL_ratio",
    "haar_LH_ratio",
    "haar_HL_ratio",
    "haar_HH_ratio",
)


def _quantile(values: list[float], q: float) -> float:
    return float(torch.tensor(values, dtype=torch.float64).quantile(q))


def _derive_gate(records: list[dict]) -> tuple[dict, dict, CAPGateConfig]:
    """Purely reconstruct the calibrated bounds from serialized observations."""
    if len(records) < 4:
        raise ValueError("gate calibration needs at least four records")
    for record in records:
        values = [
            record.get(name) for name in (*GATE_RATIO_FIELDS, "raw_saturation_fraction")
        ]
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        ):
            raise ValueError("gate calibration record contains a non-finite field")
        if any(float(record[name]) <= 0.0 for name in GATE_RATIO_FIELDS):
            raise ValueError("gate calibration ratios must be positive")
        saturation = float(record["raw_saturation_fraction"])
        if not 0.0 <= saturation <= 1.0:
            raise ValueError("gate calibration saturation must lie in [0,1]")

    def values(name: str) -> list[float]:
        return [float(record[name]) for record in records]

    lower = {
        name: min(1.0, max(1e-6, 0.9 * _quantile(values(name), 0.01)))
        for name in GATE_RATIO_FIELDS
    }
    upper = {name: max(1.0, 1.1 * _quantile(values(name), 0.99)) for name in lower}
    saturation_upper = min(
        0.10,
        _quantile(values("raw_saturation_fraction"), 0.99) + 0.01,
    )
    gate = replace(
        CAPGateConfig(),
        second_moment_ratio=lower["second_moment_ratio"],
        centered_variance_ratio=lower["centered_variance_ratio"],
        minimum_effective_rank_ratio=min(1.0, lower["effective_rank_ratio"]),
        minimum_haar_LL_ratio=lower["haar_LL_ratio"],
        minimum_haar_LH_ratio=lower["haar_LH_ratio"],
        minimum_haar_HL_ratio=lower["haar_HL_ratio"],
        minimum_haar_HH_ratio=lower["haar_HH_ratio"],
        maximum_second_moment_ratio=upper["second_moment_ratio"],
        maximum_centered_variance_ratio=upper["centered_variance_ratio"],
        maximum_effective_rank_ratio=upper["effective_rank_ratio"],
        maximum_haar_LL_ratio=upper["haar_LL_ratio"],
        maximum_haar_LH_ratio=upper["haar_LH_ratio"],
        maximum_haar_HL_ratio=upper["haar_HL_ratio"],
        maximum_haar_HH_ratio=upper["haar_HH_ratio"],
        maximum_saturation_fraction=saturation_upper,
    )
    gate.validate()
    return lower, upper, gate


def gate_calibration_consistent(payload: object) -> bool:
    """Reject a stored GO whose gate was not derived from its stored records."""
    if not isinstance(payload, dict):
        return False
    records = payload.get("records")
    if not isinstance(records, list):
        return False
    try:
        lower, upper, gate = _derive_gate(records)
    except (TypeError, ValueError):
        return False
    return (
        payload.get("status") == "cap-emf2-gate-calibration"
        and payload.get("decision") == "GO"
        and payload.get("empirical_lower") == lower
        and payload.get("empirical_upper") == upper
        and payload.get("gate") == asdict(gate)
    )


def calibrate_gate(
    pool: torch.Tensor,
    *,
    samples: int = 2_048,
    repeats: int = 12,
    seed: int = 20_260_830,
) -> dict:
    if samples < 128 or 2 * samples * repeats > len(pool):
        raise ValueError(
            "gate calibration needs globally disjoint nontrivial subset pairs"
        )
    if repeats < 4:
        raise ValueError("gate calibration needs at least four repetitions")
    generator = torch.Generator().manual_seed(seed)
    records: list[dict[str, float]] = []
    allocations: list[dict[str, str]] = []
    order = torch.randperm(len(pool), generator=generator)[: 2 * samples * repeats]
    for repeat in range(repeats):
        block = order[2 * samples * repeat : 2 * samples * (repeat + 1)]
        left_indices = block[:samples]
        right_indices = block[samples:]
        allocations.append(
            {
                "left_sha256": hashlib.sha256(
                    left_indices.numpy().tobytes()
                ).hexdigest(),
                "right_sha256": hashlib.sha256(
                    right_indices.numpy().tobytes()
                ).hexdigest(),
            }
        )
        left = pool[left_indices]
        right = pool[right_indices]
        records.append(endpoint_health(left, right))

    # A 10% multiplicative guard around the empirical 1--99% range prevents
    # the calibration set from becoming a hidden hyperparameter while keeping
    # the gate genuinely two-sided.
    lower, upper, gate = _derive_gate(records)
    return {
        "status": "cap-emf2-gate-calibration",
        "samples_per_subset": samples,
        "repeats": repeats,
        "seed": seed,
        "globally_disjoint": True,
        "allocation_sha256": allocations,
        "records": records,
        "empirical_lower": lower,
        "empirical_upper": upper,
        "gate": asdict(gate),
        "decision": "GO",
        "limits": [
            "Train split only; the CIFAR-10 test split is not opened.",
            "These are mechanical distribution-health gates, not a quality metric.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--samples", type=int, default=2_048)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20_260_830)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("gate_calibration.json")
    )
    args = parser.parse_args()
    assert_unused(args.out)
    result = calibrate_gate(
        cifar10_train_pool(args.data_root),
        samples=args.samples,
        repeats=args.repeats,
        seed=args.seed,
    )
    result["source_sha256"] = source_manifest()
    digest = write_json_atomic(args.out, result)
    print(json.dumps({k: result[k] for k in ("status", "gate", "decision")}, indent=2))
    print(f"wrote {args.out} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
