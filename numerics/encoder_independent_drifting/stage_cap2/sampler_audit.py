"""Statistical admission for the three CAP-EMF-2 time samplers.

This is a no-training audit.  It verifies support, diagonal semantics,
coefficient tails, and the joint inference-corner mass before a GPU run can be
authorized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..stage_cap.objective import sample_time_triangle
from .artifacts import assert_unused, source_manifest, write_json_atomic
from .config import SAMPLER_ARMS, screen_profile


def audit_sampler(
    arm: str,
    *,
    count: int,
    seed: int = 20_260_806,
    numerical: str = "local_1000_d0002_fp32",
) -> dict:
    if count < 10_000:
        raise ValueError("sampler admission needs at least 10,000 draws")
    frozen = screen_profile(arm, numerical, smoke=True)
    generator = torch.Generator().manual_seed(seed)
    diagonal_generator = torch.Generator().manual_seed(seed + 9_000_001)
    triangle = sample_time_triangle(
        count,
        frozen.objective,
        generator,
        diagonal_generator=diagonal_generator,
    )
    t, r = triangle.t.double(), triangle.r.double()
    h = t - r
    delta = frozen.objective.emf_delta
    coefficient = (
        (h - delta).clamp_min(0)
        * t
        / r.clamp_min(frozen.objective.resolved_coefficient_floor)
    )
    qs = torch.tensor([0.5, 0.9, 0.95, 0.99, 0.999], dtype=torch.float64)
    values = torch.quantile(coefficient, qs)
    diagonal_fraction = float(triangle.diagonal.double().mean())
    diagonal_t_mean = float(t[triangle.diagonal].mean())
    corner = float(((t > 0.95) & (h > 0.90)).double().mean())
    invariants = {
        "finite": bool(torch.isfinite(coefficient).all()),
        "nonnegative_r": bool((r >= 0).all()),
        "ordered": bool((r <= t).all()),
        "interval_valid": bool(((h >= -1e-12) & (h <= t + 1e-12)).all()),
        "diagonal_fraction": abs(diagonal_fraction - 0.5) < 0.01,
        "diagonal_exact": bool(torch.equal(r[triangle.diagonal], t[triangle.diagonal])),
        "diagonal_base_law": (
            abs(diagonal_t_mean - 0.5) < 0.01
            if arm == "ordered_uniform"
            else 0.64 < diagonal_t_mean < 0.70
        ),
    }
    checks = dict(invariants)
    if arm == "ordered_uniform":
        # P(t>.95, h>.90) = .0075 before the 50% diagonal replacement.
        checks["uniform_corner_probability"] = abs(corner - 0.00375) < 0.00035
    if arm != "legacy":
        # Applied to every successor arm, not just ordered_logitnormal.  Gating
        # only the arm whose tail was already clean left the production arm --
        # ordered_uniform -- measured but unenforced at CAP-EMF-1's level of
        # ill-conditioning.  ordered_uniform cannot reach zero here because its
        # corner mass is real, so the bound is the tail that actually damaged
        # CAP-EMF-1: no more than 2% of rows above a coefficient of 7, and
        # nothing beyond 15.
        share_above_7 = float((coefficient > 7).double().mean())
        checks["coefficient_tail_control"] = share_above_7 < (
            1e-3 if arm == "ordered_logitnormal" else 2e-2
        )
        checks["coefficient_extreme_tail_control"] = (
            float((coefficient > 15).double().mean()) < 1e-4
        )
    if arm != "legacy":
        checks["no_sampled_r_floor"] = frozen.objective.sampled_r_floor == 0.0

    failed = sorted(name for name, ok in checks.items() if not ok)
    return {
        "arm": arm,
        "count": count,
        "seed": seed,
        "numerical_candidate": numerical,
        "emf_delta": delta,
        "sampler_mode": frozen.objective.sampler_mode,
        "diagonal_sampling": frozen.objective.diagonal_sampling,
        "sampled_r_floor": frozen.objective.sampled_r_floor,
        "denominator_floor": frozen.objective.emf_denominator_floor,
        "diagonal_fraction": diagonal_fraction,
        "diagonal_t_mean": diagonal_t_mean,
        "active_fraction": float((h > delta).double().mean()),
        "mean_interval": float(h.mean()),
        "inference_corner_fraction_t95_h90": corner,
        "coefficient_quantiles": {
            f"q{100 * q:g}": float(value) for q, value in zip(qs.tolist(), values)
        },
        "coefficient_tail_fraction": {
            "gt_3": float((coefficient > 3).double().mean()),
            "gt_7": float((coefficient > 7).double().mean()),
            "gt_15": float((coefficient > 15).double().mean()),
            "gt_30": float((coefficient > 30).double().mean()),
        },
        "checks": checks,
        "failed": failed,
        "verdict": "PASS" if not failed else "FAIL",
    }


def run_all_samplers(
    count: int,
    seed: int,
    numerical: str = "local_1000_d0002_fp32",
) -> dict:
    arms = {
        arm: audit_sampler(arm, count=count, seed=seed + index, numerical=numerical)
        for index, arm in enumerate(SAMPLER_ARMS)
    }
    failed = [arm for arm, result in arms.items() if result["verdict"] != "PASS"]
    return {
        "status": "cap-emf2-sampler-audit",
        "numerical_candidate": numerical,
        "count_per_arm": count,
        "seed": seed,
        "arms": arms,
        "failed": failed,
        "decision": "GO" if not failed else "NO_GO",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20_260_806)
    parser.add_argument("--numerical", default="local_1000_d0002_fp32")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("sampler_audit.json")
    )
    args = parser.parse_args()
    assert_unused(args.out)
    result = run_all_samplers(args.count, args.seed, args.numerical)
    result["source_sha256"] = source_manifest()
    digest = write_json_atomic(args.out, result)
    print(json.dumps(result, indent=2))
    print(f"wrote {args.out} sha256={digest}")
    return 0 if result["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
