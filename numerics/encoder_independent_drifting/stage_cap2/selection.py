"""Concurrent matched-arm selection for the CAP-EMF-2 150k screen.

Individual promotion records answer whether one arm clears the absolute
development gate relative to CAP-EMF-1.  They do not answer the screen's
causal question.  This certificate consumes all three immutable records,
keeps the newly trained legacy sampler as the concurrent control, and selects
at most one ordered sampler for a matched 300k continuation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from .artifacts import assert_unused, verify_json, write_json_atomic
from .config import SAMPLER_ARMS
from .promotion import CALIBRATION_MARGIN_KIND, revalidate_promotion

SELECTION_STATUS = "cap-emf2-cross-arm-selection"
LEGACY_ARM = "legacy"
ORDERED_ARMS = ("ordered_logitnormal", "ordered_uniform")


def _reference(path: Path, anchor: Path) -> str:
    return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()


def _resolve(reference: object, anchor: Path) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise RuntimeError("selection contains an empty promotion reference")
    path = Path(reference)
    return path if path.is_absolute() else (anchor / path).resolve()


def _finite_metric(record: dict, key: str) -> float:
    value = record.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"selection metric {key!r} is not finite")
    return float(value)


def _hash64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _policy(promotions: dict[str, dict]) -> dict:
    if set(promotions) != set(SAMPLER_ARMS):
        raise RuntimeError("selection requires exactly the three frozen sampler arms")
    candidates = {record.get("candidate") for record in promotions.values()}
    preflights = {record.get("preflight_sha256") for record in promotions.values()}
    if (
        len(candidates) != 1
        or not isinstance(next(iter(candidates)), str)
        or not next(iter(candidates))
        or len(preflights) != 1
        or not _hash64(next(iter(preflights)))
    ):
        raise RuntimeError("selection arms do not share one candidate and preflight")

    legacy = promotions[LEGACY_ARM]
    legacy_metrics = legacy.get("comparison", {}).get("candidate", {})
    margin = legacy.get("comparison", {}).get("calibration_margin", {})
    fid_margin = _finite_metric(margin, "clean_fid")
    kid_margin = _finite_metric(margin, "clean_kid")
    if margin.get("kind") != CALIBRATION_MARGIN_KIND:
        raise RuntimeError(
            "selection calibration margin is not the direct real/real discrepancy"
        )
    if fid_margin < 0 or kid_margin < 0:
        raise RuntimeError("selection calibration margins must be nonnegative")
    for arm, record in promotions.items():
        if record.get("arm") != arm:
            raise RuntimeError(f"selection promotion is mislabeled for {arm}")
        if record.get("comparison", {}).get("calibration_margin") != margin:
            raise RuntimeError("selection arms use different calibration margins")

    legacy_fid = _finite_metric(legacy_metrics, "clean_fid")
    legacy_kid = _finite_metric(legacy_metrics, "clean_kid")

    eligibility: dict[str, dict[str, bool]] = {}
    for arm in ORDERED_ARMS:
        record = promotions[arm]
        metrics = record.get("comparison", {}).get("candidate", {})
        checks = {
            "individual_promotion_go": record.get("decision") == "GO",
            "fid_beats_concurrent_legacy_beyond_margin": (
                _finite_metric(metrics, "clean_fid") < legacy_fid - fid_margin
            ),
            "kid_beats_concurrent_legacy_beyond_margin": (
                _finite_metric(metrics, "clean_kid") < legacy_kid - kid_margin
            ),
        }
        eligibility[arm] = checks

    eligible = [arm for arm in ORDERED_ARMS if all(eligibility[arm].values())]
    winner: str | None = None
    tie_reason: str | None = None
    if len(eligible) == 1:
        winner = eligible[0]
    elif len(eligible) == 2:
        left, right = eligible
        left_metrics = promotions[left]["comparison"]["candidate"]
        right_metrics = promotions[right]["comparison"]["candidate"]
        left_fid = _finite_metric(left_metrics, "clean_fid")
        right_fid = _finite_metric(right_metrics, "clean_fid")
        left_kid = _finite_metric(left_metrics, "clean_kid")
        right_kid = _finite_metric(right_metrics, "clean_kid")
        if abs(left_fid - right_fid) > fid_margin:
            winner = left if left_fid < right_fid else right
        elif abs(left_kid - right_kid) > kid_margin:
            winner = left if left_kid < right_kid else right
        else:
            tie_reason = (
                "ordered arms are indistinguishable within both calibrated "
                "standard-metric margins; do not select on noisy auxiliary KID"
            )
    else:
        tie_reason = "no ordered arm clears the concurrent legacy comparison"

    checks = {
        "all_three_promotions_revalidated": all(
            record.get("revalidated") is True for record in promotions.values()
        ),
        "legacy_control_continuation_go": (
            legacy.get("control_continuation", {}).get("decision") == "GO"
        ),
        "exactly_one_ordered_winner": winner in ORDERED_ARMS,
    }
    selected = [LEGACY_ARM, winner] if all(checks.values()) else []
    return {
        "decision": "GO" if selected else "NO_GO",
        "candidate": next(iter(candidates)),
        "preflight_sha256": next(iter(preflights)),
        "calibration_margin": margin,
        "eligibility": eligibility,
        "eligible_ordered_arms": eligible,
        "ordered_winner": winner,
        "selected_arms": selected,
        "checks": checks,
        "tie_reason": tie_reason,
        "policy": (
            "retain a numerically and mechanically valid concurrent legacy "
            "control without requiring it to beat the 650k historical model; "
            "ordered arm must clear its individual gate, "
            "and beat legacy on 50k CleanFID and 50k CleanKID beyond the direct "
            "real/real calibration margins; resolve two eligible arms by "
            "margin-aware 50k CleanFID then 50k CleanKID; auxiliary metrics "
            "are reported but never select or break a tie"
        ),
    }


def _validated_promotions(paths: dict[str, Path]) -> dict[str, dict]:
    records = {arm: revalidate_promotion(path) for arm, path in paths.items()}
    if set(records) != set(SAMPLER_ARMS):
        raise RuntimeError("selection promotion path set is incomplete")
    return records


def build_selection(*, promotion_paths: dict[str, Path], out: Path) -> dict:
    assert_unused(out)
    records = _validated_promotions(promotion_paths)
    policy = _policy(records)
    payload = {
        "status": SELECTION_STATUS,
        **policy,
        "promotion_sha256": {
            arm: record["artifact_sha256"] for arm, record in records.items()
        },
        "references": {
            arm: _reference(path, out.parent) for arm, path in promotion_paths.items()
        },
        "limits": [
            "This is a one-seed developmental successive-halving decision.",
            "It authorizes only the selected matched arms from 150k to 300k.",
            "It is not a confirmation, test-set decision, or general performance claim.",
        ],
    }
    payload["artifact_sha256"] = write_json_atomic(out, payload)
    return payload


def revalidate_selection(path: Path) -> dict:
    selection = verify_json(path, SELECTION_STATUS)
    references = selection.get("references")
    if not isinstance(references, dict) or set(references) != set(SAMPLER_ARMS):
        raise RuntimeError("CAP2 selection has an incomplete promotion ledger")
    paths = {
        arm: _resolve(reference, path.parent) for arm, reference in references.items()
    }
    records = _validated_promotions(paths)
    policy = _policy(records)
    bindings = {
        "policy": all(selection.get(key) == value for key, value in policy.items()),
        "promotion_hashes": selection.get("promotion_sha256")
        == {arm: record["artifact_sha256"] for arm, record in records.items()},
    }
    invalid = sorted(name for name, valid in bindings.items() if not valid)
    if invalid:
        raise RuntimeError(f"CAP2 selection revalidation failed: {invalid}")
    selection["revalidated"] = True
    return selection


def load_selection(
    path: Path, *, promotion_path: Path, arm: str, candidate: str
) -> dict:
    selection = revalidate_selection(path)
    if selection.get("decision") != "GO":
        raise RuntimeError("CAP2 cross-arm selection did not return GO")
    if arm not in selection.get("selected_arms", []):
        raise RuntimeError(f"CAP2 arm {arm!r} was not selected for 300k")
    if selection.get("candidate") != candidate:
        raise RuntimeError("CAP2 selection uses a different numerical candidate")
    promotion = verify_json(promotion_path, "cap-emf2-promotion")
    if selection.get("promotion_sha256", {}).get(arm) != promotion["artifact_sha256"]:
        raise RuntimeError("CAP2 selection is not bound to this arm promotion")
    return selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revalidate",
        type=Path,
        help="recompute an existing selection and print its verified winner",
    )
    for arm in SAMPLER_ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}-promotion", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    build_values = [
        *(getattr(args, f"{arm}_promotion") for arm in SAMPLER_ARMS),
        args.out,
    ]
    if args.revalidate is not None:
        if any(value is not None for value in build_values):
            parser.error(
                "--revalidate is mutually exclusive with promotion inputs and --out"
            )
        result = revalidate_selection(args.revalidate)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "ordered_winner": result["ordered_winner"],
                    "selected_arms": result["selected_arms"],
                    "revalidated": result["revalidated"],
                }
            )
        )
        return 0 if result["decision"] == "GO" else 1

    missing = [
        flag
        for flag, value in (
            *(
                (
                    f"--{arm.replace('_', '-')}-promotion",
                    getattr(args, f"{arm}_promotion"),
                )
                for arm in SAMPLER_ARMS
            ),
            ("--out", args.out),
        )
        if value is None
    ]
    if missing:
        parser.error(f"build mode requires {', '.join(missing)}")
    paths = {arm: getattr(args, f"{arm}_promotion") for arm in SAMPLER_ARMS}
    assert all(isinstance(path, Path) for path in paths.values())
    assert args.out is not None
    result = build_selection(promotion_paths=paths, out=args.out)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "selected_arms": result["selected_arms"],
                "tie_reason": result["tie_reason"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.out} sha256={result['artifact_sha256']}")
    return 0 if result["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
