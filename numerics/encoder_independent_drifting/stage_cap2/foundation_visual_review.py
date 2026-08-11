"""Hash-bound human capability review for the 750k foundation grid.

Recognizability is a scientific precondition for ASFD, but it is not honestly
reducible to the repository's train-only moment/rank gates.  This command binds
one explicit operator decision to the fixed, uncurated 750k evaluation grid.
It cannot rescue a failed quantitative gate and it never opens test data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .artifacts import assert_unused, file_sha256, verify_json, write_json_atomic

STATUS = "cap-emf2-foundation-visual-review"
PASS_PHRASE = "I reviewed the fixed uncurated grid without selecting samples"


def _resolve(reference: object, anchor: Path) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise RuntimeError("evaluation has no fixed-grid reference")
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (anchor / path).resolve()


def build_review(
    evaluation_path: Path,
    *,
    decision: str,
    reviewer: str,
    acknowledgement: str,
    out: Path,
) -> dict:
    evaluation = verify_json(evaluation_path, "cap-emf2-development-evaluation")
    if int(evaluation.get("step", -1)) != 750_000:
        raise RuntimeError("visual capability review is fixed to the 750k EMA")
    if evaluation.get("checkpoint_selection") != "final recorded EMA":
        raise RuntimeError("visual review requires the declared final EMA")
    grid = evaluation.get("uncurated_grid")
    if not isinstance(grid, dict):
        raise TypeError("evaluation lacks its uncurated grid ledger")
    grid_path = _resolve(grid.get("path"), evaluation_path.parent)
    expected_sha = grid.get("sha256")
    if not grid_path.is_file() or file_sha256(grid_path) != expected_sha:
        raise RuntimeError("fixed uncurated grid is missing or changed")
    if int(grid.get("rows", -1)) != 8 or int(grid.get("columns", -1)) != 16:
        raise RuntimeError("fixed uncurated grid layout changed")
    if "no curation" not in str(grid.get("selection", "")).lower():
        raise RuntimeError("grid is not explicitly recorded as uncurated")
    normalized_decision = decision.upper()
    if normalized_decision not in {"PASS", "FAIL"}:
        raise ValueError("decision must be PASS or FAIL")
    if not reviewer.strip():
        raise ValueError("reviewer identity must be nonempty")
    if normalized_decision == "PASS" and acknowledgement != PASS_PHRASE:
        raise RuntimeError(
            "PASS requires the exact --acknowledgement phrase printed by --help"
        )
    return {
        "status": STATUS,
        "decision": normalized_decision,
        "reviewer": reviewer.strip(),
        "acknowledgement": acknowledgement,
        "evaluation": {
            "path": Path(
                os.path.relpath(evaluation_path.resolve(), out.parent.resolve())
            ).as_posix(),
            "sha256": evaluation["artifact_sha256"],
            "step": 750_000,
        },
        "grid": {
            "path": Path(os.path.relpath(grid_path, out.parent.resolve())).as_posix(),
            "sha256": expected_sha,
            "rows": 8,
            "columns": 16,
            "selection": grid["selection"],
        },
        "scope": (
            "subjective recognizability check on fixed train-reference samples; "
            "cannot override any quantitative or integrity failure"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--decision", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--acknowledgement",
        required=True,
        help=f"PASS requires exactly: {PASS_PHRASE}",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_unused(args.out)
    result = build_review(
        args.evaluation,
        decision=args.decision,
        reviewer=args.reviewer,
        acknowledgement=args.acknowledgement,
        out=args.out,
    )
    digest = write_json_atomic(args.out, result)
    print(f"wrote {args.out} sha256={digest} decision={result['decision']}")
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
