"""Hash-bound review of the fixed, uncurated 750k and 800k grids.

This is a veto against repeating the project's metric-versus-geometry failure.
It cannot rescue a failed quantitative comparison and it does not select or
replace samples.  The reviewer sees the two prospectively fixed grids and
records only whether the final grid remains recognizable and noncollapsed
relative to the admitted foundation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..stage_cap2.artifacts import (
    assert_unused,
    file_sha256,
    verify_json,
    write_json_atomic,
)
from .artifacts import source_manifest

STATUS = "asfd-final-visual-review"
PASS_PHRASE = (
    "I reviewed both fixed uncurated grids without selecting or replacing samples"
)


def _portable(path: Path, anchor: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve(reference: object, anchor: Path) -> Path:
    if not isinstance(reference, str) or not reference:
        raise RuntimeError("evaluation has no fixed-grid path")
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (anchor / path).resolve()


def _verified_grid(evaluation: dict, evaluation_path: Path) -> tuple[Path, str]:
    record = evaluation.get("uncurated_grid")
    if not isinstance(record, dict):
        raise TypeError("evaluation lacks its uncurated grid ledger")
    path = _resolve(record.get("path"), evaluation_path.parent)
    digest = record.get("sha256")
    if not path.is_file() or file_sha256(path) != digest:
        raise RuntimeError("fixed uncurated grid is missing or changed")
    if int(record.get("rows", -1)) != 8 or int(record.get("columns", -1)) != 16:
        raise RuntimeError("fixed uncurated grid layout changed")
    if "no curation" not in str(record.get("selection", "")).lower():
        raise RuntimeError("grid is not explicitly recorded as uncurated")
    return path, str(digest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-evaluation", type=Path, required=True)
    parser.add_argument("--asfd-evaluation", type=Path, required=True)
    parser.add_argument("--decision", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--acknowledgement", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_unused(args.out)
    if not args.reviewer.strip():
        raise ValueError("reviewer name must be nonempty")
    if args.acknowledgement != PASS_PHRASE:
        raise ValueError(f"acknowledgement must exactly equal: {PASS_PHRASE!r}")
    foundation = verify_json(
        args.foundation_evaluation, "cap-emf2-development-evaluation"
    )
    final = verify_json(args.asfd_evaluation, "asfd-final-evaluation")
    foundation_grid, foundation_grid_sha = _verified_grid(
        foundation, args.foundation_evaluation
    )
    final_grid, final_grid_sha = _verified_grid(final, args.asfd_evaluation)
    payload = {
        "status": STATUS,
        "decision": args.decision,
        "reviewer": args.reviewer.strip(),
        "acknowledgement": args.acknowledgement,
        "foundation_evaluation": {
            "path": _portable(args.foundation_evaluation, args.out.parent),
            "sha256": foundation["artifact_sha256"],
        },
        "asfd_evaluation": {
            "path": _portable(args.asfd_evaluation, args.out.parent),
            "sha256": final["artifact_sha256"],
        },
        "foundation_grid": {
            "path": _portable(foundation_grid, args.out.parent),
            "sha256": foundation_grid_sha,
        },
        "asfd_grid": {
            "path": _portable(final_grid, args.out.parent),
            "sha256": final_grid_sha,
        },
        "scope": (
            "Veto-only review of two fixed uncurated grids. PASS cannot rescue "
            "a failed metric, integrity, diversity, or memorization check."
        ),
        "source_sha256": source_manifest(),
    }
    digest = write_json_atomic(args.out, payload)
    print(f"wrote {args.out} sha256={digest} decision={args.decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
