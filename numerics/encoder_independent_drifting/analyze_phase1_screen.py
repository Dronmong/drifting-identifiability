"""Deterministic tables from a completed Phase-1 screen artifact.

Reads only ``phase1_screen.json``; performs no training and makes no
decisions.  Every number printed here is already in the artifact, so the
results document can be regenerated from the sealed run alone.

    uv run --python 3.12 --with numpy python \
      -m numerics.encoder_independent_drifting.analyze_phase1_screen
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = [f"A{i}" for i in range(9)]


def _median(values) -> float:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def arm_table(rows: list[dict]) -> None:
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    print(f"\n{'arm':4} {'score':>9} {'ED2':>9} {'SW1':>8} {'patchED2':>9} "
          f"{'specL1':>8} {'prec':>6} {'cover':>6} {'sec':>7} {'kernpairs':>12}")
    for arm in ARMS:
        cells = by_arm.get(arm)
        if not cells:
            continue
        print(f"{arm:4} "
              f"{_median([c.get('geometry_score') for c in cells]):9.3f} "
              f"{_median([c.get('ed2') for c in cells]):9.4f} "
              f"{_median([c.get('sw1') for c in cells]):8.4f} "
              f"{_median([c.get('patch_ed2') for c in cells]):9.4f} "
              f"{_median([c.get('spectral_l1') for c in cells]):8.4f} "
              f"{_median([c.get('precision') for c in cells]):6.3f} "
              f"{_median([c.get('coverage') for c in cells]):6.3f} "
              f"{_median([c.get('wall_seconds') for c in cells]):7.1f} "
              f"{_median([c.get('cost_kernel_pairs') for c in cells]):12.3g}")


def per_target_table(rows: list[dict]) -> None:
    by_key: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        by_key[(row["target"], row["arm"])].append(
            row.get("geometry_score"))
    targets = sorted({row["target"] for row in rows})
    shown = [a for a in ARMS if any((t, a) in by_key for t in targets)]
    print("\nmedian geometry score by target (lower is better; "
          "1.0 = as good as real data)")
    print(f"{'target':18} " + " ".join(f"{a:>8}" for a in shown))
    for target in targets:
        cells = [_median(by_key.get((target, a), [])) for a in shown]
        print(f"{target:18} " + " ".join(f"{c:8.2f}" for c in cells))


def coverage_table(rows: list[dict]) -> None:
    by_key: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        by_key[(row["target"], row["arm"])].append(row.get("coverage"))
    targets = sorted({row["target"] for row in rows})
    shown = [a for a in ARMS if any((t, a) in by_key for t in targets)]
    print("\nmedian calibrated support coverage by target")
    print(f"{'target':18} " + " ".join(f"{a:>8}" for a in shown))
    for target in targets:
        cells = [_median(by_key.get((target, a), [])) for a in shown]
        print(f"{target:18} " + " ".join(f"{c:8.3f}" for c in cells))


def anchor_table(rows: list[dict]) -> None:
    print("\nanchor presence (arms carrying an anchor)")
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("anchor_share_median") is not None:
            by_arm[row["arm"]].append(row)
    for arm in ARMS:
        cells = by_arm.get(arm)
        if not cells:
            continue
        present = sum(bool(c.get("anchor_present")) for c in cells)
        print(f"  {arm}: median gradient share "
              f"{_median([c['anchor_share_median'] for c in cells]):.4f}, "
              f"present in {present}/{len(cells)} cells")


def collision_table(rows: list[dict]) -> None:
    print("\ncollision detection by arm and source")
    for row in rows:
        report = row.get("collisions")
        if not report:
            continue
        for source, result in sorted(report.items()):
            missed = ",".join(result["failed"]) or "-"
            print(f"  {row['arm']} seed {row['seed']} {source:28} "
                  f"{result['detected']}/{result['total']} missed: {missed}")


def gate_table(gate: dict) -> None:
    print("\n=== PHASE 1 EXIT GATE ===")
    for name, condition in gate["conditions"].items():
        status = "PASS" if condition["pass"] else "FAIL"
        extra = ""
        if np.isfinite(condition.get("ratio", np.nan)):
            extra = (f"  ratio={condition['ratio']:.4f} "
                     f"[{condition['low']:.4f},{condition['high']:.4f}] "
                     f"wins={condition['wins']}/{condition['pairs']}")
        print(f"  [{status}] {name}{extra}")
    print(f"  overall: {'PASS' if gate['gate_pass'] else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path,
                        default=HERE / "phase1_screen.json")
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    rows = payload["rows"]
    print(f"artifact: {args.artifact.name}")
    print(f"status:   {payload['status']}")
    print(f"cells:    {len(rows)} rows, "
          f"{len({(r['target'], r['seed']) for r in rows})} cells, "
          f"{payload['elapsed_seconds'] / 3600:.2f} h")
    print(f"commit:   {payload['provenance']['commit'][:10]} "
          f"(dirty={payload['provenance']['git_dirty']})")
    arm_table(rows)
    per_target_table(rows)
    coverage_table(rows)
    anchor_table(rows)
    collision_table(rows)
    gate_table(payload["gate"])


if __name__ == "__main__":
    main()
