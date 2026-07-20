"""ABC secondary comparison (plan section 7, execution step 12).

A POST-GATE mechanism-attribution comparison.  It is deliberately NOT a gate:
the frozen E4 particle gate already concluded (PASS) and the frozen E5 generator
gate already concluded (FAIL); nothing here changes either verdict or selects a
policy.  Recycling the frozen test targets is therefore permitted -- no
selection, freezing, or gate decision is made from these numbers.

Scientific question: the NCJ particle winner isolates two mechanisms over the
paper field -- (1) drop the identifiability-inert P*Q gain (normalized), and
(2) remove the finite self-pair via an independent cross-fit reference batch.
The Analytical Bias Correction (ABC) is the cheaper single-batch alternative to
(2): it corrects the leading self-normalized minibatch-centroid bias
analytically on the reused/masked batch, with no second generator/particle
forward pass.  This run measures whether ABC alone recovers the cross-fit gain.

Arms (all eta = frozen NCJ eta, tau = 0.35, N/B/steps = standard):

    paper                exact P*Q, reused+masked          -- baseline
    normalized-only      constant gain, reused+masked      -- mechanism (1)
    abc                  constant+ABC, reused+masked        -- (1)+analytic (2)
    normalized-crossfit  constant gain, independent+unmask  -- (1)+cross-fit (2)

Reduced seed count keeps this a bounded context study; it makes no claim beyond
the mechanism comparison and does not supersede the frozen gate artifacts.

    uv run --with numpy --with scipy python numerics/ncj_abc_comparison.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_identifiability_improvement as R  # noqa: E402

CONTEXT_SEEDS = 6


def main() -> None:
    # Bounded context profile: identical 400-step dynamics and N/B as the
    # frozen standard study so the ED2 numbers are directly comparable, only
    # fewer paired seeds.  Named "abc-context" so it is never mistaken for a
    # gate run and skips no clean-tree guarantee (we require a clean tree here
    # explicitly below).
    base = R.PROFILES["standard"]
    profile = R.Profile("abc-context", base.steps, base.N, base.batch,
                        CONTEXT_SEEDS, CONTEXT_SEEDS, base.ref_final,
                        base.ref_cross)

    status = R._git("status", "--porcelain")
    if status:
        raise RuntimeError(
            "ABC context comparison requires a clean Git tree; status:\n"
            f"{status}")

    R.verify_registry_hashes()
    _, frozen = R.latest_frozen("standard")
    eta = float(frozen["eta"])

    arms = [
        R.PAPER,
        R.Arm("normalized-only", "constant", False, True, 0.0, eta,
              R.NORM_CLIP),
        R.Arm("abc", "abc", False, True, 0.0, eta, R.NORM_CLIP),
        R.Arm("normalized-crossfit", "constant", True, False, 0.0, eta,
              R.NORM_CLIP),
    ]

    run, rows = R.run_grid_with_sources(
        "abc-comparison", profile, R.TEST_REGISTRY, arms, CONTEXT_SEEDS,
        extra_config={
            "purpose": "post-gate ABC mechanism comparison (plan section 7)",
            "not_a_gate": True,
            "frozen_e4_reference":
                "20260720-011000-NCJ-test-standard/e4_gate.json",
            "frozen_eta": eta,
        },
        extra_sources=[HERE / "ncj_abc_comparison.py"])

    def stats(arm: str) -> dict:
        overall = R.hierarchical_stats(rows, arm)
        gm = R.hierarchical_stats(
            rows, arm, cell_filter=lambda r: r["family"] in
            ("gaussian", "gauss_mixture", "grid_mixture"))
        ng = R.hierarchical_stats(
            rows, arm, cell_filter=lambda r: r["family"] not in
            ("gaussian", "gauss_mixture", "grid_mixture"))
        winning = sum(1 for v in overall["cell_ratios"].values() if v < 1.0)
        return {
            "point_ratio": overall["point_ratio"],
            "hierarchical_ci": overall["hierarchical_ci"],
            "winning_cells_fraction": winning / len(overall["cell_ratios"]),
            "family_ratios": overall["family_ratios"],
            "gaussian_mixture_point": gm["point_ratio"],
            "gaussian_mixture_ci": gm["hierarchical_ci"],
            "non_gaussian_point": ng["point_ratio"],
            "non_gaussian_ci": ng["hierarchical_ci"],
        }

    summary = {
        "stage": "abc-comparison",
        "not_a_gate": True,
        "profile": R.asdict(profile),
        "frozen_eta": eta,
        "context_seeds": CONTEXT_SEEDS,
        "arm_stats_vs_paper": {a.label: stats(a.label)
                               for a in arms if a.label != "paper"},
        "distinct_total_kernel_pairs":
            sorted({int(r["total_kernel_pairs"]) for r in rows}),
    }
    (run.dir / "abc_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    run.log("")
    run.log("ABC comparison (ratio vs paper, lower is better):")
    for label, s in summary["arm_stats_vs_paper"].items():
        run.log(f"  {label:22s} ratio={s['point_ratio']:.4f} "
                f"CI={s['hierarchical_ci']} "
                f"win={s['winning_cells_fraction']:.2f}")
    run.finish(len(rows))
    print(f"\nwrote {run.dir / 'abc_comparison.json'}")


if __name__ == "__main__":
    main()
