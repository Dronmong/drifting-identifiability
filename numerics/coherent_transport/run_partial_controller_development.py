"""Stage 1B: partial-transport controller development (plan sections 4.2-4.6, 9).

Factorial over {planner} x {transported mass}, all on a paper-drift local base
(section 4.2, interleaved local-then-repair). Endpoint: combined precision x
coverage (support quality AND completeness), retaining ED2 and rare-mass.

Arms:
  local_only     - paper drift, no repair (baseline 3);
  psqt_full      - + independent PSQT repair, all particles (current method);
  coherent_full  - + consensus repair, all particles (Stage 1A winner);
  partial_fixed  - + deficit-fill partial repair, rho=0.20;
  partial_adapt  - + deficit-fill partial repair, adaptive rho (trust region).

Gate 1B: partial_adapt improves the combined endpoint over coherent_full and
does not merely select rho=1 everywhere (mean rho reported).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from lowdim_drift import drift_paper                                # noqa: E402
from est_plan import all_rank_matches, hard_consensus, unit_directions  # noqa: E402
from metrics import (precision_recall, rare_mass_error,            # noqa: E402
                     energy_distance2)
from partial_controller import (partial_repair, adaptive_rho)      # noqa: E402
from run_coherent_particle_screen import psqt_indep                # noqa: E402
import targets as T                                                # noqa: E402

N, STEPS, L, SEEDS = 256, 150, 32, 3
ETA_LOCAL, ETA_REPAIR = 0.05, 0.3


def run_arm(arm, tgt, seed):
    rng = np.random.default_rng(1000 + seed)
    drng = np.random.default_rng(5000 + seed)
    Y = tgt.sampler(N, drng)
    U = unit_directions(L, 2, seed=42 + seed)
    q = rng.normal(size=(N, 2)) * tgt.scale
    rhos = []
    for _ in range(STEPS):
        q = q + ETA_LOCAL * drift_paper(q, Y, 0.2, True)           # local base
        if arm == "local_only":
            pass
        elif arm == "psqt_full":
            q = q + ETA_REPAIR * psqt_indep(q, Y, U)
        elif arm == "coherent_full":
            Pi = all_rank_matches(q, Y, U)
            q = q + ETA_REPAIR * (hard_consensus(Y, Pi)[0] - q)
        elif arm == "partial_fixed":
            q = partial_repair(q, Y, 0.20)
        elif arm == "partial_adapt":
            r = adaptive_rho(q, Y); rhos.append(r)
            q = partial_repair(q, Y, r)
        if not np.all(np.isfinite(q)):
            break
    Ye = tgt.sampler(1024, np.random.default_rng(9000 + seed))
    pr = precision_recall(q, Ye)
    out = {"prec": pr["precision"], "cover": pr["coverage"],
           "combined": pr["precision"] * pr["coverage"],
           "ed2": energy_distance2(q, Ye)}
    if tgt.modes is not None:
        out["mass_l1"] = rare_mass_error(q, tgt.modes, tgt.sigmas,
                                         tgt.weights)["mass_l1"]
    if rhos:
        out["mean_rho"] = float(np.mean(rhos))
    return out


ARMS = ["local_only", "psqt_full", "coherent_full", "partial_fixed",
        "partial_adapt"]


def main():
    t0 = time.time()
    for tgt in T.suite():
        print(f"\n=== {tgt.name} ({tgt.kind}) ===")
        print(f"{'arm':14}{'prec':>6}{'cover':>7}{'comb':>7}{'ed2':>8}"
              f"{'extra':>10}")
        for arm in ARMS:
            rows = [run_arm(arm, tgt, s) for s in range(SEEDS)]
            med = {k: float(np.median([r[k] for r in rows if k in r]))
                   for k in rows[0]}
            line = (f"{arm:14}{med['prec']:>6.2f}{med['cover']:>7.2f}"
                    f"{med['combined']:>7.3f}{med['ed2']:>8.4f}")
            if "mass_l1" in med:
                line += f"  mL1={med['mass_l1']:.2f}"
            if "mean_rho" in med:
                line += f"  rho={med['mean_rho']:.2f}"
            print(line, flush=True)
    print(f"\nStage 1B wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
