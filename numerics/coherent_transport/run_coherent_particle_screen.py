"""Stage 1A: free-particle coherent-plan screen (research plan sections 9, 15, 18).

Compares, on structured-geometry targets, with FREE PARTICLES and NO neural net:
  paper        - official-style drift (local geometry learner);
  psqt_indep   - the current independent sliced correction (section 2.2);
  est_bary     - EST barycentric destination (section 4.4);
  consensus    - hard EST consensus routing (section 4.5);
  sinkhorn     - dense entropic-OT barycentric map (strong reference).

Central question (RQ1/H1): does converting sliced evidence into a coherent joint
plan reduce off-support leakage / improve support precision vs the independent
correction, while retaining coverage and rare mass? Matched target access: all
arms see one fixed planning target set per seed. Etas are declared, not tuned on
the evaluation metrics (guardrail 14.5).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from lowdim_drift import drift_paper                              # noqa: E402
from est_plan import (all_rank_matches, est_barycenter,          # noqa: E402
                      hard_consensus, unit_directions)
from metrics import (precision_recall, checkerboard_leakage,     # noqa: E402
                     rare_mass_error, energy_distance2)
import targets as T                                              # noqa: E402

N, STEPS, L, SEEDS = 256, 200, 32, 3
ETA = {"paper": 0.05, "psqt_indep": 0.3, "est_bary": 0.3,
       "consensus": 0.3, "sinkhorn": 0.3}
TAU = 0.2


def sinkhorn_map(X, Y, eps, iters=60):
    C = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    K = np.exp(-C / eps)
    u = np.ones(len(X)); v = np.ones(len(Y))
    for _ in range(iters):
        u = 1.0 / (K @ v + 1e-300)
        v = 1.0 / (K.T @ u + 1e-300)
    P = u[:, None] * K * v[None, :]
    return (P @ Y) / (P.sum(1, keepdims=True) + 1e-300)


def psqt_indep(q, Y, U):
    d = q.shape[1]
    PX = q @ U.T; PY = Y @ U.T
    Pi = all_rank_matches(q, Y, U)
    g = np.zeros_like(q)
    for l in range(U.shape[0]):
        scal = PY[Pi[l], l] - PX[:, l]
        g += scal[:, None] * U[l][None, :]
    return (d / U.shape[0]) * g


def step(arm, q, Y, U, eps):
    if arm == "paper":
        return q + ETA[arm] * drift_paper(q, Y, TAU, True)
    if arm == "psqt_indep":
        return q + ETA[arm] * psqt_indep(q, Y, U)
    if arm in ("est_bary", "consensus"):
        Pi = all_rank_matches(q, Y, U)
        Tdest = est_barycenter(Y, Pi) if arm == "est_bary" \
            else hard_consensus(Y, Pi)[0]
        return q + ETA[arm] * (Tdest - q)
    if arm == "sinkhorn":
        return q + ETA[arm] * (sinkhorn_map(q, Y, eps) - q)
    raise ValueError(arm)


def run_arm(arm, tgt, seed):
    rng = np.random.default_rng(1000 + seed)
    drng = np.random.default_rng(5000 + seed)
    Y = tgt.sampler(N, drng)                       # fixed planning target set
    U = unit_directions(L, 2, seed=42 + seed)
    q = rng.normal(size=(N, 2)) * tgt.scale        # broad init (reach is easy)
    eps = 0.02 * tgt.scale ** 2
    for _ in range(STEPS):
        q = step(arm, q, Y, U, eps)
        if not np.all(np.isfinite(q)):
            break
    Yeval = tgt.sampler(1024, np.random.default_rng(9000 + seed))
    pr = precision_recall(q, Yeval)
    row = {"precision": pr["precision"], "off_support": pr["off_support"],
           "recall": pr["recall"], "coverage": pr["coverage"],
           "ed2": energy_distance2(q, Yeval)}
    if tgt.kind == "checkerboard":
        row["leak"] = checkerboard_leakage(q)
    if tgt.modes is not None:
        row["mass_l1"] = rare_mass_error(q, tgt.modes, tgt.sigmas,
                                         tgt.weights)["mass_l1"]
    return row


ARMS = ["paper", "psqt_indep", "est_bary", "consensus", "sinkhorn"]


def main():
    t0 = time.time()
    for tgt in T.suite():
        print(f"\n=== {tgt.name} ({tgt.kind}) ===")
        hdr = f"{'arm':12}{'prec':>7}{'offsup':>8}{'recall':>7}{'cover':>7}{'ed2':>8}"
        extra = tgt.kind == "checkerboard" or tgt.modes is not None
        print(hdr + (f"{'extra':>9}" if extra else ""))
        for arm in ARMS:
            rows = [run_arm(arm, tgt, s) for s in range(SEEDS)]
            med = {k: float(np.median([r[k] for r in rows if k in r]))
                   for k in rows[0]}
            line = (f"{arm:12}{med['precision']:>7.2f}{med['off_support']:>8.2f}"
                    f"{med['recall']:>7.2f}{med['coverage']:>7.2f}{med['ed2']:>8.4f}")
            if "leak" in med:
                line += f"  leak={med['leak']:.2f}"
            if "mass_l1" in med:
                line += f"  mL1={med['mass_l1']:.2f}"
            print(line, flush=True)
    print(f"\nscreen wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
