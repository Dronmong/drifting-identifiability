"""Support-quality and distribution metrics (research plan section 7).

The Stage-1A gate is judged on SUPPORT precision / leakage, not ED2/SW1 alone
(guardrail section 14.2). Support metrics follow improved precision/recall
(Kynkaanniemi 2019) and density/coverage (Naeem 2020), with the manifold radius
calibrated from the TARGET's own k-NN distances (section 7.2: the support
threshold is calibrated on target-vs-target geometry). Pure numpy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lowdim_drift import energy_distance2, sliced_w1  # noqa: E402,F401


def _pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(
        ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0.0))


def knn_radius(P: np.ndarray, k: int = 5) -> np.ndarray:
    """Distance from each point of P to its k-th nearest neighbour within P."""
    D = _pairwise(P, P)
    np.fill_diagonal(D, np.inf)
    D.sort(axis=1)
    k = min(k, D.shape[1])
    return D[:, k - 1]


def precision_recall(gen: np.ndarray, target: np.ndarray, k: int = 5) -> dict:
    """Improved precision/recall + density/coverage, target-calibrated.

    precision = fraction of generated points inside the target manifold
      (off_support = 1 - precision = leakage);
    recall    = fraction of target points inside the generated manifold;
    density   = avg # target balls covering a generated point (/k);
    coverage  = fraction of target points with >= 1 generated point in their
      own k-NN ball (robust recall)."""
    rt = knn_radius(target, k)
    rg = knn_radius(gen, k)
    Dgt = _pairwise(gen, target)          # (Ng, Nt)
    inside_t = Dgt <= rt[None, :]         # gen i inside target j's ball
    precision = float(inside_t.any(axis=1).mean())
    Dtg = Dgt.T                            # (Nt, Ng)
    inside_g = Dtg <= rg[None, :]
    recall = float(inside_g.any(axis=1).mean())
    density = float(inside_t.sum(axis=1).mean() / k)
    coverage = float((Dgt <= rt[None, :]).any(axis=0).mean())
    return {"precision": precision, "recall": recall,
            "off_support": 1.0 - precision,
            "density": density, "coverage": coverage}


def checkerboard_leakage(
    gen: np.ndarray,
    cell: float = 1.0,
    cells: int = 4,
    origin: float = 0.0,
) -> float:
    """Fraction of generated points that fall in an OFF (empty) checkerboard
    cell or outside the declared checkerboard domain.

    A cell ``(i,j)`` is ON iff ``(i+j)`` is even. Earlier code checked parity
    without checking the domain, causing out-of-range points in even-parity
    cells to be counted as valid support.
    """
    if cell <= 0 or cells <= 0:
        raise ValueError("cell and cells must be positive")
    shifted = gen - origin
    idx = np.floor(shifted / cell).astype(int)
    in_domain = ((idx >= 0) & (idx < cells)).all(axis=1)
    on = ((idx[:, 0] + idx[:, 1]) % 2 == 0)
    return float((~(in_domain & on)).mean())


def calibrated_support_radius(
    target_calibration_a: np.ndarray,
    target_calibration_b: np.ndarray,
    quantile: float = 0.95,
) -> float:
    """Target-only global support radius from two independent target pools.

    For every point in pool A, compute its distance to the nearest point in
    pool B and take a frozen quantile. This avoids choosing the support
    threshold from candidate outputs or from the final evaluation cloud.
    """
    if not 0 < quantile <= 1:
        raise ValueError("quantile must lie in (0, 1]")
    if len(target_calibration_a) == 0 or len(target_calibration_b) == 0:
        raise ValueError("calibration pools must be nonempty")
    nearest = _pairwise(target_calibration_a, target_calibration_b).min(axis=1)
    return float(np.quantile(nearest, quantile))


def calibrated_precision_coverage(
    gen: np.ndarray,
    target_eval: np.ndarray,
    target_calibration_a: np.ndarray,
    target_calibration_b: np.ndarray,
    quantile: float = 0.95,
) -> dict:
    """Precision/coverage using a target-only independently calibrated radius."""
    radius = calibrated_support_radius(
        target_calibration_a, target_calibration_b, quantile)
    D = _pairwise(gen, target_eval)
    precision = float((D.min(axis=1) <= radius).mean())
    coverage = float((D.min(axis=0) <= radius).mean())
    return {
        "radius": radius,
        "precision": precision,
        "coverage": coverage,
        "off_support": 1.0 - precision,
    }


def rare_mass_error(gen: np.ndarray, modes: np.ndarray, sigmas: np.ndarray,
                    weights: np.ndarray, rho: float = 3.0) -> dict:
    """Per-mode empirical mass vs true weights (rare-component allocation).
    Oracle-evaluation only. Returns L1 mass error and worst rare-mode ratio."""
    sig = np.full(len(modes), sigmas) if np.isscalar(sigmas) else sigmas
    D = _pairwise(gen, modes)
    nearest = D.argmin(1)
    within = D[np.arange(len(gen)), nearest] <= rho * sig[nearest]
    counts = np.array([np.sum((nearest == k) & within)
                       for k in range(len(modes))], dtype=float)
    emp = counts / max(len(gen), 1)
    return {"mass_l1": float(np.abs(emp - weights).sum()),
            "empirical_mass": emp}


def _tests(log=print) -> None:
    rng = np.random.default_rng(3)
    fails = []

    def check(name, ok):
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            fails.append(name)

    # Same distribution -> high precision & recall.
    A = rng.normal(size=(300, 2))
    B = rng.normal(size=(300, 2))
    pr = precision_recall(A, B)
    check("1. matched samples -> precision & recall high",
          pr["precision"] > 0.8 and pr["recall"] > 0.8)

    # Off-support cloud -> low precision (high leakage).
    far = rng.normal(size=(300, 2)) + np.array([50.0, 0])
    pr2 = precision_recall(far, B)
    check("2. off-support cloud -> low precision / high off_support",
          pr2["precision"] < 0.05 and pr2["off_support"] > 0.95)

    # Generated covering only half the target modes -> reduced coverage.
    tgt = np.vstack([rng.normal(size=(150, 2)) * 0.05 + [-5, 0],
                     rng.normal(size=(150, 2)) * 0.05 + [5, 0]])
    gen_half = rng.normal(size=(300, 2)) * 0.05 + [-5, 0]
    check("3. half-covered target -> coverage ~0.5",
          abs(precision_recall(gen_half, tgt)["coverage"] - 0.5) < 0.15)

    # Checkerboard leakage: points on ON cells -> low; on OFF cells -> high.
    on_pts = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 0.5]])   # (i+j) even
    off_pts = np.array([[1.5, 0.5], [0.5, 1.5]])              # (i+j) odd
    outside = np.array([[-0.5, -0.5], [4.5, 4.5]])
    check("4. checkerboard leakage on/off/outside",
          checkerboard_leakage(on_pts) == 0.0 and
          checkerboard_leakage(off_pts) == 1.0 and
          checkerboard_leakage(outside) == 1.0)

    # Independently calibrated support: matched cloud is substantially better
    # than a far-away cloud under the same target-only threshold.
    C1 = rng.normal(size=(300, 2))
    C2 = rng.normal(size=(300, 2))
    E = rng.normal(size=(300, 2))
    matched = calibrated_precision_coverage(A, E, C1, C2)
    shifted = calibrated_precision_coverage(far, E, C1, C2)
    check("5. calibrated support separates matched and far clouds",
          matched["precision"] > shifted["precision"] + 0.7)

    if fails:
        raise SystemExit(f"metrics tests FAILED: {fails}")
    log("all metrics tests passed")


if __name__ == "__main__":
    _tests()
