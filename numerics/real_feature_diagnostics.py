"""Run Objective-7 diagnostics on real encoder feature tensors.

This script expects features that have already been extracted from a real
encoder, e.g. a `.npy` array of shape `[num_samples, feature_dim]` or a `.npz`
archive containing such an array.  It does not download models or datasets;
that keeps the numerical audit reproducible and avoids silently changing the
trusted experimental input.

Example:

    uv run --with numpy python numerics/real_feature_diagnostics.py \
      --features path/to/features.npy \
      --m 8 --num-probes 64 --taus 0.02 0.05 0.2

The output is a Markdown report with:

* feature normalization statistics;
* softmax effective sample size at the paper temperatures;
* empirical interaction-matrix rank/singular values;
* a finite dual-certificate-style lower frame constant computed from the
  Moore--Penrose left inverse, using the same `l1 -> linf` geometry as the Lean
  finite certificate;
* the same diagnostics for the column-reweighted Algorithm-2 limiting kernel.

Nothing here is a proof.  It is the real-feature counterpart to
`numerics/run_all.py`: the formulas are the finite matrices whose conditioning
the Lean theorems require.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import driftlab as dl


TAUS_DEFAULT = [0.02, 0.05, 0.2]


def out(lines: list[str], line: str = "") -> None:
    print(line)
    lines.append(line)


def strict_pairs(m: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(m) for j in range(i + 1, m)]


def load_feature_array(path: Path, key: str | None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        keys = list(archive.keys())
        if not keys:
            raise ValueError(f"{path} contains no arrays")
        chosen = key or keys[0]
        if chosen not in archive:
            raise ValueError(f"{path} has keys {keys}, not {chosen!r}")
        arr = archive[chosen]
    else:
        raise ValueError("expected .npy or .npz feature file")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D array after flattening, got shape {arr.shape}")
    mask = np.isfinite(arr).all(axis=1)
    arr = arr[mask]
    if arr.shape[0] < 2:
        raise ValueError("need at least two finite feature vectors")
    return arr


def normalize_features(
    features: np.ndarray, mode: str, rng: np.random.Generator, max_pairs: int
) -> tuple[np.ndarray, float]:
    if mode == "none":
        return features.copy(), 1.0
    centered = features - features.mean(axis=0, keepdims=True)
    if mode == "center":
        return centered, 1.0
    if mode != "mean-distance":
        raise ValueError(f"unknown normalization mode {mode!r}")
    n = centered.shape[0]
    pairs = min(max_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n - 1, size=pairs)
    j = j + (j >= i)
    d = np.linalg.norm(centered[i] - centered[j], axis=1)
    scale = float(np.mean(d[d > 0]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("mean pairwise distance is zero; features appear collapsed")
    return centered / scale, scale


def farthest_points(features: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    n = features.shape[0]
    count = min(count, n)
    chosen = [int(rng.integers(0, n))]
    dist2 = np.sum((features - features[chosen[0]]) ** 2, axis=1)
    for _ in range(1, count):
        idx = int(np.argmax(dist2))
        chosen.append(idx)
        new_dist2 = np.sum((features - features[idx]) ** 2, axis=1)
        dist2 = np.minimum(dist2, new_dist2)
    return np.array(chosen, dtype=int)


def random_points(features: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    count = min(count, features.shape[0])
    return rng.choice(features.shape[0], size=count, replace=False)


def choose_points(features: np.ndarray, count: int, method: str, rng: np.random.Generator) -> np.ndarray:
    if method == "farthest":
        return farthest_points(features, count, rng)
    if method == "random":
        return random_points(features, count, rng)
    raise ValueError(f"unknown selection method {method!r}")


def kernel_values(kernel: str, tau: float, distances: np.ndarray) -> np.ndarray:
    if kernel == "laplace":
        return np.exp(-distances / tau)
    if kernel == "gaussian":
        return np.exp(-(distances**2) / (2.0 * tau**2))
    raise ValueError(f"unknown kernel {kernel!r}")


def kernel_to_points(kernel: str, tau: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return kernel_values(kernel, tau, np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2))


def column_mass(kernel: str, tau: float, anchors: np.ndarray, y: np.ndarray) -> np.ndarray:
    return kernel_to_points(kernel, tau, anchors, y).sum(axis=0)


def interaction_matrix(
    supports: np.ndarray,
    probes: np.ndarray,
    tau: float,
    kernel: str,
    column_reweighted: bool,
    anchors: np.ndarray,
) -> np.ndarray:
    """Flattened vector-valued interaction matrix.

    Column `(i,j)` is the vector field over probes

        k(x,z_i) k(x,z_j) (z_i-z_j)

    optionally scaled by `1/sqrt(g(z_i)g(z_j))`, the column-reweighted
    Algorithm-2 population kernel from `ColumnReweightedMeanShift.lean`.
    """
    m, d = supports.shape
    pairs = strict_pairs(m)
    K = kernel_to_points(kernel, tau, probes, supports)
    if column_reweighted:
        g = column_mass(kernel, tau, anchors, supports)
        g = np.maximum(g, np.finfo(float).tiny)
    A = np.empty((probes.shape[0] * d, len(pairs)), dtype=np.float64)
    for col, (i, j) in enumerate(pairs):
        scale = K[:, i] * K[:, j]
        if column_reweighted:
            scale = scale / np.sqrt(g[i] * g[j])
        field = scale[:, None] * (supports[i] - supports[j])[None, :]
        A[:, col] = field.reshape(-1)
    return A


@dataclass
class MatrixDiagnostics:
    rank: int
    rows: int
    cols: int
    min_sv: float
    med_sv: float
    max_sv: float
    cond: float
    dual_c: float
    dual_resid_inf: float
    ceiling: float
    random_violation: float


def matrix_diagnostics(A: np.ndarray, rng: np.random.Generator, trials: int) -> MatrixDiagnostics:
    rows, cols = A.shape
    s = np.linalg.svd(A, compute_uv=False)
    tol = np.finfo(float).eps * max(rows, cols) * (s[0] if len(s) else 0.0)
    rank = int((s > tol).sum())
    min_sv = float(s[-1]) if len(s) else 0.0
    med_sv = float(np.median(s)) if len(s) else 0.0
    max_sv = float(s[0]) if len(s) else 0.0
    cond = float(max_sv / min_sv) if min_sv > 0 else float("inf")
    if rank == cols and cols > 0:
        L = np.linalg.pinv(A)
        resid = float(np.abs(L @ A - np.eye(cols)).max())
        mass = float(np.abs(L).sum())
        dual_c = 1.0 / mass if mass > 0 and np.isfinite(mass) else 0.0
    else:
        resid = float("inf")
        dual_c = 0.0
    ceiling = float(np.max(np.abs(A), axis=0).min()) if cols > 0 else 0.0
    worst = -np.inf
    if trials > 0 and cols > 0:
        for _ in range(trials):
            w = rng.standard_normal(cols)
            lhs = dual_c * np.abs(w).sum()
            rhs = np.abs(A @ w).max()
            worst = max(worst, float(lhs - rhs))
    else:
        worst = 0.0
    return MatrixDiagnostics(rank, rows, cols, min_sv, med_sv, max_sv, cond, dual_c, resid, ceiling, worst)


def ess_summary(features: np.ndarray, anchors: np.ndarray, tau: float, kernel: str) -> tuple[float, float, float]:
    K = kernel_to_points(kernel, tau, anchors, features)
    vals = np.array([dl.ess(row) for row in K])
    return float(np.quantile(vals, 0.1)), float(np.median(vals)), float(np.quantile(vals, 0.9))


def fmt(x: float) -> str:
    if np.isinf(x):
        return "inf"
    if np.isnan(x):
        return "nan"
    return f"{x:.3e}"


def emit_matrix_table(lines: list[str], title: str, rows: list[tuple[float, MatrixDiagnostics]]) -> None:
    out(lines, f"### {title}")
    out(lines)
    out(lines, "| tau | rank/cols | min sv | med sv | cond | dual c | ceiling | dual residual | max random violation |")
    out(lines, "|-----|-----------|--------|--------|------|--------|---------|---------------|----------------------|")
    for tau, d in rows:
        out(
            lines,
            f"| {tau:g} | {d.rank}/{d.cols} | {fmt(d.min_sv)} | {fmt(d.med_sv)} | "
            f"{fmt(d.cond)} | {fmt(d.dual_c)} | {fmt(d.ceiling)} | "
            f"{fmt(d.dual_resid_inf)} | {fmt(d.random_violation)} |",
        )
    out(lines)


def parse_taus(values: Iterable[str]) -> list[float]:
    return [float(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help=".npy/.npz feature array")
    parser.add_argument("--key", type=str, default=None, help="array key inside .npz; defaults to first key")
    parser.add_argument("--output", type=Path, default=Path("numerics/REAL_FEATURES.md"))
    parser.add_argument("--m", type=int, default=8, help="number of basis/support points")
    parser.add_argument("--num-probes", type=int, default=64, help="number of probe/anchor points")
    parser.add_argument("--taus", nargs="+", default=[str(t) for t in TAUS_DEFAULT], help="kernel temperatures")
    parser.add_argument("--kernel", choices=["laplace", "gaussian"], default="laplace")
    parser.add_argument("--normalize", choices=["mean-distance", "center", "none"], default="mean-distance")
    parser.add_argument("--select", choices=["farthest", "random"], default="farthest")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--max-normalization-pairs", type=int, default=20000)
    parser.add_argument("--random-violation-trials", type=int, default=5000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    raw = load_feature_array(args.features, args.key)
    features, scale = normalize_features(raw, args.normalize, rng, args.max_normalization_pairs)
    taus = parse_taus(args.taus)

    if args.m < 2:
        raise ValueError("--m must be at least 2")
    if args.m > features.shape[0]:
        raise ValueError("--m exceeds number of available feature vectors")
    if args.num_probes < 1:
        raise ValueError("--num-probes must be positive")

    support_idx = choose_points(features, args.m, args.select, rng)
    supports = features[support_idx]
    probe_count = min(args.num_probes, features.shape[0])
    probe_idx = choose_points(features, probe_count, args.select, rng)
    probes = features[probe_idx]
    anchors = probes

    lines: list[str] = []
    out(lines, "# Real encoder feature diagnostics")
    out(lines)
    out(lines, f"- source file: `{args.features}`")
    out(lines, f"- raw shape: `{tuple(raw.shape)}`")
    out(lines, f"- finite normalized shape: `{tuple(features.shape)}`")
    out(lines, f"- normalization: `{args.normalize}`; scale = `{scale:.6g}`")
    out(lines, f"- kernel: `{args.kernel}`")
    out(lines, f"- support selection: `{args.select}`; m = `{args.m}`")
    out(lines, f"- probes/anchors: `{probe_count}`")
    out(lines, f"- strict pairs: `{len(strict_pairs(args.m))}`")
    out(lines)
    out(lines, "This report is numerical only.  Full rank and a positive finite")
    out(lines, "`dual c` are evidence that the finite interaction-frame condition is")
    out(lines, "well conditioned for this feature cloud; they are not Lean proofs.")
    out(lines)

    out(lines, "## Softmax effective sample size")
    out(lines)
    out(lines, "| tau | ESS p10 | ESS median | ESS p90 |")
    out(lines, "|-----|---------|------------|---------|")
    for tau in taus:
        p10, med, p90 = ess_summary(features, probes, tau, args.kernel)
        out(lines, f"| {tau:g} | {p10:.2f} | {med:.2f} | {p90:.2f} |")
    out(lines)

    bare_rows: list[tuple[float, MatrixDiagnostics]] = []
    col_rows: list[tuple[float, MatrixDiagnostics]] = []
    for tau in taus:
        A_bare = interaction_matrix(supports, probes, tau, args.kernel, False, anchors)
        A_col = interaction_matrix(supports, probes, tau, args.kernel, True, anchors)
        bare_rows.append((tau, matrix_diagnostics(A_bare, rng, args.random_violation_trials)))
        col_rows.append((tau, matrix_diagnostics(A_col, rng, args.random_violation_trials)))

    emit_matrix_table(lines, "Bare interaction matrix", bare_rows)
    emit_matrix_table(lines, "Column-reweighted interaction matrix", col_rows)

    out(lines, "## Reading guide")
    out(lines)
    out(lines, "- `rank/cols = cols/cols` means the selected support-pair directions are")
    out(lines, "  distinguishable by the selected probes in floating point.")
    out(lines, "- `dual c` is the computable finite certificate obtained from a pseudoinverse")
    out(lines, "  left inverse.  Larger is better; values near machine zero mean the theorem")
    out(lines, "  may be logically true but numerically useless for this feature geometry.")
    out(lines, "- `ceiling` is the smallest column sup norm; no valid frame constant can")
    out(lines, "  exceed it, matching the Lean ceiling theorem.")
    out(lines, "- `max random violation` should be non-positive up to numerical noise when")
    out(lines, "  `dual c` is a valid certificate.")
    out(lines)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
