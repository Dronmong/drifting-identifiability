"""Audited finite-batch conservative finishers for one-dimensional QLD.

The functions in this module deliberately separate the new fields from the
repository's exact paper Algorithm-2 implementation.  They support two
negative-reference conventions:

``crossfit``
    The caller supplies an independently generated, detached negative cloud.

``reused_deleted``
    The query cloud is reused as the negative cloud and its diagonal terms are
    removed exactly (not approximated by a large masking penalty).

Every vectorized field has a slow loop implementation used by
``invariant_tests``.  Denominator flooring is explicit and reported; it is a
numerical safeguard, not part of the mathematical definition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import numpy as np


ReferenceMode = Literal["crossfit", "reused_deleted"]
FieldKind = Literal["mean", "sharp", "kgrad"]


@dataclass(frozen=True)
class FieldDiagnostics:
    """Numerical and work diagnostics for a single field evaluation."""

    kind: str
    reference_mode: str
    positive_denominator_min: float
    negative_denominator_min: float
    denominator_floor: float
    positive_floor_activations: int
    negative_floor_activations: int
    positive_terms_per_anchor: int
    negative_terms_per_anchor: int
    kernel_pairs: int

    @property
    def floor_activations(self) -> int:
        return self.positive_floor_activations + \
            self.negative_floor_activations


@dataclass(frozen=True)
class FieldResult:
    """A field together with the quantities needed for trajectory audits."""

    field: np.ndarray
    positive_score: np.ndarray
    negative_score: np.ndarray
    diagnostics: FieldDiagnostics


def _points(values: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 2 or len(out) == 0 or out.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty (N, d) array")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains a non-finite value")
    return out


def _validate_tau(tau: float) -> float:
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    return tau


def _validate_scale(scale: float) -> float:
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("kernel_scale must be finite and positive")
    return scale


def pairwise_displacements(queries: np.ndarray,
                           references: np.ndarray) -> tuple[np.ndarray,
                                                                   np.ndarray]:
    """Return ``references - queries`` and their Euclidean norms."""
    q = _points(queries, "queries")
    y = _points(references, "references")
    if q.shape[1] != y.shape[1]:
        raise ValueError("query and reference dimensions differ")
    displacement = y[None, :, :] - q[:, None, :]
    radius = np.linalg.norm(displacement, axis=2)
    return displacement, radius


def laplace_kernel_from_radius(radius: np.ndarray, tau: float, *,
                               kernel_scale: float = 1.0) -> np.ndarray:
    """Evaluate ``scale * exp(-radius / tau)``."""
    tau = _validate_tau(tau)
    scale = _validate_scale(kernel_scale)
    radius = np.asarray(radius, dtype=float)
    if np.any(radius < 0.0) or not np.all(np.isfinite(radius)):
        raise ValueError("radius must be finite and nonnegative")
    return scale * np.exp(-radius / tau)


def sharp_laplace_kernel_from_radius(radius: np.ndarray, tau: float, *,
                                     kernel_scale: float = 1.0) -> np.ndarray:
    """Evaluate the sharp companion ``scale*tau*(r+tau)*exp(-r/tau)``."""
    tau = _validate_tau(tau)
    scale = _validate_scale(kernel_scale)
    radius = np.asarray(radius, dtype=float)
    if np.any(radius < 0.0) or not np.all(np.isfinite(radius)):
        raise ValueError("radius must be finite and nonnegative")
    return scale * tau * (radius + tau) * np.exp(-radius / tau)


def sharp_laplace_kernel(queries: np.ndarray, references: np.ndarray,
                         tau: float, *, kernel_scale: float = 1.0) \
        -> np.ndarray:
    """Pairwise sharp Laplace companion-kernel matrix."""
    _, radius = pairwise_displacements(queries, references)
    return sharp_laplace_kernel_from_radius(
        radius, tau, kernel_scale=kernel_scale)


def log_sharp_laplace_kernel_from_radius(
        radius: np.ndarray, tau: float, *, kernel_scale: float = 1.0) \
        -> np.ndarray:
    """Stable logarithm of the sharp companion kernel."""
    tau = _validate_tau(tau)
    scale = _validate_scale(kernel_scale)
    radius = np.asarray(radius, dtype=float)
    if np.any(radius < 0.0) or not np.all(np.isfinite(radius)):
        raise ValueError("radius must be finite and nonnegative")
    return (math.log(scale) + math.log(tau) + np.log(radius + tau) -
            radius / tau)


def _mask_for_reference(n_query: int, n_reference: int,
                        deleted_diagonal: bool) -> np.ndarray:
    mask = np.ones((n_query, n_reference), dtype=bool)
    if deleted_diagonal:
        if n_query != n_reference:
            raise ValueError(
                "exact diagonal deletion requires equally sized clouds")
        if n_reference < 2:
            raise ValueError("diagonal deletion requires at least two points")
        np.fill_diagonal(mask, False)
    return mask


def _resolve_negative(queries: np.ndarray, negatives: np.ndarray | None,
                      reference_mode: ReferenceMode) \
        -> tuple[np.ndarray, bool]:
    if reference_mode == "reused_deleted":
        if negatives is not None:
            supplied = _points(negatives, "negatives")
            if supplied.shape != queries.shape or not np.array_equal(
                    supplied, queries):
                raise ValueError(
                    "reused_deleted negatives must be omitted or exactly "
                    "equal to queries")
        return queries, True
    if reference_mode == "crossfit":
        if negatives is None:
            raise ValueError("crossfit requires an explicit negative cloud")
        negative = _points(negatives, "negatives")
        if negative.shape[1] != queries.shape[1]:
            raise ValueError("query and negative dimensions differ")
        return negative, False
    raise ValueError(f"unknown reference mode: {reference_mode}")


def _safe_row_denominator(values: np.ndarray, mask: np.ndarray,
                          denominator_floor: float) \
        -> tuple[np.ndarray, float, int, int]:
    if denominator_floor <= 0.0 or not math.isfinite(denominator_floor):
        raise ValueError("denominator_floor must be finite and positive")
    counts = mask.sum(axis=1)
    if np.any(counts <= 0):
        raise ValueError("an anchor has no retained reference terms")
    raw = np.where(mask, values, 0.0).sum(axis=1) / counts
    activations = int(np.count_nonzero(raw < denominator_floor))
    safe = np.maximum(raw, denominator_floor)
    return safe, float(np.min(raw)), activations, int(counts[0])


def _score_vectorized(kind: FieldKind, queries: np.ndarray,
                      references: np.ndarray, tau: float, *,
                      deleted_diagonal: bool, denominator_floor: float,
                      kernel_scale: float) \
        -> tuple[np.ndarray, float, int, int, int]:
    displacement, radius = pairwise_displacements(queries, references)
    mask = _mask_for_reference(len(queries), len(references),
                               deleted_diagonal)
    counts = mask.sum(axis=1)
    laplace = laplace_kernel_from_radius(
        radius, tau, kernel_scale=kernel_scale)

    if kind == "mean":
        denominator_values = laplace
        numerator_values = laplace[:, :, None] * displacement
    elif kind == "sharp":
        denominator_values = sharp_laplace_kernel_from_radius(
            radius, tau, kernel_scale=kernel_scale)
        # Scaling the companion kernel also scales its gradient.
        numerator_values = laplace[:, :, None] * displacement
    elif kind == "kgrad":
        denominator_values = laplace
        unit = np.divide(
            displacement, radius[:, :, None],
            out=np.zeros_like(displacement),
            where=radius[:, :, None] > 0.0)
        numerator_values = laplace[:, :, None] * unit / tau
    else:
        raise ValueError(f"unknown field kind: {kind}")

    denominator, minimum, floor_count, terms = _safe_row_denominator(
        denominator_values, mask, denominator_floor)
    numerator = np.where(mask[:, :, None], numerator_values, 0.0).sum(
        axis=1) / counts[:, None]
    score = numerator / denominator[:, None]
    if not np.all(np.isfinite(score)):
        raise FloatingPointError("non-finite conservative score")
    return score, minimum, floor_count, terms, int(mask.sum())


def conservative_field(
        kind: FieldKind, queries: np.ndarray, positives: np.ndarray,
        negatives: np.ndarray | None = None, *, tau: float,
        reference_mode: ReferenceMode,
        denominator_floor: float = 1e-300,
        kernel_scale: float = 1.0) -> FieldResult:
    """Evaluate one of the three audited positive-minus-negative fields."""
    q = _points(queries, "queries")
    p = _points(positives, "positives")
    if p.shape[1] != q.shape[1]:
        raise ValueError("query and positive dimensions differ")
    tau = _validate_tau(tau)
    scale = _validate_scale(kernel_scale)
    negative, delete_negative = _resolve_negative(
        q, negatives, reference_mode)

    positive_score, p_min, p_floor, p_terms, p_pairs = _score_vectorized(
        kind, q, p, tau, deleted_diagonal=False,
        denominator_floor=denominator_floor, kernel_scale=scale)
    negative_score, n_min, n_floor, n_terms, n_pairs = _score_vectorized(
        kind, q, negative, tau, deleted_diagonal=delete_negative,
        denominator_floor=denominator_floor, kernel_scale=scale)
    field = positive_score - negative_score
    return FieldResult(
        field=field,
        positive_score=positive_score,
        negative_score=negative_score,
        diagnostics=FieldDiagnostics(
            kind=kind,
            reference_mode=reference_mode,
            positive_denominator_min=p_min,
            negative_denominator_min=n_min,
            denominator_floor=float(denominator_floor),
            positive_floor_activations=p_floor,
            negative_floor_activations=n_floor,
            positive_terms_per_anchor=p_terms,
            negative_terms_per_anchor=n_terms,
            kernel_pairs=p_pairs + n_pairs))


def corrected_mean_shift_field(
        queries: np.ndarray, positives: np.ndarray,
        negatives: np.ndarray | None = None, *, tau: float,
        reference_mode: ReferenceMode,
        denominator_floor: float = 1e-300) -> FieldResult:
    return conservative_field(
        "mean", queries, positives, negatives, tau=tau,
        reference_mode=reference_mode, denominator_floor=denominator_floor)


def sharp_laplace_field(
        queries: np.ndarray, positives: np.ndarray,
        negatives: np.ndarray | None = None, *, tau: float,
        reference_mode: ReferenceMode,
        denominator_floor: float = 1e-300,
        kernel_scale: float = 1.0) -> FieldResult:
    return conservative_field(
        "sharp", queries, positives, negatives, tau=tau,
        reference_mode=reference_mode, denominator_floor=denominator_floor,
        kernel_scale=kernel_scale)


def kernel_gradient_field(
        queries: np.ndarray, positives: np.ndarray,
        negatives: np.ndarray | None = None, *, tau: float,
        reference_mode: ReferenceMode,
        denominator_floor: float = 1e-300) -> FieldResult:
    return conservative_field(
        "kgrad", queries, positives, negatives, tau=tau,
        reference_mode=reference_mode, denominator_floor=denominator_floor)


def _logmeanexp_masked(log_values: np.ndarray, mask: np.ndarray) \
        -> np.ndarray:
    counts = mask.sum(axis=1)
    if np.any(counts <= 0):
        raise ValueError("an anchor has no retained reference terms")
    retained = np.where(mask, log_values, -np.inf)
    maximum = np.max(retained, axis=1)
    return maximum + np.log(np.exp(retained - maximum[:, None]).sum(
        axis=1)) - np.log(counts)


def sharp_logkde_loss(
        queries: np.ndarray, positives: np.ndarray,
        negatives: np.ndarray | None = None, *, tau: float,
        reference_mode: ReferenceMode,
        kernel_scale: float = 1.0) -> float:
    """Return ``mean(log Zsharp_neg - log Zsharp_pos)`` stably."""
    q = _points(queries, "queries")
    p = _points(positives, "positives")
    if p.shape[1] != q.shape[1]:
        raise ValueError("query and positive dimensions differ")
    negative, delete_negative = _resolve_negative(
        q, negatives, reference_mode)
    _, positive_radius = pairwise_displacements(q, p)
    _, negative_radius = pairwise_displacements(q, negative)
    positive_mask = _mask_for_reference(len(q), len(p), False)
    negative_mask = _mask_for_reference(
        len(q), len(negative), delete_negative)
    positive_log = log_sharp_laplace_kernel_from_radius(
        positive_radius, tau, kernel_scale=kernel_scale)
    negative_log = log_sharp_laplace_kernel_from_radius(
        negative_radius, tau, kernel_scale=kernel_scale)
    result = float(np.mean(
        _logmeanexp_masked(negative_log, negative_mask) -
        _logmeanexp_masked(positive_log, positive_mask)))
    if not math.isfinite(result):
        raise FloatingPointError("non-finite sharp log-KDE loss")
    return result


def _slow_score(kind: FieldKind, query: np.ndarray,
                references: np.ndarray, tau: float, *,
                deleted_index: int | None, denominator_floor: float,
                kernel_scale: float) -> tuple[np.ndarray, float, bool, int]:
    numerator = np.zeros_like(query, dtype=float)
    denominator = 0.0
    terms = 0
    for j, reference in enumerate(references):
        if deleted_index is not None and j == deleted_index:
            continue
        displacement = reference - query
        radius = float(np.linalg.norm(displacement))
        kernel = kernel_scale * math.exp(-radius / tau)
        if kind == "mean":
            denominator += kernel
            numerator += kernel * displacement
        elif kind == "sharp":
            denominator += (kernel_scale * tau * (radius + tau) *
                            math.exp(-radius / tau))
            numerator += kernel * displacement
        elif kind == "kgrad":
            denominator += kernel
            if radius > 0.0:
                numerator += kernel * displacement / (tau * radius)
        else:
            raise ValueError(kind)
        terms += 1
    if terms == 0:
        raise ValueError("an anchor has no retained reference terms")
    denominator /= terms
    numerator /= terms
    floored = denominator < denominator_floor
    return (numerator / max(denominator, denominator_floor), denominator,
            floored, terms)


def conservative_field_slow(
        kind: FieldKind, queries: np.ndarray, positives: np.ndarray,
        negatives: np.ndarray | None = None, *, tau: float,
        reference_mode: ReferenceMode,
        denominator_floor: float = 1e-300,
        kernel_scale: float = 1.0) -> FieldResult:
    """Deliberately slow reference implementation for invariant checking."""
    q = _points(queries, "queries")
    p = _points(positives, "positives")
    if q.shape[1] != p.shape[1]:
        raise ValueError("query and positive dimensions differ")
    tau = _validate_tau(tau)
    scale = _validate_scale(kernel_scale)
    negative, delete_negative = _resolve_negative(
        q, negatives, reference_mode)

    positive_scores: list[np.ndarray] = []
    negative_scores: list[np.ndarray] = []
    positive_denominators: list[float] = []
    negative_denominators: list[float] = []
    p_floor = 0
    n_floor = 0
    p_terms = 0
    n_terms = 0
    for i, query in enumerate(q):
        ps, pd, pf, p_terms = _slow_score(
            kind, query, p, tau, deleted_index=None,
            denominator_floor=denominator_floor, kernel_scale=scale)
        ns, nd, nf, n_terms = _slow_score(
            kind, query, negative, tau,
            deleted_index=i if delete_negative else None,
            denominator_floor=denominator_floor, kernel_scale=scale)
        positive_scores.append(ps)
        negative_scores.append(ns)
        positive_denominators.append(pd)
        negative_denominators.append(nd)
        p_floor += int(pf)
        n_floor += int(nf)
    positive_score = np.asarray(positive_scores)
    negative_score = np.asarray(negative_scores)
    return FieldResult(
        field=positive_score - negative_score,
        positive_score=positive_score,
        negative_score=negative_score,
        diagnostics=FieldDiagnostics(
            kind=kind,
            reference_mode=reference_mode,
            positive_denominator_min=min(positive_denominators),
            negative_denominator_min=min(negative_denominators),
            denominator_floor=float(denominator_floor),
            positive_floor_activations=p_floor,
            negative_floor_activations=n_floor,
            positive_terms_per_anchor=p_terms,
            negative_terms_per_anchor=n_terms,
            kernel_pairs=len(q) * (p_terms + n_terms)))


def _central_difference(function: Callable[[np.ndarray], float],
                        points: np.ndarray, epsilon: float = 1e-6) \
        -> np.ndarray:
    gradient = np.zeros_like(points, dtype=float)
    for index in np.ndindex(points.shape):
        plus = points.copy()
        minus = points.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (function(plus) - function(minus)) / (2 * epsilon)
    return gradient


def invariant_tests(log: Callable[[str], None] = print) -> dict[str, object]:
    """Run the fast S1 algebraic and numerical invariant suite."""
    rng = np.random.default_rng(20260721)
    q = rng.normal(size=(7, 1))
    positive = rng.normal(loc=0.4, scale=1.3, size=(9, 1))
    negative = rng.normal(loc=-0.2, scale=0.8, size=(8, 1))
    tau = 0.7
    checks: list[str] = []

    for kind in ("mean", "sharp", "kgrad"):
        for mode in ("crossfit", "reused_deleted"):
            n = negative if mode == "crossfit" else None
            fast = conservative_field(
                kind, q, positive, n, tau=tau, reference_mode=mode)
            slow = conservative_field_slow(
                kind, q, positive, n, tau=tau, reference_mode=mode)
            np.testing.assert_allclose(
                fast.field, slow.field, rtol=2e-13, atol=2e-13)
            if fast.diagnostics.floor_activations != 0:
                raise AssertionError("ordinary batch unexpectedly hit a floor")
            checks.append(f"vectorized_equals_slow:{kind}:{mode}")

        swapped = conservative_field(
            kind, q, negative, positive, tau=tau,
            reference_mode="crossfit").field
        original = conservative_field(
            kind, q, positive, negative, tau=tau,
            reference_mode="crossfit").field
        np.testing.assert_allclose(swapped, -original, rtol=1e-13, atol=1e-13)
        checks.append(f"swap_negates:{kind}")

        shift = 17.25
        translated = conservative_field(
            kind, q + shift, positive + shift, negative + shift, tau=tau,
            reference_mode="crossfit").field
        np.testing.assert_allclose(translated, original,
                                   rtol=2e-13, atol=2e-13)
        checks.append(f"translation_invariant:{kind}")

        identical = positive.copy()
        zero = conservative_field(
            kind, q, identical, identical.copy(), tau=tau,
            reference_mode="crossfit").field
        np.testing.assert_allclose(zero, 0.0, rtol=0.0, atol=0.0)
        checks.append(f"identical_cloud_zero:{kind}")

    sharp_a = sharp_laplace_field(
        q, positive, negative, tau=tau, reference_mode="crossfit",
        kernel_scale=1.0).field
    sharp_b = sharp_laplace_field(
        q, positive, negative, tau=tau, reference_mode="crossfit",
        kernel_scale=13.0).field
    np.testing.assert_allclose(sharp_a, sharp_b, rtol=2e-13, atol=2e-13)
    checks.append("sharp_positive_scale_invariant")

    deleted = sharp_laplace_field(
        q, positive, tau=tau, reference_mode="reused_deleted")
    if deleted.diagnostics.negative_terms_per_anchor != len(q) - 1:
        raise AssertionError("reused/deleted retained the wrong term count")
    checks.append("exact_diagonal_deletion")

    # The derivative of h with respect to x is k(x,y) * (y-x).
    x0 = np.asarray([[0.37]])
    y0 = np.asarray([[-1.13]])
    eps = 1e-6
    h_plus = sharp_laplace_kernel(x0 + eps, y0, tau)[0, 0]
    h_minus = sharp_laplace_kernel(x0 - eps, y0, tau)[0, 0]
    numerical_h_derivative = (h_plus - h_minus) / (2 * eps)
    expected_h_derivative = math.exp(-abs(float(x0[0, 0] - y0[0, 0])) /
                                     tau) * float(y0[0, 0] - x0[0, 0])
    np.testing.assert_allclose(numerical_h_derivative,
                               expected_h_derivative, rtol=2e-9, atol=2e-9)
    checks.append("sharp_kernel_derivative")

    # For L=mean(log Zneg-log Zpos), grad_x L = -V_sharp / n.
    loss_gradient = _central_difference(
        lambda x: sharp_logkde_loss(
            x, positive, negative, tau=tau,
            reference_mode="crossfit"), q)
    np.testing.assert_allclose(
        -len(q) * loss_gradient, sharp_a, rtol=2e-7, atol=2e-7)
    checks.append("sharp_field_equals_loss_descent")

    copied = sharp_laplace_field(
        np.ascontiguousarray(q), np.ascontiguousarray(positive),
        np.ascontiguousarray(negative), tau=tau,
        reference_mode="crossfit").field
    noncontiguous = sharp_laplace_field(
        np.asfortranarray(q.copy()), np.asfortranarray(positive.copy()),
        np.asfortranarray(negative.copy()), tau=tau,
        reference_mode="crossfit").field
    np.testing.assert_array_equal(copied, noncontiguous)
    checks.append("copy_and_layout_invariant")

    extreme = sharp_laplace_field(
        np.asarray([[0.0], [1.0]]),
        np.asarray([[1e6], [1e6 + 1.0]]),
        np.asarray([[-1e6], [-1e6 - 1.0]]), tau=0.2,
        reference_mode="crossfit", denominator_floor=1e-100)
    if extreme.diagnostics.floor_activations == 0:
        raise AssertionError("extreme safeguard test did not report floors")
    checks.append("floor_activation_reported")

    report: dict[str, object] = {
        "status": "pass",
        "check_count": len(checks),
        "checks": checks,
    }
    log(f"conservative finisher invariants: PASS ({len(checks)} checks)")
    return report


if __name__ == "__main__":
    invariant_tests()
