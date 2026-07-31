"""Positive-definite geometry kernels over fixed feature blocks (plan 6.2).

The conservative construction of the plan is

    K_geom(x, y) = sum_{s,r,b} alpha_{s,r,b} kappa_{s,b}(F_{s,r}(x),
                                                        F_{s,r}(y)),

with every ``alpha >= 0`` and every base kernel positive definite.  A
nonnegative combination of positive-definite kernels is positive definite, so
the mixture is too.  The same region index ``r`` is used on both arguments;
no soft cross-patch matching rule appears anywhere, because such a rule is
not a positive-definite kernel and must not be called one.

Base kernels
------------
``smooth_laplace``  exp(-sqrt(|u-v|^2 + eps^2) / tau).
    Positive definite in every dimension: ``t -> sqrt(t + eps^2)`` is a
    Bernstein function, so ``t -> exp(-a sqrt(t + eps^2))`` is completely
    monotone on ``[0, inf)`` and Schoenberg's theorem applies to the squared
    distance.  Unlike the paper's ``exp(-|u-v|/tau)`` it is differentiable at
    coincident points, which the kernel-gradient field needs.
``gaussian``        exp(-|u-v|^2 / (2 tau^2)).
``laplace``         exp(-|u-v| / tau); the paper's kernel, kept as the
    required raw ablation.  Its gradient is undefined at zero distance.

Bandwidths are calibrated once from TARGET-ONLY samples with a frozen
quantile (median heuristic) and never re-tuned against a candidate output or
an evaluation metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .fixed_features import Branch

BASE_KERNELS = ("smooth_laplace", "gaussian", "laplace")


def squared_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Numerically floored ``[na, nb]`` squared Euclidean distances."""
    aa = (a * a).sum(dim=1, keepdim=True)
    bb = (b * b).sum(dim=1, keepdim=True).T
    return (aa + bb - 2.0 * (a @ b.T)).clamp_min(0.0)


def base_gram(name: str, a: torch.Tensor, b: torch.Tensor, tau: float,
              eps: float) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("bandwidth must be positive")
    sq = squared_distances(a, b)
    if name == "smooth_laplace":
        if eps <= 0:
            raise ValueError("smooth_laplace needs eps > 0")
        return torch.exp(-torch.sqrt(sq + eps * eps) / tau)
    if name == "gaussian":
        return torch.exp(-sq / (2.0 * tau * tau))
    if name == "laplace":
        return torch.exp(-torch.sqrt(sq.clamp_min(1e-24)) / tau)
    raise ValueError(f"unknown base kernel {name!r}")


@dataclass
class BlockKernel:
    """A nonnegative combination of per-block positive-definite base kernels.

    ``combine="sum"`` is the plan's declared conservative form,
    ``sum_b w_b kappa_b``.  It is positive definite because a nonnegative
    combination of positive-definite kernels is.

    ``combine="product"`` is ``prod_b kappa_b^{w_b}``, i.e. a single
    exponential of the weighted sum of block distances.  It is also
    positive definite: each ``kappa_b`` here is infinitely divisible, so
    ``kappa_b^{w_b}`` is again a base kernel of the same family with
    bandwidth ``tau_b / w_b``, and a product of positive-definite kernels is
    positive definite by the Schur product theorem.  It is registered
    because the sum form averages block affinities and therefore
    *concentrates* -- the very flattening the paper reports for raw
    high-dimensional kernels.  Both are measured; neither is assumed.
    """

    base: str
    taus: torch.Tensor            # [n_blocks], > 0
    weights: torch.Tensor         # [n_blocks], >= 0, sums to 1
    eps: float
    combine: str = "sum"          # "sum" | "product"

    def __post_init__(self) -> None:
        if self.base not in BASE_KERNELS:
            raise ValueError(f"unknown base kernel {self.base!r}")
        if self.combine not in ("sum", "product"):
            raise ValueError(f"unknown combination rule {self.combine!r}")
        if len(self.taus) != len(self.weights):
            raise ValueError("taus and weights must have equal length")
        if bool((self.taus <= 0).any()):
            raise ValueError("every bandwidth must be positive")
        if bool((self.weights < 0).any()):
            raise ValueError("block weights must be nonnegative")
        total = float(self.weights.sum())
        if not total > 0:
            raise ValueError("block weights must not sum to zero")
        if abs(total - 1.0) > 1e-9:
            raise ValueError("block weights must sum to one")

    @property
    def n_blocks(self) -> int:
        return int(len(self.taus))

    def gram_from_blocks(self, left: list[torch.Tensor],
                         right: list[torch.Tensor]) -> torch.Tensor:
        if len(left) != self.n_blocks or len(right) != self.n_blocks:
            raise ValueError("block count does not match the kernel")
        total = None
        for index, (a, b) in enumerate(zip(left, right)):
            weight = self.weights[index].to(a.dtype)
            if self.combine == "sum":
                gram = base_gram(self.base, a, b, float(self.taus[index]),
                                 self.eps) * weight
                total = gram if total is None else total + gram
            else:
                # kappa_b ** w_b == same family at bandwidth tau_b / w_b.
                if float(weight) <= 0:
                    continue
                gram = base_gram(self.base, a, b,
                                 float(self.taus[index]) / float(weight),
                                 self.eps)
                total = gram if total is None else total * gram
        if total is None:
            raise ValueError("every block weight was zero")
        return total

    def gram(self, branch: Branch, left: torch.Tensor,
             right: torch.Tensor) -> torch.Tensor:
        return self.gram_from_blocks(branch.blocks(left), branch.blocks(right))

    def block_distance_stats(self, branch: Branch, left: torch.Tensor,
                             right: torch.Tensor) -> dict:
        with torch.no_grad():
            dists = []
            for a, b in zip(branch.blocks(left), branch.blocks(right)):
                dists.append(torch.sqrt(squared_distances(a, b)).flatten())
            all_d = torch.cat(dists)
            quantiles = torch.quantile(
                all_d, torch.tensor([0.05, 0.5, 0.95], dtype=all_d.dtype,
                             device=all_d.device))
        return {
            "distance_mean": float(all_d.mean()),
            "distance_q05": float(quantiles[0]),
            "distance_median": float(quantiles[1]),
            "distance_q95": float(quantiles[2]),
            "bandwidth_median": float(self.taus.median()),
        }


def _gram_from_squared(kernel: BlockKernel,
                       squared: list[torch.Tensor]) -> torch.Tensor:
    """Gram matrix from cached per-block squared distances.

    The bandwidth bisection below changes only ``tau``, so the block
    distances are computed once and reused; recomputing them per iteration
    would dominate the calibration cost.
    """
    total = None
    for index, sq in enumerate(squared):
        weight = float(kernel.weights[index])
        tau = float(kernel.taus[index])
        if kernel.combine == "sum":
            gram = _kernel_from_squared(kernel.base, sq, tau,
                                        kernel.eps) * weight
            total = gram if total is None else total + gram
        else:
            if weight <= 0:
                continue
            gram = _kernel_from_squared(kernel.base, sq, tau / weight,
                                        kernel.eps)
            total = gram if total is None else total * gram
    if total is None:
        raise ValueError("every block weight was zero")
    return total


def _kernel_from_squared(name: str, sq: torch.Tensor, tau: float,
                         eps: float) -> torch.Tensor:
    if name == "smooth_laplace":
        return torch.exp(-torch.sqrt(sq + eps * eps) / tau)
    if name == "gaussian":
        return torch.exp(-sq / (2.0 * tau * tau))
    if name == "laplace":
        return torch.exp(-torch.sqrt(sq.clamp_min(1e-24)) / tau)
    raise ValueError(f"unknown base kernel {name!r}")


def _row_ess_fraction(gram: torch.Tensor, exclude_self: bool) -> float:
    """Median row ESS as a fraction of the neighbours actually available.

    **`exclude_self` is the Phase-25 correction and it changes the number by an
    order of magnitude.**  A target-vs-target Gram has a zero-distance entry on
    every diagonal, which dominates its row.  Measured on real CIFAR at the
    bandwidth the old calibration selected:

        diagonal included   median ESS fraction = 0.0500   (the declared target)
        diagonal removed    median ESS fraction = 0.6019
        the FIELD's actual  cloud -> positives  = 0.7104

    The field weights a generated cloud point against real positives and never
    has a self-match, so the diagonal-included figure describes nothing the
    field does -- it was met by the self-term rather than by selectivity among
    distinct samples.  Every `target_ess_fraction` recorded before Phase 25 is
    mislabelled by roughly this factor of twelve; see
    `EncoderIndependentPhase25Plan.md` section 2 for what that does and does
    not invalidate.

    The denominator is the number of usable columns, so the excluded-self and
    rectangular cases are directly comparable as "fraction of the neighbours on
    offer".
    """
    if exclude_self:
        if gram.shape[0] != gram.shape[1]:
            raise ValueError(
                "exclude_self only applies to a self-paired (square) Gram")
        gram = gram - torch.diag_embed(torch.diagonal(gram))
    available = gram.shape[1] - (1 if exclude_self else 0)
    if available < 1:
        raise ValueError("no neighbours left to measure")
    # Normalize each row by its own MAXIMUM before forming weights.  Row
    # weights are scale invariant, so this is exact, and it keeps every
    # intermediate in [0, 1] with a row sum of at least 1.
    #
    # The obvious `gram / gram.sum(1).clamp_min(eps)` is subtly wrong and
    # removing the diagonal is what exposed it: when tau is small the surviving
    # entries are subnormal, the true row sum falls *below* the clamp, and the
    # clamp then rescales a legitimate row into garbage -- ESS read 1.6e28 at
    # tau = 0.00197, destroying monotonicity in tau so the bisection below
    # diverged toward zero.  The self-term always kept a row O(1), which is why
    # this defect could never fire before Phase 25.
    row_max = gram.max(dim=1, keepdim=True).values
    alive = row_max.squeeze(1) > 0
    scaled = gram / torch.where(row_max > 0, row_max,
                                torch.ones_like(row_max))
    row_sum = scaled.sum(dim=1)
    concentration = (scaled ** 2).sum(dim=1)
    # ESS = (sum w)^2 / sum w^2, both operands now well conditioned in [1, n].
    ess = torch.where(alive & (concentration > 0),
                      row_sum ** 2 / concentration.clamp_min(1e-30),
                      torch.ones_like(row_sum))
    return float(ess.median() / available)


def degenerate_row_fraction(kernel: BlockKernel, blocks: list[torch.Tensor],
                            exclude_self: bool = True) -> float:
    """Share of rows whose kernel weights have underflowed to nothing.

    The direct collapse signal.  A nonzero value means the bandwidth is below
    what the arithmetic can represent for this data, so any ESS reported for it
    is a floor artifact rather than a measurement.
    """
    with torch.no_grad():
        gram = kernel.gram_from_blocks(blocks, blocks)
        if exclude_self:
            gram = gram - torch.diag_embed(torch.diagonal(gram))
        return float((gram.sum(dim=1) <= 0).to(torch.float32).mean())


def _median_ess_from_squared(kernel: BlockKernel,
                             squared: list[torch.Tensor],
                             exclude_self: bool = True) -> float:
    return _row_ess_fraction(_gram_from_squared(kernel, squared), exclude_self)


def median_ess_fraction(kernel: BlockKernel, blocks: list[torch.Tensor],
                        exclude_self: bool = False) -> float:
    """Median row effective sample size of a Gram matrix, as a fraction.

    1.0 means the kernel weights every neighbour identically (flat, no
    geometry); the reciprocal of the available count means it has collapsed
    onto a single neighbour.  See :func:`_row_ess_fraction` for why
    ``exclude_self`` defaults to True and what changed at Phase 25.
    """
    with torch.no_grad():
        return _row_ess_fraction(kernel.gram_from_blocks(blocks, blocks),
                                 exclude_self)


def geometric_multipliers(levels: int, span: float) -> tuple[float, ...]:
    """Geometrically spaced bandwidth multipliers centred on 1.

    ``levels=5, span=4`` gives ``(0.25, 0.5, 1, 2, 4)``.  A single global
    bandwidth is only sensitive to structure at one scale, which is why the
    MMD generative literature (GMMN; MMD-GAN; KID) uses a *mixture* spanning
    orders of magnitude.  Declared, never searched.
    """
    if levels < 1:
        raise ValueError("a mixture needs at least one level")
    if span <= 0:
        raise ValueError("the span must be positive")
    if levels == 1:
        return (1.0,)
    return tuple(float(span ** (2.0 * i / (levels - 1) - 1.0))
                 for i in range(levels))


def calibrate_block_kernel(branch: Branch, target: torch.Tensor, base: str,
                           quantile: float, multiplier: float, eps: float,
                           combine: str = "sum",
                           target_ess_fraction: float | None = None,
                           ess_samples: int = 128,
                           ess_iterations: int = 24,
                           tau_multipliers: Sequence[float] | None = None,
                           exclude_self: bool = False
                           ) -> BlockKernel:
    """Frozen target-only bandwidths, one per block.

    The per-block median heuristic sets the *relative* scales, which is what
    stops a single global bandwidth from saturating some blocks and
    flattening others.  In high dimension the median heuristic on its own
    still leaves the kernel nearly flat -- the paper's reported failure mode
    -- so when ``target_ess_fraction`` is given, a single global factor is
    additionally solved for by bisection so that the median row effective
    sample size on TARGET-vs-TARGET data hits the declared fraction.

    That calibration reads target samples only.  It never sees generated
    output, an evaluation metric or a gate, so it is a declared design rule
    rather than selection on an outcome.  ESS is monotone increasing in the
    bandwidth, which is what makes the bisection well posed.

    **`exclude_self` defaults to the legacy measurement ON PURPOSE, and the
    reason is not conservatism.**  The self-paired Gram used for the solve has a
    zero-distance entry on every diagonal, so `target_ess_fraction` is met
    through that self-match rather than through selectivity among distinct
    samples: 0.05 declared, 0.60 realized among actual neighbours, 0.71 in the
    field.  The *label* is wrong by roughly 12x (see :func:`_row_ess_fraction`).

    But Phase 22 measured the performance optimum in **realized** terms, and it
    sits at 0.52-0.71 with both ends sharply worse -- so the legacy path's
    mislabelled 0.05 lands almost exactly on the good operating point, while a
    corrected solve at the same nominal 0.05 lands deep in the regime measured
    catastrophic at p = 0.0001 (KID 0.260 against 0.131).  On the checkerboard
    dataset it drives tau ~20x smaller and the field degenerate.

    So switching the default would silently degrade every caller.  The
    correction is to the *reporting*: pass ``exclude_self=True`` to measure what
    the field actually does, and choose a target near 0.5-0.7 rather than 0.05
    when calibrating that way.  See `EncoderIndependentPhase25Plan.md` section 2
    and the Phase 25 results note.
    """
    if not 0.0 < quantile <= 1.0:
        raise ValueError("bandwidth quantile must lie in (0, 1]")
    if multiplier <= 0:
        raise ValueError("bandwidth multiplier must be positive")
    with torch.no_grad():
        blocks = branch.blocks(target)
        taus = []
        for block in blocks:
            sq = squared_distances(block, block)
            n = len(block)
            off = sq[~torch.eye(n, dtype=torch.bool, device=sq.device)]
            scale = float(torch.quantile(
                torch.sqrt(off.clamp_min(0.0)), quantile))
            # A degenerate (constant) block would give tau = 0; floor it and
            # let the weight decide, rather than silently dropping the block.
            taus.append(max(scale * multiplier, 1e-6))
    tau_tensor = torch.tensor(taus, dtype=torch.float32)
    if tau_multipliers is not None:
        # Applied BEFORE the ESS solve, so the global factor is chosen for
        # the mixture as a whole rather than for one member of it.
        if len(tau_multipliers) != len(taus):
            raise ValueError(
                f"got {len(tau_multipliers)} tau multipliers for "
                f"{len(taus)} blocks")
        if any(m <= 0 for m in tau_multipliers):
            raise ValueError("every tau multiplier must be positive")
        tau_tensor = tau_tensor * torch.tensor(list(tau_multipliers),
                                               dtype=torch.float32)
    weights = torch.full((len(taus),), 1.0 / len(taus), dtype=torch.float32)
    kernel = BlockKernel(base=base, taus=tau_tensor, weights=weights, eps=eps,
                         combine=combine)
    if target_ess_fraction is None:
        return kernel
    if not 0.0 < target_ess_fraction < 1.0:
        raise ValueError("the target ESS fraction must lie in (0, 1)")

    with torch.no_grad():
        sample_blocks = branch.blocks(target[:ess_samples])
        squared = [squared_distances(block, block) for block in sample_blocks]
        low, high = 1e-4, 1e4
        for _ in range(ess_iterations):
            mid = float(np.sqrt(low * high))
            probe = BlockKernel(base, tau_tensor * mid, weights, eps, combine)
            if _median_ess_from_squared(
                    probe, squared, exclude_self) < target_ess_fraction:
                low = mid
            else:
                high = mid
    factor = float(np.sqrt(low * high))
    return BlockKernel(base, (tau_tensor * factor).clamp_min(1e-6), weights,
                       eps, combine)


def mmd2_unbiased(kernel: BlockKernel, branch: Branch, left: torch.Tensor,
                  right: torch.Tensor) -> float:
    """Unbiased MMD^2 for this kernel: the discrepancy the geometry can see.

    Used to ask whether a geometry branch is *blind* to a collision pair.  A
    kernel that cannot separate a pair is not measure-determining on that
    pair -- which is why the geometry branch is never the correctness
    authority.
    """
    with torch.no_grad():
        n, m = len(left), len(right)
        if n < 2 or m < 2:
            raise ValueError("unbiased MMD needs >= 2 samples per side")
        left_blocks = branch.blocks(left)
        right_blocks = branch.blocks(right)
        kxx = kernel.gram_from_blocks(left_blocks, left_blocks)
        kyy = kernel.gram_from_blocks(right_blocks, right_blocks)
        kxy = kernel.gram_from_blocks(left_blocks, right_blocks)
        xx = (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
        yy = (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
        return float(xx + yy - 2.0 * kxy.mean())


def min_eigenvalue(kernel: BlockKernel, branch: Branch,
                   samples: torch.Tensor) -> float:
    """Smallest eigenvalue of an empirical Gram matrix (numeric PD check)."""
    with torch.no_grad():
        gram = kernel.gram(branch, samples, samples).double()
        gram = 0.5 * (gram + gram.T)
        return float(torch.linalg.eigvalsh(gram).min())
