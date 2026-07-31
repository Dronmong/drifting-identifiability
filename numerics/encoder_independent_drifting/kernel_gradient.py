"""Movement: standard displacement vs kernel-gradient drift (plan 6.3).

The paper's normalized displacement field is

    V_std(x) = E_p[k(x,Y)(Y-x)] / E_p[k(x,Y)]
             - E_q[k(x,Y)(Y-x)] / E_q[k(x,Y)],

which always moves along raw differences ``Y - x`` no matter what geometry
the kernel measures.  The kernel-gradient field is

    V_KG(x) = E_p[grad_x k(x,Y)] / E_p[k(x,Y)]
            - E_q[grad_x k(x,Y)] / E_q[k(x,Y)]
            = grad_x log Z_p(x) - grad_x log Z_q(x),

so the declared geometry controls the update *direction* as well as the
interaction strength.  Both live behind one interface and the standard field
is a required ablation, never silently dropped.

Sign convention matches the repository's ``lowdim_drift.drift_snis``:
``positive`` are target samples and ``negative`` are the current generated
cloud, so the field points towards the target and away from the cloud.

Noise is measured by cross-fitting rather than asserted: the positive and
negative batches are split into disjoint halves, the field is evaluated twice
on the *same* probes, and the drift signal-to-noise ratio uses the
half-difference.  This is the utility statistic the adaptive controller
consumes, and it is why the controller batch must be disjoint from the field
batch.
"""

from __future__ import annotations

import torch

from .fixed_features import Branch
from .kernels import BlockKernel

DIRECTION_MODES = ("standard", "paper", "kernel_gradient",
                   "projected_kernel_gradient")
NORMALIZATIONS = ("none", "rms")
NUMERICAL_ZERO = 1e-12


def data_span_basis(probes: torch.Tensor, others: torch.Tensor,
                    tolerance: float = 1e-6) -> tuple[
                        torch.Tensor, torch.Tensor]:
    """Factorize ``span{Y_j - x_i}`` once, for reuse across branches.

    Returns the shared orthonormal basis and the per-probe completion
    direction.  Every geometry branch in an arm sees the same probes and the
    same anchors, so this factorization -- the expensive part of the
    projection -- is computed once per step instead of once per branch
    (reform R9, from the measured 10 ms/call cost of the SVD).
    """
    flat_probes = probes.reshape(len(probes), -1)
    flat_others = others.reshape(len(others), -1)
    centre = flat_others.mean(dim=0, keepdim=True)
    centred = (flat_others - centre).to(torch.float32)
    # Centring drops the rank to at most M-1, so a reduced QR would return
    # one spurious column spanning nothing; which column that is depends on
    # the row ordering, which would make the projector depend on the batch
    # permutation.  An SVD with an explicit rank cutoff is ordering
    # independent and rank correct.
    _, singular, right = torch.linalg.svd(centred, full_matrices=False)
    scale = float(centred.norm())
    if len(singular) and float(singular[0]) > 0:
        rank = int((singular > tolerance * float(singular[0])).sum())
    else:
        rank = 0
    basis = right[:rank].T                       # [d, rank], orthonormal
    if rank == 0:
        basis = flat_probes.new_zeros((flat_probes.shape[1], 0))
    extra = centre - flat_probes
    if rank:
        extra = extra - (extra @ basis) @ basis.T
    norms = extra.norm(dim=1, keepdim=True)
    unit = extra / norms.clamp_min(NUMERICAL_ZERO)
    unit = torch.where(norms > tolerance * max(scale, 1.0),
                       unit, torch.zeros_like(unit))
    return basis, unit


def apply_data_span_projection(vectors: torch.Tensor, basis: torch.Tensor,
                               unit: torch.Tensor) -> torch.Tensor:
    """Project with a basis produced by :func:`data_span_basis`."""
    flat = vectors.reshape(len(vectors), -1)
    projected = ((flat @ basis) @ basis.T if basis.shape[1]
                 else torch.zeros_like(flat))
    projected = projected + (flat * unit).sum(dim=1, keepdim=True) * unit
    return projected.reshape(vectors.shape).to(vectors.dtype)


def data_span_projection(probes: torch.Tensor, others: torch.Tensor,
                         vectors: torch.Tensor,
                         tolerance: float = 1e-6) -> torch.Tensor:
    """Project each row of ``vectors`` onto ``span{Y_j - x_i : j}``.

    Reform R3b.  The standard displacement rule is confined to this subspace
    by construction -- it can only move toward convex combinations of real
    samples -- while the raw kernel gradient is free to move anywhere, which
    is how it exploits a non-injective feature map's null directions
    (Phase-1 diagnosis section 3).  Projecting keeps the structured kernel
    deciding *which* neighbours matter and *how much*, and removes the
    off-manifold freedom.

    Computed in one shared factorization rather than per probe.  Since

        Y_j - x_i = (Y_j - Ybar) + (Ybar - x_i),

    the span is ``rowspace(Y - Ybar)`` plus the single probe-dependent
    direction ``Ybar - x_i``.  The first part is common to all probes, so one
    QR per call suffices and only a rank-one correction is per-probe.
    """
    basis, unit = data_span_basis(probes, others, tolerance)
    return apply_data_span_projection(vectors, basis, unit)


def _weights(gram: torch.Tensor, floor: float) -> tuple[
        torch.Tensor, torch.Tensor]:
    """Row-normalized affinities and the RAW (unfloored) denominators.

    The floor protects the division; the reported denominator is the true
    one, so that a collapsed kernel is visible in the diagnostics instead of
    being papered over by the floor.
    """
    denominator = gram.sum(dim=1, keepdim=True)
    return gram / denominator.clamp_min(floor), denominator.squeeze(1)


def _standard_side(probes: torch.Tensor, others: torch.Tensor,
                   gram: torch.Tensor, floor: float) -> tuple[
                       torch.Tensor, torch.Tensor]:
    weights, denominator = _weights(gram, floor)
    flat_others = others.reshape(len(others), -1)
    barycentre = weights @ flat_others
    return barycentre.reshape(probes.shape) - probes, denominator


def _paper_side(probes: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor, gram_p: torch.Tensor,
                gram_n: torch.Tensor, floor: float) -> torch.Tensor:
    """The paper's Algorithm-2 bi-softmax drift, from a Gram matrix.

    The plan's section 6.3 writes "the paper's standard normalized
    displacement field" as the row-normalized SNIS mean shift.  That is a
    *simplification*: the paper's Algorithm 2 normalizes the affinity matrix
    along **both** axes and weights each side by the other's total mass.
    This repository already contains a verbatim port of the real thing
    (``lowdim_drift.drift_paper`` / ``driftlab.compute_v_paper``) and
    explicitly labels the SNIS field "DIAGNOSTIC ONLY", with a Lean
    development around the column reweighting
    (``ColumnReweightedMeanShift.lean``) and a numerics experiment (E2)
    measuring its scale.

    The column normalization is what stops a target point that happens to be
    the nearest neighbour of many particles from dominating: it divides that
    point's influence by the total attention it receives.  Measured, this
    damps the field in dense regions relative to sparse ones by ~1.8x, which
    is precisely an anti-density-seeking mechanism.

    Working from the Gram matrix rather than from distances generalizes
    Algorithm 2 to any positive-definite block kernel, since
    ``softmax(logits) = k / sum k`` for ``k = exp(logits)``.
    """
    gram = torch.cat([gram_p, gram_n], dim=1)
    row = gram / gram.sum(dim=1, keepdim=True).clamp_min(floor)
    column = gram / gram.sum(dim=0, keepdim=True).clamp_min(floor)
    affinity = torch.sqrt((row * column).clamp_min(0.0))
    n_positive = gram_p.shape[1]
    a_positive = affinity[:, :n_positive]
    a_negative = affinity[:, n_positive:]
    weight_positive = a_positive * a_negative.sum(dim=1, keepdim=True)
    weight_negative = a_negative * a_positive.sum(dim=1, keepdim=True)
    drift = (weight_positive @ positive.reshape(len(positive), -1)
             - weight_negative @ negative.reshape(len(negative), -1))
    return drift.reshape(probes.shape)


def _log_partition(probes: torch.Tensor, others: torch.Tensor,
                   branch: Branch, kernel: BlockKernel,
                   floor: float, mask: torch.Tensor | None) -> tuple[
                       torch.Tensor, torch.Tensor]:
    gram = kernel.gram_from_blocks(branch.blocks(probes),
                                   branch.blocks(others))
    if mask is not None:
        gram = gram * mask
    denominator = gram.sum(dim=1) / len(others)
    return torch.log(denominator.clamp_min(floor)).sum(), denominator


def field(
    generated: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    feature_branch: Branch,
    kernel: BlockKernel,
    direction_mode: str = "kernel_gradient",
    normalization: str = "rms",
    denominator_floor: float = 1e-30,
    self_mask: bool = False,
    diagnostics: bool = True,
    projection: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Drift on ``generated``, plus plan section 10.1 kernel health.

    Returns a detached drift with the shape of ``generated``.  The drift is a
    stop-gradient regression target, so no generator graph is retained.
    """
    if direction_mode not in DIRECTION_MODES:
        raise ValueError(f"unknown direction mode {direction_mode!r}")
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"unknown normalization {normalization!r}")
    if denominator_floor <= 0:
        raise ValueError("the denominator floor must be positive and explicit")
    if self_mask and len(negative) != len(generated):
        raise ValueError("self masking needs the generated reuse pattern")

    probes = generated.detach()
    mask = None
    if self_mask:
        mask = 1.0 - torch.eye(len(generated), dtype=probes.dtype,
                                    device=probes.device)

    stats: dict = {}
    if direction_mode in ("standard", "paper"):
        with torch.no_grad():
            blocks = feature_branch.blocks(probes)
            gram_p = kernel.gram_from_blocks(
                blocks, feature_branch.blocks(positive))
            gram_n = kernel.gram_from_blocks(
                blocks, feature_branch.blocks(negative))
            if mask is not None:
                gram_n = gram_n * mask
            den_p = gram_p.sum(dim=1)
            den_n = gram_n.sum(dim=1)
            if direction_mode == "paper":
                drift = _paper_side(probes, positive, negative, gram_p,
                                    gram_n, denominator_floor)
            else:
                drift_p, den_p = _standard_side(
                    probes, positive, gram_p, denominator_floor)
                drift_n, den_n = _standard_side(
                    probes, negative, gram_n, denominator_floor)
                drift = drift_p - drift_n
    else:
        probes = probes.clone().requires_grad_(True)
        log_p, den_p = _log_partition(
            probes, positive, feature_branch, kernel, denominator_floor, None)
        log_n, den_n = _log_partition(
            probes, negative, feature_branch, kernel, denominator_floor, mask)
        grad_p, = torch.autograd.grad(log_p, probes, retain_graph=True)
        grad_n, = torch.autograd.grad(log_n, probes)
        drift = (grad_p - grad_n).detach()
        probes = probes.detach()
        if direction_mode == "projected_kernel_gradient":
            with torch.no_grad():
                if projection is None:
                    anchors = torch.cat([positive, negative], dim=0)
                    projection = data_span_basis(probes, anchors)
                unprojected = drift
                drift = apply_data_span_projection(drift, *projection)
                retained = float(
                    drift.norm() / unprojected.norm().clamp_min(
                        NUMERICAL_ZERO))
        with torch.no_grad():
            gram_p = kernel.gram_from_blocks(
                feature_branch.blocks(probes),
                feature_branch.blocks(positive))
        den_p, den_n = den_p.detach(), den_n.detach()

    raw_rms = float(torch.sqrt(
        (drift.reshape(len(drift), -1) ** 2).sum(dim=1).mean()))
    if normalization == "rms":
        drift = drift / max(raw_rms, NUMERICAL_ZERO)

    if diagnostics:
        stats = kernel_health(gram_p, den_p, den_n, drift, raw_rms,
                              denominator_floor)
        stats.update(kernel.block_distance_stats(
            feature_branch, generated.detach(), positive))
        if direction_mode == "projected_kernel_gradient":
            # Fraction of the raw kernel-gradient norm that survives the
            # data-span projection.  A small value means most of the field
            # was pointing off the data manifold.
            stats["projection_retained_fraction"] = retained
    return drift.detach(), stats


def kernel_health(gram: torch.Tensor, denominator_p: torch.Tensor,
                  denominator_n: torch.Tensor, drift: torch.Tensor,
                  raw_rms: float, denominator_floor: float) -> dict:
    """Plan section 10.1 per-branch kernel health."""
    with torch.no_grad():
        affinity = gram.flatten()
        quantiles = torch.quantile(
            affinity, torch.tensor([0.05, 0.5, 0.95], dtype=affinity.dtype,
                                 device=affinity.device))
        row_sum = gram.sum(dim=1, keepdim=True)
        # A row whose affinities have all underflowed carries no weights at
        # all.  Normalizing it against the floor would yield w = 0 and an
        # "effective sample size" of 1/0 -- a huge number reported as
        # excellent health, in precisely the collapsed regime the plan wants
        # flagged.  Such rows are excluded and counted instead.
        alive = (row_sum.squeeze(1) > NUMERICAL_ZERO)
        n_alive = int(alive.sum())
        weights = gram / row_sum.clamp_min(NUMERICAL_ZERO)
        if n_alive:
            live = weights[alive]
            ess = 1.0 / (live ** 2).sum(dim=1).clamp_min(NUMERICAL_ZERO)
            entropy = -(live * torch.log(live.clamp_min(1e-30))).sum(dim=1)
            ess_mean = float(ess.mean())
            entropy_mean = float(entropy.mean())
        else:
            ess_mean = float("nan")
            entropy_mean = float("nan")
        row_max = gram.max(dim=1, keepdim=True).values.clamp_min(NUMERICAL_ZERO)
        norms = drift.reshape(len(drift), -1).norm(dim=1)
    return {
        "collapsed_row_fraction": 1.0 - n_alive / max(len(gram), 1),
        "affinity_mean": float(affinity.mean()),
        "affinity_q05": float(quantiles[0]),
        "affinity_median": float(quantiles[1]),
        "affinity_q95": float(quantiles[2]),
        "affinity_zero_fraction": float((affinity < NUMERICAL_ZERO).double()
                                        .mean()),
        "affinity_saturation_fraction": float(
            ((gram / row_max) > 0.99).double().mean()),
        "ess_mean": ess_mean,
        "ess_fraction": ess_mean / gram.shape[1],
        "entropy_mean": entropy_mean,
        "entropy_fraction": entropy_mean / float(
            torch.log(torch.tensor(float(gram.shape[1])))),
        "denominator_min": float(min(denominator_p.min(),
                                     denominator_n.min())),
        "denominator_max": float(max(denominator_p.max(),
                                     denominator_n.max())),
        # Nonzero means the kernel has collapsed on some probe and the floor
        # -- not the data -- is deciding the update there.
        "denominator_floor_fraction": float(
            (torch.cat([denominator_p, denominator_n]) < denominator_floor)
            .double().mean()),
        "drift_rms_raw": raw_rms,
        "drift_norm_mean": float(norms.mean()),
        "drift_norm_std": float(norms.std(unbiased=False)),
    }


def field_with_snr(
    generated: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    feature_branch: Branch,
    kernel: BlockKernel,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Field plus a cross-fit drift signal-to-noise ratio.

    The positive and negative batches are halved, the field is recomputed on
    the same probes from each disjoint half, and

        SNR = mean_i |V_i|^2 / (mean_i |V^A_i - V^B_i|^2 / 4)

    estimates signal over minibatch-noise variance without reusing an
    example for both the estimate and its own error bar.  Normalization is
    forced off inside the halves so the ratio is not trivially one.
    """
    drift, stats = field(generated, positive, negative, feature_branch,
                         kernel, **kwargs)
    half_kwargs = dict(kwargs)
    half_kwargs["normalization"] = "none"
    half_kwargs["diagnostics"] = False
    half_kwargs["self_mask"] = False
    # Each half has its own anchors, so it must build its own data span; a
    # cached full-batch basis would project onto the wrong subspace.
    half_kwargs["projection"] = None
    np_half, nn_half = len(positive) // 2, len(negative) // 2
    if np_half < 1 or nn_half < 1:
        stats["drift_snr"] = float("nan")
        return drift, stats
    drift_a, _ = field(generated, positive[:np_half], negative[:nn_half],
                       feature_branch, kernel, **half_kwargs)
    drift_b, _ = field(generated, positive[np_half:2 * np_half],
                       negative[nn_half:2 * nn_half], feature_branch, kernel,
                       **half_kwargs)
    with torch.no_grad():
        mean = 0.5 * (drift_a + drift_b)
        signal = float((mean.reshape(len(mean), -1) ** 2).sum(dim=1).mean())
        diff = (drift_a - drift_b).reshape(len(drift_a), -1)
        noise = float((diff ** 2).sum(dim=1).mean()) / 4.0
    stats["drift_signal"] = signal
    stats["drift_noise"] = noise
    stats["drift_snr"] = signal / (noise + NUMERICAL_ZERO)
    return drift, stats


def drift_spectrum(drift: torch.Tensor, bands: int = 3) -> dict[str, float]:
    """Radially binned power fractions of the input-space drift.

    Reports where in frequency the update actually acts, so a claim that a
    branch supplies high-frequency structure can be checked rather than
    assumed (plan section 10.1, "spectral distribution of the input
    gradient").
    """
    with torch.no_grad():
        spectrum = torch.fft.rfft2(drift.to(torch.float32))
        power = (spectrum.real ** 2 + spectrum.imag ** 2).mean(dim=(0, 1))
        h, w = power.shape
        fy = torch.fft.fftfreq(drift.shape[-2]).abs()[:h]
        fx = torch.fft.rfftfreq(drift.shape[-1])[:w]
        radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        radius = radius / radius.max().clamp_min(NUMERICAL_ZERO)
        total = power.sum().clamp_min(NUMERICAL_ZERO)
        out = {}
        for band in range(bands):
            low, high = band / bands, (band + 1) / bands
            selector = (radius >= low) & (
                radius <= high if band == bands - 1 else radius < high)
            out[f"drift_power_band{band}"] = float(
                power[selector].sum() / total)
    return out
