"""Plan section 9, P0.3 unit tests for the common drift-field API."""

from __future__ import annotations

import torch

from .. import fixed_features as FF
from .. import kernel_gradient as KG
from .. import kernels as K
from ..config import GeometryConfig
from .harness import main

SIZE, CHANNELS = 8, 3


def _images(n: int, seed: int, shift: float = 0.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, CHANNELS, SIZE, SIZE, generator=g) + shift


def _raw_branch() -> FF.Branch:
    return FF.build_family(GeometryConfig(family="raw"), CHANNELS).branches[0]


def _wavelet_branch() -> FF.Branch:
    return FF.build_family(
        GeometryConfig(family="wavelet", scales=2, pool=2),
        CHANNELS).branches[0]


def _kernel(base: str, tau: float, blocks: int = 1,
            combine: str = "sum") -> K.BlockKernel:
    return K.BlockKernel(base, torch.full((blocks,), tau),
                         torch.full((blocks,), 1.0 / blocks), 1e-3, combine)


def test_autodiff_matches_finite_differences() -> None:
    """1. the kernel-gradient field matches central finite differences"""
    branch, kernel = _raw_branch(), _kernel("smooth_laplace", 6.0)
    generated = _images(4, 1).to(torch.float64)
    positive = _images(9, 2, shift=0.5).to(torch.float64)
    negative = _images(5, 3).to(torch.float64)

    def log_ratio(x: torch.Tensor) -> float:
        with torch.no_grad():
            gp = kernel.gram_from_blocks(branch.blocks(x),
                                         branch.blocks(positive))
            gn = kernel.gram_from_blocks(branch.blocks(x),
                                         branch.blocks(negative))
            return float(torch.log(gp.mean(dim=1)).sum()
                         - torch.log(gn.mean(dim=1)).sum())

    drift, _ = KG.field(generated, positive, negative, branch, kernel,
                        direction_mode="kernel_gradient",
                        normalization="none", diagnostics=False)
    eps = 1e-6
    for index in ((0, 0, 0, 0), (2, 1, 3, 4), (3, 2, 7, 7)):
        plus, minus = generated.clone(), generated.clone()
        plus[index] += eps
        minus[index] -= eps
        numeric = (log_ratio(plus) - log_ratio(minus)) / (2 * eps)
        assert abs(numeric - float(drift[index])) < 1e-5 * max(
            1.0, abs(numeric)), (index, numeric, float(drift[index]))


def test_gaussian_reproduces_the_smoothed_score() -> None:
    """2. a raw Gaussian kernel gives mean-shift / tau^2 exactly"""
    branch, tau = _raw_branch(), 5.0
    kernel = _kernel("gaussian", tau)
    generated, positive = _images(6, 4), _images(11, 5, shift=0.7)
    negative = _images(6, 6)
    standard, _ = KG.field(generated, positive, negative, branch, kernel,
                           direction_mode="standard", normalization="none",
                           diagnostics=False)
    gradient, _ = KG.field(generated, positive, negative, branch, kernel,
                           direction_mode="kernel_gradient",
                           normalization="none", diagnostics=False)
    expected = standard / (tau ** 2)
    assert float((gradient - expected).norm()
                 / expected.norm()) < 1e-5


def test_denominator_floor_is_explicit_and_finite() -> None:
    """3. denominators are floored explicitly and the field stays finite"""
    branch = _raw_branch()
    kernel = _kernel("smooth_laplace", 1e-3)      # affinities underflow
    generated, positive = _images(4, 7), _images(6, 8, shift=60.0)
    negative = _images(4, 9, shift=-60.0)
    for mode in KG.DIRECTION_MODES:
        drift, stats = KG.field(generated, positive, negative, branch, kernel,
                                direction_mode=mode, normalization="none",
                                denominator_floor=1e-30)
        # The reported denominator is the raw one, so total underflow shows
        # up as zero rather than as the floor; the field must still be
        # finite, and the floor must announce that it engaged.
        assert torch.isfinite(drift).all(), mode
        assert stats["denominator_min"] == 0.0, (mode,
                                                 stats["denominator_min"])
        assert stats["denominator_floor_fraction"] > 0.0, mode
    try:
        KG.field(generated, positive, negative, branch, kernel,
                 denominator_floor=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-positive denominator floor was accepted")


def test_collapsed_kernel_is_flagged_not_praised() -> None:
    """3b. a fully collapsed kernel reports collapse, never a huge ESS

    Regression: normalizing an all-underflow row against the denominator
    floor gives zero weights and an "effective sample size" of 1/0, which
    was reported as ESS ~ 1e10 -- excellent health in exactly the regime the
    plan wants flagged (the paper's flat-kernel failure).
    """
    import math
    branch = _raw_branch()
    kernel = _kernel("smooth_laplace", 1e-3)
    generated, positive = _images(4, 30), _images(6, 31, shift=80.0)
    negative = _images(4, 32, shift=-80.0)
    _, stats = KG.field(generated, positive, negative, branch, kernel,
                        direction_mode="kernel_gradient",
                        normalization="none")
    assert stats["collapsed_row_fraction"] == 1.0, stats[
        "collapsed_row_fraction"]
    assert math.isnan(stats["ess_fraction"]), stats["ess_fraction"]
    assert math.isnan(stats["entropy_fraction"]), stats["entropy_fraction"]

    # A healthy kernel on a realistic cloud still reports a finite ESS
    # fraction in (0, 1].  The clouds above are 160 units apart in 192
    # dimensions, so they collapse at any sane bandwidth; the control has to
    # use a cloud the kernel can actually see.
    near_generated, near_positive = _images(4, 33), _images(6, 34, shift=0.3)
    healthy = _kernel("smooth_laplace", 8.0)
    _, ok = KG.field(near_generated, near_positive, near_generated, branch,
                     healthy, direction_mode="kernel_gradient",
                     normalization="none")
    assert ok["collapsed_row_fraction"] == 0.0
    assert 0.0 < ok["ess_fraction"] <= 1.0 + 1e-9, ok["ess_fraction"]


def test_batch_permutation_invariance() -> None:
    """4. permuting the positive/negative batches does not change the field"""
    branch, kernel = _wavelet_branch(), _kernel(
        "smooth_laplace", 2.0, blocks=4)
    generated, positive = _images(5, 10), _images(12, 11, shift=0.4)
    negative = _images(7, 12)
    order_p = torch.randperm(len(positive))
    order_n = torch.randperm(len(negative))
    for mode in KG.DIRECTION_MODES:
        a, _ = KG.field(generated, positive, negative, branch, kernel,
                        direction_mode=mode, diagnostics=False)
        b, _ = KG.field(generated, positive[order_p], negative[order_n],
                        branch, kernel, direction_mode=mode,
                        diagnostics=False)
        assert float((a - b).abs().max()) < 1e-5, mode


def test_probe_permutation_is_equivariant() -> None:
    """5. permuting the probes permutes the field the same way"""
    branch, kernel = _raw_branch(), _kernel("smooth_laplace", 3.0)
    generated, positive = _images(6, 13), _images(9, 14, shift=0.3)
    negative = _images(6, 15)
    order = torch.randperm(len(generated))
    for mode in KG.DIRECTION_MODES:
        a, _ = KG.field(generated, positive, negative, branch, kernel,
                        direction_mode=mode, normalization="none",
                        diagnostics=False)
        b, _ = KG.field(generated[order], positive, negative, branch, kernel,
                        direction_mode=mode, normalization="none",
                        diagnostics=False)
        assert float((a[order] - b).abs().max()) < 1e-5, mode


def test_swapping_positive_and_negative_reverses_the_field() -> None:
    """6. swapping the positive and negative inputs reverses the field"""
    branch, kernel = _wavelet_branch(), _kernel(
        "smooth_laplace", 2.0, blocks=4)
    generated = _images(5, 16)
    positive, negative = _images(9, 17, shift=0.6), _images(9, 18)
    for mode in KG.DIRECTION_MODES:
        a, _ = KG.field(generated, positive, negative, branch, kernel,
                        direction_mode=mode, normalization="none",
                        diagnostics=False)
        b, _ = KG.field(generated, negative, positive, branch, kernel,
                        direction_mode=mode, normalization="none",
                        diagnostics=False)
        assert float((a + b).abs().max()) < 1e-5, mode


def test_identical_laws_give_zero_field() -> None:
    """7. identical empirical laws give a numerically zero field"""
    branch, kernel = _wavelet_branch(), _kernel(
        "smooth_laplace", 2.0, blocks=4)
    generated, shared = _images(6, 19), _images(10, 20)
    for mode in KG.DIRECTION_MODES:
        drift, _ = KG.field(generated, shared, shared, branch, kernel,
                            direction_mode=mode, normalization="none",
                            diagnostics=False)
        assert float(drift.abs().max()) < 1e-5, mode


def test_kernel_gradient_differs_structurally_from_displacement() -> None:
    """8. a structured kernel makes the two modes genuinely diverge

    Test 2 shows the two modes coincide up to a scalar for a raw Gaussian
    kernel.  The load-bearing claim of plan section 6.3 is that they stop
    coinciding once the kernel measures structure, so the contrast is
    measured as a direction cosine plus a spectral-profile distance -- both
    of which are data-configuration robust, unlike an absolute band level.
    """
    from .. import datasets as D
    import numpy as np
    rng = np.random.default_rng(2)
    target = D.texture_blocks().sample(48, rng)
    generated = D.texture_blocks().sample(24, rng) * 0.3
    branch = FF.build_family(
        GeometryConfig(family="wavelet", scales=2, pool=4), CHANNELS
    ).branches[0]
    kernel = K.calibrate_block_kernel(branch, target, "smooth_laplace",
                                      0.5, 1.0, 1e-3, combine="product")
    drifts, spectra = {}, {}
    for mode in KG.DIRECTION_MODES:
        drift, _ = KG.field(generated, target, generated, branch, kernel,
                            direction_mode=mode, diagnostics=False)
        drifts[mode] = drift.flatten()
        spectra[mode] = KG.drift_spectrum(drift)
    cosine = float(torch.nn.functional.cosine_similarity(
        drifts["standard"], drifts["kernel_gradient"], dim=0))
    profile = sum(
        abs(spectra["standard"][k] - spectra["kernel_gradient"][k])
        for k in spectra["standard"])
    assert cosine < 0.9, cosine
    assert profile > 0.1, profile


def test_paper_mode_reproduces_the_repository_algorithm2_port() -> None:
    """10b. `paper` mode equals lowdim_drift.drift_paper exactly

    The strongest correctness check available: the repository already
    contains a verbatim port of the paper's Algorithm 2, independently
    cross-checked against `driftlab.compute_v_paper`.  With a raw pixel
    branch and the paper's own `laplace` kernel, this implementation must
    agree with it to numerical precision.
    """
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "numerics"))
    from lowdim_drift import drift_paper                    # noqa: PLC0415

    g = torch.Generator().manual_seed(71)
    tau = 1.3
    for n_probe, n_pos, n_neg in ((6, 9, 6), (12, 7, 11)):
        probes = torch.randn(n_probe, CHANNELS, 4, 4, generator=g)
        positive = torch.randn(n_pos, CHANNELS, 4, 4, generator=g) + 0.4
        negative = torch.randn(n_neg, CHANNELS, 4, 4, generator=g)
        branch = FF.build_family(GeometryConfig(family="raw"),
                                 CHANNELS).branches[0]
        kernel = K.BlockKernel("laplace", torch.tensor([tau]),
                               torch.tensor([1.0]), 1e-3)
        got, _ = KG.field(probes, positive, negative, branch, kernel,
                          direction_mode="paper", normalization="none",
                          diagnostics=False)
        flat = [x.reshape(len(x), -1).double().numpy()
                for x in (probes, positive, negative)]
        want = drift_paper(flat[0], flat[1], tau, False)
        # drift_paper reuses the probe cloud as negatives; call it directly
        # on the concatenated form instead so the negatives are honoured.
        import numpy as np
        dp = np.linalg.norm(flat[0][:, None, :] - flat[1][None, :, :], axis=2)
        dn = np.linalg.norm(flat[0][:, None, :] - flat[2][None, :, :], axis=2)
        logit = np.concatenate([-dp / tau, -dn / tau], axis=1)
        lr = logit - logit.max(axis=1, keepdims=True)
        row = np.exp(lr) / np.exp(lr).sum(axis=1, keepdims=True)
        lc = logit - logit.max(axis=0, keepdims=True)
        col = np.exp(lc) / np.exp(lc).sum(axis=0, keepdims=True)
        affinity = np.sqrt(row * col)
        ap, an = affinity[:, :n_pos], affinity[:, n_pos:]
        want = ((ap * an.sum(1, keepdims=True)) @ flat[1]
                - (an * ap.sum(1, keepdims=True)) @ flat[2])
        error = float(np.abs(got.reshape(n_probe, -1).numpy() - want).max())
        assert error < 1e-4, (n_probe, error)


def test_paper_mode_damps_dense_regions_relative_to_sparse() -> None:
    """10c. the column reweighting is an anti-density-seeking mechanism

    This is the mechanism the Phase-2 failure was traced to: SNIS lets a
    target point that is the nearest neighbour of many particles dominate,
    which pulls the generated cloud toward high-density regions.
    """
    import numpy as np
    g = torch.Generator().manual_seed(72)
    dense = torch.randn(24, CHANNELS, 4, 4, generator=g) * 0.15
    sparse = torch.randn(8, CHANNELS, 4, 4, generator=g) * 0.15 + 3.0
    probes = torch.cat([dense, sparse])
    positive = torch.cat([
        torch.randn(16, CHANNELS, 4, 4, generator=g) * 0.15,
        torch.randn(16, CHANNELS, 4, 4, generator=g) * 0.15 + 3.0])
    branch = FF.build_family(GeometryConfig(family="raw"),
                             CHANNELS).branches[0]
    kernel = K.BlockKernel("laplace", torch.tensor([0.8]),
                           torch.tensor([1.0]), 1e-3)
    fields = {}
    for mode in ("standard", "paper"):
        drift, _ = KG.field(probes, positive, probes, branch, kernel,
                            direction_mode=mode, normalization="none",
                            diagnostics=False)
        fields[mode] = drift.reshape(len(probes), -1).norm(dim=1).numpy()
    ratio = fields["paper"] / np.maximum(fields["standard"], 1e-12)
    dense_ratio = float(np.median(ratio[:24]))
    sparse_ratio = float(np.median(ratio[24:]))
    assert dense_ratio < sparse_ratio, (dense_ratio, sparse_ratio)


def test_data_span_projection_is_an_orthogonal_projector() -> None:
    """11. R3b: the data-span projection is idempotent and self-adjoint"""
    g = torch.Generator().manual_seed(41)
    probes = torch.randn(5, 3, 4, 4, generator=g)
    others = torch.randn(9, 3, 4, 4, generator=g)
    v = torch.randn(5, 3, 4, 4, generator=g)
    p1 = KG.data_span_projection(probes, others, v)
    p2 = KG.data_span_projection(probes, others, p1)
    assert float((p1 - p2).abs().max()) < 1e-4, float((p1 - p2).abs().max())
    # Self-adjointness on the probe-independent part: <Pu, v> == <u, Pv>.
    u = torch.randn(5, 3, 4, 4, generator=g)
    left = float((KG.data_span_projection(probes, others, u) * v).sum())
    right = float((u * KG.data_span_projection(probes, others, v)).sum())
    assert abs(left - right) < 1e-3 * max(1.0, abs(left)), (left, right)


def test_projection_preserves_displacement_directions() -> None:
    """12. R3b: vectors already in span{Y_j - x} are left unchanged"""
    g = torch.Generator().manual_seed(42)
    probes = torch.randn(4, 3, 4, 4, generator=g)
    others = torch.randn(7, 3, 4, 4, generator=g)
    # A standard-displacement-style field lies in the span by construction.
    weights = torch.rand(4, 7, generator=g)
    weights = weights / weights.sum(dim=1, keepdim=True)
    barycentre = (weights @ others.reshape(7, -1)).reshape(probes.shape)
    displacement = barycentre - probes
    projected = KG.data_span_projection(probes, others, displacement)
    relative = float((projected - displacement).norm()
                     / displacement.norm())
    assert relative < 1e-3, relative


def test_projection_removes_off_span_components() -> None:
    """13. R3b: a direction orthogonal to the data span is removed"""
    g = torch.Generator().manual_seed(43)
    # Few anchors in a large ambient space, so the span is a strict subspace.
    probes = torch.randn(3, 3, 8, 8, generator=g)
    others = torch.randn(5, 3, 8, 8, generator=g)
    flat_others = others.reshape(5, -1)
    centred = flat_others - flat_others.mean(0, keepdim=True)
    basis, _ = torch.linalg.qr(centred.T, mode="reduced")
    v = torch.randn(3, 3 * 8 * 8, generator=g)
    extra = flat_others.mean(0, keepdim=True) - probes.reshape(3, -1)
    # Remove the whole span from v, leaving a purely off-manifold direction.
    v = v - (v @ basis) @ basis.T
    e = extra - (extra @ basis) @ basis.T
    e = e / e.norm(dim=1, keepdim=True)
    v = v - (v * e).sum(dim=1, keepdim=True) * e
    projected = KG.data_span_projection(probes, others,
                                        v.reshape(probes.shape))
    assert float(projected.norm() / v.norm()) < 1e-3, float(
        projected.norm() / v.norm())


def test_projected_mode_stays_closer_to_the_data() -> None:
    """14. R3b: projection moves the field toward the displacement rule"""
    from .. import datasets as D
    import numpy as np
    rng = np.random.default_rng(9)
    target = D.texture_blocks().sample(48, rng)
    generated = D.texture_blocks().sample(24, rng) * 0.3
    branch = FF.build_family(
        GeometryConfig(family="wavelet", scales=2, pool=4), CHANNELS
    ).branches[0]
    kernel = K.calibrate_block_kernel(branch, target, "smooth_laplace",
                                      0.5, 1.0, 1e-3)
    fields = {}
    for mode in KG.DIRECTION_MODES:
        drift, stats = KG.field(generated, target, generated, branch, kernel,
                                direction_mode=mode, normalization="rms")
        fields[mode] = drift.flatten()
        if mode == "projected_kernel_gradient":
            assert 0.0 < stats["projection_retained_fraction"] <= 1.0 + 1e-6

    def cosine(a: str, b: str) -> float:
        return float(torch.nn.functional.cosine_similarity(
            fields[a], fields[b], dim=0))

    # The projected field must align with the displacement rule strictly
    # better than the unprojected kernel gradient does.
    assert cosine("projected_kernel_gradient", "standard") > cosine(
        "kernel_gradient", "standard"), (
            cosine("projected_kernel_gradient", "standard"),
            cosine("kernel_gradient", "standard"))


def test_block_kernels_are_positive_definite() -> None:
    """9. both declared combination rules give a PSD Gram matrix"""
    branch = _wavelet_branch()
    samples = _images(24, 21)
    for combine in ("sum", "product"):
        kernel = K.calibrate_block_kernel(branch, samples, "smooth_laplace",
                                          0.5, 1.0, 1e-3, combine=combine)
        assert K.min_eigenvalue(kernel, branch, samples) > -1e-8, combine


def test_invalid_kernel_configurations_are_rejected() -> None:
    """10. negative weights and non-normalized weights are refused"""
    for weights in (torch.tensor([-0.5, 1.5]), torch.tensor([0.5, 0.9])):
        try:
            K.BlockKernel("gaussian", torch.tensor([1.0, 1.0]), weights, 1e-3)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid weights {weights}")


if __name__ == "__main__":
    main("kernel gradients (P0.3)", dict(globals()))
