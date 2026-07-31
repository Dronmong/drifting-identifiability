"""Plan section 9, P0.2 unit tests for the fixed compositional features."""

from __future__ import annotations

import numpy as np
import torch

from .. import fixed_features as FF
from ..config import GeometryConfig
from .harness import main

SIZE, CHANNELS = 16, 3


def _images(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, CHANNELS, SIZE, SIZE, generator=g)


def _family(name: str, **kwargs) -> FF.FeatureFamily:
    return FF.build_family(GeometryConfig(family=name, **kwargs), CHANNELS)


def test_shapes_by_scale_and_location() -> None:
    """1. output shape follows the declared scale and location grid"""
    family = _family("wavelet", scales=3, orientations=4, pool=4)
    images = _images(5, 1)
    shapes = family.block_shapes(images)
    wavelet = [k for k in shapes if k.startswith("wavelet_s")]
    assert len(wavelet) == 3, shapes
    for name in wavelet:
        # 4x4 pooled locations, each carrying channels x orientations.
        assert len(shapes[name]) == 16, (name, len(shapes[name]))
        assert set(shapes[name]) == {CHANNELS * 4}, shapes[name]
    for branch in family.branches:
        for block in branch.blocks(images):
            assert block.shape[0] == 5


def test_no_trainable_parameters() -> None:
    """2. no family carries a trainable parameter"""
    for name in FF.FAMILIES:
        family = _family(name, second_order=True, covariance_terms=True)
        assert family.trainable_parameter_count() == 0, name
        for tensor in family.tensors:
            assert not tensor.requires_grad, name


def test_deterministic_outputs() -> None:
    """3. features are deterministic and seed-reproducible"""
    images = _images(4, 2)
    for name in FF.FAMILIES:
        a = _family(name).branches[0].flat(images)
        b = _family(name).branches[0].flat(images)
        assert torch.equal(a, b), name


def test_input_gradients_are_nonzero() -> None:
    """4. every branch has a nonzero gradient w.r.t. the input"""
    for name in FF.FAMILIES:
        family = _family(name, second_order=True, covariance_terms=True)
        for branch in family.branches:
            images = _images(3, 5).requires_grad_(True)
            branch.flat(images).square().sum().backward()
            assert images.grad is not None and float(
                images.grad.abs().sum()) > 0, (name, branch.name)


def test_smooth_modulus_is_differentiable_at_zero() -> None:
    """5. the smooth modulus has a finite gradient at a zero response"""
    zero = torch.zeros(2, 2, requires_grad=True)
    other = torch.zeros(2, 2, requires_grad=True)
    FF.smooth_modulus(zero, other, 1e-3).sum().backward()
    assert torch.isfinite(zero.grad).all()


def test_translation_and_deformation_sensitivity() -> None:
    """6. modulus features are deformation-stable yet position-sensitive

    The declared design is Mallat's: a smooth modulus followed by local
    pooling is *more stable* than raw pixels under a small displacement,
    while remaining sensitive to a large one.  The coarse pyramid branch is
    deliberately position-sensitive too, so it is not the contrast here.
    """
    from .. import datasets as D
    images = D.pinwheel().sample(16, np.random.default_rng(7))
    small = torch.roll(images, shifts=(1, 1), dims=(2, 3))
    large = torch.roll(images, shifts=(6, 6), dims=(2, 3))
    wavelet = _family("wavelet", scales=3, pool=4).branches[0]
    raw = _family("raw").branches[0]

    def relative_change(branch: FF.Branch, other: torch.Tensor) -> float:
        a, b = branch.flat(images), branch.flat(other)
        return float((a - b).norm() / a.norm().clamp_min(1e-12))

    small_wavelet = relative_change(wavelet, small)
    small_raw = relative_change(raw, small)
    large_wavelet = relative_change(wavelet, large)
    assert small_wavelet < small_raw, (small_wavelet, small_raw)
    assert large_wavelet > small_wavelet, (large_wavelet, small_wavelet)
    assert large_wavelet > 0.05, large_wavelet


def test_localized_features_separate_permuted_layouts() -> None:
    """7. localized features distinguish permuted patch layouts

    Measured as a detection ratio -- the between-law energy distance over
    the within-law null in the *same* feature space -- so branches of
    different dimension are compared fairly.  Global pooling is not exactly
    blind here (a filter straddling a quadrant seam sees which patches are
    adjacent), which is why the claim is relative, not absolute.
    """
    from .. import datasets as D
    from ..metrics import energy_distance2
    rng = np.random.default_rng(3)
    layout = D.patch_layout()
    left_a = layout.sample(96, rng)
    left_b = layout.sample(96, rng)
    right = D._finish(
        np.stack([D._assemble_layout(o)
                  for o in D.LAYOUT_COLLISION_ORDERS])[
                      rng.integers(0, 2, 96)], rng, 0.06)
    localized = _family("wavelet", pool=4).branches[0]
    globally_pooled = FF.Branch(
        "global_mean",
        lambda x: [FF.local_pool(
            FF.smooth_modulus(*FF.wavelet_response(
                x, FF.gabor_bank(4)), 1e-3), 1).reshape(len(x), -1)])

    def detection(branch: FF.Branch) -> float:
        null = energy_distance2(branch.flat(left_a), branch.flat(left_b))
        signal = energy_distance2(branch.flat(left_a), branch.flat(right))
        return signal / max(abs(null), 1e-12)

    assert detection(localized) > 3 * detection(globally_pooled), (
        detection(localized), detection(globally_pooled))


def test_orthonormal_wavelet_preserves_pixel_distance() -> None:
    """8. NEGATIVE CONTROL: the Haar basis changes no Euclidean distance"""
    images = _images(16, 11)
    transformed = FF.haar_transform(images)
    pixel = torch.cdist(images.reshape(16, -1), images.reshape(16, -1))
    haar = torch.cdist(transformed.reshape(16, -1),
                       transformed.reshape(16, -1))
    assert float((pixel - haar).abs().max()) < 1e-3, float(
        (pixel - haar).abs().max())


def test_pyramid_levels_decimate() -> None:
    """9. the pyramid halves resolution per level and stays complete"""
    images = _images(2, 13)
    gauss = FF.gaussian_pyramid(images, 3)
    assert [tuple(g.shape[-2:]) for g in gauss] == [(16, 16), (8, 8), (4, 4)]
    laplace = FF.laplacian_pyramid(images, 3)
    reconstructed = laplace[-1]
    for level in reversed(range(len(laplace) - 1)):
        up = torch.nn.functional.interpolate(
            reconstructed, size=laplace[level].shape[-2:], mode="bilinear",
            align_corners=False)
        reconstructed = laplace[level] + up
    assert float((reconstructed - images).abs().max()) < 1e-5


def test_gabor_bank_is_admissible_and_normalized() -> None:
    """10. oriented filters have zero mean and unit norm"""
    bank = FF.gabor_bank(4)
    assert bank.shape == (4, 2, 5, 5)
    assert float(bank.sum(dim=(2, 3)).abs().max()) < 1e-5
    norms = bank.reshape(8, -1).norm(dim=1)
    assert float((norms - 1.0).abs().max()) < 1e-5


def test_random_conv_is_seed_determined_not_learned() -> None:
    """11. random convolutional features are a function of the seed alone"""
    a = FF.random_conv_stack(CHANNELS, 8, 2, 4321)
    b = FF.random_conv_stack(CHANNELS, 8, 2, 4321)
    c = FF.random_conv_stack(CHANNELS, 8, 2, 4322)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(a[0], c[0])


if __name__ == "__main__":
    main("fixed features (P0.2)", dict(globals()))
