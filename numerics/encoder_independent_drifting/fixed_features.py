"""Branch B: fixed compositional image geometry (plan section 6.2).

Nothing in this module is trained.  Every filter is generated
deterministically from a declared seed and carries ``requires_grad=False``;
:func:`trainable_parameter_count` is asserted to be zero by the unit tests.
The families here exist to make finite-batch image geometry usable, not to
supply any source-law authority -- they are **not** claimed to be injective,
and after a modulus or a pooling step they generally are not.

Structure
---------
A :class:`FeatureFamily` exposes *branches*.  A branch is the unit the
adaptive mixture reweights (plan section 6.4: "each j represents a scale,
feature family, or bandwidth").  Each branch extracts a list of *blocks*
``F_{s,r}``, indexed by scale ``s`` and spatial region ``r``; the geometry
kernel sums nonnegative-weighted base kernels over those blocks (plan
section 6.2).  The same region ``r`` is used on both images, so no soft
cross-patch matching is ever described as a positive-definite kernel.

Position sensitivity is deliberate.  Coefficients are pooled onto a coarse
grid rather than globally averaged, and a coarse position-sensitive image
pyramid is retained, so that two images with identical patch marginals but
different global arrangement remain distinguishable.  Invariance belongs in
this branch only; the spectral anchor keeps source-space distinctions.

Negative control: ``haar_control`` is a full orthonormal Haar transform.  A
radial kernel on it induces exactly the pixel-space distances, so it must
show no geometry gain -- if it does, the measurement is wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .config import GeometryConfig, MASTER_SEED, derive_seed

FAMILIES = (
    "raw", "pyramid", "wavelet", "scattering", "randconv", "dictionary",
    "haar_control",
)


# ---------------------------------------------------------------------------
# Fixed linear operators
# ---------------------------------------------------------------------------


_BINOMIAL = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def blur(images: torch.Tensor) -> torch.Tensor:
    """Separable fixed binomial low-pass, reflect padded, shape preserving."""
    c = images.shape[1]
    k = _BINOMIAL.to(images.dtype).to(images.device)
    pad = (len(k) - 1) // 2
    row = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    col = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(F.pad(images, (pad, pad, 0, 0), mode="reflect"),
                 row, groups=c)
    return F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), col, groups=c)


def gaussian_pyramid(images: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Blur-and-decimate pyramid; stops early if a level would go below 2px."""
    out = [images]
    for _ in range(max(levels - 1, 0)):
        current = out[-1]
        if min(current.shape[-2:]) < 4:
            break
        out.append(blur(current)[:, :, ::2, ::2])
    return out


def laplacian_pyramid(images: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Band-pass pyramid: level i minus the upsampled level i+1.

    The coarsest Gaussian level is appended, so the representation is
    complete (it determines the input) up to interpolation error.
    """
    gauss = gaussian_pyramid(images, levels)
    out = []
    for i in range(len(gauss) - 1):
        up = F.interpolate(gauss[i + 1], size=gauss[i].shape[-2:],
                           mode="bilinear", align_corners=False)
        out.append(gauss[i] - up)
    out.append(gauss[-1])
    return out


def gabor_bank(orientations: int, size: int = 5, sigma: float = 1.0,
               xi: float = 2.2, dtype=torch.float32) -> torch.Tensor:
    """Fixed oriented band-pass filters, ``[orientations, 2, size, size]``.

    Channel 0 is the (mean-removed, hence admissible) even part and channel 1
    the odd part; together they act as a complex analytic wavelet.  Both are
    L2-normalized so branch scales are comparable.
    """
    if orientations < 1 or size < 3 or size % 2 == 0:
        raise ValueError("need >= 1 orientation and an odd size >= 3")
    half = size // 2
    grid = torch.arange(-half, half + 1, dtype=torch.float64)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    envelope = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    filters = []
    for index in range(orientations):
        theta = math.pi * index / orientations
        proj = xx * math.cos(theta) + yy * math.sin(theta)
        even = envelope * torch.cos(xi * proj)
        even = even - envelope * (even.sum() / envelope.sum())
        odd = envelope * torch.sin(xi * proj)
        even = even / even.norm().clamp_min(1e-12)
        odd = odd / odd.norm().clamp_min(1e-12)
        filters.append(torch.stack([even, odd]))
    return torch.stack(filters).to(dtype)


def wavelet_response(images: torch.Tensor,
                     bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise oriented response; returns (even, odd), each
    ``[B, C, orientations, H, W]``."""
    b, c, h, w = images.shape
    orientations, parts, size, _ = bank.shape
    pad = size // 2
    if pad >= min(h, w):
        raise ValueError("filter support exceeds the image; decimate first")
    weight = bank.reshape(orientations * parts, 1, size, size)
    weight = weight.repeat(c, 1, 1, 1)          # [C*O*2, 1, k, k]
    padded = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(padded, weight.to(images.dtype), groups=c)
    out = out.reshape(b, c, orientations, parts, h, w)
    return out[:, :, :, 0], out[:, :, :, 1]


def smooth_modulus(even: torch.Tensor, odd: torch.Tensor,
                   eps: float) -> torch.Tensor:
    """``|u|_eps = sqrt(even^2 + odd^2 + eps^2)`` -- differentiable at zero.

    ``eps`` is a declared smoothing parameter; the unit tests and the results
    document report its sensitivity.
    """
    if eps <= 0:
        raise ValueError("modulus epsilon must be positive")
    return torch.sqrt(even ** 2 + odd ** 2 + eps ** 2)


def local_pool(x: torch.Tensor, grid: int) -> torch.Tensor:
    """Average-pool the trailing two dimensions onto a ``grid x grid`` map.

    Deliberately coarse rather than global: the output stays position
    sensitive, which is what makes global-arrangement collisions detectable.
    """
    shape = x.shape
    flat = x.reshape(-1, 1, shape[-2], shape[-1])
    target = min(grid, shape[-2], shape[-1])
    pooled = F.adaptive_avg_pool2d(flat, target)
    return pooled.reshape(*shape[:-2], target, target)


def haar_transform(images: torch.Tensor) -> torch.Tensor:
    """Full orthonormal 2-D Haar transform (shape preserving).

    Orthonormal, so Euclidean distances are exactly pixel distances.  This is
    the negative control for "a wavelet basis alone changes nothing".
    """
    x = images.clone()
    h, w = x.shape[-2:]
    size = min(h, w)
    root = 1.0 / math.sqrt(2.0)
    while size >= 2 and size % 2 == 0:
        block = x[..., :size, :size]
        low = (block[..., 0::2, :] + block[..., 1::2, :]) * root
        high = (block[..., 0::2, :] - block[..., 1::2, :]) * root
        block = torch.cat([low, high], dim=-2)
        low = (block[..., :, 0::2] + block[..., :, 1::2]) * root
        high = (block[..., :, 0::2] - block[..., :, 1::2]) * root
        block = torch.cat([low, high], dim=-1)
        x = x.clone()
        x[..., :size, :size] = block
        size //= 2
    return x


def random_conv_stack(channels_in: int, channels: int, layers: int,
                      seed: int, dtype=torch.float32) -> list[torch.Tensor]:
    """Fixed random 3x3 convolution weights with NNGP-style fan-in scaling.

    Not trained and not pretrained: the weights are a deterministic function
    of the declared seed, so this is a random-feature approximation of a
    convolutional NNGP-type kernel, not a learned encoder.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2 ** 63 - 1))
    weights = []
    fan_in_channels = channels_in
    for _ in range(max(layers, 1)):
        w = torch.randn(channels, fan_in_channels, 3, 3, generator=generator,
                        dtype=torch.float64)
        w = w / math.sqrt(fan_in_channels * 9)
        weights.append(w.to(dtype))
        fan_in_channels = channels
    return weights


def random_conv_features(images: torch.Tensor,
                         weights: list[torch.Tensor]) -> list[torch.Tensor]:
    """Apply the fixed stack; returns the activation map after each layer."""
    out = []
    x = images
    for index, w in enumerate(weights):
        x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), w.to(x.dtype))
        x = F.gelu(x)                       # smooth, so gradients exist
        if index + 1 < len(weights) and min(x.shape[-2:]) >= 4:
            x = blur(x)[:, :, ::2, ::2]
        out.append(x)
    return out


# ---------------------------------------------------------------------------
# Branches and families
# ---------------------------------------------------------------------------


@dataclass
class Branch:
    """One mixture-level geometry branch."""

    name: str
    extract: Callable[[torch.Tensor], list[torch.Tensor]]

    def blocks(self, images: torch.Tensor) -> list[torch.Tensor]:
        out = self.extract(images)
        if not out:
            raise ValueError(f"branch {self.name} produced no blocks")
        return out

    def flat(self, images: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.blocks(images), dim=1)


def bandwidth_mixture(branch: Branch, levels: int) -> Branch:
    """Repeat every block ``levels`` times so a kernel can span scales.

    The block kernel already sums nonnegative-weighted base kernels over
    blocks, and a nonnegative combination of positive-definite kernels is
    positive definite -- so a mixture over *bandwidths* is obtained for free
    by presenting the same features several times and giving each copy its
    own tau (see `kernels.geometric_multipliers`).  Nothing is recomputed:
    the repeated entries are references to one tensor, so only the Gram
    evaluation is paid for again.
    """
    if levels < 1:
        raise ValueError("a mixture needs at least one level")

    def extract(images: torch.Tensor) -> list[torch.Tensor]:
        return [block for block in branch.blocks(images)
                for _ in range(levels)]

    return Branch(f"{branch.name}_mix{levels}", extract)


@dataclass
class FeatureFamily:
    name: str
    branches: list[Branch]
    config: GeometryConfig
    tensors: list[torch.Tensor]

    @property
    def branch_names(self) -> list[str]:
        return [b.name for b in self.branches]

    def trainable_parameter_count(self) -> int:
        return sum(int(t.requires_grad) for t in self.tensors)

    def block_shapes(self, images: torch.Tensor) -> dict[str, list[int]]:
        return {b.name: [int(x.shape[1]) for x in b.blocks(images)]
                for b in self.branches}


def _regions(pooled: torch.Tensor) -> list[torch.Tensor]:
    """Split a ``[B, ..., G, G]`` pooled map into one block per region."""
    b, g = pooled.shape[0], pooled.shape[-1]
    flat = pooled.reshape(b, -1, pooled.shape[-2] * pooled.shape[-1])
    return [flat[:, :, r] for r in range(g * g)]


def build_family(config: GeometryConfig, channels: int,
                 seed_label: str = "geometry") -> FeatureFamily:
    """Assemble the declared fixed geometry family.  No training, ever."""
    if config.family not in FAMILIES:
        raise ValueError(f"unknown geometry family {config.family!r}")
    bank = gabor_bank(config.orientations).requires_grad_(False)
    tensors: list[torch.Tensor] = [bank]
    branches: list[Branch] = []
    pool, eps, scales = config.pool, config.modulus_eps, config.scales

    def pyramid_branches() -> list[Branch]:
        out = []
        for level in range(scales):
            def extract(images: torch.Tensor, level=level) -> list[
                    torch.Tensor]:
                pyr = laplacian_pyramid(images, scales)
                return _regions(local_pool(pyr[min(level, len(pyr) - 1)], pool))
            out.append(Branch(f"pyramid_l{level}", extract))
        return out

    def wavelet_branches() -> list[Branch]:
        out = []
        for level in range(scales):
            def extract(images: torch.Tensor, level=level) -> list[
                    torch.Tensor]:
                pyr = gaussian_pyramid(images, scales)
                base = pyr[min(level, len(pyr) - 1)]
                even, odd = wavelet_response(base, bank.to(base.dtype))
                return _regions(local_pool(smooth_modulus(even, odd, eps),
                                           pool))
            out.append(Branch(f"wavelet_s{level}", extract))
        return out

    def scattering_branches() -> list[Branch]:
        out = []
        for level in range(max(scales - 1, 1)):
            def extract(images: torch.Tensor, level=level) -> list[
                    torch.Tensor]:
                pyr = gaussian_pyramid(images, scales)
                base = pyr[min(level, len(pyr) - 1)]
                even, odd = wavelet_response(base, bank.to(base.dtype))
                mod = smooth_modulus(even, odd, eps)
                b, c, o, h, w = mod.shape
                mod = mod.reshape(b, c * o, h, w)
                if min(h, w) >= 4:
                    mod = blur(mod)[:, :, ::2, ::2]
                even2, odd2 = wavelet_response(mod, bank.to(mod.dtype))
                return _regions(local_pool(smooth_modulus(even2, odd2, eps),
                                           pool))
            out.append(Branch(f"scatter2_s{level}", extract))
        return out

    def covariance_branch() -> Branch:
        def extract(images: torch.Tensor) -> list[torch.Tensor]:
            pyr = gaussian_pyramid(images, scales)
            mods = []
            size = pyr[0].shape[-2:]
            for level in range(min(scales, len(pyr))):
                even, odd = wavelet_response(
                    pyr[level], bank.to(pyr[level].dtype))
                mod = smooth_modulus(even, odd, eps)
                b, c, o, h, w = mod.shape
                mod = mod.reshape(b, c * o, h, w)
                if (h, w) != size:
                    mod = F.interpolate(mod, size=size, mode="bilinear",
                                        align_corners=False)
                mods.append(mod)
            blocks = []
            for i in range(len(mods)):
                for j in range(i + 1, len(mods)):
                    prod = local_pool(mods[i] * mods[j], pool)
                    blocks.extend(_regions(prod))
            if not blocks:
                blocks = _regions(local_pool(mods[0], pool))
            return blocks
        return Branch("wavelet_cov", extract)

    def randconv_branch() -> Branch:
        weights = random_conv_stack(
            channels, config.randconv_channels, config.randconv_layers,
            derive_seed(MASTER_SEED, seed_label, config.randconv_seed_label))
        for w in weights:
            w.requires_grad_(False)
        tensors.extend(weights)

        def extract(images: torch.Tensor) -> list[torch.Tensor]:
            maps = random_conv_features(images, weights)
            blocks: list[torch.Tensor] = []
            for level, activation in enumerate(maps):
                blocks.extend(_regions(local_pool(activation, pool)))
            return blocks
        return Branch("randconv", extract)

    def global_pyramid_branch() -> Branch:
        def extract(images: torch.Tensor) -> list[torch.Tensor]:
            pyr = gaussian_pyramid(images, scales)
            return [pyr[-1].reshape(len(images), -1)]
        return Branch("pyramid_global", extract)

    if config.family == "raw":
        branches = [Branch("raw", lambda x: [x.reshape(len(x), -1)])]
    elif config.family == "haar_control":
        branches = [Branch(
            "haar", lambda x: [haar_transform(x).reshape(len(x), -1)])]
    elif config.family == "pyramid":
        branches = pyramid_branches()
    elif config.family == "wavelet":
        branches = wavelet_branches()
    elif config.family == "scattering":
        branches = wavelet_branches() + scattering_branches()
        if config.covariance_terms:
            branches.append(covariance_branch())
    elif config.family == "randconv":
        branches = [randconv_branch()]
    elif config.family == "dictionary":
        branches = pyramid_branches() + wavelet_branches()
        if config.second_order:
            branches += scattering_branches()
        if config.covariance_terms:
            branches.append(covariance_branch())
        branches.append(randconv_branch())

    if config.include_global_pyramid and config.family not in (
            "raw", "haar_control"):
        branches = branches + [global_pyramid_branch()]

    return FeatureFamily(config.family, branches, config, tensors)
