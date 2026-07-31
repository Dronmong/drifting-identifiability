"""Source-collision pairs and an honest two-sample detection test.

Each entry is a genuinely different pair ``p != q`` that some plausible
image statistic is blind to.  The suite exists so that "the anchor keeps
source distinctions" is a measured claim rather than a slogan, and so that
the geometry branches can be *shown* to be non-injective instead of being
quietly assumed injective.

Detection is a permutation two-sample test on a supplied discrepancy: the
observed discrepancy is compared with its own pooled-relabelling null, so a
discrepancy that is merely large in absolute terms does not count as a
detection.  The blindness predictions in ``blind_to`` are the plan's
expectations and are reported against the measured outcome, not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from . import datasets as D

Sampler = Callable[[int, np.random.Generator], torch.Tensor]


@dataclass
class CollisionPair:
    name: str
    left: Sampler
    right: Sampler
    blind_to: str
    note: str


def _layout_collision() -> Sampler:
    protos = np.stack([D._assemble_layout(o)
                       for o in D.LAYOUT_COLLISION_ORDERS])

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        which = rng.integers(0, len(protos), n)
        return D._finish(protos[which], rng, 0.06)
    return sampler


def _phase_scramble(base: Sampler, noise: float = 0.0) -> Sampler:
    """Randomize Fourier phase while preserving each image's power spectrum.

    Hermitian symmetry is enforced by scrambling in the real-FFT domain and
    inverting, so the output is real and its radial power spectrum matches
    the input's to numerical precision.
    """
    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        images = base(n, rng)
        spectrum = torch.fft.rfft2(images)
        magnitude = spectrum.abs()
        phase = torch.tensor(
            rng.uniform(-np.pi, np.pi, size=tuple(spectrum.shape)),
            dtype=torch.float32)
        # The DC term must stay real for the image to stay real-valued.
        phase[..., 0, 0] = 0.0
        scrambled = magnitude * torch.exp(1j * phase)
        out = torch.fft.irfft2(scrambled, s=images.shape[-2:])
        if noise:
            out = out + torch.tensor(
                rng.normal(scale=noise, size=tuple(out.shape)),
                dtype=torch.float32)
        return out.to(torch.float32)
    return sampler


def _single_color_layout(index: int) -> Sampler:
    """One colour layout only.

    The full ``color_layout`` target is a 50/50 mixture of a layout and its
    own channel swap, so swapping channels maps that distribution to itself
    -- it is a symmetry, not a collision.  A colour collision therefore has
    to be built from a single component.
    """
    protos = D.color_layout().prototypes.numpy()

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        return D._finish(np.repeat(protos[index:index + 1], n, axis=0),
                         rng, 0.07)
    return sampler


def _channel_swap(base: Sampler) -> Sampler:
    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        return base(n, rng)[:, [2, 1, 0]]
    return sampler


def _drop_rare() -> Sampler:
    target = D.rare_object()
    protos = target.prototypes.numpy()

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        return D._finish(np.repeat(protos[0:1], n, axis=0), rng, 0.06)
    return sampler


def _low_pass(base: Sampler) -> Sampler:
    from .fixed_features import blur

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        return blur(blur(base(n, rng)))
    return sampler


def _translated(base: Sampler, shift: int = 4) -> Sampler:
    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        return torch.roll(base(n, rng), shifts=(shift, shift), dims=(2, 3))
    return sampler


def suite() -> list[CollisionPair]:
    layout = D.patch_layout()
    phase = D.phase_structured()
    rare = D.rare_object()
    texture = D.texture_blocks()
    pinwheel = D.pinwheel()
    return [
        CollisionPair(
            "patch_layout_permutation", layout.sample, _layout_collision(),
            "globally pooled patch statistics",
            "identical patch multiset, different global arrangement"),
        CollisionPair(
            "phase_scramble", phase.sample, _phase_scramble(phase.sample),
            "power-spectrum-only statistics",
            "identical per-image power spectrum, destroyed phase structure"),
        CollisionPair(
            "color_swap", _single_color_layout(0),
            _channel_swap(_single_color_layout(0)),
            "luminance-only statistics",
            "identical luminance, permuted colour channels"),
        CollisionPair(
            "rare_mode_drop", rare.sample, _drop_rare(),
            "bulk statistics",
            "5% rare mode present vs absent"),
        CollisionPair(
            "high_frequency_removal", texture.sample, _low_pass(texture.sample),
            "coarse-scale statistics",
            "fine texture removed by a fixed low-pass"),
        CollisionPair(
            "translation_orbit", pinwheel.sample, _translated(pinwheel.sample),
            "fully translation-invariant statistics",
            "same image content at a different location"),
    ]


def permutation_test(discrepancy: Callable[[torch.Tensor, torch.Tensor],
                                           float],
                     left: torch.Tensor, right: torch.Tensor,
                     permutations: int, rng: np.random.Generator,
                     alpha: float = 0.05) -> dict:
    """Two-sample permutation test; ``detected`` iff the p-value <= alpha."""
    observed = float(discrepancy(left, right))
    pooled = torch.cat([left, right], dim=0)
    n = len(left)
    exceed = 0
    null_values = []
    for _ in range(permutations):
        order = rng.permutation(len(pooled))
        value = float(discrepancy(pooled[order[:n]], pooled[order[n:]]))
        null_values.append(value)
        if value >= observed:
            exceed += 1
    p_value = (exceed + 1) / (permutations + 1)
    return {
        "observed": observed,
        "null_median": float(np.median(null_values)),
        "p_value": p_value,
        "detected": bool(p_value <= alpha),
    }


def run_suite(discrepancy: Callable[[torch.Tensor, torch.Tensor], float],
              samples: int, permutations: int,
              rng: np.random.Generator, alpha: float = 0.05) -> dict:
    """Run every collision pair against one discrepancy functional."""
    rows = []
    for pair in suite():
        left = pair.left(samples, rng)
        right = pair.right(samples, rng)
        result = permutation_test(
            discrepancy, left, right, permutations, rng, alpha)
        rows.append({"collision": pair.name, "blind_to": pair.blind_to,
                     **result})
    detected = sum(bool(row["detected"]) for row in rows)
    return {
        "rows": rows,
        "detected": detected,
        "total": len(rows),
        "detection_rate": detected / max(len(rows), 1),
        "failed": [row["collision"] for row in rows if not row["detected"]],
    }
