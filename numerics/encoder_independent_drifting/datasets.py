"""Structured low-resolution image targets (plan section 9, Phase 1).

Each family is chosen to expose a specific geometry failure, and every one
of the plan's listed probes is present:

``checkerboard``    noisy checkerboard;
``pinwheel``        radial spokes;
``rings_islands``   rings plus disconnected islands;
``texture_blocks``  local oriented texture blocks;
``patch_layout``    identical patch histograms, different global arrangement;
``color_layout``    colour-swapped layouts;
``phase_structured``phase structure on top of a fixed power spectrum;
``rare_object``     rare small-object modes (5% of mass);
``deformed``        translated and slightly deformed copies.

Samplers return ``[n, C, H, W]`` float32 in roughly ``[-1, 1]``.
``component_label`` is an ORACLE diagnostic used only for occupancy
evaluation; no objective, controller or bandwidth may read it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

SIZE = 16
CHANNELS = 3


@dataclass
class ImageTarget:
    name: str
    sampler: Callable[[int, np.random.Generator], torch.Tensor]
    kind: str
    n_components: int = 0
    prototypes: torch.Tensor | None = None   # ORACLE, evaluation only
    component_weights: np.ndarray | None = None
    # Declared, never auto-detected: every recorded result in this program is
    # a CPU result, and a target that silently followed whatever torch build
    # happened to be installed would make artifacts incomparable in silence.
    device: torch.device | str | None = None

    def sample(self, n: int, rng: np.random.Generator) -> torch.Tensor:
        out = self.sampler(n, rng)
        # Sampling is driven by a numpy Generator, so the draw is identical
        # whichever device trains; only the transfer differs.  Moving here
        # rather than at every call site keeps one place responsible.
        if self.device is not None:
            out = out.to(self.device)
        # Resolution is validated as square-and-consistent rather than fixed
        # at SIZE, so real-image targets at other resolutions are admissible;
        # channel count and rank are still enforced.
        if out.ndim != 4 or out.shape[1] != CHANNELS:
            raise ValueError(f"{self.name} produced shape {tuple(out.shape)}")
        if out.shape[2] != out.shape[3]:
            raise ValueError(f"{self.name} produced non-square images")
        return out

    def component_label(self, images: torch.Tensor) -> np.ndarray:
        """Nearest declared prototype.  Oracle diagnostic only."""
        if self.prototypes is None:
            raise ValueError(f"{self.name} has no oracle components")
        flat = images.reshape(len(images), -1)
        proto = self.prototypes.reshape(len(self.prototypes), -1)
        distance = torch.cdist(flat, proto)
        return distance.argmin(dim=1).numpy()


def _grid() -> tuple[np.ndarray, np.ndarray]:
    axis = (np.arange(SIZE) - (SIZE - 1) / 2) / (SIZE / 2)
    return np.meshgrid(axis, axis, indexing="ij")


def _finish(images: np.ndarray, rng: np.random.Generator,
            noise: float) -> torch.Tensor:
    out = images + rng.normal(size=images.shape) * noise
    return torch.tensor(np.clip(out, -1.5, 1.5), dtype=torch.float32)


def _prototypes(build: Callable[[int], np.ndarray], count: int
                ) -> torch.Tensor:
    return torch.tensor(np.stack([build(k) for k in range(count)]),
                        dtype=torch.float32)


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


def checkerboard(noise: float = 0.10, cell: int = 4) -> ImageTarget:
    """Two checkerboard phases; a global-arrangement probe with equal
    marginal colour histograms."""

    def build(phase: int) -> np.ndarray:
        idx = np.arange(SIZE) // cell
        board = ((idx[:, None] + idx[None, :] + phase) % 2) * 2.0 - 1.0
        return np.stack([board, board, -board])

    protos = _prototypes(build, 2)

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        phase = rng.integers(0, 2, n)
        base = protos.numpy()[phase]
        return _finish(base, rng, noise)

    return ImageTarget("checkerboard", sampler, "arrangement", 2, protos,
                       np.array([0.5, 0.5]))


def pinwheel(blades: int = 5, noise: float = 0.08) -> ImageTarget:
    """Radial spokes at a few discrete rotations."""
    yy, xx = _grid()
    radius = np.sqrt(xx ** 2 + yy ** 2)
    angle = np.arctan2(yy, xx)

    def build(k: int) -> np.ndarray:
        shifted = angle + 2 * np.pi * k / 4 + 1.6 * radius
        spoke = np.cos(blades * shifted)
        body = np.tanh(3.0 * spoke) * np.exp(-2.0 * radius ** 2)
        return np.stack([body, np.roll(body, 1, axis=0), -body])

    protos = _prototypes(build, 4)

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        which = rng.integers(0, 4, n)
        return _finish(protos.numpy()[which], rng, noise)

    return ImageTarget("pinwheel", sampler, "manifold", 4, protos,
                       np.full(4, 0.25))


def rings_islands(noise: float = 0.07) -> ImageTarget:
    """A ring plus one disconnected bright island in a declared corner."""
    yy, xx = _grid()
    radius = np.sqrt(xx ** 2 + yy ** 2)
    ring = np.exp(-((radius - 0.65) ** 2) / (2 * 0.10 ** 2)) * 2 - 1
    corners = [(3, 3), (3, 12), (12, 3), (12, 12)]

    def build(k: int) -> np.ndarray:
        image = np.stack([ring, ring * 0.2, -ring])
        row, col = corners[k]
        image[:, row - 1:row + 2, col - 1:col + 2] = 1.0
        return image

    protos = _prototypes(build, 4)

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        which = rng.integers(0, 4, n)
        return _finish(protos.numpy()[which], rng, noise)

    return ImageTarget("rings_islands", sampler, "hole", 4, protos,
                       np.full(4, 0.25))


def _grating(orientation: float, frequency: float, size: int) -> np.ndarray:
    axis = np.arange(size) - (size - 1) / 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    proj = xx * np.cos(orientation) + yy * np.sin(orientation)
    return np.cos(2 * np.pi * frequency * proj / size)


def texture_blocks(noise: float = 0.06) -> ImageTarget:
    """Four quadrants of oriented texture; the assignment is the component.

    High-frequency content lives here, so a method that only matches coarse
    statistics is visibly wrong.
    """
    half = SIZE // 2
    styles = [(0.0, 3.0), (np.pi / 2, 3.0), (np.pi / 4, 5.0),
              (-np.pi / 4, 5.0)]

    def build(k: int) -> np.ndarray:
        order = [(k + j) % 4 for j in range(4)]
        image = np.zeros((CHANNELS, SIZE, SIZE))
        for index, (row, col) in enumerate(
                [(0, 0), (0, half), (half, 0), (half, half)]):
            orientation, frequency = styles[order[index]]
            patch = _grating(orientation, frequency, half)
            image[:, row:row + half, col:col + half] = patch[None]
            image[index % CHANNELS, row:row + half, col:col + half] *= 0.4
        return image

    protos = _prototypes(build, 4)

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        which = rng.integers(0, 4, n)
        return _finish(protos.numpy()[which], rng, noise)

    return ImageTarget("texture_blocks", sampler, "texture", 4, protos,
                       np.full(4, 0.25))


_LAYOUT_PATCHES = None


def _layout_patches() -> np.ndarray:
    """Four fixed 8x8 RGB patches shared by every arrangement."""
    global _LAYOUT_PATCHES
    if _LAYOUT_PATCHES is None:
        half = SIZE // 2
        patches = []
        for index, (orientation, frequency) in enumerate(
                [(0.0, 2.0), (np.pi / 2, 2.0), (np.pi / 4, 4.0),
                 (0.0, 6.0)]):
            patch = _grating(orientation, frequency, half)
            rgb = np.stack([patch, patch * (1 - 0.3 * index),
                            -patch * (1 - 0.15 * index)])
            patches.append(rgb)
        _LAYOUT_PATCHES = np.stack(patches)
    return _LAYOUT_PATCHES


def _assemble_layout(order: tuple[int, ...]) -> np.ndarray:
    half = SIZE // 2
    patches = _layout_patches()
    image = np.zeros((CHANNELS, SIZE, SIZE))
    slots = [(0, 0), (0, half), (half, 0), (half, half)]
    for slot, patch_index in zip(slots, order):
        row, col = slot
        image[:, row:row + half, col:col + half] = patches[patch_index]
    return image


LAYOUT_ORDERS = ((0, 1, 2, 3), (1, 0, 3, 2))
LAYOUT_COLLISION_ORDERS = ((3, 2, 1, 0), (2, 3, 0, 1))


def patch_layout(noise: float = 0.06) -> ImageTarget:
    """Fixed patch multiset, declared global arrangements.

    Every arrangement has the *same* patch histogram, so a purely local or
    globally pooled statistic cannot tell the target from its collision
    partner in ``collision_suite``.
    """
    protos = torch.tensor(
        np.stack([_assemble_layout(o) for o in LAYOUT_ORDERS]),
        dtype=torch.float32)

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        which = rng.integers(0, len(LAYOUT_ORDERS), n)
        return _finish(protos.numpy()[which], rng, noise)

    return ImageTarget("patch_layout", sampler, "arrangement",
                       len(LAYOUT_ORDERS), protos,
                       np.full(len(LAYOUT_ORDERS), 1 / len(LAYOUT_ORDERS)))


def color_layout(noise: float = 0.07) -> ImageTarget:
    """Colour-swapped layouts: identical luminance, different channels."""
    half = SIZE // 2

    def build(k: int) -> np.ndarray:
        image = np.zeros((CHANNELS, SIZE, SIZE))
        top, bottom = (0, 2) if k == 0 else (2, 0)
        image[top, :half] = 1.0
        image[bottom, half:] = 1.0
        image[1] = -0.5
        return image

    protos = _prototypes(build, 2)

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        which = rng.integers(0, 2, n)
        return _finish(protos.numpy()[which], rng, noise)

    return ImageTarget("color_layout", sampler, "color", 2, protos,
                       np.full(2, 0.5))


def phase_structured(noise: float = 0.05) -> ImageTarget:
    """Localized blobs: a definite phase structure over a fixed spectrum.

    The collision partner in ``collision_suite`` keeps the power spectrum and
    randomizes the phase, which destroys the blobs.
    """
    yy, xx = _grid()
    centres = [(-0.4, -0.4), (0.4, 0.4), (-0.4, 0.4), (0.4, -0.4)]

    def build(k: int) -> np.ndarray:
        cy, cx = centres[k]
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 0.22 ** 2))
        body = blob * 2 - 1
        return np.stack([body, -body, body * 0.3])

    protos = _prototypes(build, 4)

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        which = rng.integers(0, 4, n)
        return _finish(protos.numpy()[which], rng, noise)

    return ImageTarget("phase_structured", sampler, "phase", 4, protos,
                       np.full(4, 0.25))


RARE_WEIGHT = 0.05


def rare_object(noise: float = 0.06) -> ImageTarget:
    """95% plain background, 5% background with a small bright object."""
    yy, xx = _grid()
    background = np.stack([0.4 * xx, 0.4 * yy, -0.4 * xx])

    def build(k: int) -> np.ndarray:
        image = background.copy()
        if k == 1:
            image[:, 6:10, 6:10] = 1.2
        return image

    protos = _prototypes(build, 2)
    weights = np.array([1.0 - RARE_WEIGHT, RARE_WEIGHT])

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        which = rng.choice(2, size=n, p=weights)
        return _finish(protos.numpy()[which], rng, noise)

    return ImageTarget("rare_object", sampler, "rare", 2, protos, weights)


def deformed(noise: float = 0.05) -> ImageTarget:
    """A base pattern under small translations and a mild smooth warp."""
    yy, xx = _grid()
    base = np.stack([
        np.tanh(3 * np.cos(4 * np.pi * xx)),
        np.tanh(3 * np.sin(4 * np.pi * yy)),
        np.tanh(2 * np.cos(3 * np.pi * (xx + yy))),
    ])

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        images = np.repeat(base[None], n, axis=0)
        shift_y = rng.integers(-2, 3, n)
        shift_x = rng.integers(-2, 3, n)
        warped = np.empty_like(images)
        for i in range(n):
            rolled = np.roll(images[i], (shift_y[i], shift_x[i]), axis=(1, 2))
            amplitude = rng.normal(scale=0.12)
            warp = 1.0 + amplitude * np.cos(2 * np.pi * yy)
            warped[i] = rolled * warp[None]
        return _finish(warped, rng, noise)

    return ImageTarget("deformed", sampler, "deformation")


def suite() -> list[ImageTarget]:
    return [checkerboard(), pinwheel(), rings_islands(), texture_blocks(),
            patch_layout(), color_layout(), phase_structured(), rare_object(),
            deformed()]


def named(name: str) -> ImageTarget:
    for target in suite():
        if target.name == name:
            return target
    raise ValueError(f"unknown target {name!r}")
