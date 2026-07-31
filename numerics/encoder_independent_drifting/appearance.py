"""Measures that separate *quality* from *coverage*, and texture from blur.

Phase 22 found the metric ordering opposed to the visual ordering:
`C_sharper` scored worst on KID, FID and the spectral tail and produced the
most photograph-like samples in the program, while the KID/FID winners
produced garish coloured worm patterns.  The second moments ruled out a scale
effect (all six arms at 0.97-1.00), so the split is one of **rank**:

    real CIFAR   tail 0.1962   ~80% of variance in the top 32 PCs
    A_control    tail 0.1611   closest to real, garish samples
    C_sharper    tail 0.0015   a ~32-dim family, plausible blurry samples

That is high precision / low recall against low precision / high recall.  KID
and FID are distribution-level moment matches and reward coverage, so they
rank A above C.  **This program has computed no precision or recall in 22
phases**, which is exactly why such a trade-off could sit behind every
configuration decision undetected.

Two measures live here.

`precision_recall`  Kynkaanniemi et al. 2019.  Estimates each set's manifold
    as a union of balls -- one per sample, radius the distance to that
    sample's k-th nearest neighbour within its own set -- and asks what
    fraction of the other set falls inside.  Precision is the share of
    generated samples landing in the real manifold (quality); recall is the
    share of real samples landing in the generated manifold (coverage).
    Unlike FID they are separately readable, which is the whole point.

`spectrum_slope`  Natural images have a radially-averaged power spectrum
    close to a power law, P(f) ~ f^-alpha.  Blur steepens the slope;
    high-frequency texture flattens it.  This discriminates "blurry
    photograph" from "garish texture" directly, which no statistic currently
    in the program does, and it needs no learned network.

    **Calibrate against the data, never against a textbook value.**  Full
    resolution natural images sit near alpha = 2, but CIFAR-10 at 32x32 is
    itself a heavy downsample of larger photographs and measures **alpha =
    3.60** on this program's eval split.  Only the gap to the real split is
    meaningful; the absolute number is a property of the dataset.

**A property of precision that has to travel with it.**  Precision rewards
*typicality*, not realism.  A moment-matched Gaussian -- structureless, and
the distribution that saturated ED2 -- scores precision **0.923** against real
data's 0.739, because it concentrates in the dense centre of the feature cloud
while real data spends some of its mass in the sparse tails.  Its recall is
0.001.  So precision alone cannot be read as "looks like data"; it is only
interpretable **paired with recall**, and a high-precision/near-zero-recall
result is the signature of a distribution that has collapsed onto the typical
set.  Validated in `diagnose_phase23.py`.
"""

from __future__ import annotations

import numpy as np
import torch

# Declared, not tuned.  k = 3 is the value Kynkaanniemi et al. use throughout.
NEIGHBOURS = 3
# Distance chunk, so an n^2 matrix never has to exist at once.
_CHUNK = 1024
# Radial band for the spectrum fit.  DC and the two lowest bins carry the
# image mean and are dominated by composition rather than texture; the bins
# nearest Nyquist carry interpolation and aliasing artifacts.  Both ends are
# excluded before fitting, at fixed indices declared here.
_SPECTRUM_LOW = 2
_SPECTRUM_HIGH_MARGIN = 2


def _knn_radii(features: np.ndarray, k: int) -> np.ndarray:
    """Distance from each sample to its k-th nearest neighbour in its own set.

    The sample itself is excluded, so ``k`` counts genuine neighbours.
    """
    n = len(features)
    if n <= k:
        raise ValueError(f"need more than k={k} samples, got {n}")
    radii = np.empty(n, dtype=np.float64)
    squared = (features ** 2).sum(axis=1)
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        block = features[start:stop]
        distance = (squared[start:stop, None] + squared[None, :]
                    - 2.0 * block @ features.T)
        np.maximum(distance, 0.0, out=distance)
        # Exclude self by pushing the diagonal out of the running.
        rows = np.arange(stop - start)
        distance[rows, rows + start] = np.inf
        partitioned = np.partition(distance, k - 1, axis=1)[:, k - 1]
        radii[start:stop] = np.sqrt(partitioned)
    return radii


def _fraction_within(query: np.ndarray, reference: np.ndarray,
                     radii: np.ndarray) -> float:
    """Share of ``query`` falling inside any reference ball."""
    squared_query = (query ** 2).sum(axis=1)
    squared_reference = (reference ** 2).sum(axis=1)
    inside = np.zeros(len(query), dtype=bool)
    for start in range(0, len(query), _CHUNK):
        stop = min(start + _CHUNK, len(query))
        distance = (squared_query[start:stop, None] + squared_reference[None, :]
                    - 2.0 * query[start:stop] @ reference.T)
        np.maximum(distance, 0.0, out=distance)
        inside[start:stop] = (np.sqrt(distance) <= radii[None, :]).any(axis=1)
    return float(inside.mean())


def precision_recall(generated: np.ndarray, real: np.ndarray,
                     k: int = NEIGHBOURS) -> dict:
    """Improved precision and recall from cached features.

    ``precision`` is the fraction of generated samples inside the real
    manifold -- how many look like plausible data.  ``recall`` is the fraction
    of real samples inside the generated manifold -- how much of the data the
    model covers.  A model can score well on one and badly on the other, which
    is the failure mode FID and KID collapse into a single number.
    """
    generated = np.asarray(generated, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    real_radii = _knn_radii(real, k)
    generated_radii = _knn_radii(generated, k)
    precision = _fraction_within(generated, real, real_radii)
    recall = _fraction_within(real, generated, generated_radii)
    harmonic = (0.0 if precision + recall <= 0 else
                2.0 * precision * recall / (precision + recall))
    return {"precision": precision, "recall": recall, "f1": harmonic,
            "k": int(k), "samples_generated": len(generated),
            "samples_real": len(real)}


def _radial_power(images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged power spectrum, averaged over images and channels."""
    grey = images.mean(dim=1) if images.dim() == 4 else images
    size = grey.shape[-1]
    spectrum = torch.fft.fftshift(torch.fft.fft2(grey.double()), dim=(-2, -1))
    power = (spectrum.abs() ** 2).mean(dim=0).cpu().numpy()
    centre = size // 2
    y, x = np.ogrid[:size, :size]
    radius = np.sqrt((y - centre) ** 2 + (x - centre) ** 2)
    index = radius.astype(int)
    counts = np.bincount(index.ravel(), minlength=centre + 1)
    totals = np.bincount(index.ravel(), weights=power.ravel(),
                         minlength=centre + 1)
    keep = counts > 0
    frequencies = np.arange(len(counts))[keep]
    return frequencies, totals[keep] / counts[keep]


def spectrum_slope(images: torch.Tensor) -> dict:
    """Power-law exponent of the radially-averaged power spectrum.

    Fits ``log P = -alpha log f + c`` over a fixed mid-frequency band.  A
    natural-image ensemble sits near ``alpha = 2``; blur raises it (energy
    lost at high frequency), added texture lowers it.  The band is declared in
    module constants and never fitted to an outcome.
    """
    frequencies, power = _radial_power(images)
    high = len(frequencies) - _SPECTRUM_HIGH_MARGIN
    band = slice(_SPECTRUM_LOW, high)
    log_f = np.log(frequencies[band].astype(np.float64))
    log_p = np.log(np.clip(power[band], 1e-30, None))
    if len(log_f) < 3:
        raise ValueError("image too small for a spectrum fit")
    slope, intercept = np.polyfit(log_f, log_p, 1)
    residual = log_p - (slope * log_f + intercept)
    return {"alpha": float(-slope), "intercept": float(intercept),
            "fit_residual_rms": float(np.sqrt((residual ** 2).mean())),
            "band_low": int(_SPECTRUM_LOW), "band_high": int(high)}


def spectrum_report(generated: torch.Tensor, real: torch.Tensor) -> dict:
    """Spectrum slopes for both sides, and the gap that matters."""
    a = spectrum_slope(generated)
    b = spectrum_slope(real)
    return {"alpha_generated": a["alpha"], "alpha_real": b["alpha"],
            "alpha_gap": a["alpha"] - b["alpha"],
            "generated": a, "real": b}
