"""Frechet Inception Distance -- the paper's own metric, now reachable.

Every phase of this program scored with energy distance and a normalized
composite.  Those are defensible and cheap, but they are not what the paper
reports, so no result here has ever been quotable against its FID ladder.
`inception_v3(weights=IMAGENET1K_V1)` downloads, so that changes.

**Read this before quoting a number.** FID is conventionally computed with
>= 10k samples per side; at the sample counts this program can afford it is
*indicative*, and biased upward, because the covariance estimate is noisy in
2048 dimensions with a few hundred samples.  :func:`frechet_distance`
therefore returns the sample count alongside the value and every report must
carry it.  Comparisons **between arms at the same sample count** are the
sound use; the absolute value is not comparable with published FIDs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_INCEPTION_INPUT = 299
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_MODEL: dict = {}


def inception_features(images: torch.Tensor, device=None,
                       batch: int = 64) -> torch.Tensor:
    """2048-dimensional pool features, from [-1, 1] images."""
    from torchvision.models import (                       # noqa: PLC0415
        Inception_V3_Weights, inception_v3,
    )
    device = device or images.device
    key = str(device)
    if key not in _MODEL:
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                             aux_logits=True)
        model.fc = torch.nn.Identity()
        _MODEL[key] = model.eval().requires_grad_(False).to(device)
    model = _MODEL[key]
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            chunk = images[start:start + batch].to(device)
            chunk = ((chunk + 1.0) * 0.5).clamp(0.0, 1.0)
            chunk = F.interpolate(chunk,
                                  size=(_INCEPTION_INPUT, _INCEPTION_INPUT),
                                  mode="bilinear", align_corners=False)
            out.append(model((chunk - mean) / std).float().cpu())
    return torch.cat(out, dim=0)


def _sqrtm(matrix: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root by eigendecomposition.

    `scipy.linalg.sqrtm` on a product of two covariances returns a complex
    array with small imaginary parts and is slow at 2048 dimensions.  The
    product `S1 @ S2` is similar to a symmetric PSD matrix, so this route is
    both faster and free of the imaginary-part cleanup that invites mistakes.
    """
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def frechet_from_features(a: np.ndarray, b: np.ndarray) -> float:
    """FID from cached 2048-d features.

    Split out from :func:`frechet_distance` so a diagnostic can score many
    subsamples without paying for the Inception forward pass each time.
    """
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a = np.cov(a, rowvar=False)
    cov_b = np.cov(b, rowvar=False)
    # Tr(S1 + S2 - 2 (S1 S2)^{1/2}), computed through the symmetric form
    # A^{1/2} S2 A^{1/2}, whose square root is PSD by construction.
    root_a = _sqrtm(cov_a)
    middle = _sqrtm(root_a @ cov_b @ root_a)
    value = (float(((mu_a - mu_b) ** 2).sum())
             + float(np.trace(cov_a) + np.trace(cov_b) - 2.0
                     * np.trace(middle)))
    return max(value, 0.0)


def kid_from_features(a: np.ndarray, b: np.ndarray) -> float:
    """Kernel Inception Distance: unbiased MMD^2, polynomial kernel.

    Binkowski et al. 2018 introduced KID precisely because FID is *biased*
    at small sample counts -- it estimates a covariance in 2048 dimensions,
    so with a few hundred samples both covariances are singular and the
    cross term is computed between nearly orthogonal subspaces.  This
    program's real-vs-real FID reads ~70 rather than 0 for that reason.

    KID has no covariance and no matrix square root.  The U-statistic below
    is unbiased at *every* sample count, so its expectation does not move
    when the sample count changes -- only its variance does.  It can go
    slightly negative on identical distributions, which is correct
    behaviour for an unbiased estimator and is deliberately not clipped.

    Kernel: ``k(x, y) = (x . y / d + 1)^3``, the standard KID choice.
    """
    d = a.shape[1]
    m, n = len(a), len(b)
    if m < 2 or n < 2:
        raise ValueError("unbiased KID needs >= 2 samples per side")
    kaa = (a @ a.T / d + 1.0) ** 3
    kbb = (b @ b.T / d + 1.0) ** 3
    kab = (a @ b.T / d + 1.0) ** 3
    return float((kaa.sum() - np.trace(kaa)) / (m * (m - 1))
                 + (kbb.sum() - np.trace(kbb)) / (n * (n - 1))
                 - 2.0 * kab.mean())


def frechet_distance(generated: torch.Tensor, real: torch.Tensor,
                     device=None) -> dict:
    """FID between two image sets, with the sample count it was computed at."""
    a = inception_features(generated, device).double().numpy()
    b = inception_features(real, device).double().numpy()
    return {"fid": frechet_from_features(a, b),
            "samples_generated": int(len(generated)),
            "samples_real": int(len(real)),
            "caveat": "indicative at this sample count; comparable between "
                      "arms measured identically, not with published FIDs"}
