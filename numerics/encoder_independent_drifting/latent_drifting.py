"""Drifting inside a self-trained latent space.

The field, kernel, R11 teacher and stop-gradient regression are unchanged --
this module only changes the *space* they act in.  Phase 22 located the
pixel-space ceiling in the arithmetic of the teacher: it is a kernel-weighted
average of the real batch and never averages fewer than ~18 images at any
viable setting, so it cannot be a sharp image.  In a learned code space the
same average decodes back onto the image manifold, which is why Phase 24
measured a d=512 ceiling of KID 0.031 with recognizable objects against the
best pixel arm's 0.131.

**The mechanism claim is testable before training anything.**  Both Phase-22
walls were attributed to distance concentration: in 3072 dimensions from
64-256 samples every positive is roughly equidistant from every cloud point,
so the kernel cannot be selective, and shrinking the bandwidth to force
selectivity destroys the estimate instead of sharpening it.  If that is right,
the *same* target-ESS calibration should realize markedly better selectivity in
a 256- or 512-dimensional code space.  :func:`concentration_report` measures it
directly, and it is recorded at the start of every run so the claim is never
merely asserted.
"""

from __future__ import annotations

import numpy as np
import torch

from .autoencoder import LATENT_GRID
from .config import GeometryConfig
from .fixed_features import build_family
from .kernels import calibrate_block_kernel, median_ess_fraction


def raw_branch(blocks: int = 1):
    """Identity-feature branch, whatever the ambient dimension.

    The ``raw`` family flattens its input and does nothing else, so the same
    branch serves pixels and codes.  Keeping the geometry identical across
    spaces is what makes the comparison a statement about dimension rather
    than about feature engineering.
    """
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace")
    return build_family(geometry, blocks).branches[0]


def distance_spread(samples: torch.Tensor) -> dict:
    """Coefficient of variation of pairwise distances.

    The direct measure of concentration.  As dimension grows with a fixed
    sample size, pairwise distances converge to a common value and the CV
    falls toward zero -- at which point no bandwidth can separate near from
    far, because there is no near and far left to separate.
    """
    flat = samples.reshape(len(samples), -1).double()
    squared = (flat * flat).sum(dim=1, keepdim=True)
    gram = squared + squared.T - 2.0 * (flat @ flat.T)
    distances = gram.clamp_min(0.0).sqrt()
    mask = ~torch.eye(len(flat), dtype=torch.bool, device=flat.device)
    off = distances[mask]
    mean = float(off.mean())
    return {"distance_mean": mean, "distance_std": float(off.std()),
            "distance_cv": float(off.std()) / max(mean, 1e-12),
            "dimension": int(flat.shape[1]), "samples": len(flat)}


def concentration_report(pixels: torch.Tensor, codes: torch.Tensor,
                         target_ess: float, positives: int) -> dict:
    """Does a code space actually escape the concentration that capped pixels?

    Calibrates the identical kernel to the identical target ESS in both spaces
    and reports what selectivity is actually realized on a batch of the size
    the field uses.  A code space that concentrates just as badly would predict
    latent drifting inherits the pixel-space walls.
    """
    out = {}
    for name, samples in (("pixel", pixels), ("code", codes)):
        branch = raw_branch()
        kernel = calibrate_block_kernel(
            branch, samples, "smooth_laplace", 0.5, 1.0, 1e-3,
            combine="sum", target_ess_fraction=target_ess)
        batch = samples[:positives]
        realized = median_ess_fraction(kernel, branch.blocks(batch))
        out[name] = {**distance_spread(samples),
                     "target_ess": float(target_ess),
                     "realized_ess": float(realized),
                     "images_averaged": float(realized * len(batch)),
                     "bandwidth_median": float(kernel.taus.median())}
    out["cv_ratio_code_over_pixel"] = (
        out["code"]["distance_cv"] / max(out["pixel"]["distance_cv"], 1e-12))
    out["realized_ess_drop"] = (
        out["pixel"]["realized_ess"] - out["code"]["realized_ess"])
    out["code_is_less_concentrated"] = bool(
        out["code"]["distance_cv"] > out["pixel"]["distance_cv"])
    return out


def encode_target(model, target, count: int, rng) -> torch.Tensor:
    """A batch of real images, as codes. Flattened to match the field's view."""
    with torch.no_grad():
        return model.encode(target.sample(count, rng)).flatten(1)


def decode_codes(model, codes: torch.Tensor, batch: int = 256
                 ) -> torch.Tensor:
    """Codes back to images, in chunks so a large eval pool fits in memory."""
    out = []
    with torch.no_grad():
        for start in range(0, len(codes), batch):
            chunk = codes[start:start + batch]
            out.append(model.decode(chunk).cpu())
    return torch.cat(out, dim=0)


def code_shape(model) -> tuple[int, int]:
    return int(model.latent_channels), int(LATENT_GRID)


def summarize_history(history: list[dict]) -> dict:
    if not history:
        return {}
    values = [h["mse"] for h in history]
    return {"ae_mse_first": float(values[0]), "ae_mse_final": float(values[-1]),
            "ae_mse_min": float(np.min(values))}
