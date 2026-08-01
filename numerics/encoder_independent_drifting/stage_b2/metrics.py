"""Stage-local evaluation metrics and safety vetoes for B2.

Keeping these small helpers local avoids importing the historical F1/B1
execution graphs into the B2 artifact boundary merely to compute diagnostics.
Inception remains report-only.
"""

from __future__ import annotations

import numpy as np
import torch

from ..appearance import precision_recall, spectrum_slope
from ..fid import frechet_from_features, inception_features, kid_from_features


def _self_distances(images: torch.Tensor) -> torch.Tensor:
    flat = images.reshape(len(images), -1).double()
    squared = (flat * flat).sum(dim=1, keepdim=True)
    distances = (squared + squared.T - 2.0 * flat @ flat.T).clamp_min(0.0).sqrt()
    diagonal = torch.eye(len(flat), dtype=torch.bool, device=flat.device)
    return distances.masked_fill(diagonal, float("inf"))


def effective_rank(images: torch.Tensor) -> float:
    flat = images.reshape(len(images), -1).double()
    values = torch.linalg.svdvals(flat - flat.mean(dim=0, keepdim=True)).square()
    total = float(values.sum())
    return 0.0 if total <= 0 else total**2 / float(values.square().sum())


def duplicate_rate(images: torch.Tensor, scale: float) -> float:
    nearest = _self_distances(images).min(dim=1).values
    return float((nearest < 0.05 * scale).to(torch.float32).mean())


def nn_diversity(images: torch.Tensor) -> float:
    choices = _self_distances(images).argmin(dim=1)
    return float(torch.unique(choices).numel()) / len(images)


def nearest_reference_distances(
    images: torch.Tensor,
    reference: torch.Tensor,
    device: torch.device | str,
    chunk: int = 2_048,
) -> tuple[torch.Tensor, torch.Tensor]:
    queries = images.detach().to(device).reshape(len(images), -1).float()
    qnorm = queries.square().sum(dim=1, keepdim=True)
    best = torch.full((len(queries),), float("inf"), device=device)
    claimed = torch.full((len(queries),), -1, dtype=torch.long, device=device)
    for start in range(0, len(reference), chunk):
        block = (
            reference[start : start + chunk]
            .to(device)
            .reshape(-1, queries.shape[1])
            .float()
        )
        bnorm = block.square().sum(dim=1).unsqueeze(0)
        squared = (qnorm + bnorm - 2.0 * queries @ block.T).clamp_min_(0.0)
        values, positions = squared.min(dim=1)
        improve = values < best
        best[improve] = values[improve]
        claimed[improve] = positions[improve] + start
    return best.sqrt().cpu(), claimed.cpu()


def matched_real_metrics(
    control: torch.Tensor, reference_features: np.ndarray, device
) -> dict:
    features = inception_features(control.cpu(), device).double().numpy()
    result = precision_recall(features, reference_features)
    result["kid"] = kid_from_features(features, reference_features)
    return result


def generated_metrics(
    images: torch.Tensor,
    reference_features: np.ndarray,
    real_for_scale: torch.Tensor,
    device,
    *,
    include_fid: bool = False,
) -> dict:
    features = inception_features(images.cpu(), device).double().numpy()
    pr = precision_recall(features, reference_features)
    scale = float(_self_distances(real_for_scale.cpu()).min(dim=1).values.median())
    spectrum_alpha = spectrum_slope(images.cpu())["alpha"]
    return {
        "recall": pr["recall"],
        "precision": pr["precision"],
        "pr_f1": pr["f1"],
        "kid": kid_from_features(features, reference_features),
        "fid": frechet_from_features(features, reference_features)
        if include_fid
        else None,
        "fid_is_small_sample_indicative": bool(include_fid),
        "effective_rank": effective_rank(images.cpu()),
        "duplicate_rate": duplicate_rate(images.cpu(), scale),
        "nn_diversity": nn_diversity(images.cpu()),
        "spectrum_alpha": spectrum_alpha,
        "second_moment": float(
            images.cpu().flatten(1).var(0).mean()
            / real_for_scale.cpu().flatten(1).var(0).mean().clamp_min(1e-12)
        ),
        "outside_fraction": float(((images < -1.0) | (images > 1.0)).float().mean()),
        "sample_outside_fraction": float(
            ((images < -1.0) | (images > 1.0)).flatten(1).any(dim=1).float().mean()
        ),
        "minimum": float(images.min()),
        "maximum": float(images.max()),
        "rms": float(images.square().mean().sqrt()),
        "samples_generated": len(images),
        "samples_reference": len(reference_features),
    }


def memorization_statistics_augmented(
    images: torch.Tensor,
    train: torch.Tensor,
    normalizer: float,
    device: torch.device | str,
) -> dict:
    if normalizer <= 0:
        raise ValueError("memorization normalizer must be positive")
    direct_distance, direct_claim = nearest_reference_distances(images, train, device)
    flipped_distance, flipped_claim = nearest_reference_distances(
        images, torch.flip(train, dims=(-1,)), device
    )
    use_flip = flipped_distance < direct_distance
    distances = torch.where(use_flip, flipped_distance, direct_distance)
    claimed = torch.where(use_flip, flipped_claim + len(train), direct_claim)
    unique = torch.unique(claimed)
    return {
        "nearest_train_or_flip_normalized": float(distances.median()) / normalizer,
        "nearest_train_or_flip_p05_normalized": float(
            np.percentile(distances.numpy(), 5)
        )
        / normalizer,
        "distinct_augmented_train_claimed": int(unique.numel()),
        "claimed_augmented_train_fraction": float(unique.numel()) / len(images),
        "flip_claim_fraction": float(use_flip.float().mean()),
    }


def apply_vetoes(metrics: dict, memorization: dict, thresholds: dict) -> dict:
    comparisons = {
        "effective_rank": {
            "value": float(metrics["effective_rank"]),
            "threshold": float(thresholds["effective_rank"]),
        },
        "one_minus_duplicate_rate": {
            "value": 1.0 - float(metrics["duplicate_rate"]),
            "threshold": float(thresholds["one_minus_duplicate_rate"]),
        },
        "nn_diversity": {
            "value": float(metrics["nn_diversity"]),
            "threshold": float(thresholds["nn_diversity"]),
        },
        "nearest_train_or_flip_normalized": {
            "value": float(memorization["nearest_train_or_flip_normalized"]),
            "threshold": float(thresholds["nearest_train_or_flip_normalized"]),
        },
    }
    for item in comparisons.values():
        item["passes"] = bool(item["value"] >= item["threshold"])
    return {
        "comparisons": comparisons,
        "passes": bool(all(item["passes"] for item in comparisons.values())),
    }
