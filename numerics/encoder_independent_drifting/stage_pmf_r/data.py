"""Strictly train-only CIFAR automobile loader for developmental S3R."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

from ..stage_pmf.config import AUTOMOBILE_LABEL

DEFAULT_ROOT = os.path.expanduser("~/.cache/cifar")


@lru_cache(maxsize=4)
def automobile_train_pool(root: str | None = None) -> torch.Tensor:
    """Open only the official training archive and return automobiles in [-1,1]."""
    from torchvision.datasets import CIFAR10

    train_set = CIFAR10(root=root or DEFAULT_ROOT, train=True, download=False)
    labels = np.asarray(train_set.targets)
    indices = np.flatnonzero(labels == AUTOMOBILE_LABEL)
    if len(indices) != 5_000:
        raise RuntimeError(f"unexpected CIFAR training automobile count: {len(indices)}")
    return (
        torch.as_tensor(train_set.data[indices], dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .div(127.5)
        .sub(1.0)
        .contiguous()
    )
