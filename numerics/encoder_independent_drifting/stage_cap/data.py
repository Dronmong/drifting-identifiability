"""CIFAR-10 automobile pools, with the test split sealed by construction.

The training process must never instantiate the test split.  That is enforced
here rather than by convention: :func:`sealed_test_pool` refuses to run without
an explicit acknowledgement argument, so an accidental import cannot open it,
and the training module never calls it.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

from .config import AUTOMOBILE_LABEL, SEALED_TEST_POOL_SIZE, TRAIN_POOL_SIZE

DEFAULT_ROOT = os.path.expanduser("~/.cache/cifar")


def _pool(train: bool, root: str | None, expected: int) -> torch.Tensor:
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=root or DEFAULT_ROOT, train=train, download=False)
    labels = np.asarray(dataset.targets)
    indices = np.flatnonzero(labels == AUTOMOBILE_LABEL)
    if len(indices) != expected:
        split = "train" if train else "test"
        raise RuntimeError(
            f"unexpected CIFAR {split} automobile count: {len(indices)} != {expected}"
        )
    return (
        torch.as_tensor(dataset.data[indices], dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .div(127.5)
        .sub(1.0)
        .contiguous()
    )


@lru_cache(maxsize=4)
def automobile_train_pool(root: str | None = None) -> torch.Tensor:
    """The 5,000 official training automobiles, scaled to [-1, 1]."""
    return _pool(True, root, TRAIN_POOL_SIZE)


def sealed_test_pool(
    root: str | None = None, *, acknowledge_sealed: bool = False
) -> torch.Tensor:
    """The 1,000 test automobiles.  Refuses to open without acknowledgement.

    Only the post-training sealed evaluation may call this, and only after the
    final checkpoint is frozen and hashed.  ``acknowledge_sealed=True`` is a
    deliberate act, not a default.
    """
    if not acknowledge_sealed:
        raise RuntimeError(
            "the CAP-EMF-1 test split is sealed; it may be opened only by the "
            "post-training evaluation, with acknowledge_sealed=True"
        )
    return _pool(False, root, SEALED_TEST_POOL_SIZE)


def flip_batch(images: torch.Tensor, flips: torch.Tensor) -> torch.Tensor:
    """Apply per-example horizontal flips from a recorded boolean mask."""
    if len(images) != len(flips):
        raise ValueError("one flip bit is required per image")
    flipped = torch.flip(images, dims=(-1,))
    return torch.where(flips[:, None, None, None].to(images.device), flipped, images)
