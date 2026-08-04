"""Unconditional CIFAR-10 pools, with the test split sealed by construction.

The training process must never instantiate the test split.  That is enforced
here rather than by convention: :func:`sealed_test_pool` refuses to run without
an explicit acknowledgement argument, so an accidental import cannot open it,
and the training module never calls it.

**The reference set for FID is the 50,000 training images**, which is the
standard CIFAR-10 convention and the reason this target was chosen over a
single class: a 5,000-image class forces a small-sample-biased FID and, at 410+
epochs, makes memorization the likely route to coherent-looking samples.  The
test split is kept for an honest held-out check, not for the headline number.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from .config import SEALED_TEST_POOL_SIZE, TRAIN_POOL_SIZE

DEFAULT_ROOT = os.path.expanduser("~/.cache/cifar")


def _pool(train: bool, root: str | None, expected: int) -> torch.Tensor:
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=root or DEFAULT_ROOT, train=train, download=False)
    if len(dataset.data) != expected:
        split = "train" if train else "test"
        raise RuntimeError(
            f"unexpected CIFAR-10 {split} count: {len(dataset.data)} != {expected}"
        )
    return (
        torch.as_tensor(dataset.data, dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .div(127.5)
        .sub(1.0)
        .contiguous()
    )


@lru_cache(maxsize=2)
def cifar10_train_pool(root: str | None = None) -> torch.Tensor:
    """The 50,000 official training images, all classes, scaled to [-1, 1].

    Also the FID reference set, following the standard CIFAR-10 protocol.
    """
    return _pool(True, root, TRAIN_POOL_SIZE)


def sealed_test_pool(
    root: str | None = None, *, acknowledge_sealed: bool = False
) -> torch.Tensor:
    """The 10,000 test images.  Refuses to open without acknowledgement.

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
