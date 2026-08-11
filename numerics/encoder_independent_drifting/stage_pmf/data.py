"""The sealed one-class CIFAR-10 automobile task for local pMF S3."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from .config import AUTOMOBILE_LABEL

DEFAULT_ROOT = os.path.expanduser("~/.cache/cifar")


@dataclass(frozen=True)
class AutomobileData:
    train: torch.Tensor
    test: torch.Tensor
    train_source_indices: torch.Tensor
    test_source_indices: torch.Tensor


def _digest(images: torch.Tensor) -> str:
    return hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()


@lru_cache(maxsize=4)
def automobile_data(root: str | None = None) -> AutomobileData:
    """Return official train/test automobiles in ``[-1,1]``.

    Labels are used once to define the scientific target and never supplied
    to the generator or objective.  The official CIFAR train and test files
    are distinct provenance domains; the test class remains report-only.
    """
    from torchvision.datasets import CIFAR10

    location = root or DEFAULT_ROOT
    train_set = CIFAR10(root=location, train=True, download=False)
    test_set = CIFAR10(root=location, train=False, download=False)
    train_labels = np.asarray(train_set.targets)
    test_labels = np.asarray(test_set.targets)
    train_indices = np.flatnonzero(train_labels == AUTOMOBILE_LABEL)
    test_indices = np.flatnonzero(test_labels == AUTOMOBILE_LABEL)
    if len(train_indices) != 5_000 or len(test_indices) != 1_000:
        raise RuntimeError(
            "unexpected CIFAR automobile counts: "
            f"train={len(train_indices)} test={len(test_indices)}"
        )

    def convert(data: np.ndarray, indices: np.ndarray) -> torch.Tensor:
        return (
            torch.as_tensor(data[indices], dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .div(127.5)
            .sub(1.0)
            .contiguous()
        )

    return AutomobileData(
        train=convert(train_set.data, train_indices),
        test=convert(test_set.data, test_indices),
        train_source_indices=torch.as_tensor(train_indices, dtype=torch.long),
        test_source_indices=torch.as_tensor(test_indices, dtype=torch.long),
    )


def manifest(data: AutomobileData) -> dict:
    return {
        "dataset": "CIFAR-10",
        "class_label": AUTOMOBILE_LABEL,
        "class_name": "automobile",
        "train_count": len(data.train),
        "test_count": len(data.test),
        "train_sha256": _digest(data.train),
        "test_sha256": _digest(data.test),
        "split_rule": "official train versus official test; label used only for target selection",
    }
