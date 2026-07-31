"""CIFAR-10 as an `ImageTarget`, with disjoint train/eval splits.

The regime search in `EncoderIndependentSecondPassAudit.md` found that no
synthetic testbed reachable here makes pixel geometry fail, while on CIFAR-10
it does: pixel k-NN content accuracy is .267 against chance .100, and fixed
wavelet geometry reaches .390.  CIFAR-16 is also admissible at 300 steps.
This module makes it a first-class target for Phase 2.

Two properties matter and are enforced here rather than left to a runner:

*Disjoint splits.*  A finite pool sampled with replacement would let an arm
memorize images that later appear in the evaluation pool.  The 50,000
training images are partitioned by index once: arms and earlier calibration
see `train` only; earlier evaluation, null, and support-calibration pools see
`eval` only.  Stage B1 additionally exposes torchvision's separate official
10,000-image `test` split for a fresh confirmation allocation.

*No labels.*  Class labels exist in the dataset and are used by nothing here
-- not an objective, not a controller, not a metric.  They were used in the
audit as an oracle diagnostic for measuring representations, which is a
different activity from training.

No pretrained network is involved at any point.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

from .datasets import ImageTarget

# Frozen split boundary (protocol section 2).  ``test`` is torchvision's
# separate official 10,000-image test set; it is first consumed by B1 and was
# not present in any B0 training, development, calibration, or confirmation
# allocation.
TRAIN_SPLIT = (0, 40_000)
EVAL_SPLIT = (40_000, 50_000)
TEST_SPLIT = (0, 10_000)
SPLITS = {"train": TRAIN_SPLIT, "eval": EVAL_SPLIT, "test": TEST_SPLIT}
DEFAULT_ROOT = os.path.expanduser("~/.cache/cifar")


@lru_cache(maxsize=8)
def cifar_pool(resolution: int, split: str,
               root: str | None = None) -> torch.Tensor:
    """Downsampled CIFAR-10 images for one split, cached, in ``[-1, 1]``."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected {sorted(SPLITS)}")
    if resolution < 4 or resolution > 32:
        raise ValueError("resolution must lie in [4, 32]")
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(
        root=root or DEFAULT_ROOT, train=(split != "test"), download=False)
    start, stop = SPLITS[split]
    data = dataset.data[start:stop]
    images = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
    images = images / 127.5 - 1.0
    if resolution != images.shape[-1]:
        images = torch.nn.functional.interpolate(
            images, size=(resolution, resolution), mode="area")
    return images.contiguous()


def cifar_target(resolution: int = 16, split: str = "train",
                 root: str | None = None) -> ImageTarget:
    """An `ImageTarget` drawing with replacement from one CIFAR split.

    Has no `prototypes`, so every oracle-only diagnostic (component
    occupancy) is skipped automatically -- CIFAR has no declared component
    structure this program is entitled to use.
    """
    pool = cifar_pool(resolution, split, root)

    def sampler(n: int, rng: np.random.Generator) -> torch.Tensor:
        return pool[rng.integers(0, len(pool), n)]

    return ImageTarget(f"cifar{resolution}_{split}", sampler, "natural")


def available(root: str | None = None) -> bool:
    """Is the dataset present locally?  Phase 2 refuses to guess."""
    path = os.path.join(root or DEFAULT_ROOT, "cifar-10-batches-py")
    return os.path.isdir(path)
