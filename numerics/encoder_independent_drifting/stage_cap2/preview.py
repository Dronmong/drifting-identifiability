"""Minimal, source-audited image-grid writer for CAP2 artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .artifacts import file_sha256


def save_fixed_grid(
    images: torch.Tensor, path: Path, *, rows: int, columns: int
) -> str:
    """Write the first ``rows * columns`` images without selection or sorting."""

    if rows <= 0 or columns <= 0:
        raise ValueError("grid dimensions must be positive")
    needed = rows * columns
    if len(images) < needed:
        raise ValueError("not enough images for the declared fixed grid")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("CAP2 preview images must have shape (N,3,H,W)")
    array = (
        ((images[:needed].detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    height, width = array.shape[1:3]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index in range(needed):
        row, column = divmod(index, columns)
        canvas[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = array[index]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)
    return file_sha256(path)
