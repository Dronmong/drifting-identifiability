"""Loader for an explicitly external, never-before-used B2 confirmation pool."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .artifacts import file_sha256


def load_fresh_pool(
    path: Path,
    source_id: str,
    image_size: int,
    float_encoding: str = "auto",
) -> tuple[torch.Tensor, dict]:
    """Load NCHW/NHWC images and bind their bytes to a declared source ID.

    ``.npz`` inputs must contain an ``images`` array; ``.npy`` inputs are read
    directly.  Byte images are mapped to ``[-1,1]``.  Floating inputs must
    already lie in that range.  The caller, not this loader, is responsible for
    documenting the external dataset's provenance; the file hash makes later
    substitution detectable.
    """
    if not source_id.strip():
        raise ValueError("fresh B2 data needs a nonempty source ID")
    if float_encoding not in ("auto", "minus-one-one", "zero-one"):
        raise ValueError("float encoding must be auto, minus-one-one, or zero-one")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "images" not in payload:
                raise ValueError("B2 NPZ must contain an 'images' array")
            images = np.asarray(payload["images"])
    elif path.suffix.lower() == ".npy":
        images = np.asarray(np.load(path, allow_pickle=False))
    else:
        raise ValueError("fresh B2 pool must be .npy or .npz")
    if images.ndim != 4:
        raise ValueError("fresh B2 images must be a rank-four array")
    if images.shape[-1] in (1, 3) and images.shape[1] not in (1, 3):
        images = np.transpose(images, (0, 3, 1, 2))
    if images.shape[1:] != (3, image_size, image_size):
        raise ValueError(
            f"fresh B2 images have shape {images.shape}; expected N×3×"
            f"{image_size}×{image_size}"
        )
    original_dtype = str(images.dtype)
    if np.issubdtype(images.dtype, np.integer):
        if images.min() < 0 or images.max() > 255:
            raise ValueError("integer B2 images must lie in [0,255]")
        tensor = torch.from_numpy(images.astype(np.float32)) / 127.5 - 1.0
        applied_encoding = "integer-zero-255-to-minus-one-one"
    else:
        tensor = torch.from_numpy(images.astype(np.float32))
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("fresh B2 images contain non-finite values")
        if float_encoding == "auto":
            raise ValueError(
                "floating B2 images are normalization-ambiguous; declare "
                "--fresh-float-encoding minus-one-one or zero-one"
            )
        if float_encoding == "zero-one":
            if float(tensor.min()) < -1e-6 or float(tensor.max()) > 1.0001:
                raise ValueError("zero-one B2 images must lie in [0,1]")
            tensor = tensor * 2.0 - 1.0
            applied_encoding = "float-zero-one-to-minus-one-one"
        else:
            if float(tensor.min()) < -1.0001 or float(tensor.max()) > 1.0001:
                raise ValueError("minus-one-one B2 images must lie in [-1,1]")
            applied_encoding = "float-minus-one-one"
    return tensor.contiguous(), {
        "source_id": source_id,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "samples": len(tensor),
        "shape": list(tensor.shape),
        "input_dtype": original_dtype,
        "requested_float_encoding": float_encoding,
        "applied_encoding": applied_encoding,
        "normalized_range": [float(tensor.min()), float(tensor.max())],
        "reuse_attestation_required": (
            "operator must attest this source was not used for B0/B1 selection, "
            "B2 design, training, calibration, or threshold choice"
        ),
    }
