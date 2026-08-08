"""Hardware identity checks shared by every CAP-EMF-2 execution stage."""

from __future__ import annotations

import os
import platform

import torch


def hardware_binding(
    device: torch.device, expected_gpu_name: str | None
) -> dict[str, object]:
    """Return an auditable binding to the declared production GPU.

    A generic ``cuda`` check is not enough: the numerical finite-difference
    audit and cost benchmark must run on the same device family as training.
    Matching is deliberately a case-insensitive substring so provider labels
    such as ``NVIDIA GeForce RTX 4090`` can be bound by ``RTX 4090``.
    """
    expected = expected_gpu_name.strip() if expected_gpu_name else None
    if device.type != "cuda":
        return {
            "torch_device": str(device),
            "actual_gpu_name": None,
            "expected_gpu_name_substring": expected,
            "matches": False,
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
    properties = torch.cuda.get_device_properties(device)
    actual = properties.name
    matches = expected is not None and expected.casefold() in actual.casefold()
    return {
        "torch_device": str(device),
        "actual_gpu_name": actual,
        "expected_gpu_name_substring": expected,
        "matches": matches,
        "compute_capability": f"sm_{properties.major}{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def require_same_hardware(device: torch.device, admitted: dict) -> dict:
    """Reject a screen launched on hardware unlike the admitted device."""
    live = hardware_binding(device, admitted.get("expected_gpu_name_substring"))
    if not live["matches"]:
        raise RuntimeError(
            "live device does not match the production GPU declared at admission: "
            f"{live}"
        )
    required_equal = (
        "actual_gpu_name",
        "compute_capability",
        "torch_version",
        "cuda_runtime",
        "cudnn_version",
        "cublas_workspace_config",
    )
    changed = {
        key: {"admitted": admitted.get(key), "live": live.get(key)}
        for key in required_equal
        if live.get(key) != admitted.get(key)
    }
    if changed:
        raise RuntimeError(
            f"live numerical environment differs from admission: {changed}"
        )
    return live
