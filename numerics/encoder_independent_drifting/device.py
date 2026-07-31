"""Device selection, with reproducibility preserved across CPU and GPU.

Every phase of this program was measured on CPU (`torch==2.7.1`, the
CPU-only wheel).  Moving to a CUDA build changes accumulation order, cuDNN
algorithm selection and — on Ampere and later — enables TF32 matmuls by
default, so results shift.  That is why the device is *declared* and
recorded in provenance rather than picked up silently, and why
:func:`equivalence_report` exists.

**The reproducibility rule: random numbers are drawn on the CPU and then
moved.**  A `torch.Generator` is device-specific and a CUDA generator does
not reproduce a CPU one, so generating on the device would change every
sample path and make a CPU/GPU comparison meaningless.  Drawing on CPU and
transferring keeps the *stochastic* part bit-identical between devices, so
any residual difference is purely floating-point arithmetic — which is the
thing worth measuring.

TF32 is disabled by default for the same reason: it silently drops matmul
precision to ~10 bits of mantissa, which is a much larger perturbation than
anything else here.  Set ``allow_tf32=True`` deliberately if the speed is
wanted and the loss of precision has been checked.
"""

from __future__ import annotations

import os

import torch

_ENV_VAR = "EID_DEVICE"


def resolve_device(request: str | None = None) -> torch.device:
    """Pick a device: explicit argument, then ``EID_DEVICE``, then CPU.

    The default is **CPU**, not "cuda if available".  Every recorded result
    in this program is a CPU result, and a library that silently switches
    device based on what is installed would make old and new artifacts
    incomparable without anything in the record saying so.
    """
    choice = request or os.environ.get(_ENV_VAR) or "cpu"
    choice = choice.strip().lower()
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {choice!r} requested but torch reports no CUDA. "
            f"Installed build: {torch.__version__} "
            f"(cuda={torch.version.cuda}). A CUDA wheel is needed, e.g. "
            "--extra-index-url https://download.pytorch.org/whl/cu126 "
            "--index-strategy unsafe-best-match --with torch==2.7.1+cu126")
    return torch.device(choice)


def configure(device: torch.device, allow_tf32: bool = False) -> dict:
    """Apply declared numerical settings and return them for provenance."""
    settings = {"device": str(device), "allow_tf32": bool(allow_tf32),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda}
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        properties = torch.cuda.get_device_properties(device)
        settings |= {"gpu_name": properties.name,
                     "gpu_memory_gib": round(properties.total_memory / 2 ** 30,
                                             2),
                     "capability": f"sm_{properties.major}{properties.minor}"}
    return settings


def randn(*shape: int, generator: torch.Generator | None = None,
          device: torch.device | str | None = None,
          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normal draw made on the CPU, then moved -- see the module docstring."""
    values = torch.randn(*shape, generator=generator, dtype=dtype)
    return values if device is None else values.to(device)


def equivalence_report(build: "callable", device_a: str = "cpu",
                       device_b: str = "cuda") -> dict:
    """Run the same seeded computation on two devices and compare.

    ``build`` takes a device and returns a tensor.  Reported as a relative
    difference, because the absolute scale is meaningless on its own; the
    number to watch is whether it sits at float32 round-off (~1e-6) or is
    large enough to change a conclusion.
    """
    first = build(resolve_device(device_a)).detach().to("cpu").double()
    second = build(resolve_device(device_b)).detach().to("cpu").double()
    difference = (first - second).norm()
    scale = first.norm().clamp_min(1e-30)
    return {"device_a": device_a, "device_b": device_b,
            "relative_difference": float(difference / scale),
            "max_absolute_difference": float((first - second).abs().max()),
            "reference_norm": float(first.norm())}
