"""Prepare the frozen external positive control for the CAP-EMF-2 preflight.

The control is NVIDIA's official class-conditional CIFAR-10 StyleGAN2-ADA
checkpoint.  Both the upstream repository revision and the network pickle are
pinned.  The pickle is loaded only after its SHA-256 has been verified.

The production protocol emits exactly 50,000 deterministic, class-balanced
32x32 RGB PNGs.  Seed ``s`` uses class ``s mod 10``, truncation 1, and constant
network noise.  This artifact is only a metric-pipeline positive control; it is
never a training target and never participates in model selection.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .artifacts import file_sha256, source_manifest, write_json_atomic
from .standard_metrics import DEFAULT_SAMPLE_COUNT, validate_png_folder

STATUS = "cap-emf2-stylegan2ada-positive-control-source"
UPSTREAM_REPOSITORY = "https://github.com/NVlabs/stylegan2-ada-pytorch"
UPSTREAM_COMMIT = "d72cc7d041b42ec8e806021a205ed9349f87c6a4"
NETWORK_URL = (
    "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/"
    "pretrained/cifar10.pkl"
)
NETWORK_SHA256 = "f8952c74e23da2186d147ad871c48780bd59500ee37c301201081ee8e0cb32f1"
CLASS_COUNT = 10
TRUNCATION_PSI = 1.0
NOISE_MODE = "const"


def source_citation() -> str:
    """Return the exact human-readable citation stored by standard_metrics."""

    return (
        "NVIDIA StyleGAN2-ADA PyTorch CIFAR-10 positive control; repository "
        f"{UPSTREAM_REPOSITORY} at commit {UPSTREAM_COMMIT}; checkpoint "
        f"{NETWORK_URL} sha256={NETWORK_SHA256}; 50,000 seeds 0..49,999; "
        "class=seed mod 10; truncation_psi=1; noise_mode=const"
    )


def balanced_class_indices(count: int, classes: int = CLASS_COUNT) -> np.ndarray:
    """Deterministic class allocation used by the frozen protocol."""

    if count <= 0 or classes <= 0:
        raise ValueError("positive-control count and class count must be positive")
    if count % classes != 0:
        raise ValueError("positive-control count must be exactly class balanced")
    return np.arange(count, dtype=np.int64) % classes


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_upstream(repo: Path, network: Path) -> dict[str, object]:
    """Reject a floating, modified, or incorrectly downloaded dependency."""

    if not (repo / ".git").is_dir():
        raise RuntimeError(f"StyleGAN2-ADA checkout is not a git repository: {repo}")
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"StyleGAN2-ADA commit mismatch: {commit} != {UPSTREAM_COMMIT}"
        )
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("StyleGAN2-ADA tracked sources have local modifications")
    if not network.is_file():
        raise RuntimeError(f"StyleGAN2-ADA checkpoint is missing: {network}")
    digest = file_sha256(network)
    if digest != NETWORK_SHA256:
        raise RuntimeError(
            f"StyleGAN2-ADA checkpoint SHA mismatch: {digest} != {NETWORK_SHA256}"
        )
    return {
        "repository": UPSTREAM_REPOSITORY,
        "commit": commit,
        "checkout": str(repo.resolve()),
        "network_url": NETWORK_URL,
        "network": str(network.resolve()),
        "network_sha256": digest,
    }


def _load_generator(repo: Path, network: Path, device: torch.device):
    repo_string = str(repo.resolve())
    sys.path.insert(0, repo_string)
    try:
        legacy = importlib.import_module("legacy")
        loaded_from = Path(legacy.__file__).resolve()
        if repo.resolve() not in loaded_from.parents:
            raise RuntimeError(f"imported StyleGAN legacy module from {loaded_from}")
        with network.open("rb") as handle:
            generator = legacy.load_network_pkl(handle)["G_ema"]
        # The pinned upstream upfirdn2d fallback forgets to mark a failed
        # extension build as initialized, so hosts without a compiler retry and
        # print a full traceback on every layer of every batch. Attempt once,
        # then freeze the documented reference implementation when unavailable.
        upfirdn2d = importlib.import_module("torch_utils.ops.upfirdn2d")
        if not upfirdn2d._init():
            upfirdn2d._inited = True
    finally:
        if sys.path and sys.path[0] == repo_string:
            sys.path.pop(0)
    generator = generator.eval().requires_grad_(False).to(device)
    if (
        int(generator.c_dim) != CLASS_COUNT
        or int(generator.img_resolution) != 32
        or int(generator.img_channels) != 3
    ):
        raise RuntimeError(
            "pinned positive-control network is not conditional RGB CIFAR-10"
        )
    return generator


def _latents(seeds: range, dimension: int) -> np.ndarray:
    # This deliberately matches NVIDIA's official generate.py per-seed RNG.
    return np.concatenate(
        [np.random.RandomState(seed).randn(1, dimension) for seed in seeds], axis=0
    )


def generate_positive_control(
    *,
    repo: Path,
    network: Path,
    output: Path,
    provenance_out: Path,
    device_name: str,
    batch: int,
) -> dict[str, object]:
    """Generate and seal the exact 50k positive-control image population."""

    if batch <= 0:
        raise ValueError("positive-control batch must be positive")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite positive-control folder {output}")
    if provenance_out.exists() or provenance_out.with_suffix(
        provenance_out.suffix + ".sha256"
    ).exists():
        raise RuntimeError(f"refusing to overwrite provenance {provenance_out}")
    staging = output.with_name(f".{output.name}.partial")
    if staging.exists():
        raise RuntimeError(f"stale positive-control staging directory exists: {staging}")

    upstream = verify_upstream(repo.resolve(), network.resolve())
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the 50k positive control requires an available CUDA device")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    staging.mkdir(parents=True)
    started = time.perf_counter()
    generator = _load_generator(repo, network, device)
    classes = balanced_class_indices(DEFAULT_SAMPLE_COUNT)
    with torch.inference_mode():
        for start in range(0, DEFAULT_SAMPLE_COUNT, batch):
            stop = min(start + batch, DEFAULT_SAMPLE_COUNT)
            seeds = range(start, stop)
            z = torch.from_numpy(_latents(seeds, int(generator.z_dim))).to(device)
            labels = np.zeros((stop - start, CLASS_COUNT), dtype=np.float32)
            labels[np.arange(stop - start), classes[start:stop]] = 1.0
            c = torch.from_numpy(labels).to(device)
            images = generator(
                z,
                c,
                truncation_psi=TRUNCATION_PSI,
                noise_mode=NOISE_MODE,
            )
            if not torch.isfinite(images).all():
                raise RuntimeError("StyleGAN2-ADA produced a non-finite image")
            arrays = (
                (images * 127.5 + 128)
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            for index, array in enumerate(arrays, start=start):
                Image.fromarray(array).save(
                    staging / f"{index:06d}.png", format="PNG", compress_level=0
                )
            if stop % 5_000 == 0 or stop == DEFAULT_SAMPLE_COUNT:
                print(f"positive control: {stop}/{DEFAULT_SAMPLE_COUNT}", flush=True)

    staged_manifest = validate_png_folder(
        staging,
        expected_count=DEFAULT_SAMPLE_COUNT,
        require_sequential_names=True,
    )
    os.replace(staging, output)
    if staging.exists() or not output.is_dir():
        raise RuntimeError("positive-control directory publication failed")
    # os.replace is an atomic rename on the same volume; it does not rewrite
    # file contents. The full decode/hash occurred immediately before it.
    image_manifest = {**staged_manifest, "directory": str(output.resolve())}

    elapsed = time.perf_counter() - started
    result = {
        "status": STATUS,
        "decision": "COMPLETE",
        "purpose": "external metric-pipeline positive control; never training data",
        "citation": source_citation(),
        "upstream": upstream,
        "sampling": {
            "count": DEFAULT_SAMPLE_COUNT,
            "seeds": [0, DEFAULT_SAMPLE_COUNT - 1],
            "class_rule": "class = seed mod 10",
            "class_counts": [int((classes == index).sum()) for index in range(10)],
            "truncation_psi": TRUNCATION_PSI,
            "noise_mode": NOISE_MODE,
            "batch": batch,
        },
        "images": image_manifest,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "deterministic_algorithms": True,
            "allow_tf32": False,
        },
        "elapsed_seconds": elapsed,
        "source_sha256": source_manifest(),
        "limits": [
            "This validates the metric stack; it is not a CAP2 comparator arm.",
            "No positive-control image may enter CAP2 training or arm selection.",
        ],
    }
    result["artifact_sha256"] = write_json_atomic(provenance_out, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stylegan-repo", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--provenance-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=100)
    args = parser.parse_args()
    result = generate_positive_control(
        repo=args.stylegan_repo,
        network=args.network,
        output=args.out_dir,
        provenance_out=args.provenance_out,
        device_name=args.device,
        batch=args.batch,
    )
    print(f"positive-control provenance sha256={result['artifact_sha256']}")
    print("citation for standard_metrics:")
    print(result["citation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
