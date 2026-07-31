"""How close to the data must a cloud start for the map to keep it there?

Phase 26 found the drifting map has at least two stable attractors: the data
distribution (real data survives 40 iterations with KID at the floor and recall
holding at 0.72) and a hyper-typical zero-recall region where training lands.
The objective's fixed point is correct; the failure is basin of attraction.

So the question is the width of the good basin.  Three interpolation paths are
run, because a single one would confound *distance from the truth* with the
artifacts of the particular way the interpolation degrades an image:

`mixture`  a fraction lambda of the CLOUD is replaced by generated samples.
    Interpolates the distribution without creating unnatural images, and asks a
    directly practical question: alongside bad samples, do the good ones get
    pulled down, or do the bad ones get pulled up?

`blend`    per-sample convex combination (1-lambda)*real + lambda*generated.
    The distance-from-truth reading.  Note that its intermediate points are
    genuine blends and therefore off-manifold in their own right -- which is
    exactly why it is not run alone.

`blur`     progressive box blur of real data.  The generated cloud's signature
    is spectrum alpha 4.43 against real 3.59, so this walks along the axis the
    generator actually fails on.  If the map can restore blurred real data, the
    generator's failure mode is recoverable; if not, blur is a trap.

Declared before running:

  the data attractor is recovered at lambda >= 0.4 on blend or mixture
      -> the basin is wide, a warm start suffices, and the concrete
         intervention is: pretrain the generator by regression onto real
         samples, then hand over to drifting
  only lambda ~ 0 returns
      -> the basin is narrow; drifting needs an initialization as good as the
         answer, which is a real and much sharper negative result
  the blur path returns toward real
      -> the map can deblur, so the generator's specific failure is escapable
  the blur path does not return
      -> blur is an absorbing direction and the warm start must be sharp, not
         merely close in pixel distance

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase27
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import cifar
from . import kernel_gradient as KG
from .config import MASTER_SEED, GeometryConfig, derive_seed
from .device import configure, resolve_device
from .diagnose_phase20 import save_grid
from .diagnose_phase25 import train_cloud
from .diagnose_phase26 import ETA, POSITIVES, score
from .diagnostics import provenance, write_json
from .fid import inception_features
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .objectives import corrected_teacher

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 45000
CLOUD = 512
ITERATIONS = 40
RECORD_AT = (0, 20, 40)
LAMBDAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# Box-blur kernel per lambda on the blur path.  Odd sizes only, declared.
BLUR_KERNELS = (1, 3, 5, 7, 9, 11)

# Attractor classification, declared.  Real data holds recall ~0.72 and the
# collapsed region sits at exactly 0.000, so these thresholds are far from
# either value rather than tuned between them.
DATA_RECALL = 0.30
COLLAPSED_RECALL = 0.05


def build_start(path: str, index: int, lam: float, real: torch.Tensor,
                generated: torch.Tensor) -> torch.Tensor:
    if path == "mixture":
        count = round(lam * len(real))
        if count == 0:
            return real.clone()
        if count >= len(real):
            return generated[:len(real)].clone()
        return torch.cat([real[:len(real) - count], generated[:count]], dim=0)
    if path == "blend":
        return (1.0 - lam) * real + lam * generated[:len(real)]
    if path == "blur":
        kernel = BLUR_KERNELS[index]
        if kernel <= 1:
            return real.clone()
        return F.avg_pool2d(real, kernel_size=kernel, stride=1,
                            padding=kernel // 2, count_include_pad=False)
    raise ValueError(f"unknown path {path!r}")


def run_path(start: torch.Tensor, source, rng, branch, kernel, reference,
             real, device) -> list[dict]:
    state = start.clone()
    history = []
    for step in range(ITERATIONS + 1):
        if step in RECORD_AT:
            history.append({"step": step, **score(state, reference, real,
                                                  device)})
        if step == ITERATIONS:
            break
        positives = source.sample(POSITIVES, rng)
        with torch.no_grad():
            drift, _ = KG.field(state, positives, state, branch, kernel,
                                direction_mode="paper", normalization="rms",
                                diagnostics=False)
            state = corrected_teacher(state + ETA * drift, positives,
                                      mode="scalar")
    return history


def classify(final: dict) -> str:
    if final["recall"] >= DATA_RECALL:
        return "data"
    if final["recall"] <= COLLAPSED_RECALL:
        return "collapsed"
    return "partial"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--paths", type=str, default="mixture,blend,blur")
    parser.add_argument("--out", type=Path, default=HERE / "phase27_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    seed = MASTER_SEED + SEED_OFFSET
    paths = args.paths.split(",")

    started = time.time()
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(seed, "p27"))
    real_eval = evaluation.sample(args.samples, rng)
    reference = inception_features(real_eval, device).double().numpy()

    print(f"=== a trained cloud to interpolate toward ({args.steps} steps) ===",
          flush=True)
    generated, train = train_cloud(args.steps, seed, device, args.resolution,
                                   args.data_root)
    branch = build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace",
                       target_ess_fraction=0.05), 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace", 0.5, 1.0, 1e-3,
        combine="sum", target_ess_fraction=0.05)
    real_cloud = train.sample(CLOUD, rng)

    rows = []
    for path in paths:
        print(f"\n=== path: {path} ===", flush=True)
        for index, lam in enumerate(LAMBDAS):
            start = build_start(path, index, lam, real_cloud, generated)
            history = run_path(
                start, train,
                np.random.default_rng(derive_seed(seed, path, lam)),
                branch, kernel, reference, real_eval, device)
            first, final = history[0], history[-1]
            entry = {"path": path, "lambda": lam,
                     "blur_kernel": BLUR_KERNELS[index] if path == "blur"
                                    else None,
                     "start": first, "final": final, "history": history,
                     "attractor": classify(final),
                     "kid_change": final["kid"] - first["kid"],
                     "recall_change": final["recall"] - first["recall"],
                     "map_improves": bool(final["kid"] < first["kid"] - 0.002)}
            rows.append(entry)
            save_grid(start[:64].cpu(),
                      HERE / f"phase27_start_{path}_{lam:.1f}.png")
            print(f"    lam={lam:.1f} start KID={first['kid']:+.5f} "
                  f"R={first['recall']:.3f} -> final KID={final['kid']:+.5f} "
                  f"R={final['recall']:.3f} alpha={final['alpha']:.3f}  "
                  f"[{entry['attractor']}]"
                  f"{'  MAP IMPROVES' if entry['map_improves'] else ''}",
                  flush=True)

    edges = {}
    for path in paths:
        group = sorted((r for r in rows if r["path"] == path),
                       key=lambda r: r["lambda"])
        returning = [r["lambda"] for r in group if r["attractor"] == "data"]
        edges[path] = max(returning) if returning else None
    blend_edge = edges.get("blend")
    mixture_edge = edges.get("mixture")
    blur_edge = edges.get("blur")
    wide = any(e is not None and e >= 0.4
               for e in (blend_edge, mixture_edge))
    verdict = {
        "basin_edge_by_path": edges,
        "basin_is_wide": bool(wide),
        "blur_recoverable": bool(blur_edge is not None and blur_edge > 0.0),
        "any_path_improved_by_map": bool(any(r["map_improves"] for r in rows)),
        "lambdas": list(LAMBDAS), "blur_kernels": list(BLUR_KERNELS),
        "thresholds": {"data_recall": DATA_RECALL,
                       "collapsed_recall": COLLAPSED_RECALL},
    }
    if wide:
        verdict["reading"] = (
            "the good basin is WIDE -- a warm start suffices; pretrain the "
            "generator by regression onto real samples, then hand over to "
            "drifting")
    elif any(e is not None and e > 0.0 for e in edges.values()):
        verdict["reading"] = (
            "the basin is NARROW but not a point -- a warm start must be close; "
            "how close is now quantified")
    else:
        verdict["reading"] = (
            "only the truth itself returns -- drifting needs an initialization "
            "as good as the answer, which is a real negative result")

    payload = {"status": "phase27-basin-of-attraction-probe",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "verdict": verdict, "rows": rows}
    digest = write_json(args.out, payload)

    print("\n=== PHASE 27: THE BASIN ===")
    print(f"{'path':10}{'lam':>6}{'start KID':>11}{'start R':>9}"
          f"{'final KID':>11}{'final R':>9}{'final a':>9}{'attractor':>11}")
    for row in rows:
        print(f"{row['path']:10}{row['lambda']:6.1f}"
              f"{row['start']['kid']:+11.5f}{row['start']['recall']:9.3f}"
              f"{row['final']['kid']:+11.5f}{row['final']['recall']:9.3f}"
              f"{row['final']['alpha']:9.3f}{row['attractor']:>11}")
    print("\n    basin edge (largest lambda still returning to the data "
          "attractor)")
    for path, edge in edges.items():
        print(f"      {path:10} {'none' if edge is None else f'{edge:.1f}'}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
