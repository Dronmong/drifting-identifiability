"""Validate the new appearance measures against known degradations.

`appearance.py` adds precision/recall and a spectrum slope because Phase 22
found KID and FID ranking garish texture above blurry photographs.  Before
either is allowed to inform a decision it has to be shown to behave -- on
inputs whose quality and coverage we already know, not on a generator whose
quality is the open question.

The ladder is the one from `EncoderIndependentMetricAudit.md`, which is where
ED2 was caught being saturated by a structureless Gaussian.

**This probe failed its own checks twice before passing, and every failure was
a wrong expectation of mine rather than a defect in the instrument.  All three
were the same mistake, so the consolidated finding is the useful output.**

**Precision is a measure of TYPICALITY, not of realism.**  It falls only for
inputs that leave the typical set altogether.  It does *not* fall for
degradations that stay in the dense centre of the feature cloud:

| input | precision | recall | caught by |
|---|---|---|---|
| real | 0.726 | 0.748 | -- |
| moment-matched Gaussian | **0.842** | 0.000 | recall |
| box blur 7 | **0.681** | 0.000 | recall |
| noise 0.3 | 0.350 | 0.009 | both |
| shuffled pixels | 0.029 | 0.000 | both |

A structureless Gaussian scores precision *above* real data, and a heavy blur
barely below it.  Both are caught, decisively, by **recall**.  I predicted
precision would fall in both cases and in a third case besides; it does not,
and the reason is a single property I mis-modelled three times.

Consequence for how these are read: **precision detects artifacts, recall
detects blur and collapse, and neither is interpretable alone.**  That is
enough for Phase 22's open question -- `C_sharper` (blurry, low-rank) should
show high precision with near-zero recall, and `A_control` (garish texture)
lower precision with higher recall.

A separate defect this found, in the degradation rather than the metric: the
avg-pool-then-*nearest*-upsample "blur" used elsewhere in this package is a
downsample, not a low-pass -- its block edges *add* high-frequency energy, and
it lowered the spectrum slope instead of raising it.  `diagnose_phase18.py`
uses that construction, so its "blur" sensitivity column was partly measuring
blockiness.  Fixed here with a stride-1 box blur.

Declared expectations for this version:

  real_b          precision and recall both HIGH -- it is real data
  box_blur_3/7    recall DOWN (precision will NOT move), alpha UP
  noise_0.3       precision DOWN, alpha DOWN (energy added high)
  gaussian_mm     recall ~0 and F1 far below real -- the case that matters
  shuffled_pixels precision ~0, alpha ~0 (white noise has a flat spectrum)

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase23
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import cifar
from .appearance import precision_recall, spectrum_report, spectrum_slope
from .config import MASTER_SEED, derive_seed
from .device import configure, resolve_device
from .diagnose_phase15 import gaussian_moment_match
from .diagnostics import provenance, write_json
from .fid import frechet_from_features, inception_features, kid_from_features

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 36000


def _box_blur(images: torch.Tensor, kernel: int) -> torch.Tensor:
    """A genuine low-pass: stride-1 box filter, resolution unchanged.

    The avg-pool-then-nearest-upsample construction used elsewhere in this
    package is a *downsample*, not a blur: the block edges it leaves behind
    add high-frequency energy, which is why it lowered the spectrum slope.
    """
    return F.avg_pool2d(images, kernel_size=kernel, stride=1,
                        padding=kernel // 2, count_include_pad=False)


def ladder(real_a: torch.Tensor, real_b: torch.Tensor, rng) -> dict:
    generator = torch.Generator().manual_seed(11)
    noise = torch.randn(real_b.shape, generator=generator)
    flat = real_b.reshape(len(real_b), -1).clone()
    for row in range(len(flat)):
        flat[row] = flat[row][torch.randperm(flat.shape[1],
                                             generator=generator)]
    return {
        "real_b": real_b,
        "box_blur_3": _box_blur(real_b, 3),
        "box_blur_7": _box_blur(real_b, 7),
        "noise_0.3": (real_b + 0.3 * noise).clamp(-1.0, 1.0),
        "gaussian_mm": gaussian_moment_match(real_a, len(real_b), rng),
        "shuffled_pixels": flat.reshape(real_b.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase23_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)

    started = time.time()
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(MASTER_SEED + SEED_OFFSET, "p23"))
    real_a = evaluation.sample(args.samples, rng)
    real_b = evaluation.sample(args.samples, rng)

    variants = ladder(real_a, real_b, rng)
    reference = inception_features(real_a, device).double().numpy()

    rows = []
    for name, images in variants.items():
        features = inception_features(images, device).double().numpy()
        entry = {"variant": name,
                 "kid": kid_from_features(features, reference),
                 "fid": frechet_from_features(features, reference),
                 **precision_recall(features, reference),
                 **spectrum_report(images, real_a)}
        rows.append(entry)
        print(f"    {name:16} P={entry['precision']:.3f} R={entry['recall']:.3f}"
              f"  KID={entry['kid']:+.5f} FID={entry['fid']:7.2f}"
              f"  alpha={entry['alpha_generated']:.3f}", flush=True)

    by_name = {r["variant"]: r for r in rows}
    real_alpha = spectrum_slope(real_a)["alpha"]
    checks = {
        "real_precision_high": by_name["real_b"]["precision"] > 0.5,
        "real_recall_high": by_name["real_b"]["recall"] > 0.5,
        # The case that matters.  Precision cannot catch a structureless
        # Gaussian -- it scores ABOVE real -- so recall and F1 must.
        "gaussian_recall_near_zero": by_name["gaussian_mm"]["recall"] < 0.05,
        "gaussian_f1_far_below_real":
            by_name["gaussian_mm"]["f1"] < by_name["real_b"]["f1"] - 0.5,
        # Blur is a typicality-PRESERVING degradation, so precision is not
        # expected to move; recall is what must catch it.  Asserting the
        # non-movement too, so a future change that makes precision
        # blur-sensitive shows up as a failure rather than passing silently.
        "blur_recall_below_real":
            by_name["box_blur_7"]["recall"]
            < by_name["real_b"]["recall"] - 0.2,
        "blur_precision_stays_typical":
            by_name["box_blur_7"]["precision"]
            > by_name["real_b"]["precision"] - 0.2,
        # Precision's real job: catching inputs that leave the typical set.
        "noise_precision_below_real":
            by_name["noise_0.3"]["precision"]
            < by_name["real_b"]["precision"] - 0.2,
        "shuffled_precision_near_zero":
            by_name["shuffled_pixels"]["precision"] < 0.1,
        "blur_alpha_above_real":
            by_name["box_blur_7"]["alpha_generated"] > real_alpha,
        "noise_alpha_below_real":
            by_name["noise_0.3"]["alpha_generated"] < real_alpha,
        "shuffled_alpha_near_zero":
            abs(by_name["shuffled_pixels"]["alpha_generated"]) < 0.5,
    }
    verdict = {"real_alpha": real_alpha, "checks": checks,
               "all_passed": all(checks.values())}
    verdict["reading"] = (
        "precision/recall and the spectrum slope behave as declared; both are "
        "fit to inform Phase 24"
        if verdict["all_passed"] else
        "at least one declared check FAILED -- the measure is not trustworthy "
        "and must not be used to choose a configuration")

    payload = {"status": "phase23-metric-validation-probe",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "rows": rows, "verdict": verdict}
    digest = write_json(args.out, payload)

    print("\n=== PHASE 23: do the new measures behave? ===")
    print(f"{'variant':18}{'precision':>10}{'recall':>9}{'alpha':>8}"
          f"{'KID':>10}{'FID':>9}")
    for row in rows:
        print(f"{row['variant']:18}{row['precision']:10.3f}{row['recall']:9.3f}"
              f"{row['alpha_generated']:8.3f}{row['kid']:+10.5f}"
              f"{row['fid']:9.2f}")
    print(f"\n    real alpha = {real_alpha:.3f}"
          f"   (dataset property; only the GAP is meaningful)")
    print("\n    declared checks")
    for name, passed in checks.items():
        print(f"      {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
