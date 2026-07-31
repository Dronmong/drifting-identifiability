"""What IS the teacher, as a function of bandwidth?

Phase 22 left a gap that no story explains.  The performance optimum sits at
realized ESS 0.52-0.71 and sharpening past it is catastrophic (p = 0.0001) --
but the count of images averaged does **not** order the arms: `E_sharper_pos`
averages 60.6 images and is the worst arm, while `B_sharp` averages 34.3 and is
fine.  So "the teacher is a blur of N images" is not the mechanism, and
something about *small tau itself* is harmful independently of how blurry the
resulting average is.

Phase 24's proposed fix (per-row adaptive bandwidth) was refuted before
implementation: cloud rows see near/far structure as well as real rows do
(within-row distance CV 0.2065 vs 0.2003) and row heterogeneity is negligible
(realized ESS q10 0.702, q90 0.741).  So the gap above is the live question, and
it is answerable without training anything to convergence.

This probe measures the teacher directly across bandwidths spanning realized
ESS 0.9 -> 0.15:

  **sharpness**    spectrum alpha and precision/recall of the teacher IMAGES.
                   Not of a trained generator -- of the target it regresses on.
  **variance**     recompute the teacher for the SAME cloud point against
                   independent positive batches and measure the spread.  The
                   candidate explanation: small tau may raise target variance
                   faster than it raises target sharpness, so the generator
                   chases a noisier target that is no more informative.
  **consistency**  mean pairwise cosine of the drift direction across those
                   independent batches.  This is the sharpest form of the
                   question -- a field whose direction does not reproduce across
                   batches carries no signal to learn from, whatever its
                   magnitude.

Bandwidths are solved on the **cloud-to-positives** Gram, which is rectangular
and therefore free of the self-match defect that made `target_ess_fraction`
mean something other than it said (see `kernels._row_ess_fraction`).  That is
the right instrument here but it reads generated output, so it is a
*characterization* tool and must never become a declared design rule.

Declared before running -- section 4, Step 2 of `EncoderIndependentPhase25Plan.md`:

  variance up, sharpness flat  -> the cliff is a target-variance effect; the fix
                                  is variance reduction at fixed selectivity
  sharpness materially up      -> the cliff is not in the teacher; it lives in
                                  the optimization
  neither moves                -> the mean-shift teacher form is the binding
                                  constraint at every bandwidth, and no
                                  kernel-family intervention can help

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from .appearance import precision_recall, spectrum_slope
from .config import MASTER_SEED, GeometryConfig, TrainConfig, derive_seed
from .device import configure, resolve_device
from .diagnose_phase20 import save_grid
from .diagnostics import provenance, write_json
from .fid import inception_features, kid_from_features
from .fixed_features import build_family
from .kernels import BlockKernel, calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher
from .run_phase16 import _tail

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 43000

# Realized ESS on the cloud->positives Gram.  0.71 is where Phase 22's best arm
# (F_mix) actually operated; 0.24 is where E_sharper_pos failed.  The ladder
# brackets both ends of the measured optimum.
TARGETS = (0.90, 0.71, 0.52, 0.35, 0.24, 0.15)
POSITIVES = 64
CLOUD = 512
BATCHES = 8


def rectangular_ess(kernel: BlockKernel, branch, cloud: torch.Tensor,
                    positives: torch.Tensor) -> tuple[float, float]:
    """Median row ESS fraction of the field's actual Gram, and the dead share.

    Max-normalized per row for the same reason `kernels._row_ess_fraction` is:
    at small tau the surviving entries are subnormal and a naive row-sum
    division turns them into garbage.
    """
    with torch.no_grad():
        gram = kernel.gram(branch, cloud, positives)
        row_max = gram.max(dim=1, keepdim=True).values
        alive = row_max.squeeze(1) > 0
        scaled = gram / torch.where(row_max > 0, row_max,
                                    torch.ones_like(row_max))
        total = scaled.sum(dim=1)
        concentration = (scaled ** 2).sum(dim=1)
        ess = torch.where(alive & (concentration > 0),
                          total ** 2 / concentration.clamp_min(1e-30),
                          torch.ones_like(total))
        return (float(ess.median()) / gram.shape[1],
                float((~alive).to(torch.float32).mean()))


def solve_for_realized(branch, base: BlockKernel, cloud: torch.Tensor,
                       positives: torch.Tensor, target: float,
                       iterations: int = 30) -> BlockKernel:
    """Bisect a global tau factor so the FIELD realizes the declared ESS."""
    low, high = 1e-4, 1e4
    for _ in range(iterations):
        mid = float(np.sqrt(low * high))
        probe = BlockKernel(base.base, base.taus * mid, base.weights, base.eps,
                            base.combine)
        realized, _ = rectangular_ess(probe, branch, cloud, positives)
        if realized < target:
            low = mid
        else:
            high = mid
    factor = float(np.sqrt(low * high))
    return BlockKernel(base.base, (base.taus * factor).clamp_min(1e-12),
                       base.weights, base.eps, base.combine)


def train_cloud(steps: int, seed: int, device, resolution: int,
                root: str | None) -> tuple[torch.Tensor, object]:
    """A representative mid-training cloud, at Phase 22's best operating point.

    An untrained generator's cloud is not the regime the teacher matters in, so
    the probe pays for a short run rather than measuring the initialization.
    """
    train = cifar.cifar_target(resolution, "train", root)
    train.device = device
    rng = np.random.default_rng(derive_seed(seed, "p25"))
    branch = build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace",
                       target_ess_fraction=0.05), 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace", 0.5, 1.0, 1e-3,
        combine="sum", target_ess_fraction=0.05)
    config = TrainConfig(steps=steps, batch=POSITIVES, image_size=resolution,
                         width=64)
    model = OneStepGenerator(config.latent_dim, 3, resolution, 64,
                             derive_seed(seed, "generator")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * 0.5 * (
                1.0 + np.cos(np.pi * step / max(steps, 1)))
        latent = sample_latent(256, config.latent_dim,
                               derive_seed(seed, "latent", step), device)
        output = model(latent)
        with torch.no_grad():
            positives = train.sample(POSITIVES, rng)
            drift, _ = KG.field(output.detach(), positives, output.detach(),
                                branch, kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            teacher = corrected_teacher(output.detach() + 0.5 * drift,
                                        positives, mode="scalar")
        loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe = sample_latent(CLOUD, config.latent_dim,
                          derive_seed(seed, "cloud"), device)
    with torch.no_grad():
        return model(probe).detach(), train


def measure(branch, kernel, cloud, train, rng, reference, real, device) -> dict:
    """Teachers from independent positive batches: sharpness, variance, cosine."""
    teachers, directions = [], []
    for index in range(BATCHES):
        positives = train.sample(POSITIVES, rng)
        with torch.no_grad():
            raw, _ = KG.field(cloud, positives, cloud, branch, kernel,
                              direction_mode="paper", normalization="none",
                              diagnostics=False)
            normalized, _ = KG.field(cloud, positives, cloud, branch, kernel,
                                     direction_mode="paper",
                                     normalization="rms", diagnostics=False)
            teacher = corrected_teacher(cloud + 0.5 * normalized, positives,
                                        mode="scalar")
        teachers.append(teacher)
        flat = raw.flatten(1)
        directions.append(flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-30))
        if index == 0:
            first, raw_rms = teacher, float(raw.flatten(1).norm(dim=1).mean())

    stack = torch.stack(teachers)                       # [BATCHES, n, C, H, W]
    # Per cloud point, spread of its teacher across independent batches,
    # relative to the real data's own spread.
    spread = stack.flatten(2).std(dim=0).mean()
    teacher_variance = float(spread / real.flatten(1).std(dim=0).mean().cpu())
    cosines = []
    for i in range(BATCHES):
        for j in range(i + 1, BATCHES):
            cosines.append((directions[i] * directions[j]).sum(dim=1))
    consistency = float(torch.stack(cosines).mean())

    features = inception_features(first.cpu(), device).double().numpy()
    pr = precision_recall(features, reference)
    return {"teacher_variance_vs_real": teacher_variance,
            "drift_direction_cosine": consistency,
            "raw_drift_norm": raw_rms,
            "teacher_alpha": spectrum_slope(first.cpu())["alpha"],
            "teacher_kid": kid_from_features(features, reference),
            "teacher_precision": pr["precision"],
            "teacher_recall": pr["recall"],
            "teacher_tail": _tail(first.cpu()),
            "sample": first.cpu()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase25_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    seed = MASTER_SEED + SEED_OFFSET

    started = time.time()
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(seed, "p25-eval"))
    real = evaluation.sample(args.samples, rng)
    reference = inception_features(real, device).double().numpy()
    real_alpha = spectrum_slope(real)["alpha"]

    print(f"=== training a representative cloud ({args.steps} steps) ===",
          flush=True)
    cloud, train = train_cloud(args.steps, seed, device, args.resolution,
                               args.data_root)
    branch = build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace"),
        3).branches[0]
    base = calibrate_block_kernel(
        branch, train.sample(256, np.random.default_rng(
            derive_seed(seed, "base"))), "smooth_laplace", 0.5, 1.0, 1e-3,
        combine="sum")
    positives_probe = train.sample(POSITIVES, np.random.default_rng(
        derive_seed(seed, "probe")))

    rows = []
    print("\n=== the teacher, across the field's realized selectivity ===",
          flush=True)
    for target in TARGETS:
        kernel = solve_for_realized(branch, base, cloud, positives_probe,
                                    target)
        realized, dead = rectangular_ess(kernel, branch, cloud,
                                         positives_probe)
        entry = measure(branch, kernel, cloud, train,
                        np.random.default_rng(derive_seed(seed, "m", target)),
                        reference, real, device)
        sample = entry.pop("sample")
        entry |= {"target_realized_ess": target, "realized_ess": realized,
                  "dead_row_fraction": dead,
                  "images_averaged": realized * POSITIVES,
                  "bandwidth_median": float(kernel.taus.median())}
        rows.append(entry)
        save_grid(sample[:64], HERE / f"phase25_teacher_ess{target:.2f}.png")
        print(f"    ESS={realized:.3f} ({entry['images_averaged']:5.1f} imgs) "
              f"tau={entry['bandwidth_median']:8.3f} "
              f"cos={entry['drift_direction_cosine']:+.4f} "
              f"var={entry['teacher_variance_vs_real']:.4f} "
              f"alpha={entry['teacher_alpha']:.3f} "
              f"P={entry['teacher_precision']:.3f} "
              f"KID={entry['teacher_kid']:+.5f} dead={dead:.3f}", flush=True)

    sharp = [r for r in rows if r["realized_ess"] <= 0.30]
    flat = [r for r in rows if r["realized_ess"] >= 0.70]
    if sharp and flat:
        variance_ratio = (np.mean([r["teacher_variance_vs_real"] for r in sharp])
                          / max(np.mean([r["teacher_variance_vs_real"]
                                         for r in flat]), 1e-12))
        cosine_ratio = (np.mean([r["drift_direction_cosine"] for r in sharp])
                        / max(np.mean([r["drift_direction_cosine"]
                                       for r in flat]), 1e-12))
        alpha_gain = (np.mean([r["teacher_alpha"] for r in flat])
                      - np.mean([r["teacher_alpha"] for r in sharp]))
        precision_gain = (np.mean([r["teacher_precision"] for r in sharp])
                          - np.mean([r["teacher_precision"] for r in flat]))
    else:
        variance_ratio = cosine_ratio = alpha_gain = precision_gain = float("nan")

    verdict = {
        "real_alpha": real_alpha, "targets": list(TARGETS),
        "variance_ratio_sharp_over_flat": float(variance_ratio),
        "cosine_ratio_sharp_over_flat": float(cosine_ratio),
        "alpha_gain_sharpening": float(alpha_gain),
        "precision_gain_sharpening": float(precision_gain),
        "variance_rises_sharply": bool(variance_ratio > 1.5),
        "consistency_collapses": bool(cosine_ratio < 0.5),
        "sharpness_improves": bool(precision_gain > 0.1 or alpha_gain > 0.3),
    }
    if verdict["sharpness_improves"]:
        verdict["reading"] = (
            "sharpening DOES produce a better teacher -- the cliff is not in "
            "the teacher and lives in the optimization")
    elif verdict["variance_rises_sharply"] or verdict["consistency_collapses"]:
        verdict["reading"] = (
            "sharpening raises target variance / destroys drift consistency "
            "without improving the teacher -- the cliff is a variance effect "
            "and the fix is variance reduction at fixed selectivity")
    else:
        verdict["reading"] = (
            "neither the teacher's quality nor its variance responds to "
            "bandwidth -- the mean-shift teacher FORM is the binding "
            "constraint and no kernel-family intervention can help")

    payload = {"status": "phase25-teacher-characterization-probe",
               "plan": "numerics/EncoderIndependentPhase25Plan.md",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "real_alpha": real_alpha, "rows": rows, "verdict": verdict}
    digest = write_json(args.out, payload)

    print("\n=== PHASE 25: WHAT THE TEACHER IS ===")
    print(f"{'ESS':>6}{'imgs':>7}{'tau':>10}{'drift cos':>11}{'variance':>10}"
          f"{'alpha':>8}{'prec':>7}{'recall':>8}{'KID':>10}{'dead':>7}")
    for row in rows:
        print(f"{row['realized_ess']:6.3f}{row['images_averaged']:7.1f}"
              f"{row['bandwidth_median']:10.3f}"
              f"{row['drift_direction_cosine']:+11.4f}"
              f"{row['teacher_variance_vs_real']:10.4f}"
              f"{row['teacher_alpha']:8.3f}{row['teacher_precision']:7.3f}"
              f"{row['teacher_recall']:8.3f}{row['teacher_kid']:+10.5f}"
              f"{row['dead_row_fraction']:7.3f}")
    print(f"\n    real alpha {real_alpha:.3f}")
    print(f"    sharp/flat teacher variance ratio  {variance_ratio:8.3f}"
          f"   (>1.5 = variance effect)")
    print(f"    sharp/flat drift cosine ratio      {cosine_ratio:8.3f}"
          f"   (<0.5 = consistency collapse)")
    print(f"    precision gained by sharpening     {precision_gain:+8.3f}")
    print(f"    alpha reduced by sharpening        {alpha_gain:+8.3f}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
