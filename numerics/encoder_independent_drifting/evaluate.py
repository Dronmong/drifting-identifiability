"""Evaluation of a trained arm against the frozen metric set.

Held-out by construction: the evaluation target pool, the two support
calibration pools and the null pool are four independent draws, and the
generator is evaluated on fresh latents it never trained on.  The
target-vs-target null is computed from the same pools for every arm in a
cell, so the normalized geometry score is comparable across arms.

No pretrained evaluator appears anywhere.  There is no FID here, and none
should be added at this scale: the plan admits external evaluators only as
frozen report-only metrics from Phase 2 onward.
"""

from __future__ import annotations

import numpy as np
import torch

from . import metrics as M
from .config import TrainConfig, derive_seed
from .datasets import ImageTarget
from .models import sample_latent
from .train import TrainOutcome, optimizer_report


# Reform R8: the null is averaged over this many independent draws.  A
# single draw made the reference unstable -- a fresh real sample scored
# 1.13-2.78 instead of the nominal 1.0, because the denominator is one noisy
# realization of a near-zero quantity (audit A1).
NULL_REPEATS = 5


def evaluation_pools(target: ImageTarget, config: TrainConfig,
                     seed: int) -> dict[str, torch.Tensor]:
    """Independent target pools, shared by every arm in a cell."""
    pools = {
        "eval": target.sample(config.eval_samples, np.random.default_rng(
            derive_seed(seed, "eval-target"))),
        "null": target.sample(config.eval_samples, np.random.default_rng(
            derive_seed(seed, "null-target"))),
        "cal_a": target.sample(256, np.random.default_rng(
            derive_seed(seed, "cal-a"))),
        "cal_b": target.sample(256, np.random.default_rng(
            derive_seed(seed, "cal-b"))),
    }
    for index in range(NULL_REPEATS):
        pools[f"null_{index}"] = target.sample(
            config.eval_samples,
            np.random.default_rng(derive_seed(seed, "null-repeat", index)))
    return pools


def null_reference(target: ImageTarget, pools: dict, seed: int) -> dict:
    """What an independent real sample scores under each metric.

    Reform R8: the median over `NULL_REPEATS` independent draws, not a
    single one.  Energy distance between two independent samples of the same
    law is a near-zero quantity with large relative fluctuation, so a
    one-draw denominator made "score = 1.0 means indistinguishable from
    real" false: a fresh real sample scored 1.13-2.78 depending on the draw
    (audit A1).  Averaging stabilizes the reference; the residual spread is
    reported as `null_spread` so the floor is visible rather than assumed.

    The `nearest_real` null is 1.0 by construction -- it is the median
    nearest-real distance of a fresh real sample measured in units of
    itself -- so an arm's normalized ratio reads directly as "how many
    target-NN scales off the data this arm sits".
    """
    draws = []
    for index in range(NULL_REPEATS):
        key = f"null_{index}"
        sample = pools.get(key, pools["null"])
        draws.append(M.null_metrics(
            sample, pools["eval"], pools["cal_a"], pools["cal_b"],
            np.random.default_rng(derive_seed(seed, "null-metrics", index)),
            target, target_null=pools["null"]))
    keys = sorted({k for draw in draws for k in draw})
    out: dict = {}
    for key in keys:
        values = [float(draw[key]) for draw in draws
                  if key in draw and np.isfinite(float(draw[key]))]
        if values:
            out[key] = float(np.median(values))
    out["null_spread"] = {
        key: float(np.std([float(draw[key]) for draw in draws
                           if key in draw and np.isfinite(float(draw[key]))]))
        for key in keys
        if any(key in draw and np.isfinite(float(draw[key]))
               for draw in draws)
    }
    return out


def evaluate_arm(outcome: TrainOutcome, target: ImageTarget,
                 config: TrainConfig, pools: dict, null: dict,
                 seed: int) -> dict:
    """Raw metrics, the normalized geometry score, health and cost."""
    latent = sample_latent(config.eval_samples, config.latent_dim,
                           derive_seed(seed, "eval-latent"))
    with torch.no_grad():
        generated = outcome.model(latent)
    rng = np.random.default_rng(derive_seed(seed, "eval-metrics"))
    row: dict = {"diverged": bool(outcome.diverged)}
    if outcome.diverged or not torch.isfinite(generated).all():
        row["diverged"] = True
        row.update({name: float("nan")
                    for name in M.GEOMETRY_SCORE_COMPONENTS})
        row["geometry_score"] = float("nan")
    else:
        raw = M.raw_metrics(generated, pools["eval"], pools["cal_a"],
                            pools["cal_b"], rng, target,
                            target_null=pools["null"])
        row.update(raw)
        # v1 kept for continuity with `phase1_screen.json`; v2 (reform R4) is
        # what any successor gate reads.
        row.update(M.normalized_geometry_score(raw, null))
        v2 = M.normalized_geometry_score_v2(raw, null)
        row["geometry_score_v2"] = v2["geometry_score"]
        row["geometry_ratios_v2"] = v2["geometry_ratios"]
        row["untrustworthy_nulls"] = v2["untrustworthy_nulls"]
        row["output_rms"] = float(generated.pow(2).mean().sqrt())
    row.update(outcome.ledger.as_dict())
    # Reform R25: the real step control, reported rather than left implicit.
    row.update(optimizer_report(config))
    row["second_moment_ratio"] = float(
        generated.flatten(1).var(0).mean()
        / pools["eval"].flatten(1).var(0).mean().clamp_min(1e-12))
    row["wall_seconds"] = outcome.wall_seconds
    row["model_parameters"] = outcome.model.parameter_count()
    row["inference_nfe"] = 1
    row.update(outcome.log.anchor_presence())
    row.update(outcome.log.summary((
        "loss_", "branch_", "anchor_", "mixture_", "drift_",
        "gradient_shares_", "correction_")))
    return row


def collision_report(outcome: TrainOutcome, samples: int, permutations: int,
                     seed: int, sources: tuple[str, ...] = ("anchor",
                                                            "geometry"),
                     ) -> dict:
    """Which source collisions this arm's own discrepancies can see.

    The anchor is asked with its *audit* bank, so the reported detection
    rate is not the training bank grading its own homework.  ``sources``
    limits the report to the branches a gate condition actually needs;
    permutation tests are expensive and the untouched ones would be
    redundant with the Phase-0 blindness table.
    """
    from . import collision_suite as CS
    from . import spectral_anchor as SA
    from .kernels import mmd2_unbiased

    out: dict = {}
    rng = np.random.default_rng(derive_seed(seed, "collision"))
    if "anchor" in sources and outcome.parts.audit_bank is not None:
        bank = outcome.parts.audit_bank

        def anchor_discrepancy(left: torch.Tensor,
                               right: torch.Tensor) -> float:
            return float(SA.anchor_loss(bank, left, right, "unbiased"))

        out["anchor"] = CS.run_suite(anchor_discrepancy, samples,
                                     permutations, rng)
    if "geometry" in sources and outcome.parts.family is not None:
        for branch in outcome.parts.family.branches:
            kernel = outcome.parts.kernels[branch.name]

            def geometry_discrepancy(left: torch.Tensor, right: torch.Tensor,
                                     branch=branch, kernel=kernel) -> float:
                return mmd2_unbiased(kernel, branch, left, right)

            out[f"geometry_{branch.name}"] = CS.run_suite(
                geometry_discrepancy, samples, permutations, rng)
    return out
