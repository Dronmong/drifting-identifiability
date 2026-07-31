"""Stage F1 machinery — protocol `EncoderIndependentF1Protocol.md` DESIGN v3.

Implements §11 step 2. Nothing here runs a confirmatory arm; `run_f1.py` does
that and only after `f1_calibration.py` returns GO and `f1_checks.py` passes.

Design points that exist because an audit caught them:

* **Unique-index allocation.** `cifar.cifar_target(...).sample` draws *with
  replacement* (`pool[rng.integers(...)]`), so building the 4 096-image replay
  bank through it would silently duplicate entries and corrupt the
  distinct-bank audit. Every F1 index set is allocated here from a permutation
  of the pool, without replacement, and asserted disjoint (§4.1, §15.5).
* **A replicate unit is an independent (source seed, teacher seed) pair**, not a
  source seed alone. v2 shared one target realization across all three
  replicates, which licensed only a conditional claim (§10, §15.5).
* **The frozen update passes `denominator_floor` explicitly and hands
  `corrected_teacher` a `report` dict**, without which the promised R11 gain is
  never written (§4.3, §15.6).
* **Checkpoint diagnostics are interval statistics**, not last-step values, with
  an extra pre-update evaluation at K = 0 so no interval is empty (§4.3).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from .appearance import precision_recall, spectrum_slope
from .config import MASTER_SEED, GeometryConfig, derive_seed
from .fid import frechet_from_features, inception_features, kid_from_features
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher
from .run_phase16 import _tail

# ---------------------------------------------------------------------------
# Frozen constants (§2.1, §3, §4, §5, §8)
# ---------------------------------------------------------------------------

SEED_OFFSET = 51000
UNITS = 3                      # independent (source, teacher) replicate units
PARTICLES = 512
POSITIVES = 64
BANK = 4096
EVAL_REFERENCE = 2048
CALIBRATION_SAMPLES = 256      # target-only kernel calibration subset
ETA = 0.5
TARGET_ESS = 0.05
# Terminal horizon raised from 200 to 20 000, decided on **measured cost before
# any outcome was read** (§5).  The protocol's estimate of ~10 s per 200-step
# rollout was 10x pessimistic: measured 4.9 ms/step on 512 particles, so 36 cells
# at K = 20 000 cost ~58 min of rollout against ~0.6 min at K = 200.
#
# The horizon is also the scientifically motivated one.  §19.5 asked for
# checkpoints "long enough to distinguish slow progress from an attractor";
# Phase 26 used 40 steps and could not.  20 000 is the same order as the 30 000
# generator-training budget of Phases 22/30 and 500x Phase 26's horizon, so a
# state that has not moved by then is an attractor on any reading.
CHECKPOINTS = (0, 10, 20, 40, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
RECALL_GATE = 0.05             # fixed; §2.1 bounds the null, never moves this
NULL_TOLERANCE = 0.025         # p_null_upper must fall below this
REPLAY_TOLERANCE = 1e-4        # relative L2 for the reproducibility check
CONTROL_RECALL_FLOOR = 0.5     # `real_data` validity precondition
RESAMPLES = 200                # particle resamples for uncertainty (§6)

ARMS = ("real_data", "random_generator", "trained_bad", "ae_reconstruction",
        "basin_interpolation", "ambient_noise")
REGIMES = ("replay", "stochastic")


def unit_seeds(unit: int) -> dict:
    """The concrete numeric seed registry for one replicate unit (§10, §15.7).

    Source and teacher seeds are independent *within* a unit, so a lucky or
    pathological target schedule cannot affect all three units.  Kernel
    calibration and the evaluation reference are deliberately shared.
    """
    base = MASTER_SEED + SEED_OFFSET + unit
    return {
        "unit": unit,
        "source_init": derive_seed(base, "source-init"),
        "source_latent": derive_seed(base, "source-latent"),
        "bank_selection": derive_seed(base, "teacher-bank"),
        "replay_schedule": derive_seed(base, "teacher-replay"),
        "stochastic_stream": derive_seed(base, "teacher-stochastic"),
        "index_allocation": derive_seed(base, "index-allocation"),
        # shared across units, listed so the registry is complete
        "kernel_calibration": derive_seed(MASTER_SEED + SEED_OFFSET, "kernel"),
        "eval_reference": derive_seed(MASTER_SEED + SEED_OFFSET, "eval-ref"),
    }


# ---------------------------------------------------------------------------
# Index allocation — unique, without replacement, asserted disjoint
# ---------------------------------------------------------------------------

@dataclass
class Allocation:
    """Disjoint unique index sets over the train pool for one unit."""

    bank: np.ndarray
    real_data: np.ndarray
    ae_inputs: np.ndarray
    calibration: np.ndarray
    identical: int
    pool_size: int
    digests: dict = dc_field(default_factory=dict)

    def assert_disjoint(self) -> None:
        named = {"bank": self.bank, "real_data": self.real_data,
                 "ae_inputs": self.ae_inputs, "calibration": self.calibration,
                 "identical": np.array([self.identical])}
        for name, values in named.items():
            if len(np.unique(values)) != len(values):
                raise AssertionError(f"{name} contains duplicate indices")
        keys = sorted(named)
        for i, left in enumerate(keys):
            for right in keys[i + 1:]:
                shared = np.intersect1d(named[left], named[right])
                if shared.size:
                    raise AssertionError(
                        f"{left} and {right} share {shared.size} indices")


def allocate(unit: int, resolution: int, root: str | None) -> Allocation:
    """Partition the train pool without replacement.

    The whole point is that no index serves two roles: a bank image must not
    also be a `real_data` particle or a kernel-calibration sample, or the
    memorization audit and the positive control are both compromised.
    """
    pool = cifar.cifar_pool(resolution, "train", root)
    seeds = unit_seeds(unit)
    order = np.random.default_rng(seeds["index_allocation"]).permutation(
        len(pool))
    cuts = np.cumsum([BANK, PARTICLES, PARTICLES, CALIBRATION_SAMPLES, 1])
    if cuts[-1] > len(pool):
        raise ValueError(f"train pool of {len(pool)} is too small for F1")
    allocation = Allocation(
        bank=np.sort(order[:cuts[0]]),
        real_data=np.sort(order[cuts[0]:cuts[1]]),
        ae_inputs=np.sort(order[cuts[1]:cuts[2]]),
        calibration=np.sort(order[cuts[2]:cuts[3]]),
        identical=int(order[cuts[3]]),
        pool_size=len(pool),
    )
    allocation.assert_disjoint()
    allocation.digests = {
        name: hashlib.sha256(np.asarray(values).tobytes()).hexdigest()[:16]
        for name, values in (("bank", allocation.bank),
                             ("real_data", allocation.real_data),
                             ("ae_inputs", allocation.ae_inputs),
                             ("calibration", allocation.calibration))}
    return allocation


def take(resolution: int, split: str, root: str | None,
         indices: np.ndarray) -> torch.Tensor:
    """Images at explicit indices — the without-replacement primitive."""
    return cifar.cifar_pool(resolution, split, root)[
        torch.as_tensor(np.asarray(indices), dtype=torch.long)]


# ---------------------------------------------------------------------------
# Teacher: kernel, replay schedule, and the frozen update
# ---------------------------------------------------------------------------

def build_kernel(allocation: Allocation, resolution: int, root: str | None,
                 device):
    """Target-only calibration on the dedicated disjoint subset (§4.3)."""
    branch = build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace",
                       target_ess_fraction=TARGET_ESS), 3).branches[0]
    samples = take(resolution, "train", root, allocation.calibration).to(device)
    kernel = calibrate_block_kernel(
        branch, samples, "smooth_laplace", 0.5, 1.0, 1e-3, combine="sum",
        target_ess_fraction=TARGET_ESS)
    return branch, kernel


def replay_schedule(unit: int, steps: int) -> np.ndarray:
    """`[steps, POSITIVES]` positions into the bank, seeded and reproducible.

    Positions, not pool indices, so the schedule is bank-relative and can be
    hashed independently of which images the bank holds.
    """
    rng = np.random.default_rng(unit_seeds(unit)["replay_schedule"])
    return np.stack([rng.choice(BANK, size=POSITIVES, replace=False)
                     for _ in range(steps)])


def frozen_update(state: torch.Tensor, positives: torch.Tensor, branch,
                  kernel) -> tuple[torch.Tensor, dict]:
    """The §4.3 update, every behaviour-changing argument explicit.

    `denominator_floor` is passed rather than inherited, and
    `corrected_teacher` receives a `report` dict — without it the R11 gain the
    protocol promises to record is never written anywhere.
    """
    drift, stats = KG.field(
        state, positives, state, branch, kernel,
        direction_mode="paper",
        normalization="rms",
        denominator_floor=1e-30,
        self_mask=False,
        diagnostics=True,
    )
    r11_report: dict = {}
    updated = corrected_teacher(
        state + ETA * drift, positives, mode="scalar", report=r11_report)
    return updated, {
        "ess_fraction": float(stats["ess_fraction"]),
        "collapsed_row_fraction": float(stats["collapsed_row_fraction"]),
        "denominator_floor_fraction": float(
            stats["denominator_floor_fraction"]),
        "drift_rms_raw": float(stats["drift_rms_raw"]),
        "correction_ratio_median": float(
            r11_report.get("correction_ratio_median", float("nan"))),
    }


def _interval(records: list[dict]) -> dict:
    """Mean over the interval for field stats, median for the R11 ratio (§4.3)."""
    if not records:
        return {}
    out = {}
    for key in ("ess_fraction", "collapsed_row_fraction",
                "denominator_floor_fraction", "drift_rms_raw"):
        out[key] = float(np.mean([r[key] for r in records]))
    out["correction_ratio_median"] = float(np.median(
        [r["correction_ratio_median"] for r in records]))
    out["interval_steps"] = len(records)
    return out


def rollout(start: torch.Tensor, unit: int, regime: str, branch, kernel,
            resolution: int, root: str | None, device,
            checkpoints=CHECKPOINTS, on_checkpoint=None) -> dict:
    """Free-particle rollout under one teacher regime.

    No generator and no optimizer participate: this is what isolates the
    particle basin from the amortization failure of Phases 28-30.
    """
    seeds = unit_seeds(unit)
    steps = max(checkpoints)
    state = start.clone().to(device)
    pool_bank = None
    schedule = None
    if regime == "replay":
        pool_bank = take(resolution, "train", root,
                         allocate(unit, resolution, root).bank).to(device)
        schedule = replay_schedule(unit, steps)
        stream = None
    elif regime == "stochastic":
        stream = np.random.default_rng(seeds["stochastic_stream"])
        target = cifar.cifar_target(resolution, "train", root)
        target.device = device
    else:
        raise ValueError(f"unknown regime {regime!r}")

    # One pre-update evaluation so the K = 0 interval is never empty (§4.3).
    if regime == "replay":
        first = pool_bank[torch.as_tensor(schedule[0], dtype=torch.long)]
    else:
        first = target.sample(POSITIVES, np.random.default_rng(
            derive_seed(seeds["stochastic_stream"], "pre")))
    with torch.no_grad():
        _, pre = frozen_update(state, first, branch, kernel)

    history, pending = [], [pre]
    for step in range(steps + 1):
        if step in checkpoints:
            entry = {"step": step, **_interval(pending)}
            if on_checkpoint is not None:
                entry |= on_checkpoint(state, step)
            history.append(entry)
            pending = []
        if step == steps:
            break
        with torch.no_grad():
            if regime == "replay":
                positives = pool_bank[
                    torch.as_tensor(schedule[step], dtype=torch.long)]
            else:
                positives = target.sample(POSITIVES, stream)
            state, record = frozen_update(state, positives, branch, kernel)
        pending.append(record)
    return {"final": state, "history": history,
            "schedule_digest": (None if schedule is None else
                                hashlib.sha256(
                                    schedule.astype(np.int32).tobytes()
                                ).hexdigest()[:16])}


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """§4.1's reproducibility statistic, with the declared eps guard."""
    return float((a - b).norm() / max(float(a.norm()), 1e-12))


# ---------------------------------------------------------------------------
# Statistics (§6) and vetoes (§7)
# ---------------------------------------------------------------------------

def effective_rank(images: torch.Tensor) -> float:
    """Participation ratio of the covariance spectrum: (sum l)^2 / sum l^2."""
    flat = images.reshape(len(images), -1).double()
    values = torch.linalg.svdvals(flat - flat.mean(dim=0, keepdim=True)) ** 2
    total = float(values.sum())
    return 0.0 if total <= 0 else total ** 2 / float((values ** 2).sum())


def _self_distances(images: torch.Tensor) -> torch.Tensor:
    """Pairwise distances with the diagonal excluded.

    The diagonal is masked, **not** added to.  `gram + eye * inf` looks
    equivalent but `0 * inf = nan`, which poisons every off-diagonal entry --
    it silently made `nearest_bank_normalized` nan, pinned `nn_diversity` at
    2/512 (every argmin returning index 0) and forced `duplicate_rate` to 0.
    """
    flat = images.reshape(len(images), -1).double()
    squared = (flat * flat).sum(dim=1, keepdim=True)
    gram = (squared + squared.T - 2.0 * flat @ flat.T).clamp_min(0.0).sqrt()
    eye = torch.eye(len(flat), dtype=torch.bool, device=flat.device)
    return gram.masked_fill(eye, float("inf"))


def duplicate_rate(images: torch.Tensor, scale: float) -> float:
    """Share of particles with another particle closer than 0.05 x scale."""
    nearest = _self_distances(images).min(dim=1).values
    return float((nearest < 0.05 * scale).to(torch.float32).mean())


def nn_diversity(images: torch.Tensor) -> float:
    """Distinct within-set nearest neighbours, as a fraction of the set."""
    choice = _self_distances(images).argmin(dim=1)
    return float(torch.unique(choice).numel()) / len(images)


def bank_statistics(images: torch.Tensor, bank: torch.Tensor,
                    real_scale: float) -> dict:
    """Normalized nearest-bank distance and distinct-bank count (§7)."""
    flat = images.reshape(len(images), -1).double()
    flat_bank = bank.reshape(len(bank), -1).double()
    a = (flat * flat).sum(dim=1, keepdim=True)
    b = (flat_bank * flat_bank).sum(dim=1, keepdim=True).T
    distance = (a + b - 2.0 * flat @ flat_bank.T).clamp_min(0.0).sqrt()
    nearest = distance.min(dim=1)
    return {
        "nearest_bank_normalized": float(nearest.values.median()) / real_scale,
        "nearest_bank_distribution": nearest.values.cpu().numpy().tolist(),
        "distinct_bank": int(torch.unique(nearest.indices).numel()),
        "claimed_bank_indices": nearest.indices.cpu().numpy().tolist(),
    }


def real_nn_scale(resolution: int, root: str | None, indices: np.ndarray
                  ) -> float:
    """Median real-to-real nearest-neighbour distance, the §7 normalizer."""
    images = take(resolution, "train", root, indices)
    return float(_self_distances(images).min(dim=1).values.median())


def score(images: torch.Tensor, reference: np.ndarray, real: torch.Tensor,
          device, resamples: int = 0, seed: int = 0) -> dict:
    """Full §6 readout. `resamples > 0` adds the particle-resample interval."""
    features = inception_features(images.cpu(), device).double().numpy()
    pr = precision_recall(features, reference)
    scale = float(_self_distances(real[:PARTICLES]).min(dim=1).values.median())
    out = {
        "recall": pr["recall"], "precision": pr["precision"],
        "kid": kid_from_features(features, reference),
        "fid": frechet_from_features(features, reference),
        "alpha": spectrum_slope(images.cpu())["alpha"],
        "tail": _tail(images.cpu()),
        "second_moment": float(images.flatten(1).var(0).mean().cpu()
                               / real.flatten(1).var(0).mean().cpu()),
        "effective_rank": effective_rank(images.cpu()),
        "duplicate_rate": duplicate_rate(images.cpu(), scale),
        "nn_diversity": nn_diversity(images.cpu()),
    }
    if resamples:
        rng = np.random.default_rng(seed)
        recalls, kids = [], []
        for _ in range(resamples):
            pick = rng.integers(0, len(features), len(features))
            resampled = features[pick]
            recalls.append(precision_recall(resampled, reference)["recall"])
            kids.append(kid_from_features(resampled, reference))
        # Conditional on the fixed reference: resampling the particles
        # re-estimates the generated k-NN radii, which is the manifold-side
        # uncertainty the binomial reference-side interval omits (§6).
        out["recall_ci"] = [float(np.percentile(recalls, 2.5)),
                            float(np.percentile(recalls, 97.5))]
        out["recall_resample_sd"] = float(np.std(recalls, ddof=1))
        out["kid_se"] = float(np.std(kids, ddof=1))
        out["uncertainty_is_conditional_on_reference"] = True
    return out


def clopper_pearson_upper(exceedances: int, trials: int,
                          confidence: float = 0.95) -> float:
    """One-sided exact upper bound on a binomial rate (§2.1 step 2)."""
    if exceedances < 0 or exceedances > trials:
        raise ValueError("exceedances must lie in [0, trials]")
    if exceedances == trials:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(confidence, exceedances + 1,
                          trials - exceedances))


def source_cloud(arm: str, unit: int, resolution: int, root: str | None,
                 device, ae_model=None, trained_bad=None) -> torch.Tensor:
    """The §3 frozen constructors. Sources needing training are passed in."""
    seeds = unit_seeds(unit)
    allocation = allocate(unit, resolution, root)
    if arm == "real_data":
        return take(resolution, "train", root, allocation.real_data).to(device)
    if arm == "random_generator":
        model = OneStepGenerator(32, 3, resolution, 64,
                                 seeds["source_init"]).to(device)
        latent = sample_latent(PARTICLES, 32, seeds["source_latent"], device)
        with torch.no_grad():
            return model(latent).detach()
    if arm == "trained_bad":
        if trained_bad is None:
            raise ValueError("trained_bad requires its regenerated tensor")
        return trained_bad.to(device)
    if arm == "ae_reconstruction":
        if ae_model is None:
            raise ValueError("ae_reconstruction requires its autoencoder")
        inputs = take(resolution, "train", root, allocation.ae_inputs).to(device)
        with torch.no_grad():
            return ae_model(inputs).detach()
    if arm == "basin_interpolation":
        # near_data is the index-paired real_data start, matching Phase 27's
        # construction (real data blended with the trained cloud) -- not the
        # autoencoder reconstructions (§3, §15.7).
        near = take(resolution, "train", root, allocation.real_data).to(device)
        if trained_bad is None:
            raise ValueError("basin_interpolation requires trained_bad")
        return 0.4 * near + 0.6 * trained_bad.to(device)
    if arm == "ambient_noise":
        generator = torch.Generator().manual_seed(
            seeds["source_latent"] % (2 ** 31))
        noise = torch.randn(PARTICLES, 3, resolution, resolution,
                            generator=generator) * 0.5
        return noise.clamp(-1.0, 1.0).to(device)
    raise ValueError(f"unknown arm {arm!r}")
