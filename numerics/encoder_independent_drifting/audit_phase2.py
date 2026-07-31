"""Implementation audit and next-phase regime finding (post-hoc; no gate).

Two jobs:

*Audit* -- question things every previous pass took for granted: whether the
metric's target level is reachable by this generator class at all, whether
the self-interaction term biases the field, and what the reforms cost.

*Regime finding* -- the Phase-1 diagnosis showed the testbed cannot reward
feature geometry, because raw pixel distance already works there.  The
program's whole motivation is the opposite regime.  Rather than guessing at
a harder testbed, A4 measures where pixel geometry provably fails and where
a structured geometry provably does better, before any arm is run.

    uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.audit_phase2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import datasets as D
from . import kernel_gradient as KG
from . import metrics as M
from .config import GeometryConfig, MASTER_SEED, TrainConfig, derive_seed
from .diagnostics import provenance, write_json
from .evaluate import evaluation_pools, null_reference
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent

HERE = Path(__file__).resolve().parent


def _cfg(steps: int = 300, **kwargs) -> TrainConfig:
    return TrainConfig(steps=steps, batch=64, controller_batch=32,
                       audit_batch=32, eval_samples=512, **kwargs)


# ---------------------------------------------------------------------------
# A1: is the metric's target level reachable by this generator class?
# ---------------------------------------------------------------------------


def a1_reachable_score(target_names=("checkerboard", "texture_blocks",
                                     "rings_islands", "pinwheel")) -> dict:
    """Score 1.0 means "as good as a fresh real sample".  Can we get there?

    The targets are prototype-plus-iid-pixel-noise, so their support is
    full dimensional (768 independent noise directions).  The generator maps
    a 32-dimensional latent through a deterministic network, so its output
    lives on a manifold of dimension at most 32.  It therefore *cannot*
    reproduce the target's noise structure, and the composite score has a
    floor that no training procedure can beat.

    Three reference points are measured:
      `fresh_real`     an independent target sample            (score 1.0 by
                       construction);
      `noiseless`      exact prototypes at the true mixture weights -- the
                       best a deterministic low-dimensional generator could
                       possibly do on these targets;
      `memorized`      the generator fitted directly to 512 target images by
                       regression, i.e. its capacity ceiling.
    """
    rows = []
    config = _cfg()
    for name in target_names:
        target = D.named(name)
        pools = evaluation_pools(target, config, MASTER_SEED)
        null = null_reference(target, pools, MASTER_SEED)
        rng = np.random.default_rng(derive_seed(MASTER_SEED, "a1", name))

        references = {"fresh_real": target.sample(config.eval_samples, rng)}
        if target.prototypes is not None:
            weights = (target.component_weights
                       if target.component_weights is not None
                       else np.full(len(target.prototypes),
                                    1 / len(target.prototypes)))
            picks = rng.choice(len(target.prototypes),
                               size=config.eval_samples, p=weights)
            references["noiseless"] = target.prototypes[picks].clone()

        # Capacity ceiling: regress the generator onto a fixed target cohort.
        cohort = target.sample(config.eval_samples, rng)
        model = OneStepGenerator(config.latent_dim, config.channels,
                                 config.image_size, config.width,
                                 derive_seed(MASTER_SEED, "generator"))
        latent = sample_latent(config.eval_samples, config.latent_dim,
                               derive_seed(MASTER_SEED, "eval-latent"))
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        for _ in range(1500):
            loss = ((model(latent) - cohort) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            references["memorized"] = model(latent)
        memorization_mse = float(loss)

        for label, sample in references.items():
            measured = M.raw_metrics(
                sample, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(MASTER_SEED, "a1m", name)),
                target, target_null=pools["null"])
            score = M.normalized_geometry_score_v2(measured, null)
            rows.append({
                "target": name, "reference": label,
                "geometry_score_v2": score["geometry_score"],
                "ed2": measured["ed2"], "precision": measured["precision"],
                "coverage": measured["coverage"],
                "nearest_real": measured.get("nearest_real"),
                "memorization_mse": (memorization_mse
                                     if label == "memorized" else None),
            })
            print(f"    A1 {name:16} {label:11} "
                  f"score={score['geometry_score']:8.3f} "
                  f"prec={measured['precision']:5.3f}", flush=True)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# A2: does the self-interaction term bias the field?
# ---------------------------------------------------------------------------


def a2_self_term_bias(target_names=("checkerboard", "texture_blocks")) -> dict:
    """The generated cloud is reused as negatives without masking.

    That puts ``k(x_i, x_i) = 1`` -- the kernel's maximum -- into the
    negative denominator.  Its own gradient vanishes for a smooth kernel, so
    it contributes nothing to the numerator while inflating the denominator,
    systematically attenuating the repulsive half of the field.  The
    repository's Phase-C result is that masking is a hazard at small
    particle counts and roughly neutral by N/K = 8, so this measures the
    size of the effect here rather than assuming either way.
    """
    rows = []
    for name in target_names:
        target = D.named(name)
        rng = np.random.default_rng(derive_seed(MASTER_SEED, "a2", name))
        calibration = target.sample(256, rng)
        positive = target.sample(64, rng)
        cloud = target.sample(64, rng) * 0.4
        for family in ("raw", "wavelet"):
            config = GeometryConfig(family=family)
            branch = build_family(config, D.CHANNELS).branches[0]
            kernel = calibrate_block_kernel(
                branch, calibration, "smooth_laplace",
                config.bandwidth_quantile, config.bandwidth_multiplier,
                config.kernel_eps, combine=config.combine,
                target_ess_fraction=config.target_ess_fraction)
            for mode in ("standard", "kernel_gradient"):
                unmasked, _ = KG.field(
                    cloud, positive, cloud, branch, kernel,
                    direction_mode=mode, normalization="none",
                    self_mask=False, diagnostics=False)
                masked, _ = KG.field(
                    cloud, positive, cloud, branch, kernel,
                    direction_mode=mode, normalization="none",
                    self_mask=True, diagnostics=False)
                cosine = float(torch.nn.functional.cosine_similarity(
                    unmasked.flatten(), masked.flatten(), dim=0))
                rows.append({
                    "target": name, "family": family, "direction_mode": mode,
                    "unmasked_rms": float(unmasked.pow(2).mean().sqrt()),
                    "masked_rms": float(masked.pow(2).mean().sqrt()),
                    "magnitude_ratio": float(
                        unmasked.norm() / masked.norm().clamp_min(1e-30)),
                    "cosine": cosine,
                })
                print(f"    A2 {name:16} {family:8} {mode:18} "
                      f"|unmasked|/|masked|={rows[-1]['magnitude_ratio']:6.3f}"
                      f" cos={cosine:6.3f}", flush=True)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# A3: what do the reforms cost?
# ---------------------------------------------------------------------------


def a3_reform_cost(target_name: str = "checkerboard") -> dict:
    """Wall-clock of the projected mode against the plain kernel gradient."""
    target = D.named(target_name)
    rng = np.random.default_rng(derive_seed(MASTER_SEED, "a3"))
    calibration = target.sample(256, rng)
    positive = target.sample(64, rng)
    cloud = target.sample(64, rng) * 0.4
    rows = []
    for family in ("raw", "wavelet"):
        config = GeometryConfig(family=family)
        branch = build_family(config, D.CHANNELS).branches[0]
        kernel = calibrate_block_kernel(
            branch, calibration, "smooth_laplace", config.bandwidth_quantile,
            config.bandwidth_multiplier, config.kernel_eps,
            combine=config.combine,
            target_ess_fraction=config.target_ess_fraction)
        for mode in KG.DIRECTION_MODES:
            KG.field(cloud, positive, cloud, branch, kernel,
                     direction_mode=mode, diagnostics=False)
            started = time.perf_counter()
            for _ in range(20):
                KG.field(cloud, positive, cloud, branch, kernel,
                         direction_mode=mode, diagnostics=False)
            rows.append({
                "family": family, "direction_mode": mode,
                "seconds_per_call": (time.perf_counter() - started) / 20,
            })
            print(f"    A3 {family:8} {mode:28} "
                  f"{rows[-1]['seconds_per_call'] * 1000:7.2f} ms/call",
                  flush=True)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# A4: where does pixel geometry actually fail?
# ---------------------------------------------------------------------------


def nuisance_confusion(samples: torch.Tensor, content: np.ndarray,
                       nuisance: np.ndarray, feature=None) -> dict:
    """Does distance rank same-content pairs closer than different-content?

    ``NCR`` is the median distance between two samples of the SAME content
    under DIFFERENT nuisance, divided by the median distance between
    DIFFERENT content under the SAME nuisance.

    ``NCR < 1``  the geometry groups by content -- it is useful;
    ``NCR > 1``  the geometry groups by nuisance -- it is actively
                 misleading, and a kernel built on it will pull the wrong
                 neighbours together.

    This is the quantity the whole encoder-independence program is about: a
    pretrained encoder is worth its cost exactly when pixel NCR > 1 and
    feature NCR < 1.
    """
    flat = (samples.reshape(len(samples), -1) if feature is None
            else feature(samples))
    flat = flat.reshape(len(flat), -1).to(torch.float64)
    distance = torch.cdist(flat, flat)
    same_content = torch.tensor(content[:, None] == content[None, :])
    eye = torch.eye(len(flat), dtype=torch.bool)
    if len(np.unique(nuisance)) > 1:
        same_nuisance = torch.tensor(nuisance[:, None] == nuisance[None, :])
        within = distance[same_content & ~same_nuisance & ~eye]
        between = distance[~same_content & same_nuisance & ~eye]
    else:
        # Real data has no declared nuisance group, so the statistic reduces
        # to within-content over between-content distance.  A value near 1
        # means the geometry carries essentially no content information.
        within = distance[same_content & ~eye]
        between = distance[~same_content & ~eye]
    if len(within) == 0 or len(between) == 0:
        return {"ncr": float("nan"), "within": float("nan"),
                "between": float("nan")}
    return {
        "ncr": float(within.median() / between.median().clamp_min(1e-30)),
        "within": float(within.median()),
        "between": float(between.median()),
    }


def nuisance_sample(resolution: int, shift: int, warp: float,
                    contents: int, count: int, rng: np.random.Generator,
                    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Content prototypes seen under a declared translation/warp nuisance."""
    axis = (np.arange(resolution) - (resolution - 1) / 2) / (resolution / 2)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    prototypes = []
    for index in range(contents):
        angle = np.pi * index / contents
        freq = 2.0 + index
        proj = xx * np.cos(angle) + yy * np.sin(angle)
        body = np.tanh(3 * np.cos(np.pi * freq * proj))
        blob = np.exp(-((xx - 0.3 * np.cos(2 * np.pi * index / contents)) ** 2
                        + (yy - 0.3 * np.sin(2 * np.pi * index / contents))
                        ** 2) / 0.08)
        prototypes.append(np.stack([body, blob * 2 - 1, -body * 0.6]))
    prototypes = np.stack(prototypes)

    offsets = list(range(-shift, shift + 1)) if shift > 0 else [0]
    content = rng.integers(0, contents, count)
    nuisance = rng.integers(0, len(offsets) ** 2, count)
    images = np.empty((count, 3, resolution, resolution))
    for index in range(count):
        base = prototypes[content[index]]
        oy = offsets[nuisance[index] // len(offsets)]
        ox = offsets[nuisance[index] % len(offsets)]
        moved = np.roll(base, (oy, ox), axis=(1, 2))
        if warp > 0:
            amplitude = warp * (0.5 + 0.5 * np.cos(
                2 * np.pi * nuisance[index] / max(len(offsets) ** 2, 1)))
            moved = moved * (1.0 + amplitude * np.cos(2 * np.pi * yy))[None]
        images[index] = moved
    images = images + rng.normal(size=images.shape) * 0.05
    return torch.tensor(images, dtype=torch.float32), content, nuisance


def a4_regime_map(count: int = 192) -> dict:
    """Sweep nuisance strength and resolution; find where pixels fail."""
    rows = []
    for resolution in (16, 24, 32):
        for shift in (0, 1, 2, 4):
            for warp in (0.0, 0.3):
                rng = np.random.default_rng(derive_seed(
                    MASTER_SEED, "a4", resolution, shift, warp))
                samples, content, nuisance = nuisance_sample(
                    resolution, shift, warp, 4, count, rng)
                row = {"resolution": resolution, "shift": shift,
                       "warp": warp, "dimension": 3 * resolution ** 2}
                row["pixel"] = nuisance_confusion(samples, content, nuisance)
                for family in ("wavelet", "scattering", "randconv"):
                    config = GeometryConfig(
                        family=family, second_order=(family == "scattering"))
                    built = build_family(config, 3)

                    def feature(images, built=built):
                        return torch.cat([b.flat(images)
                                          for b in built.branches], dim=1)

                    row[family] = nuisance_confusion(
                        samples, content, nuisance, feature)
                rows.append(row)
                print(f"    A4 res={resolution} shift={shift} warp={warp} "
                      f"pixel NCR={row['pixel']['ncr']:6.3f} "
                      f"wavelet={row['wavelet']['ncr']:6.3f} "
                      f"scatter={row['scattering']['ncr']:6.3f} "
                      f"randconv={row['randconv']['ncr']:6.3f}", flush=True)
        print(f"        k-NN content accuracy (chance .100): "
              f"pixel={row['pixel']['knn']:.3f} "
              f"wavelet={row['wavelet']['knn']:.3f} "
              f"scatter={row['scattering']['knn']:.3f} "
              f"randconv={row['randconv']['knn']:.3f} | "
              f"unsup={row['control_learned_unsupervised']['knn']:.3f} "
              f"sup={row['control_learned_supervised']['knn']:.3f}",
              flush=True)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# A5: does pixel geometry fail on REAL images?
# ---------------------------------------------------------------------------


def load_cifar(resolution: int, count: int, seed: int, root: str | None = None
               ) -> tuple[torch.Tensor, np.ndarray]:
    """CIFAR-10 downsampled to ``resolution``, scaled to roughly [-1, 1].

    Real natural images are the regime the plan is premised on and the one
    the synthetic sweep (A4) could not reach.  Used for measurement only --
    no pretrained network is involved, and the class labels are an ORACLE
    diagnostic used to define "same content", never an input to any
    objective.
    """
    import os

    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(
        root=root or os.path.expanduser("~/.cache/cifar"), download=False)
    rng = np.random.default_rng(seed)
    index = rng.choice(len(dataset.data), size=count, replace=False)
    images = torch.tensor(dataset.data[index], dtype=torch.float32)
    images = images.permute(0, 3, 1, 2) / 127.5 - 1.0
    if resolution != images.shape[-1]:
        images = torch.nn.functional.interpolate(
            images, size=(resolution, resolution), mode="area")
    return images, np.asarray(dataset.targets)[index]


def knn_content_accuracy(features: torch.Tensor, labels: np.ndarray,
                         k: int = 10) -> float:
    """Leave-one-out k-NN accuracy of the content label in this geometry.

    The primary real-data statistic.  NCR -- a ratio of median distances --
    turned out to saturate on CIFAR: pixel geometry scores 0.96 and a
    *supervised* encoder at 100% training accuracy only reaches 0.92, so the
    whole representational range is compressed into 4% and the statistic
    cannot discriminate.  k-NN accuracy measures the same property (does
    distance group by content?) but reads the local neighbourhood a kernel
    actually uses, and it separates representations by tens of points.

    Chance level is 1 / number of classes.
    """
    flat = features.reshape(len(features), -1).to(torch.float32)
    distance = torch.cdist(flat, flat)
    distance.fill_diagonal_(float("inf"))
    neighbours = distance.topk(k, largest=False).indices.numpy()
    votes = np.asarray(labels)[neighbours]
    predicted = np.array([np.bincount(row).argmax() for row in votes])
    return float((predicted == np.asarray(labels)).mean())


def _autoencoder_ncr(images: torch.Tensor, labels: np.ndarray,
                     nuisance: np.ndarray, resolution: int) -> dict:
    """Control: an UNSUPERVISED encoder trained on the same images.

    Fixed features and a learned-but-unsupervised encoder are the fair
    comparison -- neither sees a label.  If this control also sits at the
    pixel level, the limitation is unsupervised representation learning at
    this scale, not compositional features specifically.
    """
    from .reference_encoder import train_reference_encoder
    torch_rng = np.random.default_rng(derive_seed(MASTER_SEED, "a5-ae"))
    pool = images

    def sampler(n: int, rng) -> torch.Tensor:
        index = rng.integers(0, len(pool), n)
        return pool[index]

    model = train_reference_encoder(
        sampler, 3, 32, derive_seed(MASTER_SEED, "a5-ae-init"),
        steps=400, batch=64, learning_rate=2e-3, rng=torch_rng)
    with torch.no_grad():
        codes = model.encoder(images)
    out = nuisance_confusion(codes, labels, nuisance)
    out["knn"] = knn_content_accuracy(codes, labels)
    return out


def _supervised_ncr(images: torch.Tensor, labels: np.ndarray,
                    nuisance: np.ndarray, resolution: int) -> dict:
    """Control: a small SUPERVISED classifier's penultimate features.

    This is the ceiling -- what a representation trained to know the classes
    achieves on this statistic.  It exists to prove the statistic can detect
    content separation when content separation is present.  It is a
    measurement device and never enters a training objective.
    """
    import torch.nn as nn
    torch.manual_seed(derive_seed(MASTER_SEED, "a5-sup") % (2 ** 31))
    body = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(),
        nn.AdaptiveAvgPool2d(2), nn.Flatten())
    head = nn.Linear(64 * 4, 10)
    optimizer = torch.optim.Adam(
        list(body.parameters()) + list(head.parameters()), lr=2e-3)
    target = torch.tensor(labels, dtype=torch.long)
    generator = torch.Generator().manual_seed(
        derive_seed(MASTER_SEED, "a5-sup-batch") % (2 ** 31))
    for _ in range(600):
        index = torch.randint(0, len(images), (128,), generator=generator)
        loss = nn.functional.cross_entropy(
            head(body(images[index])), target[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        features = body(images)
        accuracy = float((head(features).argmax(1) == target).float().mean())
    result = nuisance_confusion(features, labels, nuisance)
    result["train_accuracy"] = accuracy
    result["knn"] = knn_content_accuracy(features, labels)
    return result


def a5_real_image_regime(resolutions=(16, 24), count: int = 2048,
                         root: str | None = None) -> dict:
    """NCR and kernel health on CIFAR-10.

    Without an explicit nuisance group the confusion ratio becomes
    median within-class distance over median between-class distance.  A
    value near 1 means the geometry carries essentially no content
    information -- the condition under which a pretrained encoder is worth
    its cost, and the condition the synthetic sweep never reached.
    """
    rows = []
    for resolution in resolutions:
        try:
            images, labels = load_cifar(
                resolution, count, derive_seed(MASTER_SEED, "a5", resolution),
                root)
        except Exception as error:                       # noqa: BLE001
            return {"available": False,
                    "error": f"{type(error).__name__}: {error}"}
        nuisance = np.zeros(len(labels), dtype=int)      # no nuisance group
        row = {"resolution": resolution, "dimension": 3 * resolution ** 2,
               "count": int(count)}
        row["pixel"] = nuisance_confusion(images, labels, nuisance)
        row["pixel"]["knn"] = knn_content_accuracy(images, labels)
        calibration = images[: count // 2]
        probe = images[count // 2:]
        for family in ("raw", "wavelet", "scattering", "randconv"):
            config = GeometryConfig(
                family=family, second_order=(family == "scattering"))
            built = build_family(config, 3)
            if family != "raw":
                def feature(batch, built=built):
                    return torch.cat([b.flat(batch) for b in built.branches],
                                     dim=1)
                row[family] = nuisance_confusion(
                    images, labels, nuisance, feature)
                row[family]["knn"] = knn_content_accuracy(
                    feature(images), labels)
            branch = built.branches[0]
            kernel = calibrate_block_kernel(
                branch, calibration, "smooth_laplace",
                config.bandwidth_quantile, config.bandwidth_multiplier,
                config.kernel_eps, combine=config.combine,
                target_ess_fraction=config.target_ess_fraction)
            _, health = KG.field(
                probe[:64], calibration[:64], probe[:64], branch, kernel,
                direction_mode="standard", normalization="none")
            row[f"health_{family}"] = {
                "ess_fraction": health["ess_fraction"],
                "affinity_median": health["affinity_median"],
                "collapsed_row_fraction": health["collapsed_row_fraction"],
                "distance_median": health["distance_median"],
                "bandwidth_median": health["bandwidth_median"],
            }
        # Controls.  Without these, "fixed geometry does not separate
        # content" cannot be told apart from "the statistic cannot detect
        # separation at this sample size".  Both controls are measurement
        # only and enter no training objective.
        row["control_learned_unsupervised"] = _autoencoder_ncr(
            images, labels, nuisance, resolution)
        row["control_learned_supervised"] = _supervised_ncr(
            images, labels, nuisance, resolution)

        rows.append(row)
        pixel = row["pixel"]["ncr"]
        print(f"    A5 CIFAR res={resolution} dim={row['dimension']:5} "
              f"pixel NCR={pixel:6.3f} "
              f"wavelet={row['wavelet']['ncr']:6.3f} "
              f"scatter={row['scattering']['ncr']:6.3f} "
              f"randconv={row['randconv']['ncr']:6.3f}", flush=True)
        print(f"        k-NN content accuracy (chance .100): "
              f"pixel={row['pixel']['knn']:.3f} "
              f"wavelet={row['wavelet']['knn']:.3f} "
              f"scatter={row['scattering']['knn']:.3f} "
              f"randconv={row['randconv']['knn']:.3f} | "
              f"unsup={row['control_learned_unsupervised']['knn']:.3f} "
              f"sup={row['control_learned_supervised']['knn']:.3f}",
              flush=True)
        print(f"        raw kernel ESS="
              f"{row['health_raw']['ess_fraction']:6.4f} "
              f"wavelet ESS={row['health_wavelet']['ess_fraction']:6.4f}",
              flush=True)
    return {"available": True, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase2_audit.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()

    stages = {
        "A1_reachable_score": a1_reachable_score,
        "A2_self_term_bias": a2_self_term_bias,
        "A3_reform_cost": a3_reform_cost,
        "A4_regime_map": a4_regime_map,
        "A5_real_image_regime": a5_real_image_regime,
    }
    wanted = set(stages) if args.only == "all" else set(args.only.split(","))
    results = {}
    for name, function in stages.items():
        if name in wanted:
            print(f"--- {name} ---", flush=True)
            results[name] = function()

    payload = {
        "status": "phase2-audit-and-regime-finding-feeds-no-gate",
        "provenance": provenance(),
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    digest = write_json(args.out, payload)
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
