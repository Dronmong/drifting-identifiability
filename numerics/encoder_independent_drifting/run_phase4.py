"""Phase 4: is the teacher contraction general, and is it mechanistic?

Executes `EncoderIndependentPhase4Protocol.md`.

P4A  generality grid, including the paper's declared temperature grid and
     Algorithm-2 self-mask, which this program had never tested;
P4B  cross-harness replication using `numerics/lowdim_drift.py` -- this
     repository's independently audited verbatim Algorithm-2 port, imported
     unmodified.  The pivot: a failure here shrinks R11 to a local fix;
P4C  analytic characterization -- measure the teacher's conditional noise,
     predict the contraction, and check the prediction against batch size;
P4D  residual-gap diagnostic (exploratory, no gate).

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from . import metrics as M
from . import oracle as O
from .config import (
    ArmConfig, FieldConfig, GeometryConfig, MASTER_SEED, MixtureConfig,
    ObjectiveConfig, TrainConfig, derive_seed,
)
from .diagnostics import paired_log_ratio, provenance, write_json
from .evaluate import evaluate_arm, evaluation_pools, null_reference
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import variance_matched_teacher
from .train import train_arm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Frozen (protocol section 3): disjoint from every earlier seed set.
PHASE4_SEED_OFFSET = 3000

# Frozen thresholds.
R11_RATIO = 0.60
DIMENSION_WITH = 0.60
DIMENSION_WITHOUT = 0.50
CROSS_HARNESS_WITHOUT = 0.50
CROSS_HARNESS_WITH = 0.80
PREDICTION_TOLERANCE = 1.5


# ---------------------------------------------------------------------------
# P4A: generality grid
# ---------------------------------------------------------------------------


def _arm(arm_id: str, variance_match: bool, *, family: str = "raw",
         self_mask: bool = False, tau: float | None = None,
         anchor: bool = False) -> ArmConfig:
    return ArmConfig(
        arm_id, anchor,
        GeometryConfig(family=family, base_kernel="smooth_laplace",
                       second_order=(family == "scattering"),
                       bandwidth_tau=tau),
        FieldConfig(direction_mode="paper", self_mask=self_mask),
        MixtureConfig(adaptive=False),
        ObjectiveConfig(lambda_anchor=1.0 if anchor else 0.0,
                        lambda_geometry=1.0,
                        teacher_variance_match=variance_match),
        note=arm_id)


def p4a_cells() -> list[dict]:
    """One factor at a time around the declared base point."""
    base = {"family": "raw", "batch": 64, "resolution": 16,
            "self_mask": False, "tau": None, "anchor": False}
    cells = [dict(base, label="base")]
    for batch in (32, 128):
        cells.append(dict(base, batch=batch, label=f"batch{batch}"))
    cells.append(dict(base, resolution=32, label="res32"))
    cells.append(dict(base, family="wavelet", label="wavelet"))
    for tau in (0.02, 0.05, 0.2):
        cells.append(dict(base, tau=tau, label=f"tau{tau}"))
    cells.append(dict(base, self_mask=True, label="selfmask"))
    cells.append(dict(base, anchor=True, label="anchor"))
    return cells


def p4a_generality(seeds: int, steps: int, root: str | None) -> dict:
    rows = []
    for cell in p4a_cells():
        train = cifar.cifar_target(cell["resolution"], "train", root)
        evaluation = cifar.cifar_target(cell["resolution"], "eval", root)
        config = TrainConfig(steps=steps, batch=cell["batch"],
                             controller_batch=32, audit_batch=32,
                             eval_samples=512,
                             image_size=cell["resolution"])
        for index in range(seeds):
            seed = MASTER_SEED + PHASE4_SEED_OFFSET + index
            pools = evaluation_pools(evaluation, config, seed)
            null = null_reference(evaluation, pools, seed)
            for variance_match in (False, True):
                arm = _arm(f"{cell['label']}_r11={variance_match}",
                           variance_match, family=cell["family"],
                           self_mask=cell["self_mask"], tau=cell["tau"],
                           anchor=cell["anchor"])
                outcome = train_arm(arm, train, config, seed)
                row = {"cell": cell["label"], "seed": seed,
                       "r11": variance_match, **{
                           k: v for k, v in cell.items() if k != "label"}}
                row.update(evaluate_arm(outcome, evaluation, config, pools,
                                        null, seed))
                rows.append(row)
            print(f"    P4A {cell['label']:12} seed{index} "
                  f"off={rows[-2].get('geometry_score_v2', float('nan')):7.3f}"
                  f"/{rows[-2].get('effective_dimension_ratio', float('nan')):5.3f}"
                  f"  on={rows[-1].get('geometry_score_v2', float('nan')):7.3f}"
                  f"/{rows[-1].get('effective_dimension_ratio', float('nan')):5.3f}",
                  flush=True)
    return {"rows": rows, "verdicts": _p4a_verdicts(rows)}


def _p4a_verdicts(rows: list[dict]) -> dict:
    out = {}
    for label in sorted({r["cell"] for r in rows}):
        on = {r["seed"]: r for r in rows
              if r["cell"] == label and r["r11"]}
        off = {r["seed"]: r for r in rows
               if r["cell"] == label and not r["r11"]}
        keys = sorted(set(on) & set(off))
        ratio = paired_log_ratio(
            [on[k].get("geometry_score_v2", np.inf) for k in keys],
            [off[k].get("geometry_score_v2", np.inf) for k in keys])

        def dim(rows_by_seed):
            values = [rows_by_seed[k].get("effective_dimension_ratio")
                      for k in keys]
            values = [float(v) for v in values
                      if isinstance(v, (int, float)) and np.isfinite(v)]
            return float(np.median(values)) if values else float("nan")

        dimension_on, dimension_off = dim(on), dim(off)
        out[label] = {
            **ratio, "dimension_with": dimension_on,
            "dimension_without": dimension_off,
            "pass": bool(np.isfinite(ratio.get("ratio", np.nan))
                         and ratio["ratio"] <= R11_RATIO
                         and dimension_on >= DIMENSION_WITH
                         and dimension_off <= DIMENSION_WITHOUT)}
    return out


# ---------------------------------------------------------------------------
# P4B: cross-harness replication in the repository's audited port
# ---------------------------------------------------------------------------


class LowDimGenerator(torch.nn.Module):
    """Small MLP generator for the low-dimensional harness."""

    def __init__(self, latent: int, dim: int, width: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed) % (2 ** 31))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, dim))
        for module in self.net:
            if isinstance(module, torch.nn.Linear):
                bound = 1.0 / max(module.in_features, 1) ** 0.5
                with torch.no_grad():
                    module.weight.copy_((torch.rand(
                        module.weight.shape, generator=generator) * 2 - 1)
                        * bound)
                    module.bias.zero_()
        self.latent = latent

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def p4b_cross_harness(seeds: int, steps: int, batch: int,
                      eta: float = 0.5) -> dict:
    """Reproduce the contraction with `lowdim_drift.drift_paper`.

    `lowdim_drift.py` is imported unmodified: its Algorithm-2 field and its
    target families are used as-is.  Only the generator and training loop are
    new, because that harness is particle-based and has none.
    """
    sys.path.insert(0, str(ROOT / "numerics"))
    import lowdim_drift as LD                              # noqa: PLC0415

    targets = [
        ("gauss_8d_K4", LD.gauss_mixture("gauss_8d_K4", 4, 8, 0.15)),
        ("gauss_16d_K6", LD.gauss_mixture("gauss_16d_K6", 6, 16, 0.15)),
        ("ring_2d", LD.ring_target("ring_2d")),
        ("moons_2d", LD.moons_target("moons_2d")),
    ]
    rows = []
    for name, target in targets:
        for index in range(seeds):
            seed = MASTER_SEED + PHASE4_SEED_OFFSET + index
            data_rng = np.random.default_rng(derive_seed(seed, "p4b", name))
            reference = target.sample(2048, data_rng)
            reference_dimension = M.effective_dimension(
                torch.tensor(reference, dtype=torch.float32))
            for variance_match in (False, True):
                rng = np.random.default_rng(
                    derive_seed(seed, "p4b-train", name))
                model = LowDimGenerator(8, target.d, 64,
                                        derive_seed(seed, "p4b-init"))
                optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
                torch_generator = torch.Generator().manual_seed(
                    derive_seed(seed, "p4b-latent") % (2 ** 31))
                for _ in range(steps):
                    positives = target.sample(batch, rng)
                    latent = torch.randn(batch, 8,
                                         generator=torch_generator)
                    output = model(latent)
                    with torch.no_grad():
                        cloud = output.detach().numpy().astype(np.float64)
                        # The repository's own audited Algorithm-2 field.
                        drift = LD.drift_paper(cloud, positives, 0.5, False)
                        teacher = torch.tensor(
                            cloud + eta * drift, dtype=torch.float32)
                        if variance_match:
                            teacher = variance_matched_teacher(
                                teacher, torch.tensor(positives,
                                                      dtype=torch.float32))
                    loss = ((output - teacher) ** 2).sum(dim=1).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                latent = torch.randn(2048, 8, generator=torch_generator)
                with torch.no_grad():
                    generated = model(latent)
                dimension = M.effective_dimension(generated)
                rows.append({
                    "target": name, "dim": target.d, "seed": seed,
                    "r11": variance_match,
                    "effective_dimension": dimension,
                    "reference_effective_dimension": reference_dimension,
                    "effective_dimension_ratio": (
                        dimension / max(reference_dimension, 1e-12)),
                    "ed2": LD.energy_distance2(
                        generated.numpy().astype(np.float64), reference),
                })
            print(f"    P4B {name:14} seed{index} "
                  f"eff_dim off={rows[-2]['effective_dimension_ratio']:5.3f} "
                  f"on={rows[-1]['effective_dimension_ratio']:5.3f}  "
                  f"ed2 off={rows[-2]['ed2']:7.4f} "
                  f"on={rows[-1]['ed2']:7.4f}", flush=True)

    verdicts = {}
    for name in sorted({r["target"] for r in rows}):
        def median(flag: bool, key: str, name=name) -> float:
            values = [r[key] for r in rows
                      if r["target"] == name and r["r11"] is flag]
            return float(np.median(values)) if values else float("nan")
        without = median(False, "effective_dimension_ratio")
        with_fix = median(True, "effective_dimension_ratio")
        verdicts[name] = {
            "dimension_without": without, "dimension_with": with_fix,
            "ed2_without": median(False, "ed2"),
            "ed2_with": median(True, "ed2"),
            "pass": bool(without < CROSS_HARNESS_WITHOUT
                         and with_fix > CROSS_HARNESS_WITH)}
    passed = sum(1 for v in verdicts.values() if v["pass"])
    return {"rows": rows, "verdicts": verdicts, "families_passed": passed,
            "pass": passed >= 3,
            "dimension_sweep": _p4b_dimension_sweep(LD, seeds, steps, batch,
                                                    eta)}


def _p4b_dimension_sweep(LD, seeds: int, steps: int, batch: int,
                         eta: float) -> dict:
    """Where does the contraction switch on?

    The regression-attenuation account predicts the contraction is governed
    by how well the teacher is determined -- i.e. by dimension relative to
    batch -- not by dimension alone or by the harness.  A 2-D target at
    batch 64 should show none; a 32-D target at the same batch should show a
    lot.  This maps the onset, which a binary pass/fail cannot.
    """
    rows = []
    for dim in (2, 4, 8, 16, 32, 64):
        target = LD.gauss_mixture(f"gauss_{dim}d", 4, dim, 0.15)
        for index in range(seeds):
            seed = MASTER_SEED + PHASE4_SEED_OFFSET + index
            data_rng = np.random.default_rng(
                derive_seed(seed, "p4b-dim", dim))
            reference = target.sample(2048, data_rng)
            reference_dimension = M.effective_dimension(
                torch.tensor(reference, dtype=torch.float32))
            record = {"dim": dim, "seed": seed,
                      "reference_effective_dimension": reference_dimension}
            for variance_match in (False, True):
                rng = np.random.default_rng(
                    derive_seed(seed, "p4b-dim-train", dim))
                model = LowDimGenerator(8, dim, 64,
                                        derive_seed(seed, "p4b-dim-init"))
                optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
                torch_generator = torch.Generator().manual_seed(
                    derive_seed(seed, "p4b-dim-latent", dim) % (2 ** 31))
                for _ in range(steps):
                    positives = target.sample(batch, rng)
                    latent = torch.randn(batch, 8, generator=torch_generator)
                    output = model(latent)
                    with torch.no_grad():
                        cloud = output.detach().numpy().astype(np.float64)
                        drift = LD.drift_paper(cloud, positives, 0.5, False)
                        teacher = torch.tensor(cloud + eta * drift,
                                               dtype=torch.float32)
                        if variance_match:
                            teacher = variance_matched_teacher(
                                teacher, torch.tensor(positives,
                                                      dtype=torch.float32))
                    loss = ((output - teacher) ** 2).sum(dim=1).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                latent = torch.randn(2048, 8, generator=torch_generator)
                with torch.no_grad():
                    generated = model(latent)
                key = "with" if variance_match else "without"
                record[f"eff_dim_{key}"] = (
                    M.effective_dimension(generated)
                    / max(reference_dimension, 1e-12))
                record[f"ed2_{key}"] = LD.energy_distance2(
                    generated.numpy().astype(np.float64), reference)
            rows.append(record)
            print(f"    P4B dim={dim:3} seed{index} "
                  f"eff_dim off={record['eff_dim_without']:5.3f} "
                  f"on={record['eff_dim_with']:5.3f}  "
                  f"ed2 off={record['ed2_without']:7.4f} "
                  f"on={record['ed2_with']:7.4f}", flush=True)

    summary = {}
    for dim in sorted({r["dim"] for r in rows}):
        subset = [r for r in rows if r["dim"] == dim]
        summary[dim] = {
            key: float(np.median([r[key] for r in subset]))
            for key in ("eff_dim_without", "eff_dim_with", "ed2_without",
                        "ed2_with")}
        summary[dim]["ed2_ratio"] = (
            summary[dim]["ed2_with"] / max(summary[dim]["ed2_without"],
                                           1e-12))
    return {"rows": rows, "summary": summary, "batch": batch}


# ---------------------------------------------------------------------------
# P4C: analytic characterization
# ---------------------------------------------------------------------------


def p4c_mechanism(seeds: int, steps: int, root: str | None,
                  batches=(32, 64, 128, 256)) -> dict:
    """Measure the teacher's conditional noise and predict the contraction.

    Regression converges to the conditional mean of the teacher given the
    latent, so it loses exactly the conditional variance:

        Var(fit) = Var(teacher) - E[Var(teacher | z)]

    giving a per-step retention ``rho = 1 - Var(teacher|z)/Var(teacher)``.
    The prediction is that ``rho`` rises with batch size, and that the
    measured effective-dimension ratio follows it in the same order.
    """
    train = cifar.cifar_target(16, "train", root)
    evaluation = cifar.cifar_target(16, "eval", root)
    rows = []
    for batch in batches:
        config = TrainConfig(steps=steps, batch=batch, controller_batch=32,
                             audit_batch=32, eval_samples=512,
                             image_size=16)
        for index in range(seeds):
            seed = MASTER_SEED + PHASE4_SEED_OFFSET + index
            rng = np.random.default_rng(derive_seed(seed, "p4c", batch))
            geometry = GeometryConfig(family="raw",
                                      base_kernel="smooth_laplace")
            branch = build_family(geometry, 3).branches[0]
            kernel = calibrate_block_kernel(
                branch, train.sample(256, rng), "smooth_laplace",
                geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
                geometry.kernel_eps, combine=geometry.combine,
                target_ess_fraction=geometry.target_ess_fraction)

            # Conditional noise of the teacher at a fixed cloud: recompute it
            # against independent target batches and split the variance.
            model = OneStepGenerator(config.latent_dim, 3, 16, config.width,
                                     derive_seed(seed, "generator"))
            latent = sample_latent(128, config.latent_dim,
                                   derive_seed(seed, "p4c-latent"))
            with torch.no_grad():
                output = model(latent)
            teachers = []
            for _ in range(16):
                positives = train.sample(batch, rng)
                drift, _ = KG.field(output, positives, output, branch,
                                    kernel, direction_mode="paper",
                                    normalization="rms", diagnostics=False)
                teachers.append((output + 0.5 * drift).reshape(128, -1))
            stack = torch.stack(teachers)                  # [R, n, d]
            conditional = stack.var(dim=0, unbiased=True).mean()
            total = stack.reshape(-1, stack.shape[-1]).var(
                dim=0, unbiased=True).mean()
            noise_fraction = float(conditional / total.clamp_min(1e-12))
            retention = max(1.0 - noise_fraction, 0.0)

            # Measured outcome at the same batch, no R11.
            pools = evaluation_pools(evaluation, config, seed)
            null = null_reference(evaluation, pools, seed)
            outcome = train_arm(_arm(f"p4c_b{batch}", False), train, config,
                                seed)
            measured = evaluate_arm(outcome, evaluation, config, pools, null,
                                    seed)
            rows.append({
                "batch": batch, "seed": seed,
                "teacher_noise_fraction": noise_fraction,
                "per_step_retention": retention,
                "predicted_dimension_ratio": retention,
                "measured_dimension_ratio": measured.get(
                    "effective_dimension_ratio"),
                "score": measured.get("geometry_score_v2"),
            })
            print(f"    P4C batch={batch:4} seed{index} "
                  f"noise_frac={noise_fraction:6.4f} "
                  f"retention={retention:6.4f} "
                  f"measured_eff_dim="
                  f"{measured.get('effective_dimension_ratio', float('nan')):5.3f}",
                  flush=True)

    def median(batch: int, key: str) -> float:
        values = [r[key] for r in rows if r["batch"] == batch
                  and isinstance(r.get(key), (int, float))
                  and np.isfinite(r[key])]
        return float(np.median(values)) if values else float("nan")

    predicted = [median(b, "predicted_dimension_ratio") for b in batches]
    measured = [median(b, "measured_dimension_ratio") for b in batches]
    order_ok = all(
        (predicted[i] < predicted[i + 1]) == (measured[i] < measured[i + 1])
        for i in range(len(batches) - 1)
        if np.isfinite(predicted[i]) and np.isfinite(predicted[i + 1])
        and np.isfinite(measured[i]) and np.isfinite(measured[i + 1]))
    magnitude_ok = all(
        np.isfinite(p) and np.isfinite(m) and m > 0
        and 1 / PREDICTION_TOLERANCE <= p / m <= PREDICTION_TOLERANCE
        for p, m in zip(predicted, measured))
    return {
        "rows": rows, "batches": list(batches),
        "median_predicted": predicted, "median_measured": measured,
        "ordering_matches": bool(order_ok),
        "magnitude_within_tolerance": bool(magnitude_ok),
        "pass": bool(order_ok and len(batches) >= 3),
    }


# ---------------------------------------------------------------------------
# P4D: residual-gap diagnostic (exploratory, no gate)
# ---------------------------------------------------------------------------


def p4d_residual_gap(steps: int, root: str | None) -> dict:
    """Covariance spectra of the corrected generator, skyline, particles."""
    train = cifar.cifar_target(16, "train", root)
    evaluation = cifar.cifar_target(16, "eval", root)
    config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                         audit_batch=32, eval_samples=512, image_size=16)
    seed = MASTER_SEED + PHASE4_SEED_OFFSET
    pools = evaluation_pools(evaluation, config, seed)
    null = null_reference(evaluation, pools, seed)

    clouds = {"real": pools["eval"]}
    outcome = train_arm(_arm("p4d_r11", True), train, config, seed)
    latent = sample_latent(config.eval_samples, config.latent_dim,
                           derive_seed(seed, "eval-latent"))
    with torch.no_grad():
        clouds["corrected_generator"] = outcome.model(latent)
    skyline = O.train_skyline(train, config, seed, pools, null)
    with torch.no_grad():
        clouds["skyline"] = skyline.model(latent)

    rng = np.random.default_rng(derive_seed(seed, "p4d"))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace")
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace",
        geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
        geometry.kernel_eps, combine=geometry.combine,
        target_ess_fraction=geometry.target_ess_fraction)
    cloud = torch.tensor(rng.normal(scale=0.5, size=(config.eval_samples, 3,
                                                     16, 16)),
                         dtype=torch.float32)
    for index in range(steps):
        drift, _ = KG.field(cloud, train.sample(64, rng), cloud, branch,
                            kernel, direction_mode="paper",
                            normalization="rms", diagnostics=False)
        cloud = cloud + 0.2 * (1.0 - index / steps) * drift
    clouds["free_particles"] = cloud

    out = {}
    for name, sample in clouds.items():
        flat = sample.reshape(len(sample), -1).to(torch.float64)
        spectrum = torch.linalg.svdvals(
            flat - flat.mean(dim=0, keepdim=True)) ** 2
        spectrum = spectrum / spectrum.sum()
        out[name] = {
            "effective_dimension": M.effective_dimension(sample),
            "top1_share": float(spectrum[0]),
            "top8_share": float(spectrum[:8].sum()),
            "top32_share": float(spectrum[:32].sum()),
        }
        print(f"    P4D {name:22} eff_dim="
              f"{out[name]['effective_dimension']:6.2f} "
              f"top1={out[name]['top1_share']:5.3f} "
              f"top8={out[name]['top8_share']:5.3f} "
              f"top32={out[name]['top32_share']:5.3f}", flush=True)
    return {"spectra": out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lowdim-steps", type=int, default=600)
    parser.add_argument("--only", type=str, default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase4.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if not cifar.available(args.root):
        raise SystemExit("CIFAR-10 is not present locally.")

    started = time.time()
    stages = {
        "P4B_cross_harness": lambda: p4b_cross_harness(
            args.seeds, args.lowdim_steps, 64),
        "P4C_mechanism": lambda: p4c_mechanism(
            args.seeds, args.steps, args.root),
        "P4A_generality": lambda: p4a_generality(
            args.seeds, args.steps, args.root),
        "P4D_residual_gap": lambda: p4d_residual_gap(args.steps, args.root),
    }
    wanted = set(stages) if args.only == "all" else set(args.only.split(","))
    results = {}
    for name, function in stages.items():
        if name in wanted:
            print(f"--- {name} ---", flush=True)
            results[name] = function()

    verdicts = {k: v.get("pass") for k, v in results.items()
                if "pass" in v}
    if "P4A_generality" in results:
        cells = results["P4A_generality"]["verdicts"]
        verdicts["P4A_generality"] = all(v["pass"] for v in cells.values())
    payload = {
        "status": "phase4-generality-and-mechanism",
        "scope": "CIFAR-10 16-32px and low-dimensional synthetic targets; "
                 "lowdim_drift.py imported unmodified; no pretrained "
                 "encoder; fresh seeds disjoint from every earlier phase",
        "config": vars(args) | {"out": str(args.out)},
        "seed_offset": PHASE4_SEED_OFFSET,
        "provenance": provenance(),
        "elapsed_seconds": time.time() - started,
        "verdicts": verdicts,
        "results": results,
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 4 ===")
    for name, verdict in verdicts.items():
        print(f"  [{'PASS' if verdict else 'FAIL'}] {name}")
    print(f"  wrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
