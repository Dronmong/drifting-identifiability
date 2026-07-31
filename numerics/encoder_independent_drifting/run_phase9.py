"""Phase 9 (protocol `EncoderIndependentPhase9Protocol.md`).

The solvable case: where does the deficit first appear?

Eleven mechanism hypotheses have been refuted and the deficit is invariant to
nine training axes.  This stops testing interventions and reduces the problem
until it is closed-form.

For a linear generator ``f(z) = Az + b`` the stop-gradient fixed point is
``E[V] = 0`` and ``E[V z^T] = 0``.  For Gaussian data ``N(0, s^2 I)`` and a
Gaussian kernel the mean-shift field is exactly radial,

    V(x) = c(sigma) x ,  c(sigma) = tau^2 [1/(sigma^2+tau^2)
                                           - 1/(s^2+tau^2)]

so ``E[V z^T] = c(sigma) A``, which for non-degenerate ``A`` vanishes only at
``sigma = s``.  **The linear Gaussian case should show no deficit.**

  9A  the solvable case: linear generator, Gaussian data
  9B  the ladder: {linear, MLP} x {Gaussian, CIFAR} -- the first cell with a
      deficit localizes it
  9C  the analytic check: measured equilibrium against the closed-form root

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase9
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .config import MASTER_SEED, derive_seed
from .diagnostics import provenance, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Frozen (protocol section 3): disjoint from every earlier phase.
SEED_OFFSET = 16000
MOMENT_BAND = (0.7, 1.3)

# The Phase-7C bandwidth optimum, expressed as a target ESS fraction and
# solved for per cell so no arm gets a hand-picked temperature.
TARGET_ESS = 0.9


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LinearGenerator(torch.nn.Module):
    """``f(z) = Az + b`` -- the case whose fixed point is solvable."""

    def __init__(self, latent: int, dim: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed) % (2 ** 31))
        weight = torch.randn(dim, latent, generator=generator) / latent ** 0.5
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(torch.zeros(dim))
        self.latent = latent

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.weight.T + self.bias


class MLPGenerator(torch.nn.Module):
    """The same map with a nonlinearity, and nothing else changed."""

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


# ---------------------------------------------------------------------------
# Data laws
# ---------------------------------------------------------------------------


def gaussian_law(dim: int, seed: int, spectrum: str) -> dict:
    """A centred Gaussian, isotropic or with a CIFAR-like decaying spectrum."""
    rng = np.random.default_rng(derive_seed(seed, "p9-law", spectrum))
    if spectrum == "isotropic":
        scale = np.ones(dim)
    elif spectrum == "decaying":
        # A power-law spectrum, the qualitative shape of natural-image PCA.
        scale = (np.arange(1, dim + 1) ** -0.75)
        scale = scale / scale.mean() ** 0.5
    else:
        raise ValueError(f"unknown spectrum {spectrum!r}")
    basis = np.linalg.qr(rng.normal(size=(dim, dim)))[0]
    root = (basis * scale) @ basis.T
    return {"kind": "gaussian", "dim": dim, "root": root,
            "spectrum": spectrum}


def cifar_law(resolution: int, root_dir: str | None) -> dict:
    from . import cifar                                    # noqa: PLC0415
    train = cifar.cifar_target(resolution, "train", root_dir)
    evaluation = cifar.cifar_target(resolution, "eval", root_dir)
    return {"kind": "cifar", "dim": 3 * resolution * resolution,
            "train": train, "eval": evaluation, "spectrum": "cifar"}


def draw(law: dict, count: int, rng) -> np.ndarray:
    if law["kind"] == "gaussian":
        return rng.normal(size=(count, law["dim"])) @ law["root"]
    return law["train"].sample(count, rng).reshape(count, -1).numpy()


def draw_eval(law: dict, count: int, rng) -> np.ndarray:
    if law["kind"] == "gaussian":
        return rng.normal(size=(count, law["dim"])) @ law["root"]
    return law["eval"].sample(count, rng).reshape(count, -1).numpy()


# ---------------------------------------------------------------------------
# Bandwidth: the same target-ESS rule everywhere (no hand-picked temperature)
# ---------------------------------------------------------------------------


def solve_tau(sample: np.ndarray, target_ess: float) -> float:
    """Smallest-error bandwidth whose median row ESS hits the target.

    Target-only, matching the repository's calibration rule: the ESS is
    computed among target samples and never looks at generated output.
    """
    distance = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
    median = float(np.median(distance[distance > 0]))
    lo, hi = median * 1e-3, median * 1e3
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        weights = np.exp(-distance / mid)
        np.fill_diagonal(weights, 0.0)
        weights = weights / np.clip(weights.sum(1, keepdims=True), 1e-300,
                                    None)
        ess = 1.0 / np.clip((weights ** 2).sum(1), 1e-300, None)
        fraction = float(np.median(ess)) / (len(sample) - 1)
        if fraction < target_ess:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


# ---------------------------------------------------------------------------
# The closed-form field, for 9C
# ---------------------------------------------------------------------------


def gaussian_coefficient(sigma2: float, s2: float, tau: float) -> float:
    """``c(sigma)`` for two centred isotropic Gaussians and a Gaussian kernel.

    Zero exactly at ``sigma = s``; positive when the cloud is too small.
    """
    return tau ** 2 * (1.0 / (sigma2 + tau ** 2) - 1.0 / (s2 + tau ** 2))


# ---------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------


def run_cell(LD, law: dict, model_kind: str, seed: int, *, steps: int,
             cloud: int, batch: int, eta: float, latent: int,
             probe: int = 2048) -> dict:
    rng = np.random.default_rng(derive_seed(seed, "p9", model_kind,
                                            law["spectrum"]))
    dim = law["dim"]
    model = (LinearGenerator(latent, dim, derive_seed(seed, "p9-init"))
             if model_kind == "linear"
             else MLPGenerator(latent, dim, 64, derive_seed(seed, "p9-init")))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    generator = torch.Generator().manual_seed(
        derive_seed(seed, "p9-latent") % (2 ** 31))

    tau = solve_tau(draw(law, 128, rng), TARGET_ESS)
    reference = draw_eval(law, probe, rng)
    reference_moment = float(reference.var(axis=0).mean())

    trace = []
    for step in range(steps):
        positives = draw(law, batch, rng)
        z = torch.randn(cloud, latent, generator=generator)
        output = model(z)
        with torch.no_grad():
            drift = torch.tensor(
                LD.drift_paper(output.detach().numpy().astype(np.float64),
                               positives, tau, False),
                dtype=torch.float32)
            rms = drift.pow(2).sum(1).mean().sqrt().clamp_min(1e-12)
            teacher = output.detach() + eta * drift / rms
        loss = ((output - teacher) ** 2).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(steps // 30, 1) == 0:
            with torch.no_grad():
                sample = model(torch.randn(512, latent,
                                           generator=generator))
            trace.append({"step": step,
                          "second_moment_ratio": float(
                              sample.var(0).mean()) / reference_moment})

    with torch.no_grad():
        generated = model(torch.randn(probe, latent, generator=generator))
    values = [t["second_moment_ratio"] for t in trace]
    quarter = max(len(values) // 4, 1)
    late = float(np.median(values[-quarter:]))
    earlier = float(np.median(values[-2 * quarter:-quarter]))
    ratio = float(generated.var(0).mean()) / reference_moment
    return {
        "model": model_kind, "law": law["spectrum"], "seed": seed,
        "tau": tau,
        "second_moment_ratio": ratio,
        "ed2": LD.energy_distance2(
            generated.numpy().astype(np.float64), reference),
        "window_growth": late - earlier,
        # 9C: how far from the closed-form root does it actually sit?
        "gaussian_coefficient": gaussian_coefficient(
            float(generated.var(0).mean()), reference_moment, tau),
        "oscillation": float(np.std(values[-quarter:])),
        "trace": trace,
    }


def _median(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
    return float(np.median(values)) if values else float("nan")


def _in_band(value: float) -> bool:
    low, high = MOMENT_BAND
    return bool(np.isfinite(value) and low <= value <= high)


def ladder(LD, laws: dict, seeds: int, steps: int, cloud: int, batch: int,
           eta: float, latent: int) -> dict:
    rows = []
    for law_name, law in laws.items():
        for model_kind in ("linear", "mlp"):
            for index in range(seeds):
                seed = MASTER_SEED + SEED_OFFSET + index
                row = run_cell(LD, law, model_kind, seed, steps=steps,
                               cloud=cloud, batch=batch, eta=eta,
                               latent=latent)
                rows.append(row)
                print(f"    {law_name:12} {model_kind:7} seed{index} "
                      f"tau={row['tau']:8.4f} "
                      f"2nd={row['second_moment_ratio']:6.3f} "
                      f"ed2={row['ed2']:9.4f} "
                      f"osc={row['oscillation']:6.4f} "
                      f"growth={row['window_growth']:+7.4f}", flush=True)
    summary = {}
    for law_name in laws:
        for model_kind in ("linear", "mlp"):
            group = [r for r in rows if r["law"] == laws[law_name]["spectrum"]
                     and r["model"] == model_kind]
            if not group:
                continue
            moment = _median(group, "second_moment_ratio")
            summary[f"{law_name}_{model_kind}"] = {
                "median_second_moment_ratio": moment,
                "median_ed2": _median(group, "ed2"),
                "median_tau": _median(group, "tau"),
                "median_oscillation": _median(group, "oscillation"),
                "median_window_growth": _median(group, "window_growth"),
                "median_gaussian_coefficient": _median(
                    group, "gaussian_coefficient"),
                "in_band": _in_band(moment),
                "deficit": bool(np.isfinite(moment)
                                and moment < MOMENT_BAND[0])}
    return {"rows": rows, "summary": summary}


def localize(summary: dict) -> dict:
    """Which factor does the first deficit appear with?"""
    def has(name: str) -> bool | None:
        entry = summary.get(name)
        return None if entry is None else entry["deficit"]

    isotropic_linear = has("gaussian_iso_linear")
    isotropic_mlp = has("gaussian_iso_mlp")
    cifar_linear = has("cifar_linear")
    cifar_mlp = has("cifar_mlp")
    factors = []
    if isotropic_mlp and not isotropic_linear:
        factors.append("nonlinearity")
    if cifar_linear and not isotropic_linear:
        factors.append("data law")
    if cifar_mlp and not (cifar_linear or isotropic_mlp):
        factors.append("nonlinearity x data interaction")
    if isotropic_linear:
        factors.append("present already in the solvable case")
    return {
        "linear_gaussian_deficit": isotropic_linear,
        "mlp_gaussian_deficit": isotropic_mlp,
        "linear_cifar_deficit": cifar_linear,
        "mlp_cifar_deficit": cifar_mlp,
        "implicated": factors or ["none -- no cell shows a deficit"],
        "meaning": (
            "the reduction removed the phenomenon; this is a null result "
            "about the reduction, not about the recipe"
            if not any([isotropic_linear, isotropic_mlp, cifar_linear,
                        cifar_mlp])
            else f"first appears with: {', '.join(factors)}"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--cloud", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase9.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(ROOT / "numerics"))
    import lowdim_drift as LD                              # noqa: PLC0415

    started = time.time()
    seed = MASTER_SEED + SEED_OFFSET
    laws = {
        "gaussian_iso": gaussian_law(args.dim, seed, "isotropic"),
        "gaussian_decay": gaussian_law(args.dim, seed, "decaying"),
        "cifar": cifar_law(args.resolution, args.data_root),
    }

    print("=== 9A/9B: the ladder ===", flush=True)
    stage = ladder(LD, laws, args.seeds, args.steps, args.cloud, args.batch,
                   args.eta, args.latent)
    where = localize(stage["summary"])

    payload = {
        "status": "phase9-frozen-protocol",
        "protocol": "numerics/EncoderIndependentPhase9Protocol.md",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "prediction": "P-lin: the linear generator on Gaussian data has no "
                      "second-moment deficit (E[V z^T] = c(sigma) A = 0 "
                      "forces sigma = s)",
        "moment_band": list(MOMENT_BAND),
        "target_ess": TARGET_ESS,
        "elapsed_seconds": time.time() - started,
        "summary": stage["summary"],
        "localization": where,
        "rows": [{k: v for k, v in r.items() if k != "trace"}
                 for r in stage["rows"]],
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 9 ===")
    print(f"{'cell':24}{'2nd_mom':>9}{'ed2':>10}{'tau':>9}{'osc':>8}"
          f"{'c(sigma)':>11}  band")
    for key, entry in stage["summary"].items():
        print(f"{key:24}{entry['median_second_moment_ratio']:9.3f}"
              f"{entry['median_ed2']:10.4f}{entry['median_tau']:9.4f}"
              f"{entry['median_oscillation']:8.4f}"
              f"{entry['median_gaussian_coefficient']:11.2e}"
              f"  {'in ' if entry['in_band'] else 'out'}")
    print(f"\n  P-lin (linear+Gaussian has no deficit): "
          f"{not where['linear_gaussian_deficit']}")
    print(f"  localization: {where['meaning']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
