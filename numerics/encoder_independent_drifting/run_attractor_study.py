"""Reform R23: characterize the attractor in a small solvable case.

Four mechanisms have been refuted, all by measuring the wrong object at
CIFAR scale.  The surviving description is that the coupled
teacher-plus-partial-fit iteration converges quickly to a fixed point whose
second moment is wrong, and that R11 selects a different fixed point.  That
is a statement about an attractor, and attractors are studied in small
solvable settings, not at 768 dimensions.

This runs the *identical* recipe against low-dimensional Gaussian mixtures,
using `numerics/lowdim_drift.py` -- the repository's independently audited
verbatim Algorithm-2 port, imported unmodified -- and characterizes where it
lands as a function of the two axes the root-cause analysis identified:
`steps_per_teacher` (R22) and the output step cap (R21).

Reported, not gated: this is a characterization, and a threshold on it would
be invented rather than derived.

    uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_attractor_study
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import metrics as M
from .config import MASTER_SEED, derive_seed
from .diagnostics import dimension_verdict, provenance, write_json
from .objectives import variance_matched_teacher

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED_OFFSET = 8000


class MLP(torch.nn.Module):
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


def run_recipe(LD, target, dim: int, seed: int, *, steps: int, batch: int,
               steps_per_teacher: int, output_step_cap: float | None,
               variance_match: bool, eta: float = 0.5,
               probe: int = 2048) -> dict:
    """The exact recipe, with R21 and R22 exposed."""
    rng = np.random.default_rng(derive_seed(seed, "attractor", dim))
    model = MLP(8, dim, 64, derive_seed(seed, "attractor-init"))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    generator = torch.Generator().manual_seed(
        derive_seed(seed, "attractor-latent", dim) % (2 ** 31))
    reference = target.sample(probe, rng)
    reference_dimension = M.effective_dimension(
        torch.tensor(reference, dtype=torch.float32))
    trace = []
    outer = max(steps // max(steps_per_teacher, 1), 1)
    for step in range(outer):
        positives = target.sample(batch, rng)
        latent = torch.randn(batch, 8, generator=generator)
        with torch.no_grad():
            frozen = model(latent)
            drift = torch.tensor(
                LD.drift_paper(frozen.numpy().astype(np.float64), positives,
                               0.5, False), dtype=torch.float32)
            rms = drift.pow(2).sum(1).mean().sqrt().clamp_min(1e-12)
            teacher = frozen + eta * drift / rms
            if variance_match:
                teacher = variance_matched_teacher(
                    teacher, torch.tensor(positives, dtype=torch.float32))
        for _ in range(max(steps_per_teacher, 1)):
            before_parameters = (
                [p.detach().clone() for p in model.parameters()]
                if output_step_cap is not None else None)
            output = model(latent)
            loss = ((output - teacher) ** 2).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if before_parameters is not None:
                with torch.no_grad():
                    realized = float((model(latent) - frozen).norm())
                    requested = float((teacher - frozen).norm())
                    if requested > 0 and realized / requested > \
                            output_step_cap:
                        scale = output_step_cap / (realized / requested)
                        for parameter, before in zip(model.parameters(),
                                                     before_parameters):
                            parameter.copy_(
                                before + (parameter - before) * scale)
        if step % max(outer // 12, 1) == 0:
            with torch.no_grad():
                sample = model(torch.randn(512, 8, generator=generator))
            trace.append({
                "outer_step": step,
                "effective_dimension_ratio": (
                    M.effective_dimension(sample)
                    / max(reference_dimension, 1e-12))})

    latent = torch.randn(probe, 8, generator=generator)
    with torch.no_grad():
        generated = model(latent)
    ratio = (M.effective_dimension(generated)
             / max(reference_dimension, 1e-12))
    flat_reference = torch.tensor(reference, dtype=torch.float32)
    return {
        "dim": dim, "seed": seed, "steps_per_teacher": steps_per_teacher,
        "output_step_cap": output_step_cap,
        "variance_match": variance_match,
        "effective_dimension_ratio": ratio,
        "dimension_verdict": dimension_verdict(ratio)["direction"],
        "second_moment_ratio": float(
            generated.var(0).mean() / flat_reference.var(0).mean()),
        "ed2": LD.energy_distance2(
            generated.numpy().astype(np.float64), reference),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dims", type=str, default="8,32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path,
                        default=HERE / "attractor_study.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(ROOT / "numerics"))
    import lowdim_drift as LD                              # noqa: PLC0415

    started = time.time()
    rows = []
    for dim in (int(x) for x in args.dims.split(",")):
        target = LD.gauss_mixture(f"gauss_{dim}d", 4, dim, 0.15)
        for index in range(args.seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            for per_teacher, cap, match in (
                    (1, None, False),      # the recipe as written
                    (1, None, True),       # + R11
                    (8, None, False),      # + R22
                    (32, None, False),
                    (1, 1.0, False),       # + R21
                    (1, 0.25, False),
                    (8, 1.0, False),       # R21 + R22
                    (8, 1.0, True)):       # all three
                row = run_recipe(
                    LD, target, dim, seed, steps=args.steps,
                    batch=args.batch, steps_per_teacher=per_teacher,
                    output_step_cap=cap, variance_match=match)
                rows.append(row)
                print(f"    dim={dim:3} seed{index} per_teacher={per_teacher:3}"
                      f" cap={str(cap):5} R11={str(match):5} "
                      f"eff_dim={row['effective_dimension_ratio']:6.3f} "
                      f"({row['dimension_verdict']:14}) "
                      f"2nd_moment={row['second_moment_ratio']:6.3f} "
                      f"ed2={row['ed2']:8.4f}", flush=True)

    summary = {}
    for row in rows:
        key = (f"per_teacher={row['steps_per_teacher']}"
               f"_cap={row['output_step_cap']}_R11={row['variance_match']}")
        summary.setdefault(key, {"ed2": [], "ratio": [], "moment": []})
        summary[key]["ed2"].append(row["ed2"])
        summary[key]["ratio"].append(row["effective_dimension_ratio"])
        summary[key]["moment"].append(row["second_moment_ratio"])
    summary = {
        key: {"median_ed2": float(np.median(v["ed2"])),
              "median_dimension_ratio": float(np.median(v["ratio"])),
              "median_second_moment_ratio": float(np.median(v["moment"])),
              "verdict": dimension_verdict(
                  float(np.median(v["ratio"])))["direction"]}
        for key, v in summary.items()}

    payload = {
        "status": "r23-attractor-characterization-reported-not-gated",
        "scope": "low-dimensional Gaussian mixtures via lowdim_drift.py "
                 "(imported unmodified); the identical recipe with R21 and "
                 "R22 exposed",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "elapsed_seconds": time.time() - started,
        "summary": summary,
        "rows": rows,
    }
    digest = write_json(args.out, payload)

    print("\n=== R23 ATTRACTOR CHARACTERIZATION ===")
    print(f"{'configuration':44}{'ed2':>10}{'eff_dim':>9}{'2nd_mom':>9}"
          f"  verdict")
    for key in sorted(summary):
        s = summary[key]
        print(f"{key:44}{s['median_ed2']:10.4f}"
              f"{s['median_dimension_ratio']:9.3f}"
              f"{s['median_second_moment_ratio']:9.3f}  {s['verdict']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
