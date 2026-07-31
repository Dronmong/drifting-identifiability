"""Why does the attractor sit at ~0.4 of the data's second moment?

Five mechanism hypotheses have been refuted.  All five were about the
*shape* of the cloud (effective dimension, anisotropy, per-step
contraction).  R23 showed the controlling variable is its *scale* -- the
second-moment ratio -- which is a different and much simpler object, and one
that is analytically tractable in the Gaussian case.

The derivation this module tests
--------------------------------
For an isotropic Gaussian target ``p = N(0, s^2 I)`` and cloud
``q = N(0, sigma^2 I)`` with a Gaussian kernel of bandwidth ``tau``, the
mean-shift field is

    V(x) = tau^2 [ 1/(sigma^2 + tau^2) - 1/(s^2 + tau^2) ] x  =  c(sigma) x

so it is radial, and its sign is positive exactly when ``sigma < s``.  With
**RMS normalization** the field becomes ``V_norm = sign(c) x / sigma``, whose
magnitude does *not* vanish as ``sigma -> s``.  The teacher is then

    T = x (1 + eta sign(c) / sigma)

and a generator following a fraction of the way each step moves ``sigma`` by
``+- lambda eta`` per step forever.  **The scale therefore cannot settle: it
limit-cycles about ``s`` with an amplitude set by ``eta``,** and a time
average taken off-centre reads as a deficit.

Two falsifiable predictions follow:

  P1  with ``normalization="none"`` the field magnitude vanishes as
      ``sigma -> s``, so the second moment should converge to the target's;
  P2  with RMS normalization the deficit should grow with ``eta``.

If both hold, the mechanism is the normalization -- a repository convention,
not part of the paper's Algorithm 2 -- and R11 has been compensating for it.

    uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_secondmoment
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
from .diagnostics import provenance, write_json
from .objectives import variance_matched_teacher

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED_OFFSET = 9000


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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def run(LD, target, dim: int, seed: int, *, steps: int, batch: int,
        eta: float, normalization: str, variance_match: bool,
        probe: int = 2048) -> dict:
    rng = np.random.default_rng(derive_seed(seed, "sm", dim, normalization))
    model = MLP(8, dim, 64, derive_seed(seed, "sm-init"))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    generator = torch.Generator().manual_seed(
        derive_seed(seed, "sm-latent", dim) % (2 ** 31))
    reference = target.sample(probe, rng)
    reference_variance = float(
        torch.tensor(reference, dtype=torch.float32).var(0).mean())
    trace = []
    for step in range(steps):
        positives = target.sample(batch, rng)
        latent = torch.randn(batch, 8, generator=generator)
        with torch.no_grad():
            frozen = model(latent)
            drift = torch.tensor(
                LD.drift_paper(frozen.numpy().astype(np.float64), positives,
                               0.5, False), dtype=torch.float32)
            if normalization == "rms":
                drift = drift / drift.pow(2).sum(1).mean().sqrt().clamp_min(
                    1e-12)
            teacher = frozen + eta * drift
            if variance_match:
                teacher = variance_matched_teacher(
                    teacher, torch.tensor(positives, dtype=torch.float32))
        loss = ((model(latent) - teacher) ** 2).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(steps // 10, 1) == 0:
            with torch.no_grad():
                sample = model(torch.randn(512, 8, generator=generator))
            trace.append({"step": step,
                          "second_moment_ratio": float(
                              sample.var(0).mean()) / reference_variance})

    latent = torch.randn(probe, 8, generator=generator)
    with torch.no_grad():
        generated = model(latent)
    return {
        "dim": dim, "seed": seed, "eta": eta,
        "normalization": normalization, "variance_match": variance_match,
        "second_moment_ratio": float(
            generated.var(0).mean()) / reference_variance,
        "effective_dimension_ratio": (
            M.effective_dimension(generated)
            / max(M.effective_dimension(
                torch.tensor(reference, dtype=torch.float32)), 1e-12)),
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
                        default=HERE / "second_moment_study.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(ROOT / "numerics"))
    import lowdim_drift as LD                              # noqa: PLC0415

    started = time.time()
    rows = []
    print(f"{'dim':>4}{'eta':>6}{'norm':>6}{'R11':>6}"
          f"{'2nd_moment':>12}{'eff_dim':>9}{'ed2':>9}")
    for dim in (int(x) for x in args.dims.split(",")):
        target = LD.gauss_mixture(f"gauss_{dim}d", 4, dim, 0.15)
        for index in range(args.seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            for normalization in ("rms", "none"):
                for eta in (0.1, 0.5, 2.0):
                    for match in (False, True):
                        if match and eta != 0.5:
                            continue
                        row = run(LD, target, dim, seed, steps=args.steps,
                                  batch=args.batch, eta=eta,
                                  normalization=normalization,
                                  variance_match=match)
                        rows.append(row)
                        print(f"{dim:4}{eta:6.1f}{normalization:>6}"
                              f"{str(match):>6}"
                              f"{row['second_moment_ratio']:12.4f}"
                              f"{row['effective_dimension_ratio']:9.3f}"
                              f"{row['ed2']:9.4f}", flush=True)

    def median(pred, key: str) -> float:
        values = [r[key] for r in rows if pred(r)]
        return float(np.median(values)) if values else float("nan")

    summary = {}
    for normalization in ("rms", "none"):
        for eta in (0.1, 0.5, 2.0):
            key = f"norm={normalization}_eta={eta}"
            summary[key] = {
                "second_moment_ratio": median(
                    lambda r, n=normalization, e=eta:
                    r["normalization"] == n and r["eta"] == e
                    and not r["variance_match"], "second_moment_ratio"),
                "ed2": median(
                    lambda r, n=normalization, e=eta:
                    r["normalization"] == n and r["eta"] == e
                    and not r["variance_match"], "ed2"),
            }
    summary["norm=rms_eta=0.5_R11"] = {
        "second_moment_ratio": median(
            lambda r: r["normalization"] == "rms" and r["variance_match"],
            "second_moment_ratio"),
        "ed2": median(
            lambda r: r["normalization"] == "rms" and r["variance_match"],
            "ed2")}

    # P1: does removing the normalization fix the second moment?
    rms_mid = summary["norm=rms_eta=0.5"]["second_moment_ratio"]
    none_mid = summary["norm=none_eta=0.5"]["second_moment_ratio"]
    # P2: does the deficit grow with eta under RMS normalization?
    deficits = [abs(1.0 - summary[f"norm=rms_eta={e}"]["second_moment_ratio"])
                for e in (0.1, 0.5, 2.0)]
    payload = {
        "status": "second-moment-mechanism-study-feeds-no-gate",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "elapsed_seconds": time.time() - started,
        "summary": summary,
        "P1_normalization_fixes_scale": bool(
            np.isfinite(none_mid) and np.isfinite(rms_mid)
            and abs(1 - none_mid) < abs(1 - rms_mid) / 2),
        "P2_deficit_grows_with_eta": bool(
            all(np.isfinite(d) for d in deficits)
            and deficits[0] < deficits[1] < deficits[2]),
        "rms_deficits_by_eta": deficits,
        "rows": rows,
    }
    digest = write_json(args.out, payload)

    print("\n=== SECOND-MOMENT MECHANISM ===")
    for key in sorted(summary):
        s = summary[key]
        print(f"  {key:26} 2nd_moment={s['second_moment_ratio']:7.4f} "
              f"ed2={s['ed2']:8.4f}")
    print(f"\n  P1 (removing RMS normalization fixes the scale): "
          f"{payload['P1_normalization_fixes_scale']}")
    print(f"  P2 (deficit grows with eta under RMS): "
          f"{payload['P2_deficit_grows_with_eta']}  "
          f"deficits={[round(d, 4) for d in deficits]}")
    print(f"  wrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
