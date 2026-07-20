"""Generate validation-pass figures from standard-profile per-seed CSVs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RUNROOT = HERE / "bench_runs_v2"
OUT = HERE / "bench_figs_v2"
OUT.mkdir(exist_ok=True)


def standard_run(which: str) -> Path:
    candidates = []
    for manifest in RUNROOT.glob(f"*-{which}/manifest.json"):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data["config"]["profile"]["name"] == "standard":
            candidates.append(manifest.parent)
    if not candidates:
        raise FileNotFoundError(f"no standard run for {which}")
    return sorted(candidates)[-1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def quartiles(values):
    return np.quantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75])


# C1: separate bandwidth-only and joint-policy conclusions.
c1 = read_csv(standard_run("C1-validated-bandwidth") / "c1_per_seed.csv")
groups = defaultdict(list)
for row in c1:
    groups[(int(row["K"]), int(row["d"]), row["arm"])].append(float(row["final_ED2"]))
targets = [(4, 1), (4, 2), (8, 2), (4, 5)]
panels = [
    ("Equal step + equal kernel budget",
     ["fine-common", "coarse-common", "average-common",
      "anneal-oracle-common", "oracle-grid-common"]),
    ("Joint bandwidth/step policies",
     ["fine-joint", "coarse-joint", "anneal-oracle-joint"]),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, (title, arms) in zip(axes, panels):
    x = np.arange(len(targets))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        med, lower, upper = [], [], []
        for K, d in targets:
            q1, q2, q3 = quartiles(groups[(K, d, arm)])
            med.append(q2); lower.append(q2 - q1); upper.append(q3 - q2)
        ax.bar(x + (i - (len(arms) - 1) / 2) * width, med, width,
               yerr=np.vstack([lower, upper]), capsize=2, label=arm)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K{K}/d{d}" for K, d in targets])
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(fontsize=7)
axes[0].set_ylabel("final squared energy distance")
fig.suptitle("Validated C1: scheduling helps, but the best policy is geometry-dependent")
fig.tight_layout()
fig.savefig(OUT / "c1_validated.png", dpi=160)


# C2: operating step vs full-generator boundary.
c2 = read_csv(standard_run("C2-validated-generator") / "c2_per_seed.csv")
groups2 = defaultdict(list)
for row in c2:
    groups2[(int(row["d"]), row["tau_regime"], row["arm"],
             float(row["eta_over_tau"]))].append(float(row["final_ED2"]))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, regime in zip(axes, ["sigma", "L"]):
    for d, marker in [(1, "o"), (2, "s")]:
        points = []
        for (dd, rr, arm, eta), vals in groups2.items():
            if dd == d and rr == regime:
                q1, q2, q3 = quartiles(vals)
                points.append((eta, q1, q2, q3, arm))
        points.sort()
        ax.errorbar([p[0] for p in points], [p[2] for p in points],
                    yerr=[[p[2] - p[1] for p in points],
                          [p[3] - p[2] for p in points]],
                    marker=marker, linestyle="-", capsize=2, label=f"d={d}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eta / tau")
    ax.set_title(f"tau = {regime}")
    ax.legend()
axes[0].set_ylabel("final squared energy distance")
fig.suptitle("Validated C2: the spectral boundary is a ceiling, not an operating point")
fig.tight_layout()
fig.savefig(OUT / "c2_validated.png", dpi=160)


# C3: paired mask/no-mask ratios pooled over target weights and batch sizes.
c3 = read_csv(standard_run("C3-validated-mask") / "c3_per_seed.csv")
paired = {}
for row in c3:
    key = (row["target_weights"], row["estimator"], int(row["per_mode"]),
           int(row["batch"]), int(row["seed"]))
    paired.setdefault(key, {})[row["mask"]] = float(row["final_ED2"])
ratio = defaultdict(list)
for key, values in paired.items():
    if "True" in values and "False" in values:
        ratio[(key[1], key[2])].append(values["True"] / values["False"])
fig, ax = plt.subplots(figsize=(6.3, 4.4))
for estimator, marker in [("snis", "o"), ("paper", "s")]:
    xs, med, lo, hi = [], [], [], []
    for per_mode in sorted({k[1] for k in ratio if k[0] == estimator}):
        q1, q2, q3 = quartiles(ratio[(estimator, per_mode)])
        xs.append(per_mode); med.append(q2); lo.append(q2 - q1); hi.append(q3 - q2)
    ax.errorbar(xs, med, yerr=[lo, hi], marker=marker, capsize=3,
                label=estimator)
ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("particles per mode N/K")
ax.set_ylabel("paired final ED ratio: mask / no mask")
ax.set_title("Validated C3: masking is hazardous only in the small-N regime")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "c3_validated.png", dpi=160)

print(f"wrote validation figures to {OUT}")
