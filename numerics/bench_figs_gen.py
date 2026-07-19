"""Generate Phase C figures from the latest bench_runs CSVs."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "numerics/bench_figs"
os.makedirs(FIG, exist_ok=True)
RUN = "numerics/bench_runs"


def latest(pattern):
    c = sorted(glob.glob(os.path.join(RUN, pattern)))
    return c[-1] if c else None


# C1: final ED per arm, grouped by target
d = latest("*C1*")
arr = np.genfromtxt(os.path.join(d, "c1_results.csv"), delimiter=",",
                    names=True, dtype=None, encoding="utf-8")
arms = ["single-fine", "paper-multi", "ladder-eq", "single-L",
        "single-best", "anneal"]
targets = sorted(set(zip(arr["K"], arr["d"], arr["L"])))
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(targets))
width = 0.13
for i, arm in enumerate(arms):
    vals = []
    for (K, dd, L) in targets:
        m = ((arr["K"] == K) & (arr["d"] == dd) & (arr["L"] == L)
             & (arr["arm"] == arm))
        vals.append(arr["ED"][m][0] if m.any() else np.nan)
    ax.bar(x + (i - 2.5) * width, vals, width, label=arm)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([f"K={K}\nd={dd}" for (K, dd, L) in targets])
ax.set_ylabel("final energy distance (log)")
ax.set_title("C1: bandwidth strategy vs final generative quality "
             "(lower is better)")
ax.legend(fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "c1_bandwidth.png"), dpi=140)
print("wrote c1_bandwidth.png")

# C3: mask on/off vs particles-per-mode
d3 = latest("*C3*")
a3 = np.genfromtxt(os.path.join(d3, "c3_results.csv"), delimiter=",",
                   names=True, dtype=None, encoding="utf-8")
fig, ax = plt.subplots(figsize=(5.5, 4))
pm = sorted(set(a3["per_mode"]))
for mask in (False, True):
    eds = [a3["ED"][(a3["per_mode"] == p) & (a3["mask"] == mask)][0]
           for p in pm]
    ax.plot(pm, eds, "o-", label=f"eye-mask={mask}")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("particles per mode  N/K")
ax.set_ylabel("final energy distance (log)")
ax.set_title("C3: the eye-mask hazard at small N/K")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "c3_mask.png"), dpi=140)
print("wrote c3_mask.png")
