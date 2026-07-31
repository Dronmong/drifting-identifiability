"""Diagnostic aggregation for plan section 10.

Health series are collected per step and reduced here.  Two conventions
matter for honesty:

* everything is reported *per branch*, so a healthy aggregate cannot hide a
  dead branch;
* the anchor-presence check is a frozen threshold on gradient share over a
  frozen fraction of training (plan section 10.2: the anchor is
  "rhetorically present but practically absent" if its share stays below the
  threshold for most of training).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

# Frozen before any arm is run (plan section 9, Phase-1 exit gate item 4).
ANCHOR_SHARE_THRESHOLD = 0.05
ANCHOR_PRESENCE_FRACTION = 0.25


class SeriesLog:
    """Append-only scalar series with median/final reductions."""

    def __init__(self) -> None:
        self.series: dict[str, list[float]] = {}
        self.steps: list[int] = []

    def add(self, step: int, values: dict) -> None:
        self.steps.append(int(step))
        for key, value in _flatten(values).items():
            self.series.setdefault(key, []).append(value)

    def summary(self, keys: tuple[str, ...] = ()) -> dict:
        out: dict[str, float] = {}
        for key, values in self.series.items():
            if keys and not key.startswith(keys):
                continue
            finite = [v for v in values if np.isfinite(v)]
            if not finite:
                continue
            out[f"median_{key}"] = float(np.median(finite))
            out[f"final_{key}"] = float(finite[-1])
        return out

    def tail_mean(self, key: str, fraction: float = 0.25) -> float:
        values = [v for v in self.series.get(key, []) if np.isfinite(v)]
        if not values:
            return float("nan")
        keep = max(1, int(len(values) * fraction))
        return float(np.mean(values[-keep:]))

    def anchor_presence(self) -> dict:
        """Was the anchor practically present, or only rhetorically?"""
        shares = [v for v in self.series.get("anchor_gradient_share", [])
                  if np.isfinite(v)]
        if not shares:
            return {"anchor_present": False, "anchor_share_median": None,
                    "anchor_share_above_threshold_fraction": None}
        above = float(np.mean([s >= ANCHOR_SHARE_THRESHOLD for s in shares]))
        return {
            "anchor_share_median": float(np.median(shares)),
            "anchor_share_above_threshold_fraction": above,
            "anchor_present": bool(above >= ANCHOR_PRESENCE_FRACTION),
        }


def _flatten(values: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}_"))
        elif isinstance(value, bool):
            out[name] = float(value)
        elif isinstance(value, (int, float)):
            out[name] = float(value)
    return out


# Reform R15.  A probe must see at least this many effective neighbours for
# the "kernel geometry" being compared to exist at all; below it the field is
# a nearest-neighbour rule.
MIN_EFFECTIVE_NEIGHBOURS = 2.0

# Reform R20.  Effective dimension has an OPTIMUM, not a monotone benefit.
# Measured on CIFAR-16: ratio .27 scores 7.5, ratio .90 scores 1.8, ratio 1.40
# scores 6.8 and ratio 3.35 scores 3.0 with coverage collapsing to .66.  A
# one-sided floor -- as used in Phase-3 C.5 and Phase-5 G5.2 -- would pass the
# over-dispersed configurations, so every gate on this quantity needs both
# bounds.
DIMENSION_BAND = (0.7, 1.3)


def dimension_verdict(ratio: float,
                      band: tuple[float, float] = DIMENSION_BAND) -> dict:
    """Reform R20: is the generated cloud's effective dimension *matched*?

    Returns the direction of any mismatch rather than a bare pass/fail, so a
    report can distinguish a collapsed cloud from an over-dispersed one --
    they score similarly badly and need opposite corrections.
    """
    low, high = band
    if not 0 < low <= high:
        raise ValueError("the dimension band must satisfy 0 < low <= high")
    if ratio is None or not np.isfinite(ratio):
        return {"matched": False, "direction": "unmeasured",
                "ratio": None, "band": [low, high]}
    if ratio < low:
        direction = "collapsed"
    elif ratio > high:
        direction = "over_dispersed"
    else:
        direction = "matched"
    return {"matched": direction == "matched", "direction": direction,
            "ratio": float(ratio), "band": [low, high]}


def kernel_admissible(stats: dict, batch: int) -> dict:
    """Reform R15: may this cell be scored as a method result?

    Phase-4 scored three cells in which 94%, 0% and 0% of probe rows had
    numerically dead affinities (median affinity 4e-20, 1e-08, 1e-02) and
    drew a conclusion about the paper's operating point from them.  A cell
    whose kernel has collapsed is not evidence about kernel geometry, and
    this makes excluding it mechanical rather than a matter of remembering.

    Excluded cells must still be **reported with their health numbers** --
    silent dropping would be its own reporting defect.
    """
    if batch <= 0:
        raise ValueError("batch must be positive")
    collapsed = float(stats.get("collapsed_row_fraction", 0.0))
    ess_fraction = stats.get("ess_fraction", float("nan"))
    ess_fraction = (float(ess_fraction) if ess_fraction is not None
                    else float("nan"))
    required = MIN_EFFECTIVE_NEIGHBOURS / batch
    reasons = []
    if collapsed > 0.0:
        reasons.append(f"collapsed_row_fraction={collapsed:.4f} > 0")
    if not np.isfinite(ess_fraction):
        reasons.append("ess_fraction is not finite (kernel fully collapsed)")
    elif ess_fraction < required:
        reasons.append(
            f"ess_fraction={ess_fraction:.5f} < {required:.5f} "
            f"({MIN_EFFECTIVE_NEIGHBOURS} effective neighbours)")
    return {
        "admissible": not reasons,
        "reasons": reasons,
        "collapsed_row_fraction": collapsed,
        "ess_fraction": ess_fraction,
        "required_ess_fraction": required,
        "affinity_median": stats.get("affinity_median"),
    }


def geometric_mean(values) -> float:
    finite = [float(v) for v in values
              if np.isfinite(v) and float(v) > 0]
    if not finite:
        return float("nan")
    return float(np.exp(np.mean(np.log(finite))))


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


def provenance() -> dict:
    """Commit, dirty flag, source hashes and environment for an artifact."""
    import torch
    status = _git("status", "--porcelain")
    sources = sorted(PACKAGE.glob("*.py")) + sorted(
        (PACKAGE / "tests").glob("*.py"))
    return {
        "commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"):
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cmdline": sys.argv,
    }


def write_json(path: Path, payload: dict) -> str:
    """Write canonical JSON plus a sidecar SHA-256, like the other runners."""
    def safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, (np.integer, np.floating)):
            return safe(value.item())
        if isinstance(value, np.ndarray):
            return safe(value.tolist())
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(safe(payload), indent=2, sort_keys=True,
                      allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def paired_log_ratio(candidate, baseline) -> dict:
    """Geometric-mean paired ratio with a paired bootstrap interval."""
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if candidate.shape != baseline.shape:
        raise ValueError("paired comparison needs matching shapes")
    keep = (np.isfinite(candidate) & np.isfinite(baseline)
            & (candidate > 0) & (baseline > 0))
    if keep.sum() == 0:
        return {"ratio": float("nan"), "low": float("nan"),
                "high": float("nan"), "wins": 0, "pairs": 0}
    logs = np.log(candidate[keep] / baseline[keep])
    rng = np.random.default_rng(20260724)
    boots = np.array([
        np.exp(np.mean(logs[rng.integers(0, len(logs), len(logs))]))
        for _ in range(2000)])
    return {
        "ratio": float(np.exp(logs.mean())),
        "low": float(np.quantile(boots, 0.025)),
        "high": float(np.quantile(boots, 0.975)),
        "wins": int((candidate[keep] < baseline[keep]).sum()),
        "pairs": int(keep.sum()),
    }
