"""Sufficient-statistic monitoring for CAP successor runs.

Every training row contributes before the ledger is emitted and reset.  This
replaces CAP-EMF-1's misleading single-batch time table and records the joint
conditioning variables that determine the one-call inference corner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import torch

T_EDGES = (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0)
R_EDGES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.6, 1.0)
H_EDGES = T_EDGES

_VALUE_FIELDS = (
    "raw_mse",
    "weighted_loss",
    "output_gradient_norm",
    "target_rms",
    "quotient_rms",
    "coefficient",
    "adaptive_weight",
)


def _label(low: float, high: float) -> str:
    return f"{low:g}-{high:g}"


def _bin_summary(
    coordinate: torch.Tensor,
    edges: tuple[float, ...],
    values: dict[str, torch.Tensor],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    total = max(int(coordinate.numel()), 1)
    for low, high in pairwise(edges):
        mask = (
            (coordinate >= low) & (coordinate < high)
            if high < 1.0
            else (coordinate >= low) & (coordinate <= high)
        )
        count = int(mask.sum())
        row: dict[str, float | int] = {"count": count, "share": count / total}
        for name, tensor in values.items():
            row[f"mean_{name}"] = (
                float(tensor[mask].double().mean()) if count else float("nan")
            )
        result[_label(low, high)] = row
    return result


def _region_summary(
    mask: torch.Tensor,
    values: dict[str, torch.Tensor],
) -> dict[str, float | int]:
    """Count-weighted sufficient statistics for one named protocol region."""
    count = int(mask.sum())
    row: dict[str, float | int] = {"count": count}
    for name, tensor in values.items():
        selected = tensor[mask].double()
        row[f"mean_{name}"] = float(selected.mean()) if count else float("nan")
    return row


@dataclass
class ObjectiveLedger:
    """Accumulate every row between two logging events."""

    rows: list[dict[str, torch.Tensor]] = field(default_factory=list)
    parameter_gradient_samples: list[dict[str, float | str]] = field(
        default_factory=list
    )

    def state_dict(self) -> dict:
        return {
            "rows": [
                {name: value.clone() for name, value in row.items()}
                for row in self.rows
            ],
            "parameter_gradient_samples": list(self.parameter_gradient_samples),
        }

    def load_state_dict(self, payload: dict | None) -> None:
        self.rows = []
        self.parameter_gradient_samples = []
        if payload is None:
            return
        for row in payload.get("rows", []):
            self.rows.append(
                {name: value.detach().cpu().clone() for name, value in row.items()}
            )
        self.parameter_gradient_samples = [
            dict(record) for record in payload.get("parameter_gradient_samples", [])
        ]

    def add(self, outcome, *, accumulation_steps: int = 1) -> None:
        if accumulation_steps <= 0:
            raise ValueError("accumulation steps must be positive")
        self.rows.append(
            {
                "t": outcome.t.detach().cpu().float(),
                "r": outcome.r.detach().cpu().float(),
                "h": outcome.interval.detach().cpu().float(),
                "diagonal": outcome.diagonal.detach().cpu().bool(),
                "active": outcome.active.detach().cpu().bool(),
                "coefficient": outcome.coefficient.detach().cpu().float(),
                "adaptive_weight": outcome.adaptive_weight.detach().cpu().float(),
                "raw_mse": outcome.per_sample_raw_mse.detach().cpu().float(),
                "weighted_loss": outcome.per_sample_weighted_loss.detach()
                .cpu()
                .float(),
                "output_gradient_norm": (
                    outcome.per_sample_output_gradient_norm.detach().cpu().float()
                    / accumulation_steps
                ),
                "target_rms": outcome.per_sample_target_rms.detach().cpu().float(),
                "quotient_rms": outcome.per_sample_quotient_rms.detach().cpu().float(),
            }
        )

    def add_parameter_gradient_sample(self, record: dict[str, float | str]) -> None:
        required = {"category", "norm", "t", "r", "h", "coefficient"}
        if not required.issubset(record):
            raise ValueError("parameter-gradient sample is missing required fields")
        self.parameter_gradient_samples.append(dict(record))

    def _stack(self) -> dict[str, torch.Tensor]:
        if not self.rows:
            raise RuntimeError("cannot summarize an empty objective ledger")
        return {
            name: torch.cat([row[name] for row in self.rows]) for name in self.rows[0]
        }

    def summary(self, *, reset: bool = True) -> dict:
        data = self._stack()
        total = int(data["t"].numel())
        values = {name: data[name] for name in _VALUE_FIELDS}
        coefficient = data["coefficient"].double()
        quantiles = torch.tensor([0.5, 0.9, 0.95, 0.99, 0.999], dtype=torch.float64)
        coefficient_quantiles = torch.quantile(coefficient, quantiles)

        global_quantiles = {}
        for name in (
            "target_rms",
            "quotient_rms",
            "adaptive_weight",
            "output_gradient_norm",
        ):
            values_q = torch.quantile(data[name].double(), quantiles)
            global_quantiles[name] = {
                f"q{100 * q:g}": float(value)
                for q, value in zip(quantiles.tolist(), values_q)
            }

        # A compact joint t/h count map.  Marginal summaries carry the expensive
        # moments; the joint grid is intended to expose support holes.
        joint: dict[str, int] = {}
        joint_statistics: dict[str, dict[str, float | int]] = {}
        for t_low, t_high in pairwise(T_EDGES):
            t_mask = (
                (data["t"] >= t_low) & (data["t"] < t_high)
                if t_high < 1.0
                else (data["t"] >= t_low) & (data["t"] <= t_high)
            )
            for h_low, h_high in pairwise(H_EDGES):
                h_mask = (
                    (data["h"] >= h_low) & (data["h"] < h_high)
                    if h_high < 1.0
                    else (data["h"] >= h_low) & (data["h"] <= h_high)
                )
                count = int((t_mask & h_mask).sum())
                if count:
                    key = f"t={_label(t_low, t_high)}|h={_label(h_low, h_high)}"
                    mask = t_mask & h_mask
                    joint[key] = count
                    cell: dict[str, float | int] = {
                        "count": count,
                        "share": count / total,
                        # r=t-h, so a joint (t,h) cell is a compact sufficient
                        # representation of the joint (t,r,h) geometry.  The
                        # realized r moments make that identity explicit.
                        "mean_r": float(data["r"][mask].double().mean()),
                        "diagonal_share": float(data["diagonal"][mask].double().mean()),
                        "active_share": float(data["active"][mask].double().mean()),
                    }
                    for name in _VALUE_FIELDS:
                        selected = data[name][mask].double()
                        cell[f"mean_{name}"] = float(selected.mean())
                        cell[f"rms_{name}"] = float(selected.square().mean().sqrt())
                    cell["q90_target_rms"] = float(
                        torch.quantile(data["target_rms"][mask].double(), 0.9)
                    )
                    cell["q90_output_gradient_norm"] = float(
                        torch.quantile(data["output_gradient_norm"][mask].double(), 0.9)
                    )
                    joint_statistics[key] = cell

        result = {
            "rows": total,
            "diagonal_fraction": float(data["diagonal"].double().mean()),
            "active_fraction": float(data["active"].double().mean()),
            "t_buckets": _bin_summary(data["t"], T_EDGES, values),
            "r_buckets": _bin_summary(data["r"], R_EDGES, values),
            "h_buckets": _bin_summary(data["h"], H_EDGES, values),
            "joint_t_h_counts": joint,
            "joint_t_r_h_statistics": joint_statistics,
            "coefficient_quantiles": {
                f"q{100 * q:g}": float(value)
                for q, value in zip(quantiles.tolist(), coefficient_quantiles)
            },
            "coefficient_tail_fraction": {
                "gt_3": float((coefficient > 3).double().mean()),
                "gt_7": float((coefficient > 7).double().mean()),
                "gt_15": float((coefficient > 15).double().mean()),
                "gt_30": float((coefficient > 30).double().mean()),
            },
            "global_quantiles": global_quantiles,
            "sampled_parameter_gradient_contributions": list(
                self.parameter_gradient_samples
            ),
            # The promotion gate consumes this exact named region.  Persisting
            # its count and all-row means avoids reconstructing it later from
            # rounded marginal bins or trusting a sampled minibatch.
            "named_regions": {
                "inference_corner": _region_summary(
                    (data["t"] > 0.95) & (data["h"] > 0.90), values
                )
            },
            "endpoint_fraction": {
                "t_gt_0.95_h_gt_0.90": float(
                    ((data["t"] > 0.95) & (data["h"] > 0.90)).double().mean()
                ),
                "t_gt_0.98": float((data["t"] > 0.98).double().mean()),
                "t_gt_0.99": float((data["t"] > 0.99).double().mean()),
                "t_gt_0.995": float((data["t"] > 0.995).double().mean()),
                "r_lt_0.02": float((data["r"] < 0.02).double().mean()),
            },
        }
        if reset:
            self.rows.clear()
            self.parameter_gradient_samples.clear()
        return result
