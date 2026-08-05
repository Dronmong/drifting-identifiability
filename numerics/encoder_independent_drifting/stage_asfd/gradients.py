"""Component gradients: summed, capped per component, never projected.

The applied update is exactly

    g = g_EMF + w1*g_B1 + w_raw*g_raw + w_self*g_self

which **is** the gradient of ``L_EMF + λ1 L_B1 + λ_raw E_raw + λ_self E_self``,
so the identifiability implication attaches to the objective the optimizer
actually descends.

An earlier draft projected each auxiliary component away from opposing the
primary gradient.  That is rejected twice over: a projected update is generally
non-conservative, so there is no potential whose stationary points the dynamics
seek; and the raw anchor is redundant precisely when it agrees with ``g_EMF``,
so projection deletes the only component that does any work.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch
from torch import nn

from .config import GradientConfig


def gradient_norm(values: list[torch.Tensor | None]) -> float:
    total = 0.0
    for value in values:
        if value is not None:
            total += float(value.detach().double().square().sum())
    return total**0.5


def gradient_cosine(
    left: list[torch.Tensor | None], right: list[torch.Tensor | None]
) -> float | None:
    dot = 0.0
    for a, b in zip(left, right):
        if a is not None and b is not None:
            dot += float((a.detach().double() * b.detach().double()).sum())
    left_norm, right_norm = gradient_norm(left), gradient_norm(right)
    if left_norm <= 0 or right_norm <= 0:
        return None
    return dot / (left_norm * right_norm)


def snapshot(model: nn.Module) -> list[torch.Tensor | None]:
    return [
        None if p.grad is None else p.grad.detach().clone() for p in model.parameters()
    ]


def capped_weight(primary: float, component: float, cap: float) -> float:
    """Scale a component to at most ``cap * ||g_primary||``.

    Returns the multiplier, so a component already under its cap is untouched.
    """
    if component <= 0:
        return 0.0
    ceiling = cap * primary
    return 1.0 if component <= ceiling else ceiling / component


@dataclass
class CorrectionOutcome:
    applied: dict[str, float] = field(default_factory=dict)
    pre_cap_ratio: dict[str, float] = field(default_factory=dict)
    post_cap_ratio: dict[str, float] = field(default_factory=dict)
    cosines: dict[str, float | None] = field(default_factory=dict)
    primary_norm: float = 0.0
    total_auxiliary_ratio: float = 0.0


def combine(
    parameters: list[nn.Parameter],
    primary: list[torch.Tensor | None],
    components: dict[str, tuple[list[torch.Tensor | None], float]],
    config: GradientConfig,
) -> CorrectionOutcome:
    """Write ``primary + sum(scaled components)`` into each parameter's ``.grad``.

    ``components`` maps a name to ``(gradient, cap)``.  Caps are independent, so
    the realized total may exceed any single cap -- that is the defined joint
    treatment, not a confound, exactly as B2.5's factorial argued for full-dose
    combined cells.
    """
    config.validate()
    primary_norm = gradient_norm(primary)
    if primary_norm <= 0:
        raise ValueError("the primary gradient vanished; nothing to correct")
    outcome = CorrectionOutcome(primary_norm=primary_norm)

    scales: dict[str, float] = {}
    for name, (values, cap) in components.items():
        norm = gradient_norm(values)
        outcome.pre_cap_ratio[name] = norm / primary_norm
        scales[name] = capped_weight(primary_norm, norm, cap)
        outcome.post_cap_ratio[name] = (norm * scales[name]) / primary_norm
        outcome.applied[name] = scales[name]
        outcome.cosines[f"primary_vs_{name}"] = gradient_cosine(primary, values)

    names = sorted(components)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            outcome.cosines[f"{left}_vs_{right}"] = gradient_cosine(
                components[left][0], components[right][0]
            )

    for position, parameter in enumerate(parameters):
        base = primary[position]
        total = None if base is None else base.detach().clone()
        for name, (values, _) in components.items():
            piece = values[position]
            if piece is None:
                continue
            scaled = piece.detach() * scales[name]
            total = scaled.clone() if total is None else total + scaled
        parameter.grad = total

    outcome.total_auxiliary_ratio = sum(outcome.post_cap_ratio.values())
    return outcome


class AbortMonitor:
    """Outcome-based aborts.  Gradient cosine is a diagnostic, not a trigger.

    A correction that never opposes the primary gradient is useless, so mild
    opposition is the working regime.  Only the anti-parallel pathology -- the
    term simply negating training -- and genuine outcome failures stop an arm.
    """

    def __init__(self, config: GradientConfig) -> None:
        config.validate()
        self.config = config
        self._cosines: dict[str, deque] = {}
        self._rank_failures: dict[str, int] = {}
        self.reasons: list[str] = []

    def observe_cosines(self, cosines: dict[str, float | None]) -> None:
        for name, value in cosines.items():
            if value is None or not name.startswith("primary_vs_"):
                continue
            window = self._cosines.setdefault(
                name, deque(maxlen=self.config.anti_parallel_window)
            )
            window.append(value)
            if len(window) == window.maxlen:
                mean = sum(window) / len(window)
                if mean < self.config.anti_parallel_cosine:
                    self.reasons.append(
                        f"{name} sustained mean cosine {mean:.3f} below "
                        f"{self.config.anti_parallel_cosine}: the term is "
                        "negating training rather than correcting it"
                    )

    def observe_rank(self, label: str, ratio: float) -> None:
        """Two consecutive checkpoints below the fraction, then abort."""
        if ratio < self.config.rank_abort_fraction:
            self._rank_failures[label] = self._rank_failures.get(label, 0) + 1
            if self._rank_failures[label] >= 2:
                self.reasons.append(
                    f"{label} effective rank ratio {ratio:.3f} below "
                    f"{self.config.rank_abort_fraction} on two checkpoints"
                )
        else:
            self._rank_failures[label] = 0

    def observe_energy_floor(self, energy: float, floor: float) -> None:
        """Below the real-versus-real floor is estimator exploitation.

        B2 scored 13.933 against a floor of 14.103. A correctly distributed
        sample cannot beat that floor; undershooting it means the generated
        cloud is *less* variable than real data in the direction the estimator
        measures, which is the same event as the rank collapse.
        """
        if energy < floor:
            self.reasons.append(
                f"raw energy {energy:.4f} below the real-real floor {floor:.4f}: "
                "estimator exploitation, not distributional improvement"
            )

    def observe_negative_ess(self, label: str, median: float, ceiling: float) -> None:
        """The ceiling lives on the field configuration, not this one.

        If the generated side's ESS approaches one, every generated sample is
        weighted almost equally and the negative barycenter is a plain batch
        mean -- at which point the energy is first-moment matching wearing a
        kernel. B2 logged this and gated only the target side.
        """
        if median > ceiling:
            self.reasons.append(
                f"{label} negative-side ESS {median:.3f} above {ceiling}: the "
                "negative barycenter has degenerated toward a batch mean"
            )

    def observe_finite(self, value: float, label: str) -> None:
        if value != value or value in (float("inf"), float("-inf")):
            self.reasons.append(f"{label} is not finite")

    @property
    def should_abort(self) -> bool:
        return bool(self.reasons)
