"""The total objective and its gradient accounting (plan section 6.5).

    L_total = lambda_A L_anchor
            + lambda_G sum_j alpha_j L_geom,j
            + lambda_R L_reg,

with ``lambda_A, lambda_G > 0``, ``lambda_R >= 0`` and every term
nonnegative.  At the ideal population level, if ``L_anchor`` denotes the
full-support characteristic-function discrepancy, separate nonnegative terms
give

    L_total = 0  =>  L_anchor = 0  =>  p = q,

which does **not** follow from ``V_anchor + V_geom = 0``: two vector fields
can cancel while neither vanishes.  The plan calls this the cancellation
loophole.  The code below only closes the algebraic cancellation loophole:
its finite random-feature anchor is not measure determining, so a computed
zero does not establish ``p = q``.

The geometry loss is the paper's stop-gradient regression

    L_geom,j = E || f(eps) - sg(f(eps) + eta_j V_j) ||^2 = eta_j^2 E |V_j|^2,

whose *value* is a nonnegative function of the field and whose *gradient*
moves the generator output along ``V_j``.  Both readings matter: the value
is what the exact-zero argument uses, the gradient is what training uses.

The anchor is differentiated directly rather than converted to a
stop-gradient target so its finite surrogate is optimized faithfully.

Cancellation is measured, not assumed: :func:`branch_gradient_report`
returns per-branch gradient norms, gradient shares and pairwise cosines with
respect to the generator output.  A persistently negative anchor/geometry
cosine is the cancellation warning the risk register asks for.
"""

from __future__ import annotations

import torch

from .spectral_anchor import SpectralBank, anchor_gradient, anchor_loss


def geometry_loss(output: torch.Tensor, drift: torch.Tensor, eta: float,
                  reference: torch.Tensor | None = None, *,
                  correction: str = "scalar", gain: float = 1.0,
                  ratio_cap: float = 10.0) -> torch.Tensor:
    """Paper-style stop-gradient regression toward ``output + eta * drift``.

    Passing ``reference`` (the real batch) enables the R11/R26 teacher
    correction; leaving it ``None`` reproduces the paper-style behaviour
    exactly.  ``correction`` defaults to the scalar form so existing callers
    that pass a reference keep R11's behaviour unchanged.

    Note that ``eta`` scales this loss by ``eta^2`` and its gradient by
    ``eta`` -- which an adaptive optimizer then normalizes away entirely
    (R24).  Under Adam this argument changes nothing about training.
    """
    if eta <= 0:
        raise ValueError("the geometry step eta must be positive")
    target = (output + eta * drift).detach()
    if reference is not None:
        target = corrected_teacher(target, reference, mode=correction,
                                   gain=gain, ratio_cap=ratio_cap).detach()
    return ((output - target) ** 2).flatten(1).sum(dim=1).mean()


def variance_matched_teacher(teacher: torch.Tensor,
                             reference: torch.Tensor) -> torch.Tensor:
    """Rescale a stop-gradient teacher to carry the reference's spread.

    Reform R11.  A single scalar about the teacher's own mean, so no sample
    changes direction: the field's decisions about *where* each sample should
    go are preserved, and only the cloud's overall scale is pinned.

    **What this does and does not do.**  It matches the teacher's second
    moment to the data's.  Earlier documents described it as "repairing
    variance collapse", which the root-cause analysis showed to be the wrong
    account: effective dimension has an optimum near 1, not a monotone
    benefit, and restoring it by other means (fitting the teacher harder)
    overshoots to 1.4-3.3x the data's, collapses coverage and does not
    improve quality.  Matching is the mechanism; the effect on effective
    dimension is a side effect that only points the same way in some regimes.

    It is confirmed empirically (Phase 3: 18/18 paired wins; Phase 5: 9/9)
    and its mechanism is **unknown** -- four explanations have been proposed
    and measured false.  What is now known is *why it is able to act at all*
    where so many reforms could not: it changes the teacher's shape, and an
    adaptive optimizer is blind to everything that only changes its scale
    (R24).  :func:`corrected_teacher` generalizes it along that axis.
    """
    return corrected_teacher(teacher, reference, mode="scalar")


TEACHER_CORRECTIONS = ("none", "scalar", "per_coordinate", "eigendirection")


def corrected_teacher(teacher: torch.Tensor, reference: torch.Tensor, *,
                      mode: str = "scalar", gain: float = 1.0,
                      ratio_cap: float = 10.0,
                      report: dict | None = None) -> torch.Tensor:
    """Reform R26: match the teacher's second moment to the reference's.

    Generalizes R11 (``mode="scalar"``) along the axis the open-questions
    pass identified.  Under an adaptive optimizer a constant rescaling of the
    gradient is invisible, so **only corrections that change the teacher's
    shape can act at all** -- which is why the scalar match works while
    ``step_eta``, RMS normalization and the output step cap do not.  If that
    principle is right, a correction resolving more directions should act
    more.

    ``scalar``          one factor for the whole cloud (R11 exactly).
    ``per_coordinate``  one factor per coordinate, in the canonical basis.
    ``eigendirection``  one factor per principal direction of the teacher's
                        own cloud, matched to the reference's spread measured
                        along that same direction.

    Every mode is applied about the teacher's own mean, so no correction
    moves the cloud's centre: the field keeps its say over *where* samples
    go, and only the spread is pinned.  ``gain`` scales the matched target
    and is declared, never searched.  ``ratio_cap`` bounds the per-direction
    factor, since a direction carrying almost no teacher variance would
    otherwise receive an unbounded gain; ``report`` collects how often it
    binds, so the guard is measured rather than assumed harmless.
    """
    if mode not in TEACHER_CORRECTIONS:
        raise ValueError(f"unknown teacher correction {mode!r}")
    if gain <= 0:
        raise ValueError("the teacher correction gain must be positive")
    if ratio_cap < 1.0:
        raise ValueError("the ratio cap must be at least 1")
    if mode == "none":
        return teacher

    flat = teacher.reshape(len(teacher), -1)
    flat_reference = reference.reshape(len(reference), -1)
    centre = flat.mean(dim=0, keepdim=True)
    centred = flat - centre
    centred_reference = (flat_reference
                         - flat_reference.mean(dim=0, keepdim=True))

    if mode == "scalar":
        spread = centred.pow(2).mean().sqrt()
        target = centred_reference.pow(2).mean().sqrt() * gain
        ratios = (target / spread.clamp_min(1e-8)).reshape(1)
        scaled = centre + centred * ratios
    elif mode == "per_coordinate":
        spread = centred.pow(2).mean(dim=0).sqrt()
        target = centred_reference.pow(2).mean(dim=0).sqrt() * gain
        ratios = (target / spread.clamp_min(1e-8)).clamp(
            1.0 / ratio_cap, ratio_cap)
        scaled = centre + centred * ratios.unsqueeze(0)
    else:
        scaled, ratios = _eigendirection_match(
            centre, centred, centred_reference, gain, ratio_cap)

    if report is not None:
        bound = ((ratios <= 1.0 / ratio_cap + 1e-9)
                 | (ratios >= ratio_cap - 1e-9)).to(torch.float32)
        report["correction_mode"] = mode
        report["correction_ratio_median"] = float(ratios.median())
        report["correction_ratio_cap_fraction"] = float(bound.mean())
    return scaled.reshape(teacher.shape)


def _eigendirection_match(centre: torch.Tensor, centred: torch.Tensor,
                          centred_reference: torch.Tensor, gain: float,
                          ratio_cap: float) -> tuple[torch.Tensor,
                                                     torch.Tensor]:
    """Per-principal-direction match, done in the teacher's own span.

    With a batch far smaller than the ambient dimension the teacher's
    covariance is heavily rank deficient, so ``Sigma_ref^{1/2}
    Sigma_teacher^{-1/2}`` is not defined.  Working in the SVD basis of the
    centred teacher avoids inverting anything: directions carrying real
    teacher variance are rescaled to the reference's spread along that same
    direction, and the numerically empty tail is left untouched.
    """
    n = len(centred)
    _, singular, right = torch.linalg.svd(centred, full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) <= 0:
        return centre + centred, torch.ones(1, dtype=centred.dtype,
                                            device=centred.device)
    keep = singular > float(singular[0]) * 1e-3
    if not bool(keep.any()):
        return centre + centred, torch.ones(1, dtype=centred.dtype,
                                            device=centred.device)
    basis = right[keep]                              # r x d, orthonormal rows
    spread = singular[keep] / max(n, 1) ** 0.5       # teacher std per direction
    target = (centred_reference @ basis.T).pow(2).mean(dim=0).sqrt() * gain
    ratios = (target / spread.clamp_min(1e-8)).clamp(
        1.0 / ratio_cap, ratio_cap)
    coefficients = centred @ basis.T                 # n x r
    # Rescale inside the retained span, leaving the orthogonal remainder as
    # it was, so the correction only ever redistributes what it can measure.
    adjusted = coefficients * (ratios - 1.0).unsqueeze(0)
    return centre + centred + adjusted @ basis, ratios


def scheduled_eta(step_eta: float, schedule: str,
                  progress: float | None) -> float:
    """Reform R16: the effective teacher step at this point in training.

    ``constant`` reproduces the paper-style fixed step.  ``linear_decay``
    mirrors the free-particle schedule ``eta * (1 - progress)``, which is the
    one configuration measured *not* to collapse under the same field.
    A floor keeps the step positive so the objective stays well defined at
    the final step.
    """
    if step_eta <= 0:
        raise ValueError("the geometry step eta must be positive")
    if schedule == "constant" or progress is None:
        return float(step_eta)
    if schedule != "linear_decay":
        raise ValueError(f"unknown eta schedule {schedule!r}")
    clipped = min(max(float(progress), 0.0), 1.0)
    return float(step_eta) * max(1.0 - clipped, 1e-3)


def reported_geometry_loss(raw_drift_rms: float, eta: float) -> float:
    """Reform R2: the geometry loss of the UNNORMALIZED field.

    ``geometry_loss`` evaluates to ``eta^2 * mean_i |V_i|^2``.  When the
    field is RMS normalized that mean is 1 by construction, so the trained
    loss is pinned at ``eta^2`` -- identically 0.25 in every one of the 243
    Phase-1 rows.  A constant cannot signal convergence, and it makes plan
    section 6.5's `L_total = 0 => L_anchor = 0` argument unreachable, since
    the geometry term can never vanish.

    Normalization is still useful for step-size control, so it is kept on
    the *update* and the reported value is recomputed from the raw field
    magnitude that ``kernel_gradient.field`` measures before normalizing.
    This quantity does move, does approach zero as the field vanishes, and
    is the one every report and stopping rule must use.
    """
    if eta <= 0:
        raise ValueError("the geometry step eta must be positive")
    return float(eta) ** 2 * float(raw_drift_rms) ** 2


def range_regularization(output: torch.Tensor,
                         limit: float = 4.0) -> torch.Tensor:
    """Nonnegative, zero inside the declared output range."""
    if limit <= 0:
        raise ValueError("the range limit must be positive")
    excess = (output.abs() - limit).clamp_min(0.0)
    return (excess ** 2).flatten(1).sum(dim=1).mean()


def total_objective(
    output: torch.Tensor,
    *,
    bank: SpectralBank | None,
    target: torch.Tensor | None,
    drifts: dict[str, torch.Tensor],
    weights: dict[str, float],
    lambda_anchor: float,
    lambda_geometry: float,
    lambda_regularization: float,
    eta: float,
    anchor_estimator: str = "biased",
    anchor_progress: float | None = None,
    teacher_reference: torch.Tensor | None = None,
    teacher_correction: str = "scalar",
    teacher_correction_gain: float = 1.0,
    teacher_correction_ratio_cap: float = 10.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Assemble the nonnegative total loss and report every component."""
    if lambda_geometry < 0 or lambda_regularization < 0:
        raise ValueError("loss weights must be nonnegative")
    if bank is not None and lambda_anchor <= 0:
        raise ValueError(
            "an enabled anchor needs lambda_A > 0 for the exact-zero argument")
    if anchor_estimator != "biased":
        raise ValueError(
            "the training objective must use the nonnegative biased "
            "estimator; the unbiased U-statistic can be negative")

    components: dict[str, float] = {}
    total = torch.zeros((), dtype=output.dtype, device=output.device)

    if bank is not None:
        if target is None:
            raise ValueError("the anchor needs target samples")
        value = anchor_loss(bank, output, target, "biased",
                            progress=anchor_progress).to(output.dtype)
        components["loss_anchor"] = float(value)
        # Reported at full declared band width, so the schedule cannot make
        # the anchor look better simply by narrowing what it looks at.
        components["loss_anchor_full_band"] = float(
            anchor_loss(bank, output.detach(), target, "biased"))
        total = total + lambda_anchor * value

    geometry_total = torch.zeros((), dtype=output.dtype, device=output.device)
    for name, drift in drifts.items():
        weight = float(weights[name])
        if weight < 0:
            raise ValueError(f"branch {name} has a negative mixture weight")
        value = geometry_loss(
            output, drift, eta, teacher_reference,
            correction=teacher_correction, gain=teacher_correction_gain,
            ratio_cap=teacher_correction_ratio_cap)
        components[f"loss_geometry_{name}"] = float(value)
        geometry_total = geometry_total + weight * value
    if drifts:
        components["loss_geometry"] = float(geometry_total)
        total = total + lambda_geometry * geometry_total

    if lambda_regularization > 0:
        value = range_regularization(output)
        components["loss_regularization"] = float(value)
        total = total + lambda_regularization * value

    components["loss_total"] = float(total)
    if float(total) < -1e-9:
        raise AssertionError("the total loss must be nonnegative by design")
    return total, components


def branch_gradient_report(
    output: torch.Tensor,
    *,
    bank: SpectralBank | None,
    target: torch.Tensor | None,
    drifts: dict[str, torch.Tensor],
    weights: dict[str, float],
    lambda_anchor: float,
    lambda_geometry: float,
    eta: float,
) -> dict:
    """Per-branch gradients w.r.t. the generator OUTPUT.

    Exact and cheap: the stop-gradient regression contributes
    ``-2 eta V_j / n`` per example, and the anchor gradient is available in
    closed form.  Reported quantities are the gradient share (plan section
    10.2's anchor-presence gate) and the pairwise cosines (the cancellation
    warning of the risk register).
    """
    n = len(output)
    gradients: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        if bank is not None:
            if target is None:
                raise ValueError("the anchor needs target samples")
            gradients["anchor"] = (
                lambda_anchor * anchor_gradient(bank, output.detach(), target))
        for name, drift in drifts.items():
            gradients[f"geometry_{name}"] = (
                -2.0 * eta * lambda_geometry * float(weights[name])
                * drift / n)

    norms = {k: float(v.norm()) for k, v in gradients.items()}
    total_norm = sum(norms.values())
    report: dict = {
        "gradient_norms": norms,
        "gradient_shares": {
            k: (v / total_norm if total_norm > 0 else 0.0)
            for k, v in norms.items()
        },
    }
    if bank is not None and drifts:
        combined = sum(v for k, v in gradients.items()
                       if k.startswith("geometry_"))
        report["anchor_geometry_cosine"] = _cosine(
            gradients["anchor"], combined)
        report["anchor_gradient_share"] = report["gradient_shares"]["anchor"]
    names = sorted(k for k in gradients if k.startswith("geometry_"))
    cosines = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            cosines[f"{left}|{right}"] = _cosine(
                gradients[left], gradients[right])
    if cosines:
        report["branch_cosines"] = cosines
        report["branch_cosine_min"] = min(cosines.values())
    return report


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = float(a.norm()) * float(b.norm())
    if denominator <= 0:
        return 0.0
    return float((a * b).sum()) / denominator
