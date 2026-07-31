"""Arm training loop and the Phase-1 arm registry (plan sections 6, 9).

Cross-fitting is structural.  Every step draws one target batch and splits
it into three *disjoint* role sets (controller / field / audit) via
``adaptive_mixture.split_roles``, which refuses overlapping roles.  The
controller only ever sees controller examples, the generator target is built
from field examples, and diagnostics are logged on audit examples.  This is
enforced even when adaptation is off, so fixed-weight and adaptive arms
consume target data identically and the compute ledger stays comparable.

Cost is counted, not estimated after the fact: :class:`WorkLedger` records
generator forwards, target examples by role, kernel pairs, feature-extraction
calls and anchor feature evaluations.  "No pretrained encoder" is not a
synonym for "cheap", and the fixed wavelet and kernel banks are inside the
training cost (plan section 9, P2.3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dataclass_field

import numpy as np
import torch

from . import kernel_gradient as KG
from . import spectral_anchor as SA
from .adaptive_mixture import MixtureController, split_roles
from .config import (
    ArmConfig, AUDIT_SEED, FieldConfig, GeometryConfig, MixtureConfig,
    ObjectiveConfig, TrainConfig, derive_seed,
)
from .datasets import ImageTarget
from .diagnostics import SeriesLog
from .fixed_features import FeatureFamily, build_family
from .kernels import BlockKernel, calibrate_block_kernel
from .models import OneStepGenerator
from .metrics import effective_dimension
from .objectives import (
    branch_gradient_report, corrected_teacher, reported_geometry_loss,
    scheduled_eta, total_objective,
)


# ---------------------------------------------------------------------------
# The real step control (reform R25)
# ---------------------------------------------------------------------------


def build_optimizer(model: torch.nn.Module,
                    config: TrainConfig) -> torch.optim.Optimizer:
    """The generator's actual step size lives here, not in ``step_eta``.

    Reform R25.  The objective's ``eta`` is inert under an adaptive optimizer
    (R24), so the optimizer choice and its learning rate are *the* step
    control.  They were hard-coded to Adam at 2e-3 for six phases; making
    them configurable is what let Phase 6A sweep them.
    """
    if config.learning_rate <= 0:
        raise ValueError("the learning rate must be positive")
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    if config.optimizer == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                               momentum=config.momentum)
    raise ValueError(f"unknown optimizer {config.optimizer!r}")


def optimizer_report(config: TrainConfig) -> dict:
    """Reform R25: the step control, for the summary of every artifact."""
    report = {"optimizer": config.optimizer,
              "learning_rate": config.learning_rate,
              "step_eta_is_inert": config.optimizer == "adam"}
    if config.optimizer == "sgd_momentum":
        report["momentum"] = config.momentum
    return report


def _correction(objective: ObjectiveConfig) -> str:
    """Resolve R11's boolean and R26's mode into one declared correction."""
    if objective.teacher_correction != "none":
        return objective.teacher_correction
    return "scalar" if objective.teacher_variance_match else "none"


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


@dataclass
class WorkLedger:
    generator_forwards: int = 0
    generator_examples: int = 0
    target_examples_field: int = 0
    target_examples_controller: int = 0
    target_examples_audit: int = 0
    kernel_pairs: int = 0
    feature_calls: int = 0
    feature_examples: int = 0
    anchor_feature_evaluations: int = 0
    optimizer_updates: int = 0

    def add_field(self, probes: int, positives: int, negatives: int,
                  blocks: int) -> None:
        self.kernel_pairs += probes * (positives + negatives) * blocks
        self.feature_calls += 3
        self.feature_examples += probes + positives + negatives

    def add_anchor(self, samples: int, features: int) -> None:
        self.anchor_feature_evaluations += samples * features

    def as_dict(self) -> dict:
        return {f"cost_{k}": v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Arm assembly
# ---------------------------------------------------------------------------


@dataclass
class ArmParts:
    family: FeatureFamily | None
    kernels: dict[str, BlockKernel]
    bank: SA.SpectralBank | None
    audit_bank: SA.SpectralBank | None
    controller: MixtureController | None

    @property
    def branch_names(self) -> tuple[str, ...]:
        return tuple(self.kernels)


def build_arm(arm: ArmConfig, calibration: torch.Tensor, channels: int,
              seed: int, reference_family: FeatureFamily | None = None,
              ) -> ArmParts:
    """Freeze every geometry bandwidth and frequency band from TARGET data.

    Calibration uses target samples only and happens once, before training.
    Nothing here may look at generated output or at an evaluation metric.
    """
    family: FeatureFamily | None = None
    kernels: dict[str, BlockKernel] = {}
    controller: MixtureController | None = None
    if arm.geometry is not None:
        if arm.geometry.family == "reference_encoder":
            if reference_family is None:
                raise ValueError("arm A8 needs a trained reference family")
            family = reference_family
        else:
            family = build_family(arm.geometry, channels,
                                  seed_label=f"{arm.arm_id}")
        if arm.geometry.base_kernel != "auto":
            base = arm.geometry.base_kernel
        else:
            base = ("laplace" if arm.field.direction_mode == "standard"
                    and arm.geometry.family == "raw" else "smooth_laplace")
        for branch in family.branches:
            # A declared temperature pins tau to a multiple of the median
            # pairwise distance (the paper's normalized units) and overrides
            # the ESS calibration; otherwise the calibration decides.
            if arm.geometry.bandwidth_tau is not None:
                kernels[branch.name] = calibrate_block_kernel(
                    branch, calibration, base,
                    arm.geometry.bandwidth_quantile,
                    arm.geometry.bandwidth_tau,
                    arm.geometry.kernel_eps,
                    combine=arm.geometry.combine,
                    target_ess_fraction=None)
            else:
                kernels[branch.name] = calibrate_block_kernel(
                    branch, calibration, base,
                    arm.geometry.bandwidth_quantile,
                    arm.geometry.bandwidth_multiplier,
                    arm.geometry.kernel_eps,
                    combine=arm.geometry.combine,
                    target_ess_fraction=arm.geometry.target_ess_fraction)
        controller = MixtureController(tuple(kernels), arm.mixture)

    bank = audit = None
    if arm.use_anchor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derive_seed(seed, "projected-scale")
                              % (2 ** 63 - 1))
        scale = SA.projected_scale(calibration, arm.anchor, generator)
        dim = calibration[0].numel()
        bank = SA.build_bank(arm.anchor, dim, scale,
                             derive_seed(seed, "anchor-bank"))
        # The audit bank is seeded off AUDIT_SEED, not off `seed`, so no
        # training decision can reach it.
        audit = SA.build_bank(arm.anchor, dim, scale,
                              derive_seed(AUDIT_SEED, "audit-bank"),
                              features=arm.anchor.audit_features)
    return ArmParts(family, kernels, bank, audit, controller)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainOutcome:
    model: OneStepGenerator
    log: SeriesLog
    ledger: WorkLedger
    parts: ArmParts
    wall_seconds: float
    controller_history: list = dataclass_field(default_factory=list)
    diverged: bool = False


def _branch_drifts(parts: ArmParts, output: torch.Tensor,
                   positives: torch.Tensor, field_config: FieldConfig,
                   ledger: WorkLedger, with_snr: bool,
                   ) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
    """Per-branch drift.

    ``with_snr`` doubles as the diagnostics switch: the inner training loop
    only needs the field, so kernel health is computed on controller and
    audit steps rather than on every step, where it would be measured and
    thrown away.
    """
    drifts: dict[str, torch.Tensor] = {}
    stats: dict[str, dict] = {}
    if parts.family is None:
        return drifts, stats
    probes = output.detach()
    # Reform R9: every branch shares the same probes and anchors, so the
    # projection's SVD is factorized once per step rather than per branch.
    projection = None
    if field_config.direction_mode == "projected_kernel_gradient":
        with torch.no_grad():
            projection = KG.data_span_basis(
                probes, torch.cat([positives, probes], dim=0))
    for branch in parts.family.branches:
        kernel = parts.kernels[branch.name]
        common = dict(
            direction_mode=field_config.direction_mode,
            normalization=field_config.normalization,
            denominator_floor=field_config.denominator_floor,
            self_mask=field_config.self_mask,
            projection=projection)
        if with_snr:
            drift, health = KG.field_with_snr(
                probes, positives, probes, branch, kernel, **common)
        else:
            drift, health = KG.field(
                probes, positives, probes, branch, kernel,
                diagnostics=False, **common)
        ledger.add_field(len(probes), len(positives), len(probes),
                         kernel.n_blocks)
        drifts[branch.name] = drift
        stats[branch.name] = health
    return drifts, stats


def _cap_output_step(model, latent: torch.Tensor,
                     before_output: torch.Tensor,
                     before_parameters: list[torch.Tensor],
                     drifts: dict[str, torch.Tensor],
                     weights: dict[str, float], eta: float, cap: float,
                     ledger: WorkLedger) -> float:
    """Reform R21: hold the realized output change to ``cap x`` the request.

    The stop-gradient recipe declares a step of ``eta * V`` but delivers
    whatever the optimizer produces through the Jacobian -- measured at 42x
    the request at initialization.  This rescales the parameter update, along
    the direction the optimizer chose, so the realized change in *output*
    space respects the declared step.

    Returns the pre-cap ratio so it can be logged.  Rescaling the update
    rather than the gradient keeps the optimizer's chosen direction intact
    and only limits its length.
    """
    if cap <= 0:
        raise ValueError("the output step cap must be positive")
    with torch.no_grad():
        after_output = model(latent)
        ledger.generator_forwards += 1
        ledger.generator_examples += len(latent)
        requested = eta * sum(
            float(weights[name]) * drift for name, drift in drifts.items())
        requested_norm = float(requested.norm())
        realized_norm = float((after_output - before_output).norm())
        if requested_norm <= 0:
            return float("nan")
        ratio = realized_norm / requested_norm
        if ratio > cap:
            scale = cap / ratio
            for parameter, before in zip(model.parameters(),
                                         before_parameters):
                parameter.copy_(before + (parameter - before) * scale)
    return ratio


def train_arm(arm: ArmConfig, target: ImageTarget, config: TrainConfig,
              seed: int, reference_family: FeatureFamily | None = None,
              log_every: int = 10) -> TrainOutcome:
    # Paired randomness (repository convention, cf. lowdim_drift.train): the
    # target-minibatch stream, the role split, the calibration sample, the
    # latents and the generator init are identical across arms in a cell, so
    # a paired comparison differs only by the objective.  The seed labels
    # deliberately omit the arm id.
    rng = np.random.default_rng(derive_seed(seed, "data"))
    calibration = target.sample(256, np.random.default_rng(
        derive_seed(seed, "calibration")))
    parts = build_arm(arm, calibration, config.channels, seed,
                      reference_family)

    # Identical initial generator across arms in a cell: the screen compares
    # objectives, not initializations.
    model = OneStepGenerator(config.latent_dim, config.channels,
                             config.image_size, config.width,
                             derive_seed(seed, "generator"))
    optimizer = build_optimizer(model, config)
    # `or` would treat an explicit 0 as "unset" and silently fall back.
    field_cloud = (config.batch if config.field_cloud is None
                   else config.field_cloud)
    if field_cloud <= 0:
        raise ValueError("the field cloud must hold at least one sample")
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(derive_seed(seed, "latent") % (2 ** 63 - 1))

    log = SeriesLog()
    ledger = WorkLedger()
    weights = ({n: 1.0 / len(parts.kernels) for n in parts.kernels}
               if parts.kernels else {})
    diverged = False
    started = time.perf_counter()

    for step in range(1, config.steps + 1):
        total_draw = (config.batch + config.controller_batch
                      + config.audit_batch)
        pool = target.sample(total_draw, rng)
        roles = split_roles(total_draw, config.controller_batch,
                            config.batch, config.audit_batch, rng)
        field_target = pool[roles.field]
        controller_target = pool[roles.controller]
        audit_target = pool[roles.audit]
        ledger.target_examples_field += len(field_target)
        ledger.target_examples_controller += len(controller_target)
        ledger.target_examples_audit += len(audit_target)

        # Reform R27: the field's cloud is a declared size, not whatever the
        # target batch happened to be.
        latent = torch.randn(field_cloud, config.latent_dim,
                             generator=torch_generator)
        # Reform R6: fraction of training elapsed, driving the anchor's
        # coarse-to-fine band schedule, and the R16 teacher-step schedule.
        progress = (step - 1) / max(config.steps - 1, 1)
        eta = scheduled_eta(arm.objective.step_eta,
                            arm.objective.eta_schedule, progress)
        for _ in range(config.inner_steps):
            output = model(latent)
            ledger.generator_forwards += 1
            ledger.generator_examples += len(latent)
            drifts, health = _branch_drifts(
                parts, output, field_target, arm.field, ledger,
                with_snr=False)
            if parts.bank is not None:
                ledger.add_anchor(len(output) + len(field_target),
                                  parts.bank.size)
            # Reform R22: the field is computed once and then optimized
            # against for `steps_per_teacher` updates.  At 1 this is exactly
            # the previous behaviour.
            for teacher_step in range(max(config.steps_per_teacher, 1)):
                if teacher_step:
                    output = model(latent)
                    ledger.generator_forwards += 1
                    ledger.generator_examples += len(latent)
                loss, components = total_objective(
                    output,
                    bank=parts.bank, target=field_target,
                    drifts=drifts, weights=weights,
                    lambda_anchor=arm.objective.lambda_anchor,
                    lambda_geometry=arm.objective.lambda_geometry,
                    lambda_regularization=(
                        arm.objective.lambda_regularization),
                    eta=eta,
                    anchor_progress=(progress if parts.bank is not None
                                     else None),
                    teacher_reference=(
                        field_target if _correction(arm.objective) != "none"
                        else None),
                    teacher_correction=_correction(arm.objective),
                    teacher_correction_gain=(
                        arm.objective.teacher_correction_gain),
                    teacher_correction_ratio_cap=(
                        arm.objective.teacher_correction_ratio_cap))
                before_output = output.detach()
                before_parameters = (
                    [p.detach().clone() for p in model.parameters()]
                    if arm.objective.output_step_cap is not None else None)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                ledger.optimizer_updates += 1
                if before_parameters is not None and drifts:
                    _cap_output_step(
                        model, latent, before_output, before_parameters,
                        drifts, weights, eta,
                        arm.objective.output_step_cap, ledger)

        # Divergence is read off the loss that was just computed, rather than
        # from an extra generator forward that the cost ledger would miss.
        if not np.isfinite(components["loss_total"]):
            diverged = True
            break

        # --- controller update: CONTROLLER examples only -------------------
        if (parts.controller is not None and arm.mixture.adaptive
                and step % arm.mixture.controller_every == 0):
            with torch.no_grad():
                controller_latent = torch.randn(
                    config.controller_batch, config.latent_dim,
                    generator=torch_generator)
                controller_output = model(controller_latent)
                ledger.generator_forwards += 1
                ledger.generator_examples += len(controller_latent)
            _, controller_health = _branch_drifts(
                parts, controller_output, controller_target, arm.field,
                ledger, with_snr=True)
            utilities = {name: float(h.get("drift_snr", 0.0))
                         for name, h in controller_health.items()}
            parts.controller.update(utilities, step)
            weights = parts.controller.as_dict()

        # --- anchor bank refresh ------------------------------------------
        if (parts.bank is not None and arm.anchor.refresh_every > 0
                and step % arm.anchor.refresh_every == 0):
            parts.bank = SA.refresh_bank(
                parts.bank, arm.anchor.refresh_fraction,
                derive_seed(seed, "anchor-refresh", step))

        # --- diagnostics: AUDIT examples only ------------------------------
        if step == 1 or step % log_every == 0 or step == config.steps:
            with torch.no_grad():
                audit_latent = torch.randn(
                    config.audit_batch, config.latent_dim,
                    generator=torch_generator)
                audit_output = model(audit_latent)
                ledger.generator_forwards += 1
                ledger.generator_examples += len(audit_latent)
            audit_drifts, audit_health = _branch_drifts(
                parts, audit_output, audit_target, arm.field, ledger,
                with_snr=True)
            record: dict = {"loss": components}
            # Reform R2: the trained geometry loss is pinned at eta^2 under
            # RMS normalization, so the honest value is recomputed from the
            # raw field magnitude and reported alongside it.
            reported: dict[str, float] = {}
            for name, health in audit_health.items():
                record[f"branch_{name}"] = health
                if "drift_rms_raw" in health:
                    reported[name] = reported_geometry_loss(
                        health["drift_rms_raw"], arm.objective.step_eta)
            if reported:
                record["loss_geometry_unnormalized"] = reported
                record["loss_geometry_unnormalized_total"] = float(sum(
                    weights[n] * v for n, v in reported.items()))
            # Reform R18: the teacher's effect on effective dimension,
            # measured on the ACTUAL trajectory.  The two refuted mechanism
            # hypotheses both probed the fixed point, where the field is zero
            # by construction and nothing can happen.
            if audit_drifts:
                with torch.no_grad():
                    combined = sum(weights[n] * d
                                   for n, d in audit_drifts.items())
                    teacher = audit_output + eta * combined
                    if _correction(arm.objective) != "none":
                        # Reform R26: the per-direction guard is measured on
                        # audit examples, not assumed harmless.
                        correction_report: dict = {}
                        teacher = corrected_teacher(
                            teacher, audit_target,
                            mode=_correction(arm.objective),
                            gain=arm.objective.teacher_correction_gain,
                            ratio_cap=(
                                arm.objective.teacher_correction_ratio_cap),
                            report=correction_report)
                        record.update({
                            k: v for k, v in correction_report.items()
                            if isinstance(v, float)})
                    before = effective_dimension(audit_output)
                    after = effective_dimension(teacher)
                record["trajectory"] = {
                    "eta_effective": eta,
                    "output_effective_dimension": before,
                    "teacher_effective_dimension": after,
                    "teacher_dimension_ratio": (
                        after / before if before > 0 else float("nan")),
                }
            if audit_drifts:
                combined = sum(
                    weights[n] * d for n, d in audit_drifts.items())
                record["drift_spectrum"] = KG.drift_spectrum(combined)
            if parts.bank is not None and parts.audit_bank is not None:
                record["anchor"] = SA.anchor_diagnostics(
                    parts.bank, parts.audit_bank, audit_output, audit_target)
            record.update(branch_gradient_report(
                audit_output, bank=parts.bank, target=audit_target,
                drifts=audit_drifts, weights=weights,
                lambda_anchor=arm.objective.lambda_anchor,
                lambda_geometry=arm.objective.lambda_geometry,
                eta=arm.objective.step_eta))
            if parts.controller is not None:
                record.update(parts.controller.diagnostics())
            log.add(step, record)

    return TrainOutcome(
        model=model, log=log, ledger=ledger, parts=parts,
        wall_seconds=time.perf_counter() - started,
        controller_history=(parts.controller.history
                            if parts.controller else []),
        diverged=diverged)


# ---------------------------------------------------------------------------
# Phase-1 arm registry (plan section 9, Phase 1 table)
# ---------------------------------------------------------------------------


def phase1_arms(combine: str = "sum") -> list[ArmConfig]:
    """The plan's A0-A8 table, frozen.

    ``combine`` selects the block-combination rule for structured families.
    It is a registered kernel choice fixed by the Phase-0 kernel-health
    measurement -- not a per-arm tuning knob -- and every structured arm
    uses the same value.  Phase 0 measured "sum" (the plan's declared
    conservative form) as the healthier rule, so it is the default here.
    """
    standard = FieldConfig(direction_mode="standard")
    gradient = FieldConfig(direction_mode="kernel_gradient")
    fixed = MixtureConfig(adaptive=False)
    adaptive = MixtureConfig(adaptive=True)
    anchor_only = ObjectiveConfig(lambda_anchor=1.0, lambda_geometry=0.0)
    both = ObjectiveConfig(lambda_anchor=1.0, lambda_geometry=1.0)
    geometry_only = ObjectiveConfig(lambda_anchor=0.0, lambda_geometry=1.0)

    if combine not in ("sum", "product"):
        raise ValueError(f"unknown block combination {combine!r}")

    def geometry(family: str, **kwargs) -> GeometryConfig:
        # The raw and Haar families have a single block, so the combination
        # rule is a no-op for them and is left at the declared default.
        if family in ("raw", "haar_control"):
            return GeometryConfig(family=family, **kwargs)
        return GeometryConfig(family=family, combine=combine, **kwargs)

    return [
        ArmConfig("A0", False, geometry("raw"), standard, fixed,
                  geometry_only,
                  note="paper-style raw pixel Laplace displacement"),
        ArmConfig("A1", False, geometry("raw"), gradient, fixed,
                  geometry_only,
                  note="raw pixel smooth Laplace, kernel gradient"),
        ArmConfig("A2", True, None,
                  FieldConfig(direction_mode="kernel_gradient"), fixed,
                  anchor_only, note="spectral anchor only, direct gradient"),
        ArmConfig("A3", False, geometry("wavelet"), standard, fixed,
                  geometry_only, note="fixed wavelet geometry, displacement"),
        ArmConfig("A4", False, geometry("wavelet"), gradient, fixed,
                  geometry_only,
                  note="fixed wavelet geometry, kernel gradient"),
        ArmConfig("A5", True, geometry("wavelet"), gradient, fixed, both,
                  note="anchor + fixed wavelet geometry"),
        ArmConfig("A6", True, geometry("randconv"), gradient, fixed, both,
                  note="anchor + fixed random convolutional geometry"),
        ArmConfig("A7", True,
                  geometry("dictionary", second_order=True), gradient,
                  adaptive, both,
                  note="anchor + geometry dictionary, adaptive mixture"),
        ArmConfig("A8", False, geometry("reference_encoder"), standard,
                  fixed, geometry_only,
                  note="LOCALLY TRAINED encoder stand-in; NOT the paper's "
                       "pretrained encoder; excluded from the exit gate"),
    ]


def arm_by_id(arm_id: str, combine: str = "product") -> ArmConfig:
    for arm in phase1_arms(combine):
        if arm.arm_id == arm_id:
            return arm
    raise ValueError(f"unknown arm {arm_id!r}")


# ---------------------------------------------------------------------------
# Phase-2 arm registry (EncoderIndependentPhase2Protocol.md section 4)
# ---------------------------------------------------------------------------


def phase2_arms() -> list[ArmConfig]:
    """B0-B3, frozen.

    Every arm uses standard displacement -- the kernel-gradient hypothesis of
    plan section 6.3 is abandoned, not re-tested -- and every arm uses the
    same ``smooth_laplace`` base kernel, so geometry is not confounded with
    kernel smoothness as it was in Phase 1.
    """
    field = FieldConfig(direction_mode="standard")
    fixed = MixtureConfig(adaptive=False)
    geometry_only = ObjectiveConfig(lambda_anchor=0.0, lambda_geometry=1.0)
    both = ObjectiveConfig(lambda_anchor=1.0, lambda_geometry=1.0)

    def geometry(family: str, **kwargs) -> GeometryConfig:
        return GeometryConfig(family=family, base_kernel="smooth_laplace",
                              **kwargs)

    return [
        ArmConfig("B0", False, geometry("raw"), field, fixed, geometry_only,
                  note="raw pixel kernel, standard displacement (baseline)"),
        ArmConfig("B1", False, geometry("wavelet"), field, fixed,
                  geometry_only,
                  note="fixed wavelet geometry, standard displacement"),
        ArmConfig("B2", False, geometry("scattering", second_order=True),
                  field, fixed, geometry_only,
                  note="fixed scattering geometry, standard displacement"),
        ArmConfig("B3", True, geometry("wavelet"), field, fixed, both,
                  note="wavelet geometry plus scheduled spectral anchor"),
    ]


def phase2_arm_by_id(arm_id: str) -> ArmConfig:
    for arm in phase2_arms():
        if arm.arm_id == arm_id:
            return arm
    raise ValueError(f"unknown Phase-2 arm {arm_id!r}")
