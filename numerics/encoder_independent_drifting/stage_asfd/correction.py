"""Executable ASFD correction attached to the CAP foundation training loop.

The online sampler is continued from the exact 750k recovery.  A frozen copy
of the 750k EMA supplies descriptor geometry, but its parameters never receive
gradients.  Every correction event adds the unprojected, independently
safety-capped gradients of three nonnegative finite-sample objectives before
CAP's ordinary global clip: a spectral source-law anchor, a raw Laplace field
energy, and the same field energy in frozen self-feature geometry.  The
uncapped components are ordinary scalar-loss gradients; norm-dependent caps
make the realized update a stabilized heuristic rather than the gradient of
one fixed combined objective.
"""

from __future__ import annotations

import math
from collections import deque
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import AnchorConfig
from ..spectral_anchor import SpectralBank, anchor_loss, build_bank, refresh_bank
from ..stage_cap.data import cifar10_train_labels, cifar10_train_pool, flip_batch
from ..stage_cap.diagnostics import effective_rank
from ..stage_cap.model import one_step_sample
from ..stage_cap2.standard_metrics import _load_model
from .artifacts import file_sha256
from .config import ASFDConfig, asfd_config
from .feature_bank import FeatureBank, verify_dataset_binding
from .features import LevelNormalization, encode, freeze_trunk, to_locations
from .field import descriptor_energy, multi_radius_energy
from .gradients import AbortMonitor, combine, gradient_norm, snapshot

CORRECTION_SEED = 20_260_909


def _normalization(payload: dict) -> dict[str, LevelNormalization]:
    result = {}
    for name, record in payload.items():
        channel = record.get("channel_scale")
        result[name] = LevelNormalization(
            channel_scale=None if channel is None else torch.tensor(channel),
            level_scale=float(record["level_scale"]),
            pc1_share=float(record["pc1_share"]),
            per_channel_applied=bool(record["per_channel_applied"]),
        )
    return result


def _taus(selected: dict) -> tuple[dict[float, float], dict[str, dict[float, float]]]:
    records = selected["bandwidths"]
    raw = {
        float(radius): float(record["tau"]) for radius, record in records["raw"].items()
    }
    feature = {
        level: {
            float(radius): float(record["tau"]) for radius, record in per_level.items()
        }
        for level, per_level in records["feature"].items()
    }
    return raw, feature


def _tensor_gradients(
    value: torch.Tensor,
    parameters: tuple[nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> list[torch.Tensor | None]:
    gradients = torch.autograd.grad(
        value,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return [None if gradient is None else gradient for gradient in gradients]


class ASFDCorrection:
    """Stateful, replayable correction hook for one continuation branch."""

    def __init__(
        self,
        *,
        teacher_checkpoint: Path,
        bank_metadata: Path,
        qualification: dict,
        coefficients: dict[str, float],
        spectral_scale: float,
        device: torch.device,
        data_root: str | None,
        continuation_start: int = 750_000,
        continuation_updates: int | None = None,
        config: ASFDConfig | None = None,
        identity_binding: dict | None = None,
        stream_seed_offset: int = 0,
    ) -> None:
        self.config = config or asfd_config()
        self.config.validate()
        self.device = torch.device(device)
        self.pool = cifar10_train_pool(data_root)
        self.bank = FeatureBank(bank_metadata)
        verify_dataset_binding(
            self.bank.payload.get("dataset_binding"),
            self.pool,
            cifar10_train_labels(data_root),
        )
        payload, teacher, _ = _load_model(teacher_checkpoint, self.device)
        # The teacher must be the EMA of the exact foundation this correction
        # continues -- ``continuation_start`` is that foundation's step, so the
        # two cannot drift apart. The step was previously written as a literal
        # 750_000, which silently assumed one horizon and made the parameter
        # decorative.
        if payload.get("kind") != "ema" or int(payload.get("step", -1)) != int(
            continuation_start
        ):
            raise RuntimeError(
                f"ASFD teacher must be the EMA at step {int(continuation_start)}, "
                f"the foundation this correction continues; got kind="
                f"{payload.get('kind')!r} step={payload.get('step')!r}"
            )
        self.teacher = freeze_trunk(teacher)
        self.teacher_checkpoint_sha256 = file_sha256(teacher_checkpoint)
        self.bank_metadata_sha256 = file_sha256(bank_metadata)
        selected = qualification.get("selected")
        if qualification.get("decision") != "GO" or not isinstance(selected, dict):
            raise RuntimeError("ASFD correction requires a GO qualification")
        if (
            qualification.get("teacher_checkpoint", {}).get("sha256")
            != self.teacher_checkpoint_sha256
            or self.bank.payload.get("teacher_checkpoint", {}).get("sha256")
            != self.teacher_checkpoint_sha256
            or self.bank.payload.get("qualification", {}).get("sha256")
            != qualification.get("artifact_sha256")
        ):
            raise RuntimeError("ASFD teacher, qualification and bank bindings differ")
        self.t_f = float(selected["t_f"])
        self.normalization = _normalization(selected["normalization"])
        self.raw_taus, self.feature_taus = _taus(selected)
        self.coefficients = {
            name: float(coefficients[name]) for name in ("b1", "raw", "self")
        }
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.coefficients.values()
        ):
            raise ValueError("ASFD coefficients must be positive and finite")
        if not math.isfinite(spectral_scale) or spectral_scale <= 0:
            raise ValueError("ASFD spectral scale must be positive and finite")
        self.stream_seed_offset = int(stream_seed_offset)
        if self.stream_seed_offset < 0:
            raise ValueError("ASFD stream seed offset must be nonnegative")
        stream_seed = CORRECTION_SEED + self.stream_seed_offset
        self.anchor_config = AnchorConfig()
        self.spectral_scale = float(spectral_scale)
        self.spectral_bank = build_bank(
            self.anchor_config,
            3 * 32 * 32,
            self.spectral_scale,
            stream_seed + 1,
        )
        self.continuation_start = int(continuation_start)
        self.continuation_updates = int(
            continuation_updates or self.config.continuation_updates
        )
        self.identity_binding = deepcopy(identity_binding or {})
        self.event_count = 0
        self.generated_forwards = 0
        self.monitor = AbortMonitor(self.config.gradients)
        self.ess_windows: dict[str, deque[float]] = {}
        self.role_rng = np.random.default_rng(stream_seed + 2)
        self.b1_rng = np.random.default_rng(stream_seed + 3)
        self.b1_flip_generator = torch.Generator().manual_seed(stream_seed + 7)
        self.prior_generator = torch.Generator().manual_seed(stream_seed + 4)
        self.feature_noise_generator = torch.Generator().manual_seed(stream_seed + 5)
        self.raw_probe_generator = torch.Generator().manual_seed(stream_seed + 6)
        self.fresh_probe_generator = torch.Generator().manual_seed(stream_seed + 8)
        self.rank_baselines: dict[str, float] = {}

    def identity(self) -> dict:
        return {
            "status": "asfd-correction-identity",
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            "feature_bank_sha256": self.bank_metadata_sha256,
            "config": asdict(self.config),
            "coefficients": dict(self.coefficients),
            "spectral_scale": self.spectral_scale,
            "continuation_start": self.continuation_start,
            "continuation_updates": self.continuation_updates,
            "stream_seed_offset": self.stream_seed_offset,
            "binding": deepcopy(self.identity_binding),
        }

    def state_dict(self) -> dict:
        return {
            "event_count": self.event_count,
            "generated_forwards": self.generated_forwards,
            "role_rng": deepcopy(self.role_rng.bit_generator.state),
            "b1_rng": deepcopy(self.b1_rng.bit_generator.state),
            "b1_flip_generator": self.b1_flip_generator.get_state(),
            "prior_generator": self.prior_generator.get_state(),
            "feature_noise_generator": self.feature_noise_generator.get_state(),
            "raw_probe_generator": self.raw_probe_generator.get_state(),
            "fresh_probe_generator": self.fresh_probe_generator.get_state(),
            "spectral_bank": {
                "frequencies": self.spectral_bank.frequencies,
                "band_index": self.spectral_bank.band_index,
                "seed": self.spectral_bank.seed,
                "refreshes": self.spectral_bank.refreshes,
            },
            "abort_monitor": self.monitor.state_dict(),
            "ess_windows": {
                name: list(values) for name, values in self.ess_windows.items()
            },
            "rank_baselines": dict(self.rank_baselines),
        }

    def load_state_dict(self, payload: dict) -> None:
        self.event_count = int(payload["event_count"])
        self.generated_forwards = int(payload["generated_forwards"])
        self.role_rng.bit_generator.state = payload["role_rng"]
        self.b1_rng.bit_generator.state = payload["b1_rng"]
        self.b1_flip_generator.set_state(payload["b1_flip_generator"])
        self.prior_generator.set_state(payload["prior_generator"])
        self.feature_noise_generator.set_state(payload["feature_noise_generator"])
        self.raw_probe_generator.set_state(payload["raw_probe_generator"])
        self.fresh_probe_generator.set_state(payload["fresh_probe_generator"])
        bank = payload["spectral_bank"]
        self.spectral_bank = SpectralBank(
            frequencies=bank["frequencies"].clone(),
            band_index=bank["band_index"].clone(),
            band_names=tuple(spec.name for spec in self.anchor_config.bands),
            scale=self.spectral_scale,
            config=self.anchor_config,
            seed=int(bank["seed"]),
            refreshes=int(bank["refreshes"]),
        )
        self.monitor.load_state_dict(payload["abort_monitor"])
        self.ess_windows = {
            str(name): deque((float(value) for value in values), maxlen=20)
            for name, values in payload.get("ess_windows", {}).items()
        }
        self.rank_baselines = {
            str(name): float(value)
            for name, value in payload.get("rank_baselines", {}).items()
        }

    def _rows(self, role: str, count: int) -> np.ndarray:
        rows = int(self.bank.payload["roles"][role]["rows"])
        return self.role_rng.integers(0, rows, size=count, dtype=np.int64)

    def _target_roles(self) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
        fields = self.config.field_config
        probe_rows = self._rows("train_probe", fields.probes)
        positive_rows = self._rows("train_positive", fields.positives)
        feature_probes = to_locations(
            self.bank.sample("train_probe", probe_rows, device=self.device)
        )
        feature_positives = to_locations(
            self.bank.sample("train_positive", positive_rows, device=self.device)
        )
        raw_probes = self.bank.raw_batch(
            "train_probe", probe_rows, self.pool, device=self.device
        )
        raw_positives = self.bank.raw_batch(
            "train_positive", positive_rows, self.pool, device=self.device
        )
        raw_noise = torch.randn(
            raw_probes.shape,
            generator=self.raw_probe_generator,
            dtype=raw_probes.dtype,
        ).to(self.device)
        raw_probes = raw_probes + fields.probe_noise_std * raw_noise
        return feature_probes, feature_positives, raw_probes, raw_positives

    def _negative_ess_ceiling(self, label: str) -> float:
        """The ceiling cannot sit below the radius it is gating.

        A radius is an ESS fraction and the bandwidth is calibrated to hit it,
        so a broad radius reaches a high ESS by construction rather than by
        degenerating. Taking the max with the absolute ceiling leaves every
        radius below it untouched.
        """
        field = self.config.field_config
        try:
            radius = float(label.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return field.negative_ess_ceiling
        relative = radius * (1.0 + field.negative_ess_excess)
        return min(
            field.negative_ess_absolute_cap,
            max(field.negative_ess_ceiling, relative),
        )

    def _record_ess(self, label: str, value: float) -> None:
        window = self.ess_windows.setdefault(label, deque(maxlen=20))
        window.append(float(value))
        if len(window) == window.maxlen:
            self.monitor.observe_negative_ess(
                label,
                sum(window) / len(window),
                self._negative_ess_ceiling(label),
            )

    def _generated_ranks(
        self,
        generated: torch.Tensor,
        generated_features: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Participation ranks in raw pixels and each frozen feature level."""
        result = {"raw": effective_rank(generated.detach())}
        for level, values in generated_features.items():
            # values are [location, sample, channel].  Rank is across samples,
            # with every declared local/global descriptor retained.
            rows = values.detach().transpose(0, 1).flatten(1)
            result[f"feature/{level}"] = effective_rank(rows)
        return result

    def _fresh_roles(
        self,
    ) -> tuple[dict, dict, dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Independent held-out target roles for checkpoint-time diagnostics."""
        fields = self.config.field_config
        probe_rows = self._rows("fresh_probe", fields.probes)
        positive_rows = self._rows("fresh_positive", fields.positives)
        negative_rows = self._rows("fresh_positive", fields.negatives)
        feature_probes = to_locations(
            self.bank.sample("fresh_probe", probe_rows, device=self.device)
        )
        feature_positives = to_locations(
            self.bank.sample("fresh_positive", positive_rows, device=self.device)
        )
        feature_negatives = to_locations(
            self.bank.sample("fresh_positive", negative_rows, device=self.device)
        )
        raw_probes = self.bank.raw_batch(
            "fresh_probe", probe_rows, self.pool, device=self.device
        )
        raw_positives = self.bank.raw_batch(
            "fresh_positive", positive_rows, self.pool, device=self.device
        )
        raw_negatives = self.bank.raw_batch(
            "fresh_positive", negative_rows, self.pool, device=self.device
        )
        noise = torch.randn(
            raw_probes.shape,
            generator=self.fresh_probe_generator,
            dtype=raw_probes.dtype,
        ).to(self.device)
        raw_probes = raw_probes + fields.probe_noise_std * noise
        return (
            feature_probes,
            feature_positives,
            feature_negatives,
            raw_probes,
            raw_positives,
            raw_negatives,
        )

    def _held_out_health(
        self,
        generated: torch.Tensor,
        generated_features: dict[str, torch.Tensor],
        train_raw_energy: float,
        train_feature_energy: float,
    ) -> dict:
        """Held-out drift, a conservative real-real floor, and diversity rank.

        Five independently resampled real-real floors are reduced by their
        minimum.  This keeps the collapse veto conservative: a model is stopped
        only when it undershoots every sampled real-real discrepancy, rather
        than one upward-noisy estimate.
        """
        ranks = self._generated_ranks(generated, generated_features)
        if not self.rank_baselines:
            self.rank_baselines = dict(ranks)
            return {
                "baseline_established": True,
                "rank": ranks,
                "rank_ratio": {name: 1.0 for name in ranks},
            }

        raw_generated_values: list[float] = []
        feature_generated_values: list[float] = []
        raw_real_values: list[float] = []
        feature_real_values: list[float] = []
        detached_generated = generated.detach().flatten(1)
        detached_features = {
            name: values.detach() for name, values in generated_features.items()
        }
        with torch.no_grad():
            for _ in range(5):
                (
                    feature_probes,
                    feature_positives,
                    feature_negatives,
                    raw_probes,
                    raw_positives,
                    raw_negatives,
                ) = self._fresh_roles()
                raw_generated, _ = multi_radius_energy(
                    raw_probes.flatten(1),
                    raw_positives.flatten(1),
                    detached_generated,
                    self.raw_taus,
                    diagnostics=False,
                )
                raw_real, _ = multi_radius_energy(
                    raw_probes.flatten(1),
                    raw_positives.flatten(1),
                    raw_negatives.flatten(1),
                    self.raw_taus,
                    diagnostics=False,
                )
                feature_generated, _ = descriptor_energy(
                    feature_probes,
                    feature_positives,
                    detached_features,
                    self.feature_taus,
                    diagnostics=False,
                )
                feature_real, _ = descriptor_energy(
                    feature_probes,
                    feature_positives,
                    feature_negatives,
                    self.feature_taus,
                    diagnostics=False,
                )
                raw_generated_values.append(float(raw_generated))
                raw_real_values.append(float(raw_real))
                feature_generated_values.append(float(feature_generated))
                feature_real_values.append(float(feature_real))
        raw_fresh = float(np.median(raw_generated_values))
        feature_fresh = float(np.median(feature_generated_values))
        rank_ratio = {
            name: value / max(self.rank_baselines[name], 1e-30)
            for name, value in ranks.items()
        }
        return {
            "baseline_established": False,
            "rank": ranks,
            "rank_baseline": dict(self.rank_baselines),
            "rank_ratio": rank_ratio,
            "raw": {
                "train_energy": train_raw_energy,
                "fresh_energy_median_5": raw_fresh,
                "train_minus_fresh": train_raw_energy - raw_fresh,
                "real_real_energy_samples": raw_real_values,
                "conservative_real_real_floor_min_5": min(raw_real_values),
                "excess_over_conservative_floor": raw_fresh - min(raw_real_values),
            },
            "feature": {
                "train_energy": train_feature_energy,
                "fresh_energy_median_5": feature_fresh,
                "train_minus_fresh": train_feature_energy - feature_fresh,
                "real_real_energy_samples": feature_real_values,
                "conservative_real_real_floor_min_5": min(feature_real_values),
                "excess_over_conservative_floor": feature_fresh
                - min(feature_real_values),
            },
        }

    def compute_components(
        self, step: int, model: nn.Module, *, include_health: bool = True
    ) -> tuple[dict[str, list[torch.Tensor | None]], dict]:
        parameters = tuple(model.parameters())
        fields = self.config.field_config
        feature_probes, feature_positives, raw_probes, raw_positives = (
            self._target_roles()
        )
        prior = torch.randn(
            fields.negatives,
            3,
            32,
            32,
            generator=self.prior_generator,
        ).to(self.device)
        generated = one_step_sample(model, prior)
        self.generated_forwards += 1

        b1_rows = self.b1_rng.integers(
            0, len(self.pool), size=fields.negatives, dtype=np.int64
        )
        b1_target = self.pool[torch.as_tensor(b1_rows)].clone()
        b1_flips = torch.rand(fields.negatives, generator=self.b1_flip_generator) < 0.5
        b1_target = flip_batch(b1_target, b1_flips).to(self.device)
        progress = min(
            max((step - self.continuation_start) / self.continuation_updates, 0.0),
            1.0,
        )
        b1_value = anchor_loss(
            self.spectral_bank,
            generated,
            b1_target,
            estimator="biased",
            progress=progress,
        )
        raw_value, raw_stats = multi_radius_energy(
            raw_probes.flatten(1),
            raw_positives.flatten(1),
            generated.flatten(1),
            self.raw_taus,
        )
        generated_descriptor_map = encode(
            self.teacher,
            generated,
            self.t_f,
            self.config.features,
            self.normalization,
            generator=self.feature_noise_generator,
        )
        generated_features = to_locations(generated_descriptor_map)
        self_value, self_stats = descriptor_energy(
            feature_probes,
            feature_positives,
            generated_features,
            self.feature_taus,
        )
        generated_input_gradient = torch.autograd.grad(
            self_value, generated, retain_graph=True, allow_unused=False
        )[0]
        input_gradient_norm = float(
            generated_input_gradient.detach().double().square().sum().sqrt()
        )
        if not math.isfinite(input_gradient_norm) or input_gradient_norm <= 0:
            raise RuntimeError(
                "ASFD frozen teacher has no live generated-input Jacobian"
            )

        components = {
            "b1": _tensor_gradients(b1_value, parameters, retain_graph=True),
            "raw": _tensor_gradients(raw_value, parameters, retain_graph=True),
            "self": _tensor_gradients(self_value, parameters, retain_graph=False),
        }
        for name, values in components.items():
            if gradient_norm(values) <= 0:
                raise RuntimeError(
                    f"ASFD {name} component has a zero parameter gradient"
                )
        stats = {
            "losses": {
                "b1": float(b1_value.detach()),
                "raw": float(raw_value.detach()),
                "self": float(self_value.detach()),
            },
            "raw_field": raw_stats,
            "self_field": self_stats,
            "generated_input_gradient_norm": input_gradient_norm,
            "generated_forwards": self.generated_forwards,
        }
        next_event = self.event_count + 1
        health_interval = max(
            1,
            min(self.config.checkpoint_every, self.continuation_updates)
            // self.config.gradients.cadence,
        )
        if include_health and (next_event == 1 or next_event % health_interval == 0):
            stats["held_out_health"] = self._held_out_health(
                generated,
                generated_features,
                float(raw_value.detach()),
                float(self_value.detach()),
            )
        return components, stats

    def apply(self, step: int, model: nn.Module) -> dict | None:
        if step % self.config.gradients.cadence:
            return None
        primary = snapshot(model)
        components, stats = self.compute_components(step, model)
        weighted = {
            name: [
                None if value is None else value * self.coefficients[name]
                for value in values
            ]
            for name, values in components.items()
        }
        caps = {
            "b1": self.config.gradients.cap_b1,
            "raw": self.config.gradients.cap_raw,
            "self": self.config.gradients.cap_self,
        }
        outcome = combine(
            list(model.parameters()),
            primary,
            {name: (values, caps[name]) for name, values in weighted.items()},
            self.config.gradients,
        )
        self.monitor.observe_cosines(outcome.cosines)
        for label, value in stats["losses"].items():
            self.monitor.observe_finite(value, f"{label} loss")
        for radius, health in stats["raw_field"].items():
            if isinstance(health, dict) and "negative" in health:
                self._record_ess(
                    f"raw/{radius}", health["negative"]["ess_fraction_median"]
                )
        for level, level_stats in stats["self_field"].items():
            if not isinstance(level_stats, dict):
                continue
            for radius, health in level_stats.items():
                if isinstance(health, dict) and "negative" in health:
                    self._record_ess(
                        f"self/{level}/{radius}",
                        health["negative"]["ess_fraction_median"],
                    )
        health = stats.get("held_out_health")
        if isinstance(health, dict) and not health.get("baseline_established", False):
            for label, ratio in health["rank_ratio"].items():
                self.monitor.observe_rank(label, float(ratio))
            self.monitor.observe_energy_floor(
                float(health["raw"]["fresh_energy_median_5"]),
                float(health["raw"]["conservative_real_real_floor_min_5"]),
            )
        if self.monitor.should_abort:
            raise RuntimeError("ASFD safety abort: " + "; ".join(self.monitor.reasons))
        self.event_count += 1
        if self.event_count % self.anchor_config.refresh_every == 0:
            self.spectral_bank = refresh_bank(
                self.spectral_bank,
                self.anchor_config.refresh_fraction,
                CORRECTION_SEED + self.stream_seed_offset + 100_000 + self.event_count,
            )
        return {
            "step": int(step),
            "event": self.event_count,
            **stats,
            "gradient": {
                "primary_norm": outcome.primary_norm,
                "pre_cap_ratio": outcome.pre_cap_ratio,
                "post_cap_ratio": outcome.post_cap_ratio,
                "applied_cap_scale": outcome.applied,
                "cosines": outcome.cosines,
                "total_auxiliary_ratio": outcome.total_auxiliary_ratio,
            },
            "spectral_bank_refreshes": self.spectral_bank.refreshes,
        }
