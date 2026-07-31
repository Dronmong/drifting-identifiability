"""Plan section 9, P0.1 unit tests for the spectral source-law anchor."""

from __future__ import annotations

import numpy as np
import torch

from .. import spectral_anchor as SA
from ..config import AnchorConfig, BandSpec
from .harness import main

DIM = 32


def _bank(features: int = 192, seed: int = 11, bands=None,
          scale: float = 1.0) -> SA.SpectralBank:
    config = AnchorConfig(features=features,
                          bands=bands or AnchorConfig().bands)
    return SA.build_bank(config, DIM, scale, seed)


def _normal(n: int, seed: int, shift: float = 0.0,
            sigma: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, DIM, generator=g) * sigma + shift


def test_nonnegative() -> None:
    """1. the biased loss is nonnegative on random inputs"""
    bank = _bank()
    for seed in range(5):
        value = float(SA.anchor_loss(bank, _normal(64, seed),
                                     _normal(64, seed + 100), "biased"))
        assert value >= 0.0, value


def test_zero_on_identical_arrays() -> None:
    """2. the biased loss is exactly zero when the arrays are identical"""
    bank = _bank()
    x = _normal(64, 3)
    assert float(SA.anchor_loss(bank, x, x, "biased")) == 0.0


def test_unbiased_is_a_diagnostic_only() -> None:
    """2b. the unbiased U-statistic can be negative, so it never trains"""
    bank = _bank()
    x = _normal(64, 4)
    assert float(SA.anchor_loss(bank, x, x, "unbiased")) < 0.0
    assert AnchorConfig().training_estimator == "biased"


def test_unbiased_is_centred_on_independent_samples() -> None:
    """2c. the unbiased estimator averages to ~0 for p == q"""
    bank = _bank(features=256, seed=17)
    values = [float(SA.anchor_loss(bank, _normal(64, s),
                                   _normal(64, s + 500), "unbiased"))
              for s in range(40)]
    biased = [float(SA.anchor_loss(bank, _normal(64, s),
                                   _normal(64, s + 500), "biased"))
              for s in range(40)]
    assert abs(float(np.mean(values))) < abs(float(np.mean(biased))) / 4


def test_gradient_matches_finite_differences() -> None:
    """3. the analytic gradient matches central finite differences"""
    bank = _bank(features=64, seed=5)
    y = _normal(12, 7).to(torch.float64)
    x = _normal(24, 8).to(torch.float64)
    analytic = SA.anchor_gradient(bank, y, x)
    eps = 1e-5
    for i, j in ((0, 0), (3, 11), (7, 20)):
        plus, minus = y.clone(), y.clone()
        plus[i, j] += eps
        minus[i, j] -= eps
        numeric = (float(SA.anchor_loss(bank, plus, x, "biased"))
                   - float(SA.anchor_loss(bank, minus, x, "biased"))) / (
                       2 * eps)
        assert abs(numeric - float(analytic[i, j])) <= 1e-6 * max(
            1.0, abs(numeric)), (i, j, numeric, float(analytic[i, j]))


def test_gradient_matches_autograd() -> None:
    """3b. the analytic gradient matches autograd exactly"""
    bank = _bank(features=96, seed=6)
    y = _normal(16, 9).requires_grad_(True)
    x = _normal(32, 10)
    SA.anchor_loss(bank, y, x, "biased").backward()
    analytic = SA.anchor_gradient(bank, y.detach(), x)
    assert float((y.grad - analytic).abs().max()) < 1e-8


def test_detects_shift_and_variance() -> None:
    """4. shifted and variance-changed Gaussians are detected"""
    bank = _bank(features=256, seed=13)
    base = _normal(256, 21)
    null = float(SA.anchor_loss(bank, _normal(256, 22), base, "unbiased"))
    shifted = float(SA.anchor_loss(bank, _normal(256, 22, shift=0.3), base,
                                   "unbiased"))
    wider = float(SA.anchor_loss(bank, _normal(256, 22, sigma=1.5), base,
                                 "unbiased"))
    assert shifted > 10 * abs(null), (shifted, null)
    assert wider > 10 * abs(null), (wider, null)


def test_band_selectivity() -> None:
    """5. a fine-scale perturbation is detected only by the matching band

    The bands select scale in R^d, not spatial frequency in an image.  The
    right probe is therefore a law that agrees with the base law coarsely but
    differs at a declared fine scale: quantizing onto a lattice of spacing
    ``delta`` leaves low frequencies almost unchanged and produces a large
    discrepancy near ``|w| ~ 2 pi / delta``.
    """
    dim, delta = 2, 0.5
    matched = 2 * np.pi / delta

    def draw(seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.randn(512, dim, generator=g)

    base = draw(31)
    quantized = torch.round(draw(32) / delta) * delta
    independent = [draw(200 + s) for s in range(32)]

    def band(multiplier: float) -> SA.SpectralBank:
        return SA.build_bank(
            AnchorConfig(features=512, bands=(BandSpec("b", multiplier),)),
            dim, 1.0, 41)

    def z_score(bank: SA.SpectralBank) -> float:
        signal = float(SA.anchor_loss(bank, quantized, base, "unbiased"))
        null = [float(SA.anchor_loss(bank, other, base, "unbiased"))
                for other in independent]
        return signal / max(float(np.std(null)), 1e-12)

    low = band(200.0 / matched)      # |w| ~ 1/200 of the matched scale
    high = band(1.0 / matched)       # |w| ~ the matched scale
    # The low band's z-score is two-sided noise, so it is bounded in
    # magnitude rather than compared as a signed ratio.
    assert z_score(high) > 10.0, z_score(high)
    assert abs(z_score(low)) < 5.0, z_score(low)


def test_fresh_bank_averaging_converges() -> None:
    """6. averaging refreshed banks approaches a dense-bank reference"""
    dense = _bank(features=8192, seed=101)
    left, right = _normal(256, 51, shift=0.2), _normal(256, 52)
    reference = float(SA.anchor_loss(dense, left, right, "unbiased"))
    bank = _bank(features=64, seed=61)
    running: list[float] = []
    errors = []
    for index in range(64):
        running.append(float(SA.anchor_loss(bank, left, right, "unbiased")))
        bank = SA.refresh_bank(bank, 1.0, 900 + index)
        if index in (3, 63):
            errors.append(abs(float(np.mean(running)) - reference))
    assert errors[-1] < errors[0], errors
    assert errors[-1] < 0.25 * abs(reference), (errors, reference)


def test_orthogonal_directions_are_unit_and_orthogonal() -> None:
    """7. structured direction blocks stay unit-norm and orthogonal"""
    g = torch.Generator().manual_seed(3)
    dirs = SA.sample_directions(DIM, DIM, "orthogonal", g)
    gram = dirs @ dirs.T
    assert float((dirs.norm(dim=1) - 1.0).abs().max()) < 1e-10
    assert float((gram - torch.eye(DIM, dtype=gram.dtype)).abs().max()) < 1e-10


def test_bank_is_reproducible_from_its_seed() -> None:
    """8. a bank and its refresh are reproducible from the recorded seed"""
    a, b = _bank(seed=77), _bank(seed=77)
    assert torch.equal(a.frequencies, b.frequencies)
    assert torch.equal(SA.refresh_bank(a, 0.5, 5).frequencies,
                       SA.refresh_bank(b, 0.5, 5).frequencies)
    assert not torch.equal(_bank(seed=78).frequencies, a.frequencies)


def test_audit_bank_is_unreachable_from_the_training_seed() -> None:
    """9. the audit bank is not a function of the training seed"""
    from ..config import AUDIT_SEED, derive_seed
    audit_a = derive_seed(AUDIT_SEED, "audit-bank")
    audit_b = derive_seed(AUDIT_SEED, "audit-bank")
    assert audit_a == audit_b
    assert audit_a != derive_seed(1234, "anchor-bank")


def test_band_schedule_opens_coarse_to_fine() -> None:
    """11. R6: the schedule starts coarse, ends uniform, never zeroes a band"""
    config = AnchorConfig(band_schedule="coarse_to_fine", schedule_floor=0.05,
                          schedule_warmup=0.5)
    start = SA.band_weights(config, 0.0)
    end = SA.band_weights(config, 1.0)
    # Coarse band fully active immediately; finer bands held at the floor.
    assert float(start[0]) == 1.0, start
    assert float(start[1]) == config.schedule_floor, start
    assert float(start[-1]) == config.schedule_floor, start
    # Bands open in order: band 1 ramps over [0, 0.25], band 2 over
    # [0.25, 0.5] for three bands at warmup 0.5.
    assert float(SA.band_weights(config, 0.125)[1]) > float(start[1])
    assert float(SA.band_weights(config, 0.125)[2]) == config.schedule_floor
    assert float(SA.band_weights(config, 0.375)[2]) > float(start[2])
    # Weights are monotone non-decreasing in progress, per band.
    previous = start
    for progress in (0.1, 0.2, 0.3, 0.4, 0.5, 1.0):
        current = SA.band_weights(config, progress)
        assert bool((current >= previous - 1e-12).all()), (previous, current)
        previous = current
    assert float(end.min()) == 1.0, end
    for progress in (0.0, 0.1, 0.3, 0.7, 1.0):
        weights = SA.band_weights(config, progress)
        assert float(weights.min()) >= config.schedule_floor - 1e-12


def test_fixed_schedule_reproduces_phase1_behaviour() -> None:
    """12. R6: schedule "fixed" and progress None are exactly uniform"""
    fixed = AnchorConfig(band_schedule="fixed")
    scheduled = AnchorConfig(band_schedule="coarse_to_fine")
    for progress in (0.0, 0.5, 1.0, None):
        assert torch.equal(SA.band_weights(fixed, progress),
                           torch.ones(len(fixed.bands), dtype=SA.PHASE_DTYPE))
    assert torch.equal(SA.band_weights(scheduled, None),
                       torch.ones(len(scheduled.bands), dtype=SA.PHASE_DTYPE))
    bank = _bank()
    x, y = _normal(64, 60), _normal(64, 61, shift=0.3)
    assert float(SA.anchor_loss(bank, x, y, "biased")) == float(
        SA.anchor_loss(bank, x, y, "biased", progress=None))


def test_schedule_weights_have_mean_one() -> None:
    """13. R6: per-frequency weights are mean-one so lambda_A keeps meaning"""
    bank = _bank(features=192)
    for progress in (0.0, 0.2, 0.6, 1.0):
        weights = SA.frequency_weights(bank, progress)
        assert abs(float(weights.mean()) - 1.0) < 1e-9, float(weights.mean())
        assert len(weights) == bank.size


def test_scheduled_gradient_matches_scheduled_loss() -> None:
    """14. R6: the analytic gradient tracks the scheduled loss"""
    bank = _bank(features=96, seed=31)
    y = _normal(16, 62).requires_grad_(True)
    x = _normal(32, 63, shift=0.4)
    SA.anchor_loss(bank, y, x, "biased", progress=0.15).backward()
    analytic = SA.anchor_gradient(bank, y.detach(), x, progress=0.15)
    assert float((y.grad - analytic).abs().max()) < 1e-8
    # And a scheduled gradient differs from the unscheduled one.
    plain = SA.anchor_gradient(bank, y.detach(), x)
    assert float((plain - analytic).abs().max()) > 0


def test_schedule_suppresses_the_noisy_high_band_early() -> None:
    """15. R6: early training weights coarse frequencies far above fine ones"""
    config = AnchorConfig(features=300, band_schedule="coarse_to_fine")
    bank = SA.build_bank(config, DIM, 1.0, 51)
    weights = SA.frequency_weights(bank, 0.0)
    coarse = float(weights[bank.band_index == 0].mean())
    fine = float(weights[bank.band_index == 2].mean())
    assert coarse > 10 * fine, (coarse, fine)


def test_projected_scale_tracks_the_target_scale() -> None:
    """10. band calibration follows the target's projected spread"""
    config = AnchorConfig()
    g = torch.Generator().manual_seed(2)
    small = SA.projected_scale(_normal(256, 71, sigma=0.5), config, g)
    g = torch.Generator().manual_seed(2)
    large = SA.projected_scale(_normal(256, 71, sigma=2.0), config, g)
    assert 3.0 < large / small < 5.0, (small, large)


if __name__ == "__main__":
    main("spectral anchor (P0.1)", dict(globals()))
