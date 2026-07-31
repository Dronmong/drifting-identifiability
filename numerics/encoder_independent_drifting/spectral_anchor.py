"""Branch A: the spectral source-law anchor (plan section 6.1).

Ideal object
------------
For a spectral measure ``rho`` on R^d the anchor discrepancy is

    L_anchor(p, q) = E_{w ~ rho} | E_p e^{i<w,X>} - E_q e^{i<w,Y>} |^2 .

If ``rho`` has full support then the induced translation-invariant kernel is
characteristic, hence ``L_anchor(p, q) = 0`` iff ``p = q``.  That is a
statement about the ideal expectation, and it is the only correctness
authority in this program.

Finite object
-------------
Everything computed here is a finite random-feature bank

    z_w(x) = [cos<w,x>, sin<w,x>]

with ``w = r * u``, ``u`` a unit direction and ``r`` from a declared
multiband radial law.  Averaging a random bank is unbiased for the
corresponding frequency expectation conditional on the empirical samples;
the V- and U-statistics below have their usual, different sample biases.  A
finite bank is **not** itself measure-determining, and no code or report in
this package may say otherwise.  The training bank is refreshed on a
declared schedule so that no single frozen finite bank becomes the claimed
identifying object, and an independent audit bank (never used for training or
selection) is retained for reporting.

Two estimators are provided.  The biased V-statistic is manifestly
nonnegative and is the only one used in the training objective, preventing
algebraic cancellation against another nonnegative loss.  This fact does not
promote a finite bank to a measure-determining criterion.  The unbiased
U-statistic removes the O(1/n) diagonal bias but can be negative, so it is a
diagnostic.

Band calibration uses target-only *projected* scales.  Calibrating from the
ambient pixel norm would put every band at a length scale growing like
sqrt(d) and reproduce the paper's flat-kernel failure.

An antithetic `+w/-w` pairing is deliberately **not** implemented: the
summand is even in ``w`` (cos is even, sin is odd and enters squared through
a mean difference that flips sign), so the pair contributes exactly the same
value and the "variance reduction" would be an accounting illusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import AnchorConfig, BandSpec

# Frequencies from heavy-tailed radial laws can be large; phases are computed
# in float64 so that cos/sin stay accurate for large <w,x>.
PHASE_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Directions and radii
# ---------------------------------------------------------------------------


def sample_directions(count: int, dim: int, mode: str,
                      generator: torch.Generator) -> torch.Tensor:
    """Unit directions, ``[count, dim]``.

    ``iid``: independent uniform-on-sphere directions.
    ``orthogonal``: stacked Haar-orthogonal blocks.  Each row is still
    marginally uniform on the sphere -- the sign correction below is what
    makes the QR factor Haar distributed -- so the declared population
    target is unchanged while redundant projections are reduced.
    """
    if count <= 0 or dim <= 0:
        raise ValueError("count and dim must be positive")
    if mode == "iid":
        raw = torch.randn(count, dim, generator=generator,
                          dtype=PHASE_DTYPE)
        return raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-300)
    if mode != "orthogonal":
        raise ValueError(f"unknown direction mode {mode!r}")
    rows: list[torch.Tensor] = []
    remaining = count
    while remaining > 0:
        block = min(remaining, dim)
        raw = torch.randn(dim, block, generator=generator, dtype=PHASE_DTYPE)
        q, r = torch.linalg.qr(raw)
        # Haar correction: QR is only unique up to the signs of diag(R).
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        rows.append((q * signs).T)
        remaining -= block
    return torch.cat(rows, dim=0)[:count]


def sample_radii(count: int, law: str, nu: float,
                 generator: torch.Generator) -> torch.Tensor:
    """Dimensionless radii ``R > 0`` from a declared law, ``[count]``.

    All three laws have a density that is positive on the whole of
    ``(0, inf)``, so the induced rotation-invariant spectral measure has full
    support and the ideal kernel is characteristic.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if law == "half_normal":
        return torch.randn(count, generator=generator,
                           dtype=PHASE_DTYPE).abs()
    if law == "student":
        if nu <= 0:
            raise ValueError("student law needs nu > 0")
        z = torch.randn(count, generator=generator, dtype=PHASE_DTYPE)
        # chi2_nu via a Gamma(nu/2, 2) draw, sampled through torch's Gamma.
        gamma = torch._standard_gamma(
            torch.full((count,), nu / 2.0, dtype=PHASE_DTYPE),
            generator=generator)
        chi2 = 2.0 * gamma
        return (z / torch.sqrt(chi2 / nu)).abs()
    if law == "heavy_cauchy":
        z1 = torch.randn(count, generator=generator, dtype=PHASE_DTYPE)
        z2 = torch.randn(count, generator=generator, dtype=PHASE_DTYPE)
        return (z1 / z2).abs()
    raise ValueError(f"unknown radial law {law!r}")


# ---------------------------------------------------------------------------
# Target-only projected-scale calibration
# ---------------------------------------------------------------------------


def projected_scale(target: torch.Tensor, config: AnchorConfig,
                    generator: torch.Generator) -> float:
    """Typical spread of ``<u, X>`` for unit ``u``, from target samples only.

    Uses pairwise projected differences rather than a projected standard
    deviation, because the anchor's frequencies act on differences.
    """
    flat = target.reshape(len(target), -1).to(PHASE_DTYPE)
    if len(flat) < 2:
        raise ValueError("projected-scale calibration needs >= 2 samples")
    dirs = sample_directions(
        config.projected_scale_probes, flat.shape[1], "iid", generator)
    proj = flat @ dirs.T                                  # [n, probes]
    pairs = min(config.projected_scale_pairs, len(flat) * (len(flat) - 1) // 2)
    left = torch.randint(0, len(flat), (pairs,), generator=generator)
    right = torch.randint(0, len(flat), (pairs,), generator=generator)
    diffs = (proj[left] - proj[right]).abs()
    scale = float(torch.quantile(
        diffs.flatten(), config.projected_scale_quantile))
    if not scale > 0:
        raise ValueError("degenerate projected scale; target has no spread")
    return scale


# ---------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------


@dataclass
class SpectralBank:
    frequencies: torch.Tensor      # [L, d], float64
    band_index: torch.Tensor       # [L], int64
    band_names: tuple[str, ...]
    scale: float                   # target-only projected scale used
    config: AnchorConfig
    seed: int
    refreshes: int = 0

    @property
    def size(self) -> int:
        return int(self.frequencies.shape[0])

    @property
    def dim(self) -> int:
        return int(self.frequencies.shape[1])

    def summary(self) -> dict:
        norms = self.frequencies.norm(dim=1)
        return {
            "features": self.size,
            "dim": self.dim,
            "seed": self.seed,
            "refreshes": self.refreshes,
            "projected_scale": self.scale,
            "band_names": list(self.band_names),
            "band_counts": [int((self.band_index == b).sum())
                            for b in range(len(self.band_names))],
            "frequency_norm_min": float(norms.min()),
            "frequency_norm_median": float(norms.median()),
            "frequency_norm_max": float(norms.max()),
        }


def _band_allocation(bands: tuple[BandSpec, ...], total: int) -> list[int]:
    weights = [max(float(b.weight), 0.0) for b in bands]
    if sum(weights) <= 0:
        raise ValueError("band weights must not all be zero")
    raw = [w / sum(weights) * total for w in weights]
    counts = [int(x) for x in raw]
    # Deterministic largest-remainder assignment of the leftover features.
    order = sorted(range(len(bands)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in range(total - sum(counts)):
        counts[order[i % len(order)]] += 1
    if min(counts) < 1:
        raise ValueError(
            "every declared band must receive at least one feature; "
            "increase AnchorConfig.features")
    return counts


def build_bank(config: AnchorConfig, dim: int, scale: float, seed: int,
               features: int | None = None) -> SpectralBank:
    """Build a frequency bank at a target-only calibrated projected scale."""
    total = int(features if features is not None else config.features)
    counts = _band_allocation(config.bands, total)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2 ** 63 - 1))
    freqs: list[torch.Tensor] = []
    index: list[torch.Tensor] = []
    for band_id, (band, count) in enumerate(zip(config.bands, counts)):
        length = band.length_multiplier * scale
        if not length > 0:
            raise ValueError(f"band {band.name} has a non-positive length")
        dirs = sample_directions(count, dim, config.direction_mode, generator)
        radii = sample_radii(count, band.law, band.nu, generator)
        freqs.append(dirs * (radii / length).unsqueeze(1))
        index.append(torch.full((count,), band_id, dtype=torch.int64))
    return SpectralBank(
        frequencies=torch.cat(freqs, dim=0),
        band_index=torch.cat(index, dim=0),
        band_names=tuple(b.name for b in config.bands),
        scale=scale,
        config=config,
        seed=int(seed),
    )


def refresh_bank(bank: SpectralBank, fraction: float,
                 seed: int) -> SpectralBank:
    """Resample a declared fraction of the bank in place of a frozen bank.

    Refreshed features are drawn from the same declared bands, so the
    population target is unchanged; refreshing exists so that no particular
    finite bank can be mistaken for the identifying object.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("refresh fraction must lie in [0, 1]")
    if fraction == 0.0:
        return bank
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2 ** 63 - 1))
    # Banks are canonically stored on CPU.  Feature evaluation moves the
    # immutable frequencies to the sample device as needed.  Keeping refresh
    # sampling on CPU makes a refresh replay bitwise-identically on CPU/CUDA
    # runs and prevents CPU-generator/GPU-index assignment failures.
    frequencies = bank.frequencies.detach().cpu().clone()
    band_index = bank.band_index.detach().cpu()
    for band_id, band in enumerate(bank.config.bands):
        slots = torch.nonzero(band_index == band_id).flatten()
        replace = round(fraction * len(slots))
        if replace <= 0:
            continue
        chosen = slots[torch.randperm(len(slots), generator=generator)[
            :replace]]
        length = band.length_multiplier * bank.scale
        dirs = sample_directions(
            len(chosen), bank.dim, bank.config.direction_mode, generator)
        radii = sample_radii(len(chosen), band.law, band.nu, generator)
        frequencies[chosen] = dirs * (radii / length).unsqueeze(1)
    return SpectralBank(
        frequencies=frequencies,
        band_index=band_index,
        band_names=bank.band_names,
        scale=bank.scale,
        config=bank.config,
        seed=int(seed),
        refreshes=bank.refreshes + 1,
    )


# ---------------------------------------------------------------------------
# Features, moments, losses
# ---------------------------------------------------------------------------


def phases(bank: SpectralBank, samples: torch.Tensor) -> torch.Tensor:
    """``<w_l, x_i>`` as ``[n, L]`` in float64."""
    flat = samples.reshape(len(samples), -1)
    if flat.shape[1] != bank.dim:
        raise ValueError(
            f"sample dimension {flat.shape[1]} != bank dimension {bank.dim}")
    flat = flat.to(PHASE_DTYPE)
    frequencies = bank.frequencies.to(
        device=flat.device, dtype=PHASE_DTYPE, non_blocking=True)
    return flat @ frequencies.T


def feature_sums(bank: SpectralBank,
                 samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unnormalized cos/sin feature sums, each ``[L]``."""
    theta = phases(bank, samples)
    return torch.cos(theta).sum(dim=0), torch.sin(theta).sum(dim=0)


def band_weights(config: AnchorConfig, progress: float | None
                 ) -> torch.Tensor:
    """Reform R6: relative weight of each declared band at a training point.

    ``progress`` runs 0 -> 1 over training; ``None`` means "no schedule", the
    Phase-1 behaviour.  Bands are ordered coarse to fine by construction
    (decreasing ``length_multiplier``), band 0 is active from the start, and
    each subsequent band ramps in over its own slice of the warmup window.
    Weights never fall below ``schedule_floor``, so no band is ever fully
    switched off and the declared population target is only reweighted, not
    truncated.
    """
    count = len(config.bands)
    if progress is None or config.band_schedule == "fixed":
        return torch.ones(count, dtype=PHASE_DTYPE)
    if config.band_schedule != "coarse_to_fine":
        raise ValueError(f"unknown band schedule {config.band_schedule!r}")
    if not 0.0 < config.schedule_warmup <= 1.0:
        raise ValueError("schedule warmup must lie in (0, 1]")
    if not 0.0 <= config.schedule_floor <= 1.0:
        raise ValueError("schedule floor must lie in [0, 1]")
    clipped = min(max(float(progress), 0.0), 1.0)
    out = torch.ones(count, dtype=PHASE_DTYPE)
    if count == 1:
        return out
    # The coarsest band carries full weight from the first step -- it is the
    # only one whose gradient is informative at initialization.  Band b >= 1
    # then ramps in over its own slice of the warmup window, so every band is
    # at full weight by ``schedule_warmup``.
    width = config.schedule_warmup / (count - 1)
    for index in range(1, count):
        ramp = (clipped - (index - 1) * width) / width
        active = min(max(ramp, 0.0), 1.0)
        out[index] = config.schedule_floor + (
            1.0 - config.schedule_floor) * active
    return out


def frequency_weights(bank: SpectralBank, progress: float | None
                      ) -> torch.Tensor:
    """Per-frequency schedule weights, normalized to mean one.

    Mean-one normalization keeps the scheduled loss on the same scale as the
    unscheduled one, so ``lambda_A`` does not silently change meaning as the
    schedule opens up.
    """
    per_band = band_weights(bank.config, progress)
    weights = per_band[bank.band_index]
    return weights / weights.mean().clamp_min(1e-30)


def biased_terms(bank: SpectralBank, generated: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
    """Per-frequency V-statistic terms, ``[L]``, each ``>= 0``."""
    n, m = len(generated), len(target)
    if n == 0 or m == 0:
        raise ValueError("both sample sets must be nonempty")
    target = target.to(generated.device)
    gc, gs = feature_sums(bank, generated)
    tc, ts = feature_sums(bank, target)
    return (gc / n - tc / m) ** 2 + (gs / n - ts / m) ** 2


def unbiased_terms(bank: SpectralBank, generated: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
    """Per-frequency U-statistic terms, ``[L]``.

    Uses ``||z_w(x)||^2 = 1`` exactly, so the diagonal correction is exact.
    Individual terms may be negative; this is a diagnostic estimator.
    """
    n, m = len(generated), len(target)
    if n < 2 or m < 2:
        raise ValueError("the unbiased estimator needs >= 2 samples per side")
    target = target.to(generated.device)
    gc, gs = feature_sums(bank, generated)
    tc, ts = feature_sums(bank, target)
    gg = (gc ** 2 + gs ** 2 - n) / (n * (n - 1))
    tt = (tc ** 2 + ts ** 2 - m) / (m * (m - 1))
    gt = (gc * tc + gs * ts) / (n * m)
    return gg + tt - 2 * gt


def anchor_loss(bank: SpectralBank, generated: torch.Tensor,
                target: torch.Tensor, estimator: str = "biased",
                progress: float | None = None) -> torch.Tensor:
    """Scalar anchor discrepancy, differentiable in ``generated``.

    ``progress`` applies the R6 coarse-to-fine band schedule.  Leave it
    ``None`` for audit and reporting, so that reported detection power is
    always measured at the full declared band width.
    """
    if estimator == "biased":
        terms = biased_terms(bank, generated, target)
    elif estimator == "unbiased":
        terms = unbiased_terms(bank, generated, target)
    else:
        raise ValueError(f"unknown estimator {estimator!r}")
    if progress is not None:
        weights = frequency_weights(bank, progress).to(
            device=terms.device, dtype=terms.dtype, non_blocking=True)
        terms = terms * weights
    return terms.mean()


def anchor_loss_by_band(bank: SpectralBank, generated: torch.Tensor,
                        target: torch.Tensor,
                        estimator: str = "biased") -> dict[str, float]:
    terms = (biased_terms(bank, generated, target) if estimator == "biased"
             else unbiased_terms(bank, generated, target))
    out: dict[str, float] = {}
    for band_id, name in enumerate(bank.band_names):
        mask = (bank.band_index == band_id).to(device=terms.device)
        out[name] = float(terms[mask].mean()) if bool(mask.any()) else 0.0
    return out


def anchor_gradient(bank: SpectralBank, generated: torch.Tensor,
                    target: torch.Tensor,
                    progress: float | None = None) -> torch.Tensor:
    """Analytic gradient of the biased anchor loss w.r.t. ``generated``.

    Shape matches ``generated``.  Derivation:
    ``dL/dy_i = (2 / (L n)) sum_l [-dc_l sin<w_l,y_i> + ds_l cos<w_l,y_i>] w_l``
    with ``dc, ds`` the cos/sin mean differences.  Cross-checked against
    autograd and central finite differences in the unit tests.
    """
    n, m = len(generated), len(target)
    target = target.to(generated.device)
    theta = phases(bank, generated)
    gc, gs = torch.cos(theta), torch.sin(theta)
    tc, ts = feature_sums(bank, target)
    dc = gc.sum(dim=0) / n - tc / m                        # [L]
    ds = gs.sum(dim=0) / n - ts / m
    if progress is not None:
        schedule = frequency_weights(bank, progress).to(
            device=theta.device, dtype=theta.dtype, non_blocking=True)
        dc, ds = dc * schedule, ds * schedule
    coeff = (-dc.unsqueeze(0) * gs + ds.unsqueeze(0) * gc)  # [n, L]
    frequencies = bank.frequencies.to(
        device=coeff.device, dtype=coeff.dtype, non_blocking=True)
    grad = (2.0 / (bank.size * n)) * (coeff @ frequencies)
    return grad.reshape(generated.shape).to(generated.dtype)


def anchor_field(bank: SpectralBank, generated: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
    """Descent direction for the anchor, i.e. minus its gradient."""
    return -anchor_gradient(bank, generated, target)


def anchor_diagnostics(bank: SpectralBank, audit: SpectralBank,
                       generated: torch.Tensor,
                       target: torch.Tensor) -> dict:
    """Plan section 10.2 anchor health."""
    with torch.no_grad():
        biased = float(anchor_loss(bank, generated, target, "biased"))
        unbiased = float(anchor_loss(bank, generated, target, "unbiased"))
        audit_biased = float(anchor_loss(audit, generated, target, "biased"))
        audit_unbiased = float(
            anchor_loss(audit, generated, target, "unbiased"))
        bands = anchor_loss_by_band(bank, generated, target, "unbiased")
        grad = anchor_gradient(bank, generated, target)
    return {
        "anchor_biased": biased,
        "anchor_unbiased": unbiased,
        "anchor_audit_biased": audit_biased,
        "anchor_audit_unbiased": audit_unbiased,
        "anchor_train_minus_audit": biased - audit_biased,
        "anchor_band_unbiased": bands,
        "anchor_grad_norm": float(grad.norm()),
        "anchor_refreshes": bank.refreshes,
    }
