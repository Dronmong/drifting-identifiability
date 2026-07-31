# Encoder-Independent Kernel Drifting — Phase 9 protocol

## The solvable case: where does the deficit first appear?

*Frozen pre-outcome design. Predictions and thresholds fixed before any
Phase-9 run. Source: `EncoderIndependentPhase8Followup.md` §6. Results go to
`EncoderIndependentPhase9Results.md`.*

---

## 1. Why this phase exists

Nine phases of intervention-testing have produced eleven refuted mechanism
hypotheses. The generator's second-moment deficit is invariant to the
optimizer, learning rate, η, kernel bandwidth, field cloud size, latent
dimension, model capacity (36×), teacher-fitting quality (2.3×), training
budget (10×) and field noise (16×). Only changing what the teacher asks for
moves it.

The marginal value of a twelfth intervention is low. What has worked twice in
this program is **shrinking the problem until it is solvable** — R23 found the
second moment by dropping to 8-dimensional Gaussian mixtures, and spectral
confinement only became askable once posed as a subspace constraint.

## 2. The derivation this phase tests

For a linear generator `f(z) = Az + b` with `z ~ N(0, I)`, the stop-gradient
regression's fixed point `E_z[(∂f/∂θ)ᵀ V] = 0` reads

```
E[V] = 0            (from b)
E[V zᵀ] = 0         (from A)
```

For Gaussian data `N(0, s²I)` and a Gaussian kernel of bandwidth `τ`, the
mean-shift field between two centred Gaussians is exactly radial:

```
V(x) = c(σ)·x ,   c(σ) = τ² [ 1/(σ²+τ²) − 1/(s²+τ²) ]
```

so `E[V zᵀ] = c(σ)·A·E[zzᵀ] = c(σ)·A`. With a non-degenerate `A` this
vanishes **only when `c(σ) = 0`, i.e. `σ = s` exactly.**

> **Prediction P-lin: the linear generator on Gaussian data has NO
> second-moment deficit.** It should land inside `[0.7, 1.3]`, and the
> derivation says close to 1.0.

## 3. Phase 9A — the solvable case

Linear generator, Gaussian data, `d = 64`, latent `k = 64`, the **real
recipe** otherwise: the audited `lowdim_drift.drift_paper` bi-softmax field,
finite batches, RMS normalization, Adam, stop-gradient regression. Isotropic
and anisotropic (CIFAR-like spectrum) covariance. 3 seeds.

**Declared outcomes:**

- **9A lands in band** → the deficit *requires* something beyond the linear
  Gaussian case, and §4's ladder localizes it. This is the derivation's
  prediction.
- **9A shows the deficit** → the mechanism lives in the regression structure
  itself and is now analytically accessible, since every object in it is
  closed-form. This is the outcome that would finally produce an explanation
  rather than a twelfth elimination.
- **9A oscillates rather than settling** → the RMS normalization's sign flip
  at `σ = s` dominates; report the amplitude, since the normalized field does
  not vanish at the fixed point even though the raw one does.

## 4. Phase 9B — the ladder

Cross **model class** × **data law**, everything else held fixed:

| | Gaussian data | CIFAR-16 data |
|---|---|---|
| **linear generator** | 9A | ✔ |
| **MLP generator** | ✔ | ✔ *(closest to the real recipe)* |

Four cells, 3 seeds each. **The first cell that shows the deficit localizes
it** — to the nonlinearity, to the data's non-Gaussianity, or to their
interaction. This is the whole point of the phase and it is reported as a
2×2, not as a single number.

**Declared reading:**

- deficit only with the MLP → **the nonlinearity** is required;
- deficit only with CIFAR → **the data law** is required;
- deficit in both single-factor cells → two independent contributions;
- deficit in no cell but present in the real conv recipe → the
  **architecture** (conv stack, GroupNorm, upsampling) is implicated, which
  would be a sharp and previously unconsidered finding.

## 5. Phase 9C — the analytic check

For the isotropic Gaussian cell, compare the **measured** equilibrium against
the closed-form `c(σ) = 0` root, and report the measured `c(σ)` at the
observed equilibrium. A large residual `|c(σ)|` at equilibrium means the
finite-batch bi-softmax field is not the population mean-shift field, which
would be worth knowing independently of the deficit.

Reported, not gated.

## 6. Declared failure branches

- **The ladder shows no deficit anywhere** → the reduction has removed the
  phenomenon and the phase is a null result about the reduction, not about
  the recipe. Say so; do not present a clean 2×2 as if it explained CIFAR.
- **The linear cell diverges or degenerates** (`A → 0`, rank collapse) →
  report the degeneracy with its numbers rather than scoring it.
- **No tuning.** Bandwidth is set by the same target-ESS rule at 0.9 (the
  Phase-7C optimum) in every cell.

## 7. What Phase 9 cannot conclude

- Nothing about ImageNet, FID, or the paper's trained model.
- A Gaussian-data result is about Gaussian data; CIFAR cells are the bridge.
- The conv architecture is *not* in the 2×2 — it is the reference point the
  ladder is compared against, using the already-recorded Phase-8 value.
- The anchor stays disabled and the geometry thread stays closed.
