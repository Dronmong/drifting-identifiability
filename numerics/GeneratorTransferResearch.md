# Making the particle→generator transition work: a research pass

Status: research spec, 2026-07-20. Motivated by the E5 failure in
`IdentifiabilityImprovementResults.md`. This document diagnoses *why* the NCJ
particle win did not transfer and reframes the objective accordingly. It does
not itself run the gated experiment; it fixes the hypotheses, the candidate
family, the fresh-split protocol, and the certified-theory target so the next
leg can be executed under discipline.

## 0. Executive summary

The NCJ particle win (E4, ratio 0.100) came almost entirely from the **no-freeze
mechanism**: dropping the identifiability-inert `P*Q` gain lets off-support
particles move at full speed instead of stalling where the affinity mass is
exponentially small (certified by T4). This is a property of the **particle
optimizer** — a constant-step Euler update `q ← q + η V`.

The learned generator is trained with **Adam**. Adam normalizes each parameter's
step by the running gradient RMS, so it **independently supplies the no-freeze
property**: a uniformly attenuated gradient still produces a full-sized Adam
step. Therefore the mechanism that won E4 is **redundant** in the generator, and
its only side effect — discarding `P*Q`'s role as a **per-sample reliability
weight** on the shared gradient — is pure cost.

Three independent pieces of evidence establish this (§2). The consequence (§3):
**the surviving lever in the Adam-trained generator is not field magnitude at
all — it is the relative per-sample weighting of the parameter-space gradient
direction.** The paper already applies a (crude) reliability weight `P*Q`;
constant gain applies the worst possible weight (uniform) for heteroscedastic
per-sample noise. The only route to a genuine generator win is a **better
reliability weight than `P*Q`** — with inverse-variance / ESS weighting as the
principled candidate — not the magnitude interventions that won E4.

## 1. What the generator update actually is

`run_identifiability_generator.py` minimizes, with the field `V` stop-gradiented,

```text
L(theta) = -(1/B) * sum_i < G_theta(z_i), V_i >,
grad_theta L = -(1/B) * sum_i V_i^T (dG(z_i)/dtheta),
theta <- Adam(theta, grad_theta L).
```

With the paper gain `V_i = (P_i Q_i) * Delta_i`, the parameter gradient is a
**`P*Q`-weighted average of per-sample directions** `Delta_i^T dG_i`. With
constant gain it is an **unweighted average**. Two facts follow:

1. **Adam is approximately scale-free.** Multiplying every `V_i` by one constant
   leaves `mhat/sqrt(vhat)` unchanged (up to `eps` and the bias-correction
   transient). So the *global magnitude* of the field — the entire currency of
   the no-freeze mechanism — is largely divided out. Retuning a single global
   step will not recover the win.
2. **What Adam does *not* divide out is the relative weighting across samples.**
   The direction of `grad_theta L` in parameter space is set by the weights
   `{P_i Q_i}`. That relative weighting is the only part of the gain that can
   change the trained generator.

## 2. Evidence that Adam neutralizes the E4 mechanism

**(a) E5 arm decomposition inverts E4.** Particle ED2 ratios vs paper become
generator ratios:

| Arm | E4 particles | E5 generator |
|---|---:|---:|
| normalized-only | 0.31 | 1.028 |
| crossfit-only | 0.80 | 1.016 |
| normalized-crossfit | 0.088 | 1.072 |

The ~3× advantage from dropping `P*Q` vanishes; the modifications become a small
uniform penalty, worst when combined.

**(b) Paper no longer freezes off-support.** Median final ED2 by init in E5:
paper on `far` = 0.0111, barely above `broad` = 0.0079. In E4 the paper field
froze on `far` starts (NCJ/paper cell ratios ~1e-3). The E5 penalty is *uniform*
across inits (~1.03–1.05), not far-concentrated — the signature of a lost
reliability weight, not a lost no-freeze mechanism.

**(c) Swapping Adam→SGD restores the E4 pattern exactly.**
`generator_optimizer_diagnostic.py` retrains the same MLP with plain SGD:

```text
target                init     paper   norm-x   ratio
NCJG-gaussian-1d      broad   0.0007   0.0006   0.841
NCJG-gaussian-1d      far     9.8767   0.0044   0.000   <- paper FREEZES
NCJG-gmix-2d-K5-uneq  far    10.5973   0.0137   0.001   <- paper FREEZES
NCJG-moons-127        far    11.1078   0.0124   0.001   <- paper FREEZES
```

Under a non-adaptive optimizer the paper field freezes off-support and constant
gain wins ~1000× on `far` — the E4 particle result. The optimizer, not the
field, is the whole difference. This is a strong falsifiable prediction borne
out: the no-freeze advantage is real but *optimizer-conditional*.

## 3. The reframed objective

> The generator gradient is a weighted average of noisy per-sample directions
> `Delta_i^T dG_i`. Adam fixes the global scale; the surviving design freedom is
> the **relative reliability weight** `w_i` on each sample. Construct a weight
> that beats the paper's `P*Q` — provably positive (so T1 identifiability is
> retained) and reducing gradient-direction variance — under a pre-registered
> fresh-split generator gate.

This is a genuinely different target from the NCJ program. It is **not** "drop
the gain," "add noise," or "cross-fit." Those were magnitude/self-pair
interventions matched to constant-step particle dynamics. The generator target
is a **variance-optimal reweighting of a shared stochastic gradient**.

### Why inverse-variance is the principled weight

Each `Delta_i` is a difference of two self-normalized minibatch centroids, so it
is a noisy estimate of the true local direction with heteroscedastic variance
`sigma_i^2` (governed by effective sample size). The minimum-variance unbiased
linear combination of independent noisy directions weights each by `1/sigma_i^2`
(Gauss–Markov / BLUE). Paper's `P*Q` is a heuristic proxy (affinity mass
correlates with, but is not, reliability); constant gain (`w_i≡1`) is the
uniform weight, worst under heteroscedasticity. An explicit reliability weight
— e.g. ESS-based `w_i = ESSpos_i * ESSneg_i` (or a jackknife variance proxy) —
is the candidate that could beat both. This is exactly the `reliability_i` the
NCJ plan (§2.2) named and deferred.

## 4. Candidate families (fresh split, ablatable)

All candidates keep a strictly positive gain, so T1's zero-set preservation and
the T2 jitter guarantee still hold verbatim; none reintroduces an axiom.

* **H1 — tempered gain.** `w_i = (P_i Q_i)^gamma`, `gamma in {0, 0.25, 0.5,
  0.75, 1}`. Spans constant (γ=0, lost E5) to paper (γ=1). Cheapest one-knob
  test of whether *any* interior reliability strength beats paper under Adam.
* **H2 — ESS / inverse-variance reliability gain.**
  `w_i = clip((ESSpos_i * ESSneg_i)^rho, w_min, w_max)`, and a jackknife
  variance proxy variant `w_i = 1/(eps + hat_var(Delta_i))`. The principled §3
  candidate; the diagnostics `ESSpos/ESSneg` are already logged by the field
  library.
* **H3 — loss-level confidence decoupling.** Replace magnitude-in-field by
  `L = -sum_i w_i < G(z_i), Delta_hat_i >` with `Delta_hat_i` the *unit*
  direction and `w_i` the reliability. Separates "which way" from "how much to
  trust," and interacts more transparently with Adam than folding both into one
  vector.
* **H4 — variance-reduced reference (only if a reliability gain shows promise).**
  Cross-fit hurt the generator (adds independent-negative variance to `Delta`);
  the generator wants *low*-variance directions. Options: keep the paper
  reused/masked reference (E4 showed cross-fit's own benefit was small), or
  Rao–Blackwellize the negative centroid over several reference draws.

Adaptive-optimizer-aware preconditioning is explicitly out of scope for v1: it
changes the paper's training semantics and confounds the reliability question.

## 5. Pre-registered experiment

* **Fresh registries.** New validation (≥12 cfgs) and test (≥16 cfgs) generator
  targets, disjoint from the E5 `identifiability_generator_registry.json` and
  from all particle registries; new master seeds. Same fresh-split discipline
  that made E4/E5 trustworthy.
* **Exact paper semantics preserved.** Same TanhMLP, Adam (`lr, betas, eps`),
  latent/init/optimizer/data streams paired across arms, stop-gradient target.
  The *only* thing an arm changes is `w_i`. A compute-matched paper baseline
  receives any extra reference forwards H4 uses.
* **Arms.** paper; paper-matched; H1 at each γ; the selected H2 weight; H3;
  (H4 only if H1/H2 validate). Validation selects one weight; the frozen test
  runs paper, paper-matched, and the frozen candidate only.
* **Gate (unchanged philosophy from E4/E5, never weakened post hoc).**
  Hierarchical bootstrap over target cells then seeds; the candidate passes only
  if geo-mean ED2 ratio ≤ 0.90 **and** hierarchical CI upper < 1 **and** both
  Gaussian/non-Gaussian subgroup CI uppers < 1 **and** ≥60% winning cells
  **and** it also beats the compute-matched paper. (The 0.90 threshold, looser
  than E4's 0.80, reflects that the generator headroom over an
  already-reliability-weighted paper baseline is expected to be modest; it is
  fixed here, before any test data is seen.)

## 6. Certified-theory target (T6)

The reframing yields a clean, honest theorem target that extends T1:

> **T6 (variance-optimal reweighting).** Among positive per-sample weights, the
> inverse-variance weight `w_i ∝ 1/Var(Delta_i)` minimizes the variance of the
> weighted-average gradient direction, at fixed expected direction. State it as
> a finite Gauss–Markov / weighted-least-squares optimality over the strict-pair
> interaction frame, reusing `interactionFrameBound_of_uniformPositiveGain`
> (T1) to certify that every candidate weight preserves the zero set. Axiom-free.

T6 does not assert the weight wins empirically; it certifies that the *chosen
lever* is the statistically correct one, exactly as T1/T4 certified the levers
of the NCJ program. Empirical confirmation remains the job of §5.

## 7. Honest stop rules and allowed conclusions

| Outcome | Defensible conclusion |
|---|---|
| A reliability weight passes the fresh generator gate | A certified, generally-better drifting field for Adam-trained low-dimensional generators |
| Best weight only ties paper | **`P*Q` is a near-optimal reliability weight under Adam**; report the characterization + the optimizer-conditionality theorem, claim no win |
| Only some families/inits improve | Conditional design rule, not a general generator improvement |
| Nothing beats or ties paper | The drift-field magnitude/weight is not the generator bottleneck; redirect to target-design or optimizer-interaction questions |

The "ties paper" outcome is a real and likely result, and it is not a failure:
combined with the confirmed optimizer-conditionality mechanism (§2) and T6, it
is a complete, publishable story about *why* population-identifiability-inert
structure is or is not optimization-relevant. Nothing here supports an ImageNet
claim.

## 8. Immediate next action

Run `generator_optimizer_diagnostic.py` at higher replication to solidify the
SGD-vs-Adam mechanism figure (committed evidence), then freeze the fresh
validation/test generator registries and implement H1 (tempered gain) — the
cheapest discriminator of whether *any* interior reliability strength beats
paper under Adam. H2/T6 follow only if H1 shows interior headroom.
