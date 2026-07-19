# Collapse Atlas: interpretation and robustness plan

*Independent audit of the first E1--E6 pass, 2026-07-18. This document does
not replace `CollapseAtlas.md` or `CollapseAtlasResults.md`; it records how the
current evidence should be interpreted and how to strengthen it.*

## 1. Executive assessment

The first atlas pass identifies a coherent and useful dynamical picture:

1. The full-field converse and the particle dynamics ask different questions.
   The converse assumes `V(p,q)(x) = 0` at every probe `x`, whereas an atomic
   particle state is stationary when the field vanishes only at the particle
   locations.
2. Point collapses are therefore genuine support-restricted equilibria. If
   `q = delta_c` and `m_p(c) = 0`, then `m_q(c) = 0` and the sole particle does
   not move.
3. The tested point collapses are nevertheless unstable to fission. After the
   symmetric perturbation

   ```text
   delta_c  ->  (delta_(c+u) + delta_(c-u))/2,
   ```

   the separation has first-order generator `I + Dm_p(c)`. Its spectral
   abscissa was positive in every tested example.
4. The most convincing practical failure mechanism is not an attracting point
   collapse. It is a wrong allocation of fixed particle mass between distant
   clusters. Correcting that allocation requires particles to cross a region
   whose Laplace coupling is suppressed on the scale `exp(-L/tau)`, producing
   very long transients.
5. The correct atomic target is locally stable for sufficiently small update
   steps in the tested positional perturbation model, but the permissible step
   size depends on the target geometry and bandwidth.
6. None of the tested natural discrepancy functionals was monotone along every
   sampled discrete trajectory.

The strongest current research message is therefore:

> Exact collapse can exist without being attracting. In separated-cluster
> problems, metastable mass imbalance may be the practically important
> obstruction even when the only attracting equilibrium is the target.

This is a numerical hypothesis generator, not yet a proof of the global
classification of support-restricted equilibria.

## 2. Evidence ledger

The following language should be used consistently when discussing the first
pass.

| Item | Directly observed | Reasonable inference | Not established |
|---|---|---|---|
| E1 | The multistart solver found the reported zeros and classifications for the tested atomic targets. | Small bandwidth resolves multiple cluster-associated sinks; larger bandwidth merges them. | That every zero was found, or that the reported landscape is exhaustive. |
| E2 | Every tested sink had positive splitting index; the symmetric-pair finite difference agreed with the predicted first-order formula. | Point collapse is generically fission-unstable, while nearly degenerate examples may escape very slowly. | Strict positivity in all dimensions and for all nondegenerate measures. The existing one-dimensional result gives nonnegativity, not by itself every desired strict case. |
| E3 | Relaxation time rose sharply with `L/tau`; the long-run imbalanced state retained a small nonzero residual; local root and least-squares solvers did not find a zero. | Inter-cluster mass correction is exponentially metastable. | A globally positive residual floor, absence of an exact wrong-mass equilibrium, or a universal exponent near `1.5`. |
| E4 | Under the implemented 720 initializations, no endpoint met the collapse or metastability thresholds and 705 met the energy-distance convergence threshold. | Generic collapse basins are small in this tested family and parameter range. | A zero collapse basin, convergence of the remaining runs, or robustness outside the implemented target/initialization family. |
| E5 | The finite-difference update Jacobian at `q = p` had spectral radius below one for the reported small steps and above one for two large-step cases. | The truth is locally stable in a flow-like step-size regime and can be destabilized by Euler overshoot. | A universal boundary proportional only to `tau`, stability to changes of support/weights/particle number, or the claimed mask comparison. |
| E6 | Each tested diagnostic increased on at least one tracked interval of the discrete update. | None is an unconditional descent objective for this tested discrete map. | Nonexistence of a Lyapunov functional, or even failure of the same quantities for the exact continuous-time flow. |

### Confidence by conclusion

* **High:** the support-restricted/full-field distinction; existence of point
  collapse equilibria; correctness of the symmetric-splitting mechanism;
  severe slowing of mass transfer as clusters separate.
* **Moderate:** generic fission instability beyond one dimension; practical
  rarity of collapse from broad random initializations; a bandwidth-dependent
  step-size ceiling.
* **Preliminary:** no exact non-collapse spurious equilibria; the numerical
  value of the metastability exponent; failure of the candidate functionals
  as continuous-flow Lyapunov functions.

## 3. Limitations of the current implementation

### 3.1 E1 is a census, not an exhaustive zero classification

`find_zeros` launches a local root solver from 250 randomized seeds and from
the target atoms. This is a sensible exploratory census, but it cannot certify
that no additional zeros exist. The result summary should use “zeros found”
rather than “all zeros,” and “matches the observed mean-shift landscape” rather
than “is exactly the landscape.”

The Jacobian is also estimated at one finite-difference scale. Near a
degenerate zero, a classification can change with the step size or numerical
precision.

### 3.2 E2 provides strong falsification pressure, not a universal theorem

The adversarial search is valuable, but it is a 24-restart local optimization
of a nonsmooth spectral objective with a penalty for `m_p(c) = 0`. The zero is
subsequently polished, which improves the result, but the procedure neither
globally minimizes the splitting index nor certifies its sign.

The stability statement must also name its perturbation class. Coincident
particles remain coincident under exact deterministic arithmetic. “Unstable”
here means unstable after allowing positional fission perturbations, not that
the dynamics spontaneously creates diversity.

### 3.3 E3's least-squares floor is not a certificate

Re-running `atlas_e3b_check.py` reproduces a residual norm of approximately
`3.684e-6`, with the optimizer reporting gradient convergence. This establishes
that one local least-squares run stopped at a small nonzero residual. It does
not establish a globally positive lower bound on the residual. Gradient
convergence can occur at a local minimum, a badly conditioned point, or a
numerical plateau.

Accordingly, the correct description is “numerical evidence for a positive
residual floor.” The word “certified” should be reserved for an analytic bound,
an interval-arithmetic exclusion, or a formally checked theorem.

The relaxation statistic is also quantized: `alpha` is the fraction of
equal-weight particles on one side of the midpoint. It changes only when a
particle crosses that boundary. This is faithful to fixed-weight particle
transport, but a fitted exponent can depend on nucleation/crossing events and
finite particle number.

There is also a logging mismatch in the censored final sweep. The loop permits
120,000 steps, but the message says `>12000*eta`. At `tau = 0.85`, the actual
maximum simulated flow time is approximately `120000 * 0.085 = 10200`.

### 3.4 E4 implements only part of its written protocol

The first run uses:

* target family P1 only;
* 60 initializations per cell rather than 200;
* `N` in `{8, 32}`, not `{8, 32, 128}`;
* two absolute spreads, rather than the three `L`-relative spreads in the
  specification;
* an energy-distance threshold alone for “converged,” although the
  specification also requests a small stationarity residual.

The implementation computes the usual squared energy-distance expression
without applying a square root. This is a valid nonnegative discrepancy and
has the same zeros, but thresholds and units should explicitly say which
convention is used.

“Wandering” currently means only “not classified at the finite horizon.” Such
runs could be slowly converging, metastable above the chosen residual
threshold, periodic, or genuinely nonconvergent.

### 3.5 E5 does not yet run the promised self-mask comparison

The current field always includes particle self-interaction. The documented
`mask_self` repeat is absent. E5 also tests one target family, two bandwidths,
and positional perturbations at fixed weights and fixed particle count.

### 3.6 E6 mixes flow behavior with time-discretization behavior

The candidate quantities are inspected after explicit Euler steps with
`eta = 0.1*tau`, sampled every ten updates. An increase disproves monotonicity
for that discrete map at that step size, but it might be caused by integration
overshoot even if the continuous orbital derivative were nonpositive.

The experiment also tests only 12 one-dimensional P1 trajectories at two
bandwidths. It rules out the strongest empirical story for these candidates,
but not all possible Lyapunov constructions.

### 3.7 Reproducibility artifacts need separation

`CollapseAtlasResults.md` is append-only and already contains repeated E3
sections from different passes. A clean result should distinguish raw runs
from a generated summary and record the exact code commit, dependency versions,
configuration, and seed. The planned `atlas_figs` products are not present in
the first committed run.

## 4. Robustness program

### R0 — make every run auditable

Before expanding the numerical search:

1. Introduce an explicit experiment configuration containing target family,
   dimension, `L`, `tau`, `N`, step size, horizon, tolerances, seed, masking,
   and sampling mode.
2. Write each run to a new directory, for example:

   ```text
   numerics/atlas_runs/<run-id>/
     manifest.json
     e1_zeros.csv
     e2_splitting.csv
     e3_trajectories.npz
     e4_endpoints.csv
     e5_spectra.csv
     e6_witnesses.npz
     summary.md
   ```

3. Put the git commit, Python version, NumPy/SciPy versions, platform, wall
   time, and command line in `manifest.json`.
4. Generate `CollapseAtlasResults.md` from raw tables instead of appending
   directly to it.
5. Add basic invariant tests:
   * weights remain positive and sum to one;
   * `V(p,p)` is zero to numerical tolerance;
   * translation and joint scaling behave as expected;
   * duplicated atoms with split weights represent the same law;
   * the symmetric-pair linearization error decays at first order with the
     perturbation radius.
6. Use dimensionless tolerances scaled by `tau`, target diameter, and field
   magnitude rather than fixed absolute thresholds alone.

### R1 — strengthen E1 zero finding

#### One dimension

1. Evaluate `m_p` on an adaptive interval grid and bracket every sign change.
2. Add tangency detection using local minima of `|m_p|`, because an even-order
   zero need not change sign.
3. Polish bracketed roots with a bracketing solver rather than Newton alone.
4. Derive a finite outer radius outside which the field has a fixed inward
   sign, eliminating missed far-field zeros.
5. For theorem-quality experiments, use interval arithmetic on each remaining
   interval to prove either existence/uniqueness of one zero or exclusion of a
   zero.

#### Two and higher dimensions

1. Combine random multistart with structured grids, atom/cluster seeds,
   continuation in `tau`, and deflation after each discovered root.
2. Check stability under increasing the seed count and enlarging the search
   box.
3. Validate each root with several precisions and an interval-Newton or
   Krawczyk test where feasible.
4. Compare the finite-difference Jacobian across a geometric sequence of step
   sizes and against automatic or analytic differentiation away from atoms.
5. Record the smallest singular value of the Jacobian so that nearly
   degenerate roots are not given a brittle sink/source label.

This will not make arbitrary high-dimensional enumeration easy, but it turns
“we found these roots” into a measured-coverage statement and can provide
actual certificates in low-dimensional boxes.

### R2 — turn E2 into a serious adversarial test

1. Verify the splitting formula over perturbation radii from roughly
   `1e-3*tau` down to the roundoff-dominated regime and fit the expected
   first-order error slope.
2. Report both:
   * the continuous-time spectral abscissa of `I + Dm_p(c)`;
   * the discrete-time spectral radius of `I + eta*(I + Dm_p(c))`.
3. Record non-normal transient amplification and eigenvector conditioning,
   not just eigenvalue real parts.
4. Remove translation, rotation, and scale symmetries from the adversarial
   parameterization. Bound atom locations and enforce a minimum meaningful
   weight so the optimizer cannot win through numerical degeneracy.
5. Replace the penalty-only search by constrained optimization where possible,
   and cross-check it with global methods such as differential evolution or
   basin hopping followed by exact-zero polishing.
6. Run hundreds or thousands of independently seeded configurations in
   dimensions 2 and 3, varying atom count and weight concentration.
7. Re-evaluate every near-zero splitting index in higher precision and, for a
   compact candidate, attempt an interval proof of its sign.
8. Test perturbation classes beyond an equal symmetric pair: unequal masses,
   multiple fission directions, added low-weight particles, and perturbations
   that change particle number.

The nearby Lean objective should remain separate: prove the one-dimensional
nonnegative splitting index, then add explicit hypotheses for strictness. That
theorem would provide a certified anchor while the atlas searches for the true
higher-dimensional boundary.

### R3 — distinguish exponential metastability from solver artifacts

1. Sweep `L` and `tau` independently. The same value of `L/tau` obtained in
   different ways should give compatible scaled behavior if that ratio is the
   controlling variable.
2. Repeat across many target geometries, within-cluster variances, initial
   mass imbalances, particle counts, and seeds.
3. Store full trajectories: particle positions, per-particle velocities,
   cluster assignment, smooth cluster membership, residual norms, and crossing
   times.
4. Treat runs that do not halve within the horizon as right-censored data.
   Fit the relaxation law with uncertainty intervals rather than dropping the
   censored point from an ordinary linear regression.
5. Compare candidate laws such as

   ```text
   T = C exp(c L/tau),
   T = C (L/tau)^a exp(c L/tau),
   T = C exp(c L/tau) N^b.
   ```

   Use held-out parameter cells or information criteria rather than reporting
   one fitted slope.
6. Repeat with decreasing `eta` at fixed flow time to verify that the result
   converges to continuous-time behavior.
7. Implement the planned two-bandwidth field and measure whether a component
   with bandwidth comparable to `L` changes exponential relaxation into a
   substantially faster regime.

#### Exact-equilibrium search

The no-spurious-equilibrium question needs a separate protocol:

1. Quotient out obvious permutation symmetries and inspect the singular values
   of the field Jacobian at the stalled state.
2. Run root and least-squares solvers from many perturbations of the stalled
   state and from independently constructed wrong-mass states.
3. Continue putative solution branches as `L/tau` changes instead of solving
   each parameter point independently.
4. Use arbitrary precision for the best candidates and compare the residual
   with conditioning and roundoff estimates.
5. Whenever a reduced symmetric cluster ansatz is available, solve the reduced
   equations and use interval arithmetic to exclude a zero in a specified
   box.
6. Ultimately replace the numerical “floor” by an analytic lower bound or a
   formal support-restricted converse. Until then, report local solver evidence
   only.

### R4 — complete the basin experiment

1. Implement the written P1 and P2 matrix with at least `N` in `{8, 32, 128}`
   and spreads defined relative to `L`.
2. Use at least 200 independent starts per cell; for rare collapse events,
   adaptively increase the count and report binomial confidence intervals.
3. Define “converged” using both a law discrepancy and a small on-support
   residual. State whether energy distance or its square is used.
4. Separate endpoint classes:
   * target-accurate and stationary;
   * target-accurate but still moving;
   * wrong-mass metastable;
   * point/cluster collapsed;
   * periodic or recurrent;
   * unresolved at the time horizon.
5. Extend every unresolved run by a large factor and estimate velocity decay
   before calling it “still in transit.”
6. Record particle-count quantization limits: for some `N`, equal-weight
   particles cannot represent the target weights exactly.
7. Add deliberately collapse-biased initializations. Generic Gaussian starts
   estimate practical basin size, while local starts around a collapse test
   the fission-stability prediction directly.
8. Only after the deterministic population atlas is stable, repeat selected
   cells with minibatch noise and the paper estimator.

### R5 — map local stability at the truth correctly

1. Implement both population self-interaction and the paper-style self mask.
2. Check the Jacobian with multiple finite-difference scales and an analytic or
   automatic derivative where differentiability permits.
3. Compute the continuous-time generator first. Then predict and verify the
   explicit-Euler step-size boundary from its spectrum.
4. Locate `eta*(tau)` by bisection rather than a five-point grid and attach an
   uncertainty interval based on derivative/precision checks.
5. Repeat across target families, dimensions, separations, atom counts, and
   weight imbalance.
6. Distinguish stability at fixed particle labels and weights from stability
   in measure topology. Include split/merge perturbations and representations
   of the same law with duplicated atoms.
7. Track eigenvalues near one that arise from symmetries or poorly observed
   modes; do not conflate neutral directions with contraction.

### R6 — test Lyapunov candidates at the flow level

For each differentiable candidate `F`, evaluate its orbital derivative

```text
d/dt F(q_t) = grad F(q) . V(q)
```

directly at many states. This is more decisive than observing an increase over
ten Euler steps.

1. Derive or automatically differentiate the atomic formulas for MMD squared,
   the normalizer gap surrogate, and smooth companion defects.
2. For nonsmooth quantities such as max norms and energy distance at
   collisions, use directional derivatives or smooth approximations.
3. Save explicit witness configurations where the orbital derivative is
   strictly positive and verify them at higher precision.
4. Repeat the trajectory test while taking `eta -> 0`. If the increase
   disappears, it was a discretization effect; if it approaches a positive
   orbital derivative, it is evidence against a continuous-flow Lyapunov
   claim.
5. Test whether violations concentrate near saddle crossings, fission events,
   or inter-cluster transport, as the original hypothesis predicts.
6. Phrase the conclusion narrowly: a positive witness rules out that specific
   functional for that flow, not the existence of every possible Lyapunov
   functional.

## 5. Recommended implementation order

### Priority 0: repair experimental provenance

* Separate raw data from generated summaries.
* Add manifests and deterministic run identifiers.
* Correct the E3 horizon label.
* Replace “certifies” and exhaustive language with evidence-calibrated wording.
* Mark the E4 and E5 specification items that have not yet been run.

### Priority 1: validate the two central discoveries

1. Strengthen E2 with a perturbation-scale study, constrained/global
   adversarial search, and the one-dimensional Lean theorem.
2. Strengthen E3 with replicated/censored relaxation fits and the explicit
   two-bandwidth intervention.
3. Treat “no exact wrong-mass equilibrium” as an open theorem target, not an
   output of local least squares.

These are the highest-value tasks because they determine whether the emerging
story—fission-unstable collapse but exponential mass metastability—is real and
general.

### Priority 2: complete the promised atlas

* Run the full E4 parameter matrix with uncertainty intervals.
* Add the self-mask and continuous-generator analyses to E5.
* Produce the missing plots from stored raw data.

### Priority 3: convert failures into theorem witnesses

* For E1/E2, interval-certify selected low-dimensional roots and splitting
  signs.
* For E3, derive quantitative upper and lower relaxation bounds.
* For E6, extract high-precision states with positive continuous orbital
  derivative.

## 6. Acceptance standard for a second atlas pass

A robust second pass should satisfy all of the following:

1. Every headline number is generated from immutable raw data and a recorded
   configuration.
2. Re-running the documented command recreates the tables without appending
   duplicates.
3. Conclusions include replication uncertainty and distinguish censored runs.
4. Root/eigenvalue findings are stable across solver, precision, and
   finite-difference scales.
5. Continuous-time conclusions are separated from finite-step map behavior.
6. Population, self-masked, minibatch, and model-level claims are never mixed.
7. “Certified” is used only for analytic, interval-verified, or formally proved
   statements.
8. Each conjecture has a stored counterexample-search protocol and a precise
   theorem statement it would motivate.

Under that standard, the atlas can become more than an exploratory simulation:
it can serve as a reliable bridge from the completed converse theorem to new
results about the actual particle dynamics.
