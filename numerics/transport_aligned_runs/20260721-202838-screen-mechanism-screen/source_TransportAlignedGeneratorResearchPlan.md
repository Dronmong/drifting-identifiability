# Transport-aligned generator research plan

## Decision

The next program will not add another gain, guard, bandwidth, or suffix to the
current two-layer `TanhMLP`.  It will test whether the remaining error comes
from repeatedly reconstructing a one-dimensional transport coupling on fresh
minibatches and then asking a non-monotone shared network to remember it.

The primary candidate is **Persistent Quantile Transport (PQT)**: a monotone
piecewise-linear generator from a scalar uniform latent variable whose knot
values are updated by globally ordered target samples.  The coupling is part
of the generator coordinate system, so ranks do not switch between updates.
The paper Laplace field remains an optional, separately attributed finisher.

This is an architecture-level experiment, not a claim about dimensions above
one or ImageNet.  If it wins, the correct result is that transport alignment
materially improves the repository's one-dimensional learned-generator
benchmark.  A higher-dimensional analogue would require a separate convex
potential or minibatch-OT program.

## What the previous programs actually established

The following table separates the mechanism from the headline ratio.  Ratios
below one are favorable; comparisons are the strongest appropriately tuned
paper baseline unless noted.

| program | strongest result | what caused it | limiting evidence |
|---|---:|---|---|
| geometry/bandwidth/mask | `1.031` aggregate | large conditional wins on curved or missing-mode targets | a tuned fixed bandwidth absorbs the static rule; triggers misfire |
| NCJ particles | `.100` | remove off-support attenuation and self-coupling | learned MLP transfer is `1.072`; Adam already solves the particle freeze and uses `P*Q` as reliability weighting |
| QLD-v1 | `.9105` on its frozen registry; `.8359` on the LB-QCD registry | exact one-dimensional monotone mass assignment for 70% of training | unequal weights, rare intervals, and far starts remain weak |
| gated LB-QCD | **`.8218`** frozen, CI `[.7649,.8915]` | QLD plus a 1024-sample rank update only for under-resolved separated regions | the router adds only about 1.7% over QLD on the frozen registry, costs `6.59x` generator examples, and remains poor from far initialization |
| OASQD | `1.0031` vs QLD | lower-variance stratified rank-gradient estimator | estimator improvement does not improve the endpoint; controller mostly inactive |
| conservative/sharp fields | transient gains only | sharper local correction | the paper suffix erases the best transient checkpoint |
| calibrated bridge | `.9842` vs QLD on ED2 | retains some transient gain | no matching SW1 gain and no robust endpoint improvement |
| QGD / safety guard | about `1.000` vs LB-QCD | constrains the paper suffix against quantile regression | persistent pressure is unstable; conservative safety is almost never active |
| mode recovery | no resolution beyond roughly `.25-.34` in hard regimes | coarse fields improve reach | bandwidth, particles, and steps do not make a shared generator split and concentrate |

Three facts dominate this history.

1. **Global ordered assignment is the only broadly successful new training
   signal.**  QLD supplies most of LB-QCD's confirmed gain.
2. **Improving the estimator after assignment is not enough.**  OASQD cuts
   gradient variance without improving the endpoint, and LB-QCD's expensive
   virtual batch adds only a small increment.
3. **The current generator is misaligned with the one-dimensional problem.**
   It maps a fresh two-dimensional Gaussian latent batch through a non-monotone
   MLP.  Every QLD update sorts current outputs and creates a new empirical
   pairing.  The optimizer must simultaneously learn the target distribution,
   a stable latent ordering, and a fission map.  None of the suffix programs
   removed that burden.

This makes persistent transport coordinates the highest-value untested
mechanism.  It also explains why another field interpolation is low priority:
the last several programs changed the requested output displacement but not
the map asked to realize it.

## Literature cross-check

The architecture is standard enough to be credible but its use here is a new
experimental intervention rather than a theorem imported into the project.

- [Neural Spline Flows](https://proceedings.neurips.cc/paper_files/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html)
  establishes monotone splines as flexible invertible generative transforms.
- [Unconstrained Monotonic Neural Networks](https://papers.nips.cc/paper_files/paper/2019/hash/2a084e55c87b1ebcdaad1f62fdbbac8e-Abstract.html)
  constructs monotone invertible transforms by enforcing a positive
  derivative.
- [Optimal transport mapping via input-convex neural networks](https://proceedings.mlr.press/v119/makkuva20a.html)
  motivates the later higher-dimensional extension: gradients of convex
  potentials parameterize transport maps.
- [Multisample Flow Matching](https://proceedings.mlr.press/v202/pooladian23a.html)
  shows why coupling quality matters to optimization: minibatch couplings can
  reduce gradient variance and straighten probability paths.
- [Improving minibatch optimal transport via partial transport](https://proceedings.mlr.press/v162/nguyen22e.html)
  is a warning against treating arbitrary minibatch OT couplings as the
  population map.  PQT therefore uses a persistent scalar order coordinate and
  reports its finite-sample target-table bias explicitly.

## Candidate A: Persistent Quantile Transport

Let `u ~ Uniform(0,1)`.  Store an ordered grid

```text
0 < u_1 < ... < u_K < 1
q_1 <= ... <= q_K.
```

The generator is the monotone piecewise-linear interpolation
`G(u) = interp(u; u_k, q_k)`.  For an iid target batch, sort the observations
and form an empirical quantile vector `q_hat` on the same probability grid.
Update

```text
q <- (1-alpha_t) q + alpha_t q_hat.
```

Convex combinations of ordered vectors remain ordered, so monotonicity is an
invariant rather than a penalty.  The target order is now attached to a fixed
latent coordinate.  There is no output sort, no label switching, no
non-convex backpropagation, and no need for the generator to discover a
one-dimensional topology already known analytically.

Two variants must be kept separate:

- `pqt-B128`: same target samples per update as paper/QLD;
- `gated-pqt-M1024`: use 1024 target samples only when the existing audited
  LB-QCD resolution diagnostic fires, otherwise use 128.

The first tests architecture alignment at matched target-sample count.  The
second tests maximum effectiveness against the repository's best method with
the same routing and target-sample policy as LB-QCD.  Neither is allowed to
reuse evaluation samples.

## Candidate B: transport-aligned Laplace refinement

Only if pure PQT leaves a consistent local-shape error, add a short paper-field
suffix.  For sampled `u`, backpropagate the paper displacement through the two
adjacent spline knots, sum the knot gradients, take one Adam update, and
project back to the monotone cone.  This arm is secondary because every prior
suffix program showed that extra local refinement can erase a better global
state.

## Attribution ladder

Run the following arms on the mutable LB-QCD development registry:

1. existing selected paper (`tau=.5`);
2. existing QLD-v1;
3. existing gated LB-QCD;
4. `pqt-B128`;
5. `gated-pqt-M1024`;
6. optionally `gated-pqt-M1024 + 10% paper suffix` after pure PQT is measured.

The comparison must report both paper-relative and LB-QCD-relative ratios.  A
PQT win over paper alone is insufficient because LB-QCD is the active best
implementation.

## Development gates

The primary metric remains the target-balanced geometric mean of cell-median
ED2.  The first candidate advances only if all are true:

- `candidate / gated-LBQCD ED2 <= .90` on the development registry;
- `candidate / selected-paper ED2 <= .75`;
- SW1 improves against gated LB-QCD;
- every predefined target family is at most `1.05` relative to gated LB-QCD;
- both missing and concentrated initializations improve;
- no extra divergences;
- the matched `pqt-B128` arm is reported even if the gated arm wins;
- target samples, generator evaluations, kernel pairs, sort work, memory, and
  wall time are all reported.

The `.90` incremental gate is deliberately stronger than the 1.7% router gain.
It prevents a cosmetic improvement from being promoted as an architectural
breakthrough.

## Falsification diagnostics

Before interpreting a favorable endpoint, check:

1. **Finite-table bias:** repeat with `K` in `{128,256,1024}`.  A result that
   exists only at 1024 knots is a resolution result, not a general architecture
   result.
2. **Sample ledger:** compare `pqt-B128` with QLD at equal target samples and
   compare gated PQT with LB-QCD under the same routing policy.
3. **Tail stress:** report heavy-tail and contaminated targets separately;
   endpoint clamping can create an artificially good central metric while
   dropping tails.
4. **Fresh latent evaluation:** generate all metric samples from untouched
   uniform latents; never evaluate stored target observations or training
   quantile batches.
5. **Initialization stress:** include `far` as a diagnostic even though it is
   not part of the current primary claim.  PQT should remove the translation
   failure if the diagnosis is correct.
6. **Metric triangulation:** ED2, SW1, empirical W2/quantile RMSE, weighted mode
   reach, and mass error must tell a compatible story.

## Promotion protocol

Development may use the existing mutable LB-QCD registry.  If a candidate
passes the development gates:

1. freeze code, hyperparameters, work ledger, and primary gate;
2. generate a new target registry not used by QLD or LB-QCD development;
3. rerun selected paper, paper bandwidth oracle, QLD, LB-QCD, and PQT from
   paired seeds;
4. bootstrap targets hierarchically with seed resampling within each
   target/initialization cell;
5. claim only the dimensionality, architecture, target families, and
   initialization regimes actually tested.

## Stop rules

- If `pqt-B128` does not beat QLD, the persistent-coordinate hypothesis is
  rejected; do not rescue it with larger tables.
- If only gated PQT wins, classify the mechanism as accumulated target
  resolution rather than generator alignment.
- If ED2 improves but SW1 or tail diagnostics worsen, repair boundary/tail
  parameterization before confirmation.
- If PQT succeeds only because it is a nonparametric empirical-CDF estimator,
  state that plainly.  The follow-up must compress it into a learned monotone
  spline or UMNN before making a transferable model-design claim.

## Immediate implementation order

1. implement the monotone quantile-map core with invariants for ordering,
   interpolation, untouched-latent evaluation, and exact work accounting;
2. add a development runner that consumes the existing LB-QCD registry but
   writes to a new artifact root;
3. run a small mechanism screen for `B128` and gated `M1024`;
4. inspect table bias and tail errors before tuning any learning-rate schedule;
5. only then decide whether the paper suffix or a learned monotone spline is
   necessary.

