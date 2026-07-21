# Quantile-to-Laplace Drifting: candidate and validation plan

**Status (2026-07-20): promising but conditional exploratory candidate, not a
confirmed paper improvement.** The algorithm below was selected after looking
at pilot results. Its current numbers therefore motivate a frozen test; they
are not a valid final claim by themselves. It is specifically a fission method
for concentrated/missing-mode starts, not a universal cure for arbitrary
initialization.

## 1. The problem isolated by the repository

The empirical ledger now rules out a large class of easy modifications.
Bandwidth tuning, step-size tuning, masking, scalar gain changes, jitter,
cross-fitting, finite Sinkhorn balancing, and longer runs can all help a chosen
regime, but none gives a robust learned-generator improvement over the tuned
paper method. Two facts recur:

1. Adam absorbs most changes that only rescale the local field.
2. From a concentrated or homogeneous initialization, a local radial field can
   move the whole generated cloud but often cannot assign different particles
   to different narrow target modes.

The missing operation is therefore not another local scale. It is a
**mass-balanced symmetry-breaking assignment**: different generated samples
must receive different destinations before the paper field performs local
refinement.

## 2. Candidate: Quantile-to-Laplace Drifting (QLD)

QLD is presently a one-dimensional algorithm. Let an equal-size minibatch be

```text
x_i = G_theta(z_i),       i = 1,...,B
y_j ~ p,                  j = 1,...,B.
```

### Phase I: quantile fission

Sort both minibatches. If `rank_x(i)` is the rank of `x_i`, use the stop-gradient
field

```text
V^Q_i = y_(rank_x(i)) - x_i.
```

This is the empirical one-dimensional quadratic optimal-transport coupling.
Unlike a local kernel average, it allocates exactly one target order statistic
to every generated order statistic. A collapsed or concentrated cloud thus
receives a spectrum of distinct destinations instead of one common mean
direction. Exact output ties are ordered by an infinitesimal latent-dependent
term solely to select a subgradient.

### Phase II: Laplace refinement

After a fixed fraction of the training horizon, switch completely to the
paper's Algorithm 2 field:

```text
V_i = V_paper(x_i; y_1,...,y_B, tau).
```

The frozen candidate emerging from the pilot is:

| item | value |
|---|---:|
| total updates | 1,200 |
| batch size | 128 |
| quantile phase | first 70% of updates |
| paper refinement | final 30% of updates |
| refinement bandwidth | `tau = 0.5` |
| optimizer/model | repository `TanhMLP` with the same Adam path as the paper arm |

This is a phase switch, not an additive blend. The clean switch makes the
mechanism attributable: global mass allocation first, local kernel sharpening
second.

## 3. Why this uses the formalization rather than ignoring it

The formal development says that the population Laplace drifting field has the
correct zero set under the proved hypotheses. QLD preserves that field as its
terminal refinement stage rather than replacing it with an unverified learned
force. The first phase is also distribution-directed: in one dimension the
rank coupling is the exact empirical `W2` coupling and its population metric
vanishes exactly when the laws agree.

The formal obligations for a later Lean layer are consequently concrete:

1. define the one-dimensional quantile field and its empirical transport
   energy;
2. prove its zero-energy law-equality statement under the relevant moment
   assumptions;
3. prove the rank displacement is a descent direction for empirical `W2^2`;
4. state the scheduled dynamics honestly--the two phases share the same target
   equilibrium, but a finite switch is not itself a convergence theorem.

No such theorem is being assumed in the experiment, and no new axiom is needed
to run it.

## 4. Exploratory evidence

Implementation: [`sliced_fission_probe.py`](sliced_fission_probe.py).

### 4.1 Negative result in two dimensions

A random-projection sliced-rank field was tested on the hard `K=32`, `d=2`
missing-mode target. The best tuned paper arm matched or beat sliced-only,
warm-start, and blended variants. In particular, the best paper resolution was
about `0.83`, while the warm-start resolution at the same favorable bandwidth
was about `0.47`. **The naive multidimensional extension is rejected.**

### 4.2 One-dimensional pilot and post-selection screen

In a `K=16` line-mixture pilot, sliced-only training reduced median ED-squared
from the best paper value of about `0.0259` to `0.0191`. A subsequent pilot
selected the 70% quantile / 30% Laplace schedule above.

The selected schedule was then rerun on four additional line-mixture targets.
For an intentionally difficult comparison, the paper arm received the best
bandwidth *separately on every target* from `{0.2, 0.5, 1, 2, 4}`.

| target | oracle paper ED2 | QLD ED2 | ratio | paper SW1 | QLD SW1 |
|---|---:|---:|---:|---:|---:|
| K8, spacing .75, sigma .08 | .0165 | .0125 | .76 | .1633 | .1279 |
| K12, spacing 1.2, sigma .04 | .0299 | .0182 | .61 | .3316 | .2784 |
| K20, spacing .8, sigma .06 | .0271 | .0106 | .39 | .2918 | .2010 |
| K24, spacing 1.5, sigma .03 | .0442 | .0519 | 1.17 | .9191 | .6796 |

The target-balanced geometric ED2 ratio was **0.678**, a 32.2% exploratory
improvement. QLD won ED2 on three of four targets and SW1 on all four. The loss
on the widest, sharpest `K=24` target is important: the method is promising,
not uniformly dominant.

These targets were examined after algorithm selection but the whole process
was not pre-registered, and six seeds are too few for a final uncertainty
statement. This table is a mechanism screen, not a publishable result.

### 4.3 Initialization stress test and hard boundary

After freezing the 70/30 schedule, it was tested without retuning on four new
targets and all four repository initialization regimes. The paper again
received an oracle bandwidth in every target/initialization cell.

| initialization | target-balanced geometric ED2 ratio |
|---|---:|
| broad | .818 |
| missing | .670 |
| concentrated | .907 |
| far | 2.494 |
| all four combined | 1.055 |

Thus the algorithm does **not** improve generally across arbitrary starts. It
is strongest on the missing-mode condition it was designed to address and is
moderately favorable on ordinary broad/concentrated starts, but it can fail
catastrophically when a high-mode-count generator begins far outside the
target. A residual-triggered adaptive switch and pure quantile training were
both tried on the two failed far cells; neither beat the tuned paper arm. This
failure is structural within the current update budget, not repaired by a
cosmetic schedule change.

## 5. Literature and novelty boundary

The components have strong precedents:

- one-dimensional rank matching is classical optimal transport;
- sliced-Wasserstein generative training predates drifting;
- recent work already proposes sliced-Wasserstein fields within the drifting
  framework;
- Sinkhorn-Drifting already develops globally balanced affinities;
- recent drifting variants also study friction schedules, divergence mixtures,
  and conservative KDE-gradient fields.

Accordingly, **do not claim that a quantile or sliced-Wasserstein drifting field
is new**. A potentially new contribution would have to be the particular
diagnosis and hybrid construction: use exact quantile transport only as a
finite fission stage, then switch to the paper's Laplace field whose
identifiability has been formally certified, together with a demonstrated
low-dimensional performance gain. A dedicated novelty review is still needed
before making even that claim.

Primary references checked for this decision:

- Deng et al., [*Generative Modeling via Drifting*](https://arxiv.org/abs/2602.04770)
- Gretton et al., [*On the Wasserstein Gradient Flow Interpretation of
  Drifting Models*](https://arxiv.org/abs/2605.05118)
- [*Sinkhorn-Drifting Generative Models*](https://arxiv.org/abs/2603.12366)
- [*Attraction, Repulsion, and Friction: DMF*](https://arxiv.org/abs/2604.18194)
- [*Finite-Particle Convergence Rates for Conservative and Non-Conservative
  Drifting Models*](https://arxiv.org/abs/2605.22795)
- Wu et al., [*Sliced Wasserstein Generative Models*](https://arxiv.org/abs/1706.02631)

The other literature-driven alternatives were not selected:

- finite Sinkhorn balancing was already implemented in this repository and
  did not fix homogeneous-swarm collapse;
- friction and scalar-gain schedules change amplitude, a lever Adam repeatedly
  neutralized here;
- an exploratory output-space natural/Gauss--Newton step looked strong against
  a weak bandwidth but lost once the paper bandwidth was tuned;
- a pure sliced field gives up the useful local refinement and was less robust
  than the phase-switched candidate.

## 6. Frozen confirmatory experiment

The next run must freeze QLD exactly as specified above. Do not tune the warm
fraction, refinement bandwidth, projection rule, update count, or optimizer on
the confirmatory targets. The primary claim is explicitly **fission under
concentrated or missing-mode initialization**. Broad and far starts remain
reported stress tests; they cannot be silently pooled away, and far-start
robustness is not claimed.

### 6.1 Target registry

Create a new, immutable test registry, disjoint from every target used above,
with at least 16 one-dimensional distributions spanning:

- equal-weight Gaussian mixtures with `K` from 6 to 32;
- unequal and highly imbalanced mixture weights;
- heterogeneous component widths;
- moderate overlap and sharply separated components;
- contaminated mixtures with a small remote component;
- connected unimodal and heavy-tailed controls where fission may be harmful;
- missing-mode and concentrated initialization regimes as the frozen primary
  fission benchmark;
- broad and far initialization as separately reported robustness diagnostics.

The target-level unit of analysis is the distribution/initialization pair, not
an individual random seed.

### 6.2 Baselines and fairness

Use the exact repository implementation of paper Algorithm 2. Select one paper
bandwidth policy on a separate validation registry, then freeze it. Also report
an oracle-per-target bandwidth as a deliberately advantaged diagnostic
baseline, but do not confuse it with a deployable method.

Use identical generator architecture, Adam settings, minibatches, target draws,
initial parameters, update count, and paired random seeds. Record wall time and
kernel/assignment evaluations; sorting adds `O(B log B)` work and must not be
hidden. Run at least 20 paired seeds per target.

### 6.3 Metrics and gate

Primary metric: final ED-squared, aggregated as the geometric mean of
candidate/paper median ratios across targets.

Proposed confirmatory gate:

1. target-balanced ED2 ratio at most `0.80`;
2. paired target-bootstrap 95% upper confidence bound below `1.0`;
3. QLD wins on at least 60% of primary target/initialization cells;
4. no predefined primary target family has a median ratio above `1.10`;
5. divergence/censoring is no worse than the paper arm.

Secondary outcomes: sliced W1, mode reach, mode resolution, target-mass error,
time to cover, and wall time. They explain a result but cannot rescue a failed
primary gate.

### 6.4 Decision rule

- **Pass:** report a low-dimensional ED2 improvement over the paper algorithm,
  explicitly scoped to the frozen one-dimensional benchmark family.
- **Mixed:** identify a measurable applicability rule using a new split; do not
  retrofit a trigger on the test registry.
- **Fail:** retain QLD as a documented negative and stop tuning it on the same
  family.

## 7. Route beyond one dimension

Do not promote the random-slice implementation merely because the 1-D result
works. It has already failed its first 2-D mechanism probe. If the frozen 1-D
gate passes, open a new development split comparing genuinely multidimensional
mass assignments:

1. adaptive or max-sliced projections chosen from a held-out batch;
2. a full entropic-OT coupling used only during the fission phase;
3. projection ensembles with an explicit diversity/orthogonality condition;
4. learned features only after a data-space benchmark, with a separate test
   for feature-map non-injectivity.

The multidimensional success criterion remains a paper-beating distributional
metric, not merely prettier trajectories or improved coverage at worse ED2.

## 8. Reproduction commands

```powershell
uv run --with numpy --with scipy python numerics/sliced_fission_probe.py
uv run --with numpy --with scipy python numerics/sliced_fission_probe.py --fresh
uv run --with numpy --with scipy python numerics/sliced_fission_probe.py --init-screen
uv run --with numpy --with scipy python numerics/sliced_fission_probe.py --repair-screen
```

The commands reproduce, respectively, the rejected 2-D mechanism probe, the
one-dimensional missing-mode table, the all-initialization stress test, and the
failed adaptive-switch repair attempt.
