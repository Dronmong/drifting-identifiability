# MeanFlow-Drifting: mechanism analysis and source assessment

*Review of `meanflowdrifting.md` (1 138 lines) against this repository's
measured results. Written 2026-08-02, after B0/B1/B2, B3, and B2.5 unit 500.
This is an analysis document, not a protocol. It authorizes nothing.*

---

## 1. What the document proposes, structurally

Stripped of notation, the architecture is: **a capable generator carries
generation, a distributional term corrects it, and a fixed invertible map
supplies geometry.** §21 of the source states this outright — *"It is a pure
Drifting Model: **false** while the pMF loss remains active."*

That is the B0/B1/B2 pattern with two substitutions: pixel MeanFlow for flow
matching (multi-step → one-step), and a balanced Sinkhorn barycentric velocity
for the Laplace field.

**This matters more than the document claims.** Our confirmations already
validate the architecture rather than hypothesize it:

| our result | what it establishes |
|---|---|
| B0 (3/3) | a prescribed bridge carries generation without an encoder |
| B1 (3/3) | a fixed source-space anchor reduces its discrepancy at no coverage cost |
| B2 (2/3) | a drift correction does the same on its own axis |
| B3 | drifting **cannot** carry generation alone at matched capacity (recall 0.0000 at 3.86 M) |

§15's role table — *"pMF supplies the map, drift supplies global correction"* —
is the shape of measurements we already have.

---

## 2. The mechanism, component by component

| component | what it actually does |
|---|---|
| **pMF** | learns `z → x` in one call, dense samplewise supervision across the whole `(r,t)` triangle. **The only component that makes samples.** |
| **Sinkhorn velocity** | `V = T_p(a) − T_q(a)`, a difference of *barycentric projections* under two balanced entropic plans. Acts only on the endpoint law. |
| **`A = DH`** | invertible Haar reweighting. Changes the *metric* transport is computed in, hence the direction of `V`. |

---

## 3. Why cross-minus-self is the load-bearing structure

The document asserts this; the derivation matters and is omitted there.

Entropic OT is **biased**: `min ⟨π,C⟩ + ε·KL(π‖u⊗u)` does not vanish at
`p = q`. The entropic term rewards diffuse plans, so a cross-term-only
objective would systematically **inflate the generated law's spread**.
Subtracting the self term — generated against an *independent* generated batch
— cancels that bias to first order. That cancellation is exactly what gives the
**Sinkhorn divergence**, unlike entropic OT, a definite population zero
(Feydy et al. 2019).

So the self term is **not hand-designed repulsion**; it is the debiasing term,
and definiteness of the zero depends on it. This is a structurally different
object from B2's Laplace field, whose negative term is a normalized barycentric
mean with no comparable guarantee.

---

## 4. Why the independent second batch is not optional

§17.2 forbids same-batch self estimation, citing an empirical ablation. The
reason is structural:

> If `x' = x`, the self plan `π^qq` is (near) the identity permutation — every
> point transports to itself at zero cost. Then `T_q(aᵢ) ≈ aᵢ`, so
> `Vᵢ ≈ T_p(aᵢ) − aᵢ`. **The debiasing term has been silently deleted**, and the
> objective reverts to biased entropic OT with its variance-inflating pull.

This also explains why diagonal masking is *not* an equivalent repair: masking
forces mass onto second-nearest neighbours, producing a different estimator
rather than the debiased one. §17.2's prohibition is correct for a reason
stronger than the cited ablation.

---

## 5. §14.2's critique of B2 — sharpened, and partly rebutted

The document argues that minimizing `E‖V‖²` can be gamed because particles
"move away from the probes until the local kernel interaction becomes weak."

**That specific channel is blocked by B2's construction**, and the document
misses why: the field is a difference of two *row-normalized* barycentric
means. Softmax normalization means distance alone cannot shrink either term —
moving all generated samples far away still yields a convex combination of
them. B2 additionally drew probes from *target + Gaussian noise* (full support)
precisely so the model could not avoid regions where its field is nonzero.

**But a real gaming channel exists, and it is more precise than stated:**

> `V(a) = Σⱼ aⱼyⱼ − Σₖ bₖzₖ` can be driven toward zero by making the generated
> **conditional barycentric mean** match the target's at each probe, *without
> the distributions matching*. Matching conditional means is a far weaker
> constraint than matching laws — and it is **easier to satisfy with a
> lower-rank generated set.**

That is a better explanation of B2's measured 38–40% effective-rank collapse
than anything in our own B2 analysis, and it makes §14.3's case for Sinkhorn
precise: the **column marginal constraint** forces mass onto every target
point, so conditional-mean matching while ignoring target regions is no longer
available.

§14.2's conclusion is right, for a reason it does not identify.

---

## 6. What the stop-gradient identity buys, and where it stops

`∇_θ L_DT = 2η ∇_θ F(q_θ)` is the document's strongest theoretical
contribution: it converts a distributional velocity into descent on a **scalar
energy**, which is exactly what B2's differentiated field energy could not
guarantee. It also reframes the stop-gradient — used throughout this repository
since Phase 1 as an implementation detail — as structurally necessary.

The identity holds only when `V` *is* `−∇δF/δq`. With finite ε, finite batches
and finitely converged plans, `V` is an estimator whose bias scales with both ε
and batch size (Genevay et al. 2018). §21 marks this correctly. **The guarantee
is asymptotic; the experiment is empirical.** The theory motivates the design
rather than certifying it.

---

## 7. Why single-class CIFAR is the *right* test — stronger than §15.2's argument

The document says one-class makes the encoder-free limitation "testable rather
than hidden." That understates the design.

The failure this mechanism targets is **weak interaction between distant
modes** (§14.1). On full CIFAR-10, ten semantic classes sit far apart in pixel
space — the regime where fixed geometry is weakest and a learned encoder helps
most. On **one class** the data manifold is far more connected, so fixed
Haar/pixel geometry is *most likely to be adequate*, and global balancing can be
tested without confounding it against cross-class semantics.

This is a clean falsification design: **if balanced Sinkhorn under fixed
geometry fails on one class, it fails everywhere.**

The converse must travel with it: success on one class does **not** extrapolate
to ten, precisely because multi-mode is where fixed geometry is weakest. §21
already marks the ImageNet claim *"unknown and unlikely."*

---

## 8. The largest unanalyzed risk: pMF/drift coupling

No section addresses this directly.

pMF regresses over the **whole `(r,t)` triangle**. The drift acts only at the
**endpoint** `(r=0, t=1)`. They look separable — but the MeanFlow identity

```text
u = v − (t − r)·du/dt
```

**couples them**: perturbing the endpoint map changes `u` at `t = 1`, which
through the consistency identity constrains `du/dt` along the entire
trajectory. The drift's endpoint correction therefore propagates backward
through the triangle.

That can help (one consistent correction enforced everywhere) or fight the
identity (the drift demands an endpoint the triangle cannot consistently
support). §20's long-short-flow-map note concedes joint convergence is
unproven; §9's **gradient-cosine logging is the right instrument**, and a
persistently negative cosine should be treated as a **primary early-abort
signal**, not merely a trigger for a PCGrad rescue arm.

---

## 9. A correction to an earlier objection of mine

I initially criticised the document for proposing fixed multiscale geometry
despite Phases 3–4 measuring it 37–44% worse than raw pixels at 49–81× the
kernel cost. **That conflated two different things and was wrong.**

§7 proposes avgpool views; §15.2 explicitly retracts them — *"Average pooling
alone is not injective"* — and replaces them with a complete Haar transform.
The document self-corrects. And the Haar construction has a property the
Phase 3–4 wavelet features did not: **with uniform weights, orthonormality
makes the cost exactly raw-pixel L².** It degrades gracefully to the identity.

Phases 3–4 *replaced* the geometry; this *reweights an invertible basis*. The
prior negative is far less applicable, and the design is strictly safer than
what we tested.

---

## 10. Source assessment

**The classical OT citations are correct and load-bearing.** Feydy et al. 2019
is exactly right for the definite population zero of Sinkhorn divergences
(positive-definiteness, OT↔MMD interpolation). Genevay et al. 2018 is right for
learning generative models with Sinkhorn divergences *and* for the entropic
bias motivating cross-minus-self. Sanjabi et al. (1810.02733) is right for
convergence/robustness of regularized-OT training. Liutkus 2019 and Du 2023 are
correctly placed as adjacent-not-primary. The OT literature is well handled.

**MeanFlow (2505.13447) and SiD2 (2410.19324) are recognizable.**

**Unverified at the time of writing:** the pixel-MeanFlow line (2601.22158),
the drifting paper (2602.04770), Sinkhorn-Drifting (2603.12366), W-Flow
(2605.11755) and the WGF-interpretation paper (2605.05118) carry 2026 arXiv
IDs this reviewer could not confirm. Three of them carry the central mechanism
argument, and the claim that *"W-Flow's ablations report balanced Sinkhorn beats
the drifting proxy, MMD, and KL"* does heavy lifting in §14.3 and §16. **These
should be read directly rather than inherited.**

**§16's "existing image evidence" row now has a matched answer it does not
use.** B3 measured drifting at 3.86 M parameters reaching recall 0.0000 while,
on drifting's own energy axis, B2 (15.30) and B1 (16.13) both beat the drifting
arm (~17.05). That *strengthens* the document's case and should be cited.

---

## 11. Continuity with this repository's calibration results

The proposal is built on our measurements, not parallel to them:

- **λ calibration is B2's protocol verbatim** — 0.25 event-level gradient
  ratio, ~2.5% cadence-averaged, frozen before the matched fork;
- **rectangular support** (§17.5: 64 generated vs 128–256 real) matches
  Phase 22's `D_pos` finding that more positives helped;
- the **ESS ≈ 0.60 outcome-blind bandwidth rule** is ours;
- **§17.4's ε warning is our bandwidth lesson, independently rediscovered**:
  *"published values such as ε = 0.05 are meaningful only relative to the
  authors' feature scaling"*, fixed by normalizing cost so median pairwise = 1.
  This is exactly τ/median ≈ 0.19 — same error class, correctly pre-empted.

---

## 12. Assessment and recommended sequence

The mechanism reasoning is sound and better-founded than B2's; the
stop-gradient reframing is a genuine theoretical contribution; §21–22's claim
ledger and falsification table are the strongest discipline any proposal in
this project has carried.

Two limitations are **not** blockers, having been resolved by the operator:
hardware (a rented GPU makes the 45 M U-ViT plus JVP a cost question, not a
feasibility one) and scope (single-class is a deliberate proof of concept, and
§7 above argues it is the correct regime rather than an under-scoping).

What remains:

1. **`B0 + Sinkhorn drift` is the cheapest decisive test** of the document's
   central claim (balanced Sinkhorn > Laplace energy). It reuses a frozen,
   validated foundation and three completed baselines, isolates the correction
   from the foundation change, and would establish whether the Sinkhorn
   machinery is worth carrying into a pMF build.
2. **B2.5's interaction term bears directly on §16.** Unit 500 shows `B1B2`
   recovering effective rank from 0.577 to 0.794 of B0 while retaining B2 in
   the objective. If that holds across units 501–502, the Laplace field's
   weakness is a **correctable side-effect** rather than fundamental, and §16's
   demotion is premature — the answer would be "Laplace energy plus a geometry
   anchor," not "replace Laplace with Sinkhorn."
3. **The §18 staging controls the correction but not the foundation.** S0 = pMF
   alone and S1 = pMF + drift isolate the drift; no arm keeps a validated
   foundation with the new correction. A failure at S1 would be unattributable
   between "Sinkhorn is weak" and "pMF is the bottleneck."

None of the above argues against building the mechanism. It argues for
establishing the correction's value on an existing foundation first, at roughly
a tenth the cost and against three baselines that already exist.

---

## 13. Reconciled decision and chronological build order

This section records the final recommendation after cross-checking the review
against `meanflowdrifting.md`, the B0/B1/B2/B3 artifacts, the partial B2.5
artifact, and the cited Sinkhorn/WGF literature. It supersedes any reading of
Section 12 as an authorization to launch the full pMF experiment immediately.

### 13.1 Decision

Proceed with the MeanFlow--Sinkhorn direction, but separate its two untested
components:

1. first establish that a fully balanced Sinkhorn correction improves the
   already validated B0 generator;
2. only after that correction passes, replace the multi-step B0 foundation
   with a one-call pixel MeanFlow foundation;
3. introduce nonuniform Haar geometry only after identity-pixel Sinkhorn has
   established the value of balancing itself.

This sequence minimizes attribution ambiguity. A direct pMF + Haar + Sinkhorn
run changes the generator, time objective, discrepancy, estimator, geometry,
and compute profile simultaneously. A failure would be scientifically weak
because no component could be isolated.

The B0 experiment is a mechanism screen, not the final one-step result. B0
still uses its existing multi-step bridge. Its purpose is to determine whether
the new distribution correction is worth carrying into the expensive pMF
build.

### 13.2 Corrections that must travel with the decision

The following qualifications are part of the protocol, not editorial details:

- Standard entropic cross-transport is biased, but its pathology must not be
  universally described as variance inflation. Barycentric averaging commonly
  creates shrinkage, blur, or collapse. The cross-minus-self construction is
  retained because it debiases the chosen Sinkhorn functional.
- An independent second generated batch is mandatory for the proposed
  one-sided detached-target estimator. At finite epsilon a same-support self
  plan is not literally always the identity, so the rationale is avoidance of
  a diagonal-shortcut/degenerate empirical estimator, not the claim that the
  mathematical self term ceases to exist.
- The population Laplace field is identifying in the theorem's stated scope.
  B2's rank risk comes from finite probes, finite batches, and stochastic
  optimization; it must not be described as a failure of the population
  converse.
- B3 rules out the repository's matched-capacity raw-pixel smooth-Laplace
  proxy, not every possible drifting or WGF generator.
- Finite epsilon defines the exact regularized Sinkhorn objective. The
  approximation errors relevant here are finite sampling, use of empirical
  independent supports, and finite solver convergence. Calling epsilon itself
  “estimator bias” is only meaningful relative to unregularized OT.
- If `A = D H` includes unequal frozen subband normalization, equal nominal
  Haar weights do not reproduce raw pixel L2. The exact raw-identity fallback
  requires `D = c I` (or a separately implemented identity cost).
- A one-class failure is strong evidence against this configuration, not a
  universal impossibility theorem. A success is likewise a focused proof of
  concept and does not extrapolate to ten classes or ImageNet.
- ArXiv `1810.02733` is Genevay, Chizat, Bach, Cuturi, and Peyre's *Sample
  Complexity of Sinkhorn Divergences*. It is not a Sanjabi et al. citation.

### 13.3 Stage S0: identity-pixel Sinkhorn implementation

Build the correction without changing B0's generator or raw-pixel geometry.
For a primary generated endpoint batch `x`, an independent no-gradient
generated batch `x_self`, and an independent real batch `y`:

1. compute quadratic raw-pixel costs `C_qp(x,y)` and `C_qq(x,x_self)`;
2. solve both rectangular entropic OT problems in the log domain with uniform
   row and column marginals;
3. require a frozen maximum *relative* marginal residual and an iteration cap;
4. form conditional barycenters from `B * pi_qp` and `B * pi_qq`;
5. set `V = T_p - T_q`;
6. regress `x` toward the detached target `x + eta * V`;
7. permit gradients only through the primary `x` branch.

No diagonal mask is allowed. The two generated noise tensors, outputs, and
roles must be independently sampled. Plans, real samples, self samples,
velocity, and target must be detached.

Before integration, test:

- row and column marginal residuals on square and rectangular matrices;
- agreement with analytically trivial constant/symmetric cases;
- conditional barycentric rows summing to one;
- finite-difference agreement between the detached-target parameter gradient
  and the corresponding finite-sample Sinkhorn-energy direction on a tiny
  deterministic system, under exactly the same entropic convention;
- absence of gradients through the self and real branches;
- deterministic replay and artifact serialization.

### 13.4 Stage S1: cheapest decisive matched screen

Clone the validated B0 state and compare, on the same data cursor, random
states, optimizer state, update count, and evaluation allocations:

| arm | continuation |
|---|---|
| `B0-control` | existing B0 objective only |
| `B0-Laplace` | existing B0 + frozen B2 correction |
| `B0-Sinkhorn-I` | existing B0 + identity-pixel balanced Sinkhorn correction |

The primary causal comparison is `B0-Sinkhorn-I` versus `B0-control`.
`B0-Laplace` is the mechanism incumbent and must use the already frozen B2
configuration rather than a retuned version.

Freeze Sinkhorn cost scaling, epsilon, solver tolerance, velocity step, event
cadence, and gradient-ratio coefficient before the confirmatory units. Select
epsilon using numerical transport diagnostics only: marginal convergence,
normalized plan entropy, maximum conditional weight, velocity scale, and
finite-value rate. Do not select it using FID, recall, or sample grids.

The screen promotes Sinkhorn only if it:

- decreases held-out Sinkhorn discrepancy or its preregistered finite-sample
  proxy;
- retains B0 coverage, effective rank, and memorization vetoes;
- improves at least one quality/coverage axis over the matched control with a
  consistent direction across the frozen units;
- has valid two-batch and solver diagnostics throughout;
- is competitive with B2 on mechanism improvement without reproducing B2's
  rank loss.

Lower training Sinkhorn loss alone is insufficient.

### 13.5 Stage S2: geometry attribution

Only if identity-pixel Sinkhorn passes, add a complete invertible Haar basis.
Compare:

- identity-pixel Sinkhorn;
- orthonormal Haar with exact identity scaling `D = cI` as a code-equivalence
  control;
- positively reweighted/scaled complete Haar.

The first two must agree numerically up to floating-point and solver tolerance.
That equivalence test is required before interpreting the nonuniform arm.
Every Haar subband remains present with a strictly positive weight. Subband
statistics are estimated from the training split and frozen outcome-blind.

This stage asks whether fixed multiscale conditioning helps. It must not be
mixed into Stage S1, where the scientific question is whether two-sided
balancing itself helps.

### 13.6 Stage S3: one-step pMF foundation

Implement and preflight pixel MeanFlow separately:

- direct pixel prediction;
- complete `(r,t)` triangle sampling;
- validated JVP and stop-gradient placement;
- one-call endpoint identity;
- no VAE, learned feature encoder, perceptual training loss, or teacher;
- a fixed one-class train/test split and memorization audit.

Do not attach Sinkhorn until the pMF-only foundation produces recognizable,
diverse samples. This gate prevents a failed foundation from being
misdiagnosed as a failed correction.

### 13.7 Stage S4: final one-step matched fork

Clone a successful pMF foundation exactly and run:

| arm | continuation |
|---|---|
| `pMF-control` | pMF only |
| `pMF-Sinkhorn-I` | pMF + the frozen identity Sinkhorn mechanism from S1 |
| `pMF-Sinkhorn-H` | optional, only if nonuniform Haar passed S2 |

Log endpoint-correction versus pMF gradient cosine and norm ratio. Shared
parameters couple the endpoint correction to the whole MeanFlow triangle.
Persistent negative cosine is an early warning, but an abort requires a frozen
combination of sustained conflict and degradation/instability; a negative
cosine by itself is not proof of harmful optimization.

The final success claim requires improved matched quality or coverage, valid
diversity and memorization checks, and an exported sampler whose forward
counter is exactly one. Sinkhorn, Haar, JVPs, reference samples, and the second
generated batch remain training-only.

### 13.8 B2.5's role

Complete B2.5 if its remaining frozen units are affordable, but do not let it
block S0--S1. Unit 500's apparent rank recovery is encouraging but not a
confirmation. If all units confirm, the conclusion is that the Laplace term's
finite-sample rank side effect can be protected by an anchor. That would make
`B1+B2` a stronger incumbent; it would not remove the conservative scalar-flow
and global-balancing reasons for testing Sinkhorn.

### 13.9 Final claim boundary

Before S4 succeeds, the project has a proposed one-step hybrid and a sequence
of component tests. After S4 succeeds, the strongest honest claim is:

> On a preregistered one-class pixel task, a fully balanced distributional
> drift improved a matched one-call pixel MeanFlow generator without a VAE,
> learned feature encoder, perceptual training loss, or extra inference call.

It remains a MeanFlow--Drifting hybrid while the pMF loss is active. A short
distribution-only takeover may later test the stronger “pure Sinkhorn
drifting” claim, but it is not part of the initial promotion gate.
