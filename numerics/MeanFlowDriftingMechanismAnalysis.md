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
