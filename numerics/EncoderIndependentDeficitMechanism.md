# Encoder-Independent Kernel Drifting — the mechanism

## The deficit is a shape phenomenon: a clumped cloud balances the field at a smaller radius

*Extended pass on the Phase-9 result. Code: `diagnose_phase10.py`. Artifact:
`phase10_probe.json` (+ `.sha256`). Development seeds
(`MASTER_SEED + 17000..`), 3 seeds, CIFAR-16, target-ESS 0.9. Nothing feeds
a gate. This is the first mechanism proposal in this program that survives
its own controls.*

---

## 0. The result

Phase 9 showed the deficit survives down to `f(z) = Az + b`, which ruled out
the architecture but sharpened a contradiction: in the Gaussian cells `q = p`
is *reachable* and is a fixed point, yet the generator converges elsewhere.
Two measurements resolve it.

**(1) The field is very nearly unbiased.** Along clouds shaped like the data,
the raw radial component crosses zero at:

| | unmasked *(repo default)* | masked |
|---|---:|---:|
| 64 positives | **α = 0.844** | 1.056 |
| 2048 positives | **α = 0.830** | 1.094 |

A ~15% inward bias from the self-term, stable across a 32× change in batch —
**nowhere near the ~55% needed** to explain a generator sitting at 0.42.

**(2) The generator's cloud is not data-shaped.** Comparing clouds at
*matched second moment*, so every difference is shape and not scale:

| cloud | 2nd moment | **radial** | nn spacing | nn CV | **spectral tail** |
|---|---:|---:|---:|---:|---:|
| free particles | 0.995 | −0.0020 | 13.45 | **0.023** | **0.415** |
| data-shaped | 0.995 | −0.0019 | 9.93 | 0.245 | 0.142 |
| **generator** | **0.417** | **+0.0006** | **3.47** | 0.337 | **0.0034** |
| **data-shaped, same scale** | **0.417** | **+0.0264** | 6.36 | 0.245 | 0.142 |

> **At the generator's own scale its cloud is radially balanced (+0.0006),
> while a cloud of the data's shape at the identical scale is pushed outward
> 43× harder (+0.0264).** The generator is not failing to reach equilibrium —
> it is at a *different* equilibrium, one belonging to its cloud's shape.

---

## 1. The mechanism

Algorithm 2 balances attraction to the data against repulsion from the
cloud's own neighbours. **Repulsion is set by how close the cloud's points
are to each other**, so the balance point depends on the cloud's packing, not
only on its variance.

The generator's cloud is **clumped**: at 0.417 of the data's variance its
nearest-neighbour spacing is 3.47 against the data-shaped comparator's 6.36
at the *same* variance — its points sit 1.8× closer together while carrying
identical total spread. It achieves that by concentrating almost all of its
energy in few directions: spectral tail **0.0034 against 0.142**, a factor of
42.

A clumped cloud therefore reaches the repulsion it needs at a **small overall
radius**. That is the deficit.

Free particles do the opposite. Being unconstrained points under a repulsive
interaction they settle into a near-regular packing — nearest-neighbour CV
**0.023** against an i.i.d. sample's 0.245, an order of magnitude more
uniform — and spread across *more* directions than the data itself (tail
0.415 against 0.142). They over-fill, and land at second moment 0.995.

**The generator cannot do this.** It is a smooth pushforward of a
low-dimensional Gaussian; its cloud is constrained to a smooth image and
cannot adopt a near-regular packing in 768 dimensions.

---

## 2. Why this explains what nine phases could not

| observation | phase | explained |
|---|---|---|
| deficit invariant to optimizer, lr, η | 6A, R24 | none of them changes the cloud's shape |
| invariant to bandwidth and cloud size | 7A | ditto |
| invariant to latent dimension (8→512) | N4 | tail only moved 0.0015→0.0058; more latent room does not buy packing |
| invariant to capacity (36×) | 8A | a wider net makes a *smoother* map, not a more spread one |
| invariant to teacher-fitting quality (2.3×) | 8B | fitting the same teacher better keeps the same shape |
| invariant to training budget and field noise | N6, P5 | equilibrium property of the shape |
| **present already for `f(z)=Az+b`** | **9** | a linear map is maximally smooth — the strongest constraint of all |
| **the nonlinearity deepens it** (1.3–3.5×) | 9 | an MLP concentrates energy further |
| spectral confinement moves particles 1.012→0.70 | P4 | the same mechanism, applied to particles — and it saturates because confinement does not clump them |
| **R11 works, isotropically** | 3,5,6,7,8 | inflating the teacher forces spacing open; it raised the tail 15× (0.0034→0.047) |
| richer corrections do not beat the scalar | 6B | the defect is a scalar radius, set by packing |
| over-correction hurts | 6B | past the balance point the field pulls back in |
| free particles are fine | 6C, 7B | they can pack regularly; the generator cannot |

Twelve standing observations, including seven previously unexplained
negatives, under one mechanism. Crucially it also predicts the **sign** of
the two things that did move: R11 (inflates spacing → helps) and spectral
confinement (reduces spread → hurts).

---

## 3. Controls, and two hypotheses this pass killed

**The field-bias hypothesis is refuted.** The obvious reading of Phase 9 was
that the bi-softmax field is biased inward. It is — but only by 15%, stable
from 64 to 2048 positives, which cannot produce a 58% deficit.

**The self-term hypothesis was killed in the previous pass and stays dead.**
In 64-D isotropic Gaussian geometry the self-term carries 21% of the
repulsion and 3.7× the masked field; at CIFAR it is 7%. The Q1 table above
confirms it at the level that matters: masking moves the radial zero from
0.844 to 1.056, a real effect and far too small to be the deficit.

**What this pass does not claim.** The relationship between packing and
equilibrium radius is *demonstrated* (the 43× radial difference at matched
scale) but not *quantified* — there is no functional law here, only a
direction. §4 is how to get one.

---

## 4. The next experiment

### 10A — measure the law

Interpolate clouds between generator-shaped and data-shaped at fixed second
moment and find the radial zero as a function of the shape statistic. The
question is whether the equilibrium radius is a **monotone function of the
spectral tail** (or of nearest-neighbour spacing), and if so, what function.

That would convert "clumping shrinks the equilibrium" into a quantitative
prediction: given a generator's tail fraction, predict its second moment.
Testable against every number this program has already recorded — the
generator at 0.0034 → 0.42, R11 at 0.047 → 0.95, particles at 0.415 → 1.0
are three points on that curve already, and they are ordered correctly.

### 10B — intervene on shape directly, not on scale

If packing is the mechanism, then a term that **opens the cloud's spacing**
should do R11's job from the right direction and without inflating anything:

- a repulsion / diversity penalty on the generator's own batch;
- or a spectral-tail floor.

**Declared gate, in the shape of the last three:** if a shape intervention
without R11 reaches a second-moment ratio in `[0.7, 1.3]` with ED² within 25%
of the best R11 cell, then **R11 is superseded by acting on shape**, and the
program's long-standing empirical reform finally has a principled
replacement.

This is the first proposed intervention in nine phases that is derived from a
measured mechanism rather than guessed, and 10A's law is what would let its
required strength be *computed* rather than tuned.

### Why not another hyperparameter sweep

Nine axes are known inert (§2). The mechanism says why: none of them changes
the cloud's packing. That is now a prediction, not an excuse — any proposed
axis can be screened in seconds by asking whether it moves the tail fraction.

---

## 5. Scope and caveats

- One dataset (CIFAR-16), one geometry, one bandwidth rule, 3 seeds.
- The packing→radius relationship is demonstrated at two scales, not swept;
  the "43×" is a single matched-scale comparison, not a curve.
- Nearest-neighbour spacing and spectral tail are correlated here and this
  pass does not separate them — 10A must, since they suggest different
  interventions.
- The Q1 radial-zero table uses clouds that are exact rescalings of real
  data; real generators differ in more ways than scale.
- Nothing here concerns ImageNet, FID, or the paper's trained model. The
  anchor stays disabled; the geometry thread stays closed.

## 6. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.diagnose_phase10 `
  --stage all --seeds 3 --steps 600
```
