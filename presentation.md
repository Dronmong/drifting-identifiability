# Presentation: What We Proved About Drifting Models

Audience: mathematically/ML-aware, but not necessarily familiar with Drifting Models.

Goal: explain what the original paper introduced, what limitations/open points remained, and what this project built on top of it.

---

## 1. One-slide takeaway

The Drifting Models paper proposes a powerful training objective: move generated samples along a data-dependent “drift” field so the generated distribution evolves toward the data distribution.

The paper proves and empirically demonstrates the safe direction:

```text
p = q  ⇒  drift V = 0.
```

Our project studied the harder identifiability direction:

```text
V = 0  ⇒  p = q ?
```

We proved this implication under explicit, machine-checked finite-basis/frame conditions, clarified the fixed-anchor form of Algorithm 2, handled feature-space and CFG edge cases honestly, and quantified the sample/conditioning costs numerically.

The result is not “we beat the paper’s FID.” The result is:

```text
We turned a heuristic identifiability story into a verified theorem stack,
and we identified exactly where the practical gaps still are.
```

---

## 2. What the original paper had

The paper, *Generative Modeling via Drifting*, introduces a one-step generative modeling paradigm.

Instead of iteratively denoising at inference time, the generator is trained so that its output distribution moves under a drifting field toward the data distribution.

At a high level:

```text
noise z ── generator fθ ──> generated sample y

training computes a drift V(y)
and asks the generator to move y toward y + V(y).
```

The paper’s key ingredients:

- a population drift field defined from attraction to data and repulsion from generated samples;
- a minibatch estimator of that field, Algorithm 2;
- feature-space drifting losses for images;
- multiple temperatures;
- classifier-free guidance-like modifications;
- strong empirical performance on ImageNet 256×256.

Reported headline performance:

```text
ImageNet 256×256, 1 NFE:
FID ≈ 1.54 latent-space model
FID ≈ 1.61 pixel-space model
```

So the paper is empirically very strong.

---

## 3. The mathematical equilibrium question

The paper’s intuitive equilibrium story is:

```text
If generated distribution q matches data distribution p,
then the drift should vanish.
```

That is the easy/safe direction:

```text
p = q  ⇒  V = 0.
```

The harder question is the reverse:

```text
If the drift vanishes, must q = p?
```

This matters because the training objective tries to minimize drift. If zero drift can happen at some wrong distribution, then minimizing drift does not necessarily mean matching the target distribution.

So our target question was:

```text
Under what conditions is zero drift identifiable?

V = 0  ⇒  p = q.
```

---

## 4. What was heuristic/open in the paper

The paper contains the right intuition, especially in the appendix finite-basis calculation.

Roughly, the appendix says:

1. Represent two distributions using finite coefficients:

   ```text
   p = Σᵢ aᵢ φᵢ,     q = Σⱼ bⱼ φⱼ.
   ```

2. Expand the drift as a bilinear sum:

   ```text
   V ≈ Σᵢ Σⱼ aᵢ bⱼ Uᵢⱼ.
   ```

3. Use antisymmetry:

   ```text
   Uᵢⱼ = -Uⱼᵢ.
   ```

4. Regroup into coefficient minors:

   ```text
   Σ_{i<j} (aᵢbⱼ - aⱼbᵢ) Uᵢⱼ.
   ```

5. If the interaction vectors `Uᵢⱼ` are independent, zero drift forces all minors to vanish.

6. For normalized probability coefficients, vanishing minors imply:

   ```text
   a = b.
   ```

This is the correct skeleton, but several things needed to be made precise:

- What exactly are the basis functions?
- Are `p` and `q` genuine probability measures?
- Which facts are paper-established and which are extra assumptions?
- When are the interaction vectors actually nondegenerate?
- How does the normalized drift used in the paper connect to the bilinear numerator?
- Does the finite minibatch estimator converge to this same population field?
- What happens in feature space?
- What happens with CFG, where the target can become signed/affine?

Our project focused on turning this into an audited theorem stack.

---

## 5. Limitations/open directions the paper itself points to

The paper is primarily an algorithmic and empirical contribution, not a complete identifiability theory.

Some limitations/open directions visible from the paper:

### 1. The reverse implication is not the main theorem

The paper’s equilibrium intuition strongly motivates:

```text
p = q ⇒ V = 0.
```

But the harder training-objective question is:

```text
V = 0 ⇒ p = q.
```

The appendix gives a finite-basis route for this, but leaves the exact usable conditions as a mathematical question.

### 2. Feature extractors matter a lot

The paper emphasizes that the feature encoder plays an important role in high-dimensional image generation. It compares SSL encoders, MAE-style encoders, latent-space encoders, and multi-scale feature sets.

This raises a mathematical guardrail:

```text
matching in feature space means matching pushforward feature distributions,
not automatically matching image distributions.
```

### 3. Kernel/temperature/feature design remains empirical

The paper uses multiple temperatures and feature normalization to make the drift robust across feature scales.

But the practical choice of:

- kernel;
- temperature grid;
- feature encoder;
- feature normalization;
- generator architecture;
- CFG scale;

is still an empirical design problem.

### 4. Algorithm 2 is a finite-batch estimator

The population drift is conceptually clean, but the actual training loop uses a finite minibatch estimator with:

- row/column softmax coupling;
- reused generated negatives;
- an implementation self-mask;
- feature-space normalization;
- multiple feature maps.

The paper’s performance shows this works. Our project asks what this estimator is mathematically converging to.

---

## 6. What we built: a trusted Lean boundary

The project has a strict trust boundary.

Allowed:

- facts directly established in the paper;
- standard external analytic/probabilistic facts when explicitly classified;
- Lean/Mathlib foundations.

Not allowed:

- axiomatizing the desired theorem;
- axiomatizing an equivalent injectivity condition;
- silently using characteristic-kernel/RKHS assumptions in promoted results;
- treating feature-space equality as source-distribution equality without an explicit lifting condition;
- treating CFG affine targets as probability laws without nonnegativity.

The result is machine checked in Lean, with automated axiom audits.

Important distinction:

```text
Conditional Gaussian/RKHS research modules exist,
but promoted theorems are audited not to depend on them.
```

---

## 7. Core theorem: finite-basis identifiability

We formalized the appendix argument as a real theorem.

For a finite probability-density basis:

```text
φ₀, …, φ_{m-1}
```

with simplex coefficients:

```text
aᵢ ≥ 0,  Σᵢ aᵢ = 1
bᵢ ≥ 0,  Σᵢ bᵢ = 1
```

define:

```text
p = Σᵢ aᵢ φᵢ
q = Σᵢ bᵢ φᵢ.
```

If the strict-pair interaction vectors satisfy a nondegeneracy/frame condition, then:

```text
zero normalized population drift at the selected probes
⇒
a = b
⇒
p = q.
```

This is the original target, but scoped honestly:

- finite basis;
- genuine probability measures;
- explicit probes;
- explicit nonzero normalizers;
- explicit interaction nondegeneracy.

---

## 8. The algebraic heart

The formal proof follows the appendix intuition:

```text
V = ΣᵢΣⱼ aᵢbⱼ Uᵢⱼ
```

Antisymmetry gives:

```text
Uᵢⱼ = -Uⱼᵢ.
```

So:

```text
V = Σ_{i<j} (aᵢbⱼ - aⱼbᵢ) Uᵢⱼ.
```

If the `Uᵢⱼ` are independent or satisfy a positive frame bound, then:

```text
aᵢbⱼ - aⱼbᵢ = 0     for all i<j.
```

For normalized coefficient vectors, this implies:

```text
a = b.
```

This final normalized-minor argument is proved directly in Lean, not assumed.

---

## 9. Quantitative stability, not just exact equality

The project does not only prove:

```text
V = 0 ⇒ p = q.
```

It also proves a stability estimate.

If the interaction family has frame constant `c > 0`, then coefficient error is controlled by drift error:

```text
‖a-b‖₁ ≤ (2B/c) ‖V_probes‖.
```

For finite squared probe loss:

```text
Loss = Σₙ ‖V(xₙ)‖²,
```

we get:

```text
‖a-b‖₁ ≤ (2B/c) sqrt(Loss).
```

This matters because training will not reach exactly zero drift. It gives a way to translate “small drift” into “small coefficient error,” provided the frame constant is usable.

---

## 10. Concrete nondegenerate model classes

The project also gives concrete model classes where the frame condition is discharged.

### Atomic empirical basis

For finite support points `zᵢ`, the interaction vectors are computed exactly.

Under a distinct strict-pair-sum condition:

```text
zᵢ + zⱼ are all distinct for i<j,
```

a Vandermonde argument proves linear independence of the interaction vectors.

This gives an end-to-end theorem:

```text
zero drift ⇒ p = q
```

for an explicit Gaussian empirical finite basis.

### Smooth two-component basis

The project also constructs a non-atomic smooth example:

- two normalized `C∞` bump densities;
- disjoint ordered supports;
- Gaussian reference measure;
- one probe;
- direct sign-based frame proof.

This gives a fully instantiated continuum-supported result:

```text
zero finite normalized population-drift loss ⇒ p = q.
```

---

## 11. What Algorithm 2 actually estimates

One of the most important findings is about the paper’s minibatch Algorithm 2.

Naively, one might expect Algorithm 2 to estimate the bare population mean-shift drift.

But with fixed anchors and no self-mask, the softmax structure simplifies differently.

The row-softmax factor cancels inside the centroid, leaving a self-normalized importance sampling estimator with weight:

```text
wᵢ(y) = k(xᵢ,y) / sqrt(g(y)),

g(y) = Σ_r k(anchor_r, y).
```

So the fixed-anchor/no-mask form targets a column-reweighted field, not
generally the bare field.

In short:

```text
fixed-anchor Algorithm 2 estimator → column-reweighted population drift,
not necessarily bare mean-shift drift.
```

This is not a bug; it is a structural clarification.

We then proved identifiability/stability for this modified field under its own frame condition.

---

## 12. Finite-sample guarantees

The project proves a finite-sample bridge:

If a random estimator `Vhat` is close to the population probe drift `V`, then the coefficient error bound holds with high probability.

Schematically:

```text
estimation error small
⇒ coefficient error controlled.
```

For self-normalized estimators, the project proves `1/N`-type concentration using the reviewed sample-mean axiom.

For the fixed-anchor/sample-split no-mask route:

```text
sampled centroids
→ SNIS consistency
→ column-reweighted population field
→ coefficient stability / identifiability.
```

For the implementation self-mask:

- the `1e6` mask is treated as a deterministic perturbation of a deleted/leave-masked-out estimator;
- the perturbation is astronomically tiny at paper temperatures;
- the deleted estimator has its own indexed SNIS consistency theorem.

These probability theorems freeze the anchor batch. The paper instead reuses
the random generated negatives as anchors (`x = y_neg`), making every column
weight jointly batch-dependent. A concentration theorem for that coupled
estimator remains open.

---

## 13. Denominator-tail improvement

The first certified finite-sample bound used a deterministic denominator floor:

```text
dmin = N · wmin.
```

This was brutally pessimistic, especially at small temperature, because `wmin` pays the worst possible kernel value.

We proved a high-probability denominator refinement:

```text
P{Σ w_l(Y_l) < Σ μw_l - t} ≤ N σw² / t².
```

Then we plugged it into the ratio-estimator bound.

Numerically, in the two-atom certified class:

| Quantity | Value at tau = 0.2 |
|---|---:|
| old deterministic certified N | ~8.6e8 |
| refined denominator-tail certified N | ~1.3e6 |
| LLN-typical benchmark | ~2.9e5 |
| observed extrapolation | ~1.9e5 |
| paper training batch | 64 |

So the new theorem removes the catastrophic denominator-floor pathology.

But certification is still much stricter than practical training.

---

## 14. Feature-space guardrail

The paper trains in feature space.

This is practically essential, but mathematically subtle.

If a feature map is:

```text
φ : X → F,
```

then matching in feature space generally gives:

```text
φ♯p = φ♯q,
```

not necessarily:

```text
p = q.
```

If `φ` is non-injective, distinct source distributions can collapse to the same feature distribution.

Our project formalizes this guardrail:

```text
feature-space drift zero ⇒ equality of feature laws
```

and source-law equality requires an extra condition, such as:

- `φ` is a measurable embedding;
- a finite feature family is measure-determining;
- a quantitative feature-stability certificate controls a source discrepancy.

This prevents a common overclaim:

```text
feature matching ≠ automatic data-distribution matching.
```

---

## 15. CFG guardrail

Classifier-free guidance introduces another subtlety.

The paper’s CFG-style formula can define an affine target:

```text
q = α p_cond - (α - 1) u.
```

This vector sums to one, but its coefficients can be negative.

So CFG is not automatically a probability-measure theorem.

We formalized this using affine coefficient vectors:

```text
Σᵢ qᵢ = 1,
but qᵢ may be negative.
```

The proved safe theorem is:

```text
zero drift matching effective negative density to conditional density
⇒ generated density equals the CFG affine target.
```

It does not claim:

```text
generated = conditional,
```

and it does not claim the CFG target is a probability law unless a separate nonnegativity condition is supplied.

Numerically, for uniform unconditional `u`, the CFG target is nonnegative only on a small simplex region:

```text
fraction ≈ (1/α)^(m-1).
```

At `α = 4`, this shrinks quickly with basis size.

So under this uniform-Dirichlet finite-basis model, the affine/signed treatment
is typical rather than a corner case. This is not a claim about the empirical
distribution of real ImageNet conditionals.

---

## 16. Practical numerical findings

The numerical suite does not prove new theorems. It prices the proved conditions.

Main findings:

### 1. Frame constants are the binding constraint

For larger finite bases, naive certificates collapse rapidly.

Example:

```text
m = 5: certificate can be around 1e-32
m = 8: float64 underflow / effectively zero
```

Probe design helps enormously, but does not remove the fundamental conditioning issue.

### 2. Algorithm 2 targets the modified field

Monte Carlo experiments confirm:

```text
MSE vs column-reweighted field decays like 1/N.
MSE vs bare field has a nonzero bias floor.
```

### 3. Self-mask perturbation is negligible

At paper temperatures:

```text
exp(-1e6 / tau_tilde)
```

is so tiny that masked and deleted estimators are indistinguishable in float64.

### 4. Certification scale is much larger than training scale

Even after the denominator-tail repair:

```text
certified N ~ 1e6
observed extrapolated N ~ 2e5
paper uses N = 64
```

This does not mean the paper fails. It means formal certification is much more demanding than practical gradient signal.

---

## 17. What we accomplished compared to the paper

The paper gave:

- a new generative modeling paradigm;
- a drift-based training algorithm;
- a heuristic/appendix-level identifiability mechanism;
- strong empirical ImageNet performance;
- feature-space and CFG extensions.

Our project added:

1. Machine-checked finite-basis identifiability:

   ```text
   V = 0 ⇒ p = q
   ```

   under explicit nondegeneracy/frame conditions.

2. Genuine probability-measure formalization:

   not just density algebra.

3. Quantitative stability:

   ```text
   small drift ⇒ small coefficient error.
   ```

4. Concrete certified model classes:

   atomic Gaussian empirical bases and a smooth two-component bump basis.

5. Algorithm 2 target clarification:

   it estimates a column-reweighted field.

6. Finite-sample estimator bridge:

   SNIS consistency, mask perturbation, deleted-estimator analysis, denominator-tail refinement.

7. Feature-space guardrail:

   feature law equality is the safe conclusion unless additional lifting assumptions hold.

8. CFG affine-density treatment:

   CFG generally exits the probability simplex, so it needs affine/signed formalism.

9. Numerical diagnostics:

   the theory is priced at paper-like temperatures and batch sizes.

---

## 18. What we did not accomplish

We should be very clear about this.

We did not show that our certified model outperforms the paper’s model.

We did not reproduce or improve the paper’s FID.

We did not prove that the full neural feature-space training setup identifies source image distributions.

We did not prove that the certified finite-basis conditions are expressive enough for full ImageNet generation.

We did not yet run diagnostics on actual paper encoder feature tensors.

What we have is a rigorous theorem/diagnostic layer, not a replacement for the paper’s empirical training pipeline.

---

## 19. Why this is still valuable

The project answers a foundational question:

```text
When is zero drift actually a certificate of distribution matching?
```

The answer is:

```text
Yes, under explicit frame/nondegeneracy, normalization,
probability, and estimator-consistency assumptions.
```

It also identifies the main practical bottlenecks:

- interaction-frame conditioning;
- feature-map injectivity or measure-determining stability;
- sample complexity for certification;
- CFG nonnegativity;
- modified-field rather than bare-field estimation.

This gives a roadmap for improving or auditing Drifting Models:

```text
Do not just ask whether the drift loss is small.
Ask whether the induced interaction matrix is well conditioned.
```

---

## 20. What would be the next empirical step?

The next high-value experiment is to run the real-feature diagnostic on actual encoder features.

The project now includes:

```text
numerics/real_feature_diagnostics.py
```

It expects externally supplied feature tensors:

```text
[num_samples, feature_dim]
```

and reports:

- softmax effective sample size;
- bare interaction-matrix rank and singular values;
- column-reweighted interaction-matrix rank and singular values;
- finite dual-certificate constants;
- random frame-violation sanity checks.

This would answer:

```text
Are the paper’s learned encoder features well-conditioned
for our identifiability theorem?
```

The paper release provides code and MAE feature extractor checkpoints, but not precomputed feature arrays in this workspace. So the remaining engineering step is to extract `.npy`/`.npz` feature tensors from the released encoder.

---

## 21. Suggested closing slide

The paper showed that Drifting Models work impressively well.

Our project asked:

```text
When does the drift objective have the right zeros?
```

We proved:

```text
Under explicit finite-basis frame conditions,
zero population drift identifies the target distribution.
```

We also showed:

```text
Algorithm 2 targets a column-reweighted field,
feature-space equality needs lifting assumptions,
CFG is generally affine/signed,
and certification is much stricter than training.
```

The result is a rigorous foundation and diagnostic toolkit for Drifting Models, not yet a performance replacement for the original model.

---

## 22. One-minute verbal summary

The Drifting Models paper gives a compelling training algorithm and strong ImageNet results. Its intuitive equilibrium claim is that the drift should vanish when the generated distribution matches the data. We investigated the reverse direction: if the drift vanishes, does that force distribution matching?

We formalized the appendix finite-basis argument in Lean. The key condition is that the pairwise interaction vectors form a nondegenerate, quantitatively well-conditioned frame. Under that condition, zero drift forces all coefficient minors to vanish, and probability normalization gives equality of the distributions. We also proved stability: small finite probe drift controls coefficient error.

Then we connected this population theorem to a fixed-anchor/sample-split form of the estimator. The main surprise is that this form does not generally estimate the bare population field; it estimates a column-reweighted field. We built the corresponding theorem for that modified field, proved self-normalized importance-sampling consistency, handled the implementation mask as a deleted-estimator perturbation, and repaired the main denominator-floor looseness with a high-probability denominator-tail theorem. The paper's exact reused-negative random-anchor coupling remains open.

Finally, we formalized two important guardrails. In feature space, the safe conclusion is equality of feature laws, not source laws, unless the feature map is measure-determining. And with CFG, the target is generally affine or signed, not necessarily a probability distribution.

So the contribution is not better FID. It is a rigorous, audited answer to when drift minimization really identifies the target, plus diagnostics that show where the practical bottlenecks are.
