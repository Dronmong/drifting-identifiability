# ASFD mechanism and architecture audit

**Subject:** [`AnchoredSelfFeatureDriftingResearchPlan.md`](AnchoredSelfFeatureDriftingResearchPlan.md)
**Date:** 2026-08-03
**Status:** analysis only; no experiment authorized, no plan file modified

---

## 0. Verdict

The plan is **structurally valid and unusually well-disciplined**, and its
central architectural decision — keep an independently squared raw Laplace
energy with a strictly positive coefficient so that correctness never routes
through the learned feature map — is correct and is the right way to import
deep-kernel machinery into this project.

Three defects are load-bearing and should be repaired before implementation:

| # | defect | severity |
|---|---|---|
| **1** | §7.1's gradient projection dissolves the identifiability anchor it is meant to protect | **high** |
| **2** | §7.1's shared auxiliary cap gives the two Stage-D arms unmatched raw dosage | **high** |
| **3** | §5.6's three radii are all broad, inheriting the exact blindness that produced B2's rank collapse | **high** |

Three more are significant:

| # | defect | severity |
|---|---|---|
| **4** | §5.2's two feature levels are adjacent blocks in an architecture with no bottleneck | medium-high |
| **5** | §5.2's layer choice contradicts the directly analogous ablation (TFD) and follows a less analogous one | medium-high |
| **6** | §1's coverage protector is raw-pixel while the pressure it must resist is feature-space | medium-high |

And a measurement finding that changes how the incumbent should be read:

> **B2 drove its held-out raw drift energy *below* the real-versus-real
> sampling floor** (13.933 against a floor of 14.103, B2.5 unit 500). A
> correctly distributed sample cannot beat that floor. The plan's §3.2 premise
> and B2.5's §6 condition 1 both reward reproducing this.

Details, evidence, and concrete repairs follow. Everything cited from this
repository was read from the artifacts, not from prior summaries.

---

## 1. Source verification — all five 2026 arXiv IDs are real

I could not confirm several 2026 arXiv IDs when I reviewed
[`MeanFlowDriftingMechanismAnalysis.md`](MeanFlowDriftingMechanismAnalysis.md)
(§ "Unverified at the time of writing"). Three of this plan's load-bearing
citations resolve cleanly, and one of them substantially changes the analysis:

| citation | resolves | matches the plan's use? |
|---|---|---|
| `2605.07327` Teacher-Feature Drifting | ✅ Zhang et al. | yes, and it contains ablations the plan should have used |
| `2605.10727` Kernel-Gradient Drifting | ✅ Esteban-Casadevall et al. | yes — identifiability for characteristic kernels, Riemannian/Fisher-Rao extensions |
| `2603.14366` PixelREPA | ✅ Shin et al. | direction yes, mechanism no (§9 below) |

That the sources are real is good news for the plan and bad news for §5.2 and
§5.6, because TFD's published ablations **contradict two of the plan's frozen
choices** (§7, §8).

TFD's verified content, for the record:

- features from **five** levels — encoder block 6, encoder block 11,
  bottleneck, decoder block 7, decoder block 12 (ImageNet-64); for SDXL,
  `down_blocks.2`, `mid_block`, `up_blocks.0`;
- ablation (Fig. 4b): "combining intermediate and deep teacher representations
  is important for stable performance"; **deep-only performed worse**; five
  levels beat three;
- features at `x_tf = x + σ_tf·ξ` with **σ_tf = 0.1**, selected by ablation
  (Fig. 4a): 0.02 converged slower to a worse FID, 0.5 also degraded;
- multi-radius `{0.02, 0.05, 0.2}`, best in Table 4, "supporting the use of
  averaged multi-radius drifting";
- average pooling at size 4, spatial structure retained, not globally pooled;
- an **anchor-margin coverage loss is essential** (Fig. 3): without it "the
  one-step generator tends to produce visually similar samples within the same
  class, leading to missing-mode behavior";
- results FID 1.58 on ImageNet-64, 18.4 on SDXL.

---

## 2. What the plan gets right

Recorded first, because most of what follows is criticism and the balance
matters.

**The separation of E1/E2/E3 in §2 is the correct decomposition**, and the
refusal to call an architecture containing hidden states "encoder-free" is the
right call. The claim actually made — *no external or separately pretrained
representation is needed, and exact population correctness does not rely on the
self-feature map being injective* — is defensible and is what the repository can
support.

**§3.1's identifiability argument is sound as stated**, and I verified both
Lean references:

- [`LaplaceEuclideanConverse.lean:29`](../DriftingIdentifiability/LaplaceEuclideanConverse.lean#L29)
  gives `laplaceZeroDrift_identifies_euclidean` with hypothesis
  `ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q` and no side conditions.
  `ZeroDrift` is genuinely **pointwise** (`intro x` / `h x`), not a.e. — the
  repository keeps a separate `ZeroDriftAE`, so the distinction is tracked.
- The energy→pointwise bridge therefore needs the probe law to have full
  support on ℝⁿ. B2's probe law is target + N(0, 0.05²), a Gaussian
  convolution of a compactly supported measure, which does have full support.
  ✅ The premise is met, not assumed.
- [`FeatureSpaceIdentifiability.lean`](../DriftingIdentifiability/FeatureSpaceIdentifiability.lean)
  does contain the collision results the plan cites (`featureLaw_dirac_eq_of_collision`,
  `featureLaw_collision_distinct_source_diracs`, and the
  `source_eq_of_*_of_embedding` family that shows what injectivity buys).

**Separate squares before averaging (§1, §5.5) is correct and non-obvious.**
Squaring an aggregate field permits two wrong fields to cancel; the zero set of
a sum of squares is the intersection of the zero sets. The plan states this,
tests it (§10, "a counterexample showing aggregate-then-square can cancel"), and
extends it across locations as well as levels — `E_self` is a mean of squared
norms over `(j, r, ℓ, z)`, so cancellation is blocked on every index. This
matches [`stage_b2/core.py:232`](encoder_independent_drifting/stage_b2/core.py#L232)
("never square the mean field").

**§5.1's warning about `torch.no_grad()` is the single most important
implementation note in the document.** Freezing parameters and blocking the
input Jacobian are different operations, and getting this wrong yields a
silently dead branch that still trains, still logs a decreasing loss, and still
produces an artifact. Flagging it in the design rather than discovering it in
preflight is right.

**§9's cost claim is arithmetically correct**, which I checked because the plan
asks for it to be measured rather than assumed. At width 512, one trunk forward
on 64 images at 256 tokens and depth 12 is ≈ 8×10¹¹ FLOPs; the kernel work over
66 locations × 2 levels × two 64×64 distance matrices in 512 dims is ≈ 10⁹ —
about 0.1%. Distances are shared across the three radii (only the softmax
temperature changes), which the implementation should exploit and the plan
should say. The real cost is the differentiable frozen-trunk forward/backward on
generated samples, ≈ 1 extra training step's work at 1-in-10 cadence, so
**+10–15% wall time**. The estimate holds.

**The Stage A → B → C → D → E ordering and §11's decision tree are correct**,
including the refusal to rescue a failed feature audit with DINO/CLIP. §14's
"do not implement the semantic correction first" is the right instinct and is
consistent with how B3 closed the capacity confound.

---

## 3. Defect 1 — the gradient projection dissolves the anchor it protects

### 3.1 The mechanism

§7.1 specifies: *"Project each auxiliary component so it does not oppose g₀ to
first order, then cap the norm of the combined auxiliary update at
`0.25 * ||g0||`."*

§3.1 separately argues that identifiability survives because
`L_corr = 0 ⟹ E_raw = 0 ⟹ p = q` **as long as λ_raw > 0**.

These two sections are about different objects and the plan does not connect
them. §3.1 reasons about a **loss**. §7.1 applies an **update direction that is
not the gradient of that loss, or of any loss.** Projection-based gradient
surgery produces a state-dependent, generally non-conservative vector field;
there is no scalar potential whose stationary points the dynamics seek. At a
fixed point of the projected dynamics you have

    g₀ + Σᵢ Π(λᵢgᵢ, g₀) = 0,

which implies nothing whatsoever about `E_raw`. The positivity of λ_raw is
doing no work, because λ_raw is not the coefficient that is applied.

### 3.2 Why this is worse than a generic rigor complaint

The projection does not remove a random part of the raw correction. It removes
**exactly the informative part**.

The raw anchor is redundant precisely when it agrees with the EMF gradient — if
both want the same parameter move, the anchor changes nothing. The anchor does
work only when it disagrees. §7.1 deletes the disagreeing component and keeps
the agreeing one. The term is therefore loudest in the logs exactly when it is
doing least, and silent when it would matter.

§7.1 anticipates the symptom — *"Log post-projection and post-cap component
shares so an apparent win cannot conceal that the raw anchor was numerically
erased"* — but treats it as a reporting obligation. It is a validity problem.
Logging that the anchor was erased does not un-erase it, and §3.1's claim is
stated unconditionally.

### 3.3 This is a regression against the validated stack

B1, B2, and B2.5 all use a **plain weighted sum**. B2.5's protocol §1 is
explicit that sequential backward passes are used only to avoid retaining two
trajectory graphs, and that *"gradient linearity makes this the gradient of
L = L_flow + λ₁L_B1 + λ₂L_B2"* — a real loss with real stationary points.

ASFD introduces projection as a new, unvalidated mechanism, and introduces it in
the one place where the repository's only formal claim lives.

### 3.4 Repair

Drop the projection. Use a weighted sum, as B1/B2/B2.5 do. If gradient conflict
is a real concern, treat it the way this repository has treated it everywhere
else — **as a measurement and an abort criterion, not a correction**. §7.2
already logs all pairwise cosines; predeclare a threshold on the sustained
`cos(g₀, g_r)` and `cos(g₀, g_s)` and abort the arm if it is violated, rather
than silently repairing it every step.

If projection is retained despite this, then §3.1 must be rewritten to say that
the identifiability implication applies to a loss the optimizer does not
descend, and the claim ledger in §12 must lose the anchor argument.

---

## 4. Defect 2 — Stage D's arms have unmatched raw dosage

### 4.1 The arithmetic

§7.1 sets event-level unprojected ratios of 0.15 (B1), 0.10 (raw), 0.10
(semantic), and caps the **combined** auxiliary at `0.25·||g0||`. §7.1 then
says the incumbent *"uses the same B1/raw coefficients and the same final
combined cap."*

| arm | nominal auxiliary | cap | realized raw share (aligned case) |
|---|---:|---:|---:|
| `EMF-raw` | 0.15 + 0.10 = 0.25 | 0.25 | 0.100 |
| `EMF-ASFD` | 0.15 + 0.10 + 0.10 = 0.35 | 0.25 | **0.071** |

Whenever the cap binds, adding a third component under a fixed total budget can
only shrink the other two. The ASFD arm therefore runs at roughly **29% less
raw anchor and 29% less B1** than the arm it is compared against.

A Stage-D difference between `EMF-raw` and `EMF-ASFD` then confounds *"added a
semantic term"* with *"reduced the raw and spectral dose by about a third."*
And the confound points the wrong way for the plan's own safety story: the arm
carrying the new, unvalidated pressure is also the arm with the weakest
protection against it.

### 4.2 The repository already decided this question

B2.5's protocol §2 states the principle in terms that apply verbatim:

> The combined cell is **not interleaved**. Both corrections fire every ten
> steps, exactly as in their single-correction cells. This is required for a
> true `2 × 2` factorial: "B1 present" and "B2 present" have the same treatment
> level in the single and combined cells.
>
> Consequently the full combined cell can have greater correction compute and a
> larger total correction gradient than either single arm. That is not a
> confound in the factorial interaction; it is the defined joint treatment.

A shared total cap is the norm-space equivalent of the interleaving that B2.5
rejected. It was rejected for the right reason and the rejection should carry.

### 4.3 Repair

**Cap each component separately** — 0.15, 0.10, 0.10 — and let the total float
to 0.35 in the ASFD arm. "Raw present" then means the same thing in both arms
and the contrast is clean. Report the realized total as a Pareto cost, exactly
as B2.5 §2 does.

If a hard total ceiling is genuinely required by stability, add a third
development arm `EMF-raw-diluted` at ASFD's realized raw/B1 dose, so the
dilution can be subtracted. That costs a run; per-component caps cost nothing.

---

## 5. Defect 3 — the radii are all broad, and that is the mechanism behind B2's rank collapse

This is the finding I have most confidence in, because the repository's own
artifacts measure it.

### 5.1 The Laplace barycenter is a strong contraction

The field is a difference of two normalized barycenters. At B2's frozen
τ = 7.085 and realized ESS ≈ 0.58 over 64 positives, each barycenter averages
≈ 37 samples. From
[`stage_b2/b2_confirmatory.json`](encoder_independent_drifting/stage_b2/b2_confirmatory.json):

| quantity | value |
|---|---:|
| probe↔positive distance median | 36.49 |
| real-vs-real field L2 per probe (floor) | 3.196 |
| model field L2 per probe | 4.342 |

Typical inter-image distance is ≈ 36.5, but the *difference between two
independent real barycenters* is ≈ 3.2. The barycentric image of the data is
contracted by roughly an order of magnitude relative to the data itself.

That is the mechanism: **matching barycenters at bandwidth τ constrains only a
heavily smoothed statistic, and is structurally blind to distributional
structure finer than τ.** Many under-dispersed clouds share one barycentric
image. Descending the energy therefore has a cheap direction available —
contract — and nothing in the objective opposes it.

### 5.2 The measured signature of contraction is present in B2

Same artifact, model-health at audit events, all three confirmation units:

| unit | positive ESS med | **negative ESS med** | pos dist med | **neg dist med** |
|---|---:|---:|---:|---:|
| 300 | 0.581 | **0.666** | 36.49 | **32.98** |
| 301 | 0.582 | **0.664** | 36.46 | **33.02** |
| 302 | 0.581 | **0.682** | 36.47 | **32.81** |

Generated samples sit **closer to real probes than real images sit to each
other**, and the weights over them are **flatter** than over real positives.
Both are contraction signatures, both are consistent in all three units, and
negative-side ESS rises across units (0.666 → 0.682, and to 0.702 at the later
audit replicate). B2's 38–40% effective-rank loss is not a mysterious
pathology; it is what descending a bandwidth-limited barycenter discrepancy
does.

### 5.3 B2 went below the sampling floor

From [`stage_b25/b25_unit_500.json`](encoder_independent_drifting/stage_b25/b25_unit_500.json),
in-domain, step 30 000. `floor_negatives` are disjoint **real** images
substituted for generated ones at identical shapes, so the floor is a genuine
finite-sample floor:

| cell | raw energy | real-real floor | excess |
|---|---:|---:|---:|
| B0 | 20.904 | 14.103 | +6.801 |
| B1 | 16.151 | 14.103 | +2.048 |
| **B2** | **13.933** | 14.103 | **−0.170** |
| B1B2 | 15.631 | 14.103 | +1.528 |

**A correctly distributed sample cannot beat the real-versus-real floor.**
Scoring below it means the generated cloud produces less field discrepancy than
an independent real sample does — i.e. it is less variable than real data in
exactly the direction the estimator measures. That is estimator exploitation,
and it is the same event as the rank collapse.

Two consequences the plan should absorb:

1. **§3.2's premise is stated in the wrong direction.** It reads B2's rank loss
   as a limitation of raw-pixel geometry that a semantic branch could repair.
   The measurements say it is a limitation of *broad-bandwidth barycenter
   matching*, which a semantic branch at broad bandwidth reproduces.
2. **B2.5 §6 condition 1** ("retains at least 80% of B2's nonnegative raw-drift
   reduction from B0") **rewards reproducing the artifact.** A floor-relative,
   two-sided criterion is correct: the energy should approach the floor from
   above, and undershooting it should count against, not for.

### 5.4 The signal-to-noise consequence, and a nearly free fix

At 64-vs-64, the floor is 54% of B2's total measured energy (10.21 of 18.85)
and 67% in B2.5 (14.103 of 20.904). **Most of what the correction differentiates
is sampling noise, not distributional discrepancy.** The floor falls roughly as
1/n, so moving the field batch 64 → 256 should cut it ≈ 4× and lift the
signal-to-floor ratio from ≈ 0.85 to ≈ 3.3.

For ASFD this is close to free on the positive side, because §5.7 already
caches target features — a larger positive bank costs storage, not compute.
The roles need not be symmetric: only the negative batch costs a generator
forward. **Asymmetric batches (large cached positives, moderate negatives) are
the highest value-per-dollar change available to this design**, and they also
relax the constraint in §5.5 below.

### 5.5 The radii themselves

§5.6 predeclares `R = {0.35, 0.60, 0.85}` as target median ESS fractions. All
three are broad. At 64 positives these are ≈ 22, ≈ 38, and ≈ 54 effective
neighbours. There is **no local regime in the set**, so all three fields are
blind to sub-bandwidth structure in the same way and averaging them does not
repair it.

Compare TFD's verified `{0.02, 0.05, 0.2}` — a **10× span**, deliberately
spanning local to global. The plan's set spans well under 2× in τ. It is
labelled multi-radius but is close to three redundant broad fields.

§5.6's own health floor (5th-percentile ESS ≥ 0.10, 95th-percentile max weight
≤ 0.50) then creates a constraint the plan never surfaces: **at 64 samples,
ESS fraction 0.10 is 6.4 neighbours, so the batch size bounds how local the
smallest radius can be.** The radius set and the batch size are coupled and
must be chosen together.

### 5.6 Repair

- Enlarge the cached positive bank per event (256+); keep the negative batch at
  64 if compute requires.
- Re-space the radii to span a genuine range — e.g. `{0.12, 0.40, 0.85}` — with
  the smallest chosen as the most local value that clears the §5.6 health floor
  at the chosen batch size. Preregister the floor-clearing procedure, not the
  numbers.
- Apply the same reasoning to the **raw** branch, which has the same defect and
  three units of evidence for it. A single broad raw radius is what produced
  the collapse; the raw energy should be multi-radius too.
- Make the gate floor-relative and two-sided on both branches.

---

## 6. Defect 4 — there is no bottleneck, and the two levels are adjacent

§5.2 declares two levels: *"`bottleneck`: output of the last encoder block"* and
*"`early_decoder`: output of the first decoder block after its long-skip
fusion."*

I checked the architecture in
[`stage_pmf_r/model.py`](encoder_independent_drifting/stage_pmf_r/model.py):

```
half = config.depth // 2          # line 43
self.skip_fusions = ...           # line 50
...
skips.append(tokens)              # line 125, per encoder block
for fusion, block, skip in zip(self.skip_fusions, self.decoder, reversed(skips)):
    tokens = fusion(torch.cat((tokens, skip), dim=-1))   # line 130
```

This is a **U-ViT with long skips and no spatial downsampling**. Every block
runs at the same width and the same token count. CAP-EMF-1 (S3R §10.2) keeps
this: width 512, depth 12, patch 2, 256 tokens throughout.

Two consequences:

1. **"Bottleneck" is a borrowed name for something that does not exist here.**
   In a convolutional U-Net the bottleneck is spatially compressed and
   semantically abstract, which is why the self-perceptual paper found the
   midblock useful. In this trunk, block 6 has the same shape as block 1. The
   plan imports the intuition without the architecture that produces it.
2. **The two chosen levels are blocks 6 and 7 of 12** — adjacent, separated by
   one transformer block and one skip fusion. Their features will be strongly
   correlated. Squaring separately prevents cancellation between them but does
   not buy information that is not there; §5.5's average over `j` will be close
   to a single level at double weight.

§6.2.6 checks the semantic field against the **raw** field (`|cosine| < 0.995`)
but **there is no check that the two feature levels are non-redundant with each
other.** That is the gap most likely to make the whole branch a no-op.

**Repair:** add an inter-level redundancy check (field cosine or feature-space
CKA between levels, with a preregistered ceiling), and spread the levels — see
§7.

**Also, an implementation trap:** the current trunk prepends two conditioning
tokens (`torch.cat((time_token, interval_token, patches), dim=1)`, line 121), so
the image grid is `tokens[:, 2:]`, not `tokens`. CAP-EMF-1 §10.2 proposes
switching to AdaLN-Zero conditioning, in which case there are exactly 256 tokens
and no slice. The extraction code must be written against whichever lands.
**§6.1.1's parity test will not catch an off-by-two token slice**, because it
only asserts that ordinary model outputs are unchanged — and they are, since
extraction is a read-only hook. Add a direct assertion that the extracted grid
reshapes to 16×16 and that its spatial autocorrelation is nontrivial.

---

## 7. Defect 5 — the layer selection follows the less analogous source

§5.2 justifies two levels by citing the self-perceptual diffusion paper: *"the
midblock was better than indiscriminately summing every layer"*, and adds *"Do
not add every layer. Both the self-perceptual and TFD ablations show that
feature selection matters and that more layers are not monotonically better."*

The second sentence misreports TFD. TFD's Fig. 4b finds the **opposite** for
this use: five levels spanning encoder-6, encoder-11, bottleneck, decoder-7,
decoder-12 beat three-level variants, deep-only performed worse, and
*"combining intermediate and deep teacher representations is important for
stable performance."*

The plan is choosing between two sources that disagree, and it picks the one
whose task is further away:

| | self-perceptual (Lin & Yang) | TFD (Zhang et al.) |
|---|---|---|
| objective | paired perceptual **regression** | **drifting** field over features |
| model | multi-step diffusion | one-step generator |
| what the layer choice governs | which activations to regress toward | which feature spaces carry a **distributional** field |

ASFD is the second thing. §4.4 already identifies TFD as "the closest work."
The layer decision should follow it.

There is also a mechanism reason the disagreement is not arbitrary. A
regression loss on many layers over-constrains, because each layer adds a
separate pointwise target for the same sample. A **distributional** field over
many layers does not: each level contributes an independent nonnegative
discrepancy whose zero set contains the true one, so extra levels tighten the
intersection rather than over-determining a point. That is the same argument the
plan uses for separate squares — it just points toward more levels, not fewer.

**Repair:** use three or four levels spanning the trunk, e.g. blocks 3, 6, 9,
12 (post-fusion where applicable). Cost is near zero: positives are cached, and
the generated-side backward through the trunk is shared across all levels —
only the kernel work scales, and §9's own arithmetic puts that at ~0.1% of a
forward. Then let §6.2 and the inter-level redundancy check decide which
survive, on target-only data.

---

## 8. Defect 6 — the coverage protector is in the wrong space

§1 point 5 relies on *"the existing spectral anchor and gradient protection to
prevent the semantic branch from buying apparent fidelity by losing coverage."*

B1 is a random-Fourier-feature characteristic-function criterion in **raw pixel
space** ([`b1.py:3`](encoder_independent_drifting/b1.py#L3)):
`A(p,q) = E_w |φ_p(w) − φ_q(w)|²`. Its own docstring records the limit: *"The
finite random-feature V-statistic used here is only a nonnegative stochastic
optimization surrogate; it is not itself measure determining."*

Three facts sit badly together:

1. **TFD reports its feature-space coverage term is essential** (Fig. 3):
   without it the one-step generator produces *"visually similar samples within
   the same class, leading to missing-mode behavior."* That is a
   semantic-space collapse.
2. **B1 has never been tested against a feature-space pressure.** It was
   validated as non-inferior to B0 on coverage, and in B2.5 it recovered rank
   against a **raw-pixel** drift term. Transferring it to a different space is
   an assumption.
3. **A within-class semantic collapse is close to the worst case for a
   raw-pixel RFF criterion.** Samples that vary in colour, background, and
   texture but repeat one car pose can keep the pixel-space characteristic
   function close to the target's while collapsing semantically.

§11's decision tree does list this as a failure branch — *"If semantic energy
falls but rank/coverage falls... a later experiment may test TFD-style anchor
margin."* Given that TFD's ablation reports the term is necessary, and given
that this repository measured a 38–40% rank collapse from a term dosed at
λ = 1.9×10⁻⁴ on a 1-in-10 cadence, the no-coverage-term configuration is not a
neutral first experiment — it is the configuration most likely to fail, for a
reason already published and already measured locally.

The counterargument is real and should be recorded: TFD's drifting loss is the
**primary** objective, whereas ASFD's semantic term is auxiliary, capped, and
cadence-averaged near 2.5%. A pressure that dominates training is not a pressure
that contributes 2.5%. But B2's own dosing was comparably light and the rank
loss was 38–40% anyway, so light dosing is not the protection it appears to be.

**Repair (pick one, preregistered):**

- **(a)** Add a feature-space coverage term to the first ASFD arm. Costs the
  one-factor-at-a-time property; buys a first experiment that is not a
  predicted failure.
- **(b)** Keep the clean single factor, but add **feature-space** effective
  rank and within-class nearest-neighbour similarity as *monitored abort
  criteria* with preregistered thresholds — not merely as reported metrics —
  and predeclare the anchor-margin arm as the immediate follow-up rather than
  "a later experiment."

(b) preserves the plan's discipline and is what I would choose. Either way,
§8's Stage D gate should measure rank in **feature space** as well as raw
pixels; the current gate's *"retains at least 90% of the incumbent effective
rank"* is raw-pixel only, and would not see the failure TFD describes.

---

## 9. Defect 7 — PixelREPA is cited for the wrong risk

§4.5 reads PixelREPA as showing *"that forcing a pixel-space generator toward a
compressed semantic target can worsen FID and collapse diversity"*, and adds
four safeguards against it.

The verified paper says something more specific: REPA fails for JiT because of
an **information asymmetry** — denoising happens in high-dimensional image space
while the semantic target is heavily compressed — and PixelREPA *repairs* it
with a Masked Transformer Adapter (shallow adapter + partial token masking),
improving JiT-B/16 FID 3.66 → 3.17.

Two corrections follow:

1. **The named mechanism largely does not apply to ASFD.** The compression is
   the problem, and ASFD's descriptor is not compressed: 66 vectors × 512
   channels ≈ 33 792 numbers against a 3 072-dimensional image. The descriptor
   is an order of magnitude *larger* than the input. §4.5's four safeguards are
   aimed at a risk this design mostly does not carry.
2. **The paper that supplies the warning also supplies a fix**, and the fix —
   partial token masking with a shallow adapter — is cheap and directly
   applicable. It appears only in passing in §11 as a possible later
   experiment.

This matters because it means the plan's risk budget is misallocated: it is
defending hard against compression-induced collapse, which is unlikely here,
and lightly against bandwidth-induced contraction (§5) and self-referential
blindness (§10), which are likely.

---

## 10. Defect 8 — self-referential blindness, and a one-sided audit

### 10.1 The mechanism

`h_{θ*}` is a frozen copy of the generator's own EMA trunk, and the fork
continues from the same `θ*`. At the moment the semantic branch switches on,
the feature map and the generator are the **same function**.

The trunk allocates representational capacity according to what its training
objective rewards — direct-`x` regression under MSE — and MSE is dominated by
low frequencies. So the trunk's features systematically underweight exactly the
bands where an MSE-trained generator is weakest. The feature geometry is
therefore least sensitive in the directions where the foundation most needs
correction.

This is not speculative for this project. S3R's EMF arm measured final raw Haar
variance ratios of LL 0.509, LH 0.410, HL 0.512, **HH 0.159** — the known defect
is diagonal high-frequency detail. CAP-EMF-1's no-loop rule (S3R §10.5)
explicitly admits *"recognizable but detail-poor"* as an outcome that still
proceeds. If the foundation lands there, its trunk's features will be weakly
sensitive to HH, and the semantic branch cannot supply what the metric cannot
see.

A partial mitigation exists and should be recorded: the trunk is trained on
**real** noised images, so its features do encode real-image structure, not only
the generator's output manifold. This is a weaker version of the "student
teaching itself" problem than it first appears — but it is not absent, and the
band argument above survives it, because it is about capacity allocation under
MSE, not about which images were seen.

### 10.2 The audit is one-sided in the wrong direction

§6.2.5 requires: *"the response to small high-frequency noise is not more than
four times the response of a normalized raw-pixel control."*

That is an **upper** bound only. It guards against the hypersensitivity seen in
the pretrained-ResNet harness of Phases 17–18. It has **no lower bound**, so a
feature map with *zero* high-frequency sensitivity passes it trivially — and
zero HF sensitivity is the failure this architecture is most likely to have.

**Repair:** make the check two-sided and per-band. Inject fixed-energy
perturbations in each Haar band (LL/LH/HL/HH), measure the feature-distance
response relative to the raw-pixel control, and require the ratio to lie in a
preregistered interval — e.g. `[0.25, 4.0]` — in **every** band. Report the
profile against S3R's Haar table so the trunk's blind spots are visible before
any training is spent. This is cheap, target-only, and directly tests the
mechanism the whole branch depends on.

**A second decorrelation option**, if budget allows: extract features from a
trunk checkpoint that is *not* the fork point — an earlier checkpoint, or an
independently seeded foundation. This decorrelates the feature map's blind spots
from the generator's, at the cost of a checkpoint or a second run. Worth
recording as the fallback if §6.2's band profile comes back degenerate.

---

## 11. On `t_f = 0.10` — the plan is right, and should still measure it

I expected this to be a defect and it is not.

§5.3 sets `x_tf = 0.9x + 0.1ξ` evaluated at absolute time 0.10, interval 0.
TFD's verified ablation selects `σ_tf = 0.1` from `{0.02, 0.1, 0.5}` on FID.
Up to the 0.9 signal scaling — which is *required* here, because this trunk is
time-conditioned and 0.9x + 0.1ξ is exactly what it expects at t = 0.10 — the
noise-to-signal ratios agree. The plan's construction is the in-distribution one
for a flow trunk and matches the validated value for a diffusion trunk. §4.3's
reasoning from CleanDIFT is also correct: a clean t = 0 feature map would be an
unjustified default for a time-conditioned trunk.

Two things are still worth saying.

**The value was selected by FID ablation on a different kind of model.** TFD's
teacher is a multi-step diffusion model; this trunk is a one-call direct-`x`
EMF model. The transfer is an assumption, and the plan cannot re-ablate on FID
without touching held-out quality.

**But it can select `t_f` legitimately.** §6.2's geometry audit is target-only
and outcome-blind — the same status as the ESS bandwidth calibration the plan
already performs. Running §6.2 over a small `t_f` grid and selecting on
benign-vs-random AUC and the band profile of §10.2 is calibration, not tuning,
and it does not create the interpretability hazard §5.3 is worried about (which
is about varying `t_f`, layers, and kernel *together* in the training
experiment). Freeze one `t_f` for the run; choose it with a measurement rather
than an analogy.

**One tension the plan does not name.** Along `t_f`, injectivity and semantic
abstraction pull in opposite directions. At small `t_f` a direct-`x` trunk's
task is near-identity, so hidden states are near-invertible codes — maximal
injectivity, minimal abstraction. At large `t_f` the trunk must use global
context — more abstraction, more collision risk. ASFD wants abstraction from the
semantic branch and relies on the raw branch for correctness, so it can afford
to sit further toward abstraction than an injectivity-seeking design would.
Naming the tradeoff makes `t_f` selection principled rather than inherited.

---

## 12. Smaller items

**§5.4's single scalar normalization per level.** `S_j` is one number shared
across all locations in a level. Transformer hidden states routinely have a few
outlier channels that dominate the L2 norm, in which case the "semantic" Laplace
distance is effectively a 1–2 dimensional statistic. §6.2.3 partly guards this
(effective rank ≥ 16, no PC above 50% variance) — but it is specified on the
**global** mean/std descriptor, and 64 of the 66 vectors per level are local
tokens. Apply the same rank and PC checks to the local-token descriptor, and
predeclare the remedy (frozen per-channel target standardization) if PC1 exceeds
the ceiling at token level.

**Position-locked local fields, and the flip problem.** §5.5 vectorizes
locations as a batch dimension, so token (3,5) is only ever compared with token
(3,5). 64 of 66 vectors per level — **97% of the semantic energy** — are
position-locked. Under the horizontal flip that §5.7 records as an augmentation
bit, token (i,j) maps to (i, 15−j), so a flipped car reads as far from an
unflipped one at every local location. A substantial fraction of `E_self` will
measure pose and position rather than semantics.

Worse, §6.2.1 explicitly undoes known flips and translations *for the audit*
(*"Do not compare unregistered token locations, which would mistake a coordinate
permutation for a semantic failure"*) — but the **training** field gets no such
registration. The audit therefore measures the feature map under conditions the
objective never enjoys. At minimum, report the energy decomposition between the
2 global and 64 local vectors so the split is visible; consider reweighting
toward the permutation-invariant global descriptors, or a location-marginal
formulation.

**Cached-bank bias.** §5.7 caches two noised views per target image while
negatives are redrawn fresh at every event. Over ≈ 16 000 events the positive
barycenter is a *fixed* random function whose deviation from the population
barycenter is a fixed bias the generator can partly match. The Stage-D gate
correctly evaluates on a fresh bank, so the failure is caught — but §7.2's
mandatory diagnostics should log the **train-bank minus fresh-bank energy gap**
at every checkpoint, since that gap is the direct measurement of the bias and is
currently not on the list.

**Negative-side ESS is uncalibrated.** [`stage_b2/core.py:214`](encoder_independent_drifting/stage_b2/core.py#L214)
already records `positive` and `negative` weight health separately — good — but
`B2Config` has no threshold on the negative side, and §5.6's health limits are
written against the target-only calibration. The generated side is the one that
degenerates (§5.2 above: 0.666 → 0.702). If negative ESS approaches 1, the
negative barycenter approaches a plain batch mean and `E_self` silently
degenerates to first-moment matching. **Gate it at runtime**, per level and
radius, in both branches.

**Stage E has no decision rule.** §8's Stage D has a seven-condition gate;
Stage E says only *"Report:"*. Every confirmation in this repository has had a
preregistered k-of-n rule — B0 3/3, B1 3/3, B2 2/3. Stage E should have one, and
it should account for its own weakness: two units against a **1 000-image**
sealed class-test split, where B0/B1/B2 used 2 048-sample references and three
units and were still described as *"coarse consistency, not high-powered
inference."* Seven simultaneous conditions across two units is a rule a
genuinely better method can easily fail.

**Novelty framing.** §13's assessment is appropriately hedged and, now that TFD
and Kernel-Gradient Drifting both verify, correctly scoped: the intersection
that remains open is *self-founded* rather than teacher-distilled feature
drifting, with an independent identifying source-space anchor. §12's claim
ledger does not overreach. No change needed.

---

## 13. What this implies for ordering

The plan's §14 order is right in shape. Two amendments.

**Finish B2.5 before spending rented-GPU budget.** Unit 500 is complete and
units 501/502 are paused at ~11 h local. It is the only experiment that
measures whether the spectral anchor and the raw drift term are complementary,
which is the exact question ASFD's §3.2 premise depends on. Its one completed
unit is already informative and is **not** what a summary of "B1B2 looks good"
would suggest:

| cell | recall | precision | KID | FID | rank | rank/B0 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.2012 | 0.836 | 0.0717 | 125.3 | 14.00 | 1.000 |
| B1 | 0.1973 | 0.814 | 0.0547 | 112.1 | 12.86 | 0.918 |
| B2 | 0.1738 | 0.822 | 0.0662 | 122.3 | 8.07 | 0.577 |
| **B1B2** | **0.2148** | 0.793 | **0.0529** | **110.7** | 11.11 | 0.794 |

B1B2 is best on recall, KID and FID. But against B2.5 §6's four preregistered
conditions it **fails two of four**: it retains 75.6% of B2's raw-drift
reduction (threshold 80%) and reaches rank/B0 = 0.794 (threshold 0.85). It
passes both appearance conditions. **Unit 500 is not "promising" under the rule
as written.** The factorial interactions are strongly positive
(`I_rank` = +4.18, `I_recall` = +0.0449, `I_rawE` = +6.45), so B1 does rescue
most of B2's geometry loss — just not to the declared bar, and at the cost of a
quarter of the drift reduction.

That is directly relevant twice over. It weakens §3.2's motivation (the rank
problem is ~20% with the anchor present, not 42%), and it is the only local
evidence on whether stacking correctness terms compounds or trades off — which
is precisely what ASFD proposes to do with a third term.

**Order the qualification gate to fail fast.** Within Stage B, run §10.2's
per-band sensitivity profile and §6's inter-level redundancy check *first*.
Both are cheap, target-only, and test the two assumptions the entire branch
rests on: that the trunk sees the bands the foundation is missing, and that the
chosen levels carry distinct information. If either fails, the plan's §11
"cancel the ASFD arm" branch fires before any bank construction or coefficient
calibration is spent.

---

## 14. Consolidated repair list

Ordered by expected value, not by section number.

| # | change | section | cost |
|---|---|---|---|
| 1 | Replace projection with a weighted sum; abort on outcomes, not on cosine (refined in **A.6**) | §7.1 | none |
| 2 | Per-component gradient caps (0.15/0.10/0.10), not a shared total | §7.1 | none |
| 3 | Re-space radii to span local→global; apply to the raw branch too | §5.6 | none |
| 4 | Enlarge the cached positive bank (256+); asymmetric negative batch | §5.7 | storage |
| 5 | Floor-relative, two-sided drift-energy gates on both branches | §8 | none |
| 6 | Three or four spread feature levels (blocks 3/6/9/12), not two adjacent | §5.2 | ~0.1% compute |
| 7 | Add an inter-level redundancy check to the qualification gate | §6.2 | negligible |
| 8 | Make §6.2.5 two-sided and per-Haar-band | §6.2 | negligible |
| 9 | Runtime gate on negative-side ESS, per level and radius | §5.6, §7.2 | negligible |
| 10 | Feature-space rank/NN-similarity as monitored abort criteria; predeclare the anchor-margin follow-up | §8, §11 | none |
| 11 | Select `t_f` on a small grid by the target-only §6.2 audit | §5.3 | one audit pass |
| 12 | Preregister a k-of-n Stage E decision rule | §8 | none |
| 13 | Log the train-bank vs fresh-bank energy gap | §7.2 | negligible |
| 14 | Local-token rank/PC checks; predeclare per-channel standardization remedy | §5.4, §6.2 | negligible |
| 15 | Report the global-vs-local energy split; note the flip/registration mismatch | §5.5, §6.2 | negligible |
| 16 | Assert the extracted grid reshapes to 16×16 (conditioning-token slice) | §6.1 | negligible |

Items 1–5 cost nothing and address the three high-severity defects. If only one
change is made, make it item 1: without it, the plan's central architectural
claim — that correctness never routes through the learned feature map — is not
implemented by the plan's own training procedure.

---

## 15. Summary

The proposal is well-founded. Its core insight is right, its literature is real
and mostly used correctly, its identifiability argument checks out against the
Lean sources, and its experimental discipline is consistent with how this
repository has operated.

The failures are concentrated in the step from mathematics to optimizer. §3.1
reasons about a loss; §7.1 applies something that is not that loss's gradient,
and the anchor the plan is built around is the first casualty. The dosing
scheme then makes the two arms non-comparable in the one quantity that matters.
And the radius set reproduces the exact blindness that the repository has now
measured three times, most sharply in B2 scoring *below* the real-versus-real
floor.

None of these require abandoning the design. All three are fixed by changes that
cost no compute.

**Appendix A** below is the revised specification: drop-in replacement text for
every affected section of the plan, written so that the plan is implementable
as amended.

---

# Appendix A — Revised specification

This appendix replaces the corresponding sections of
[`AnchoredSelfFeatureDriftingResearchPlan.md`](AnchoredSelfFeatureDriftingResearchPlan.md).
Sections not listed here are unchanged and stand as written. Nothing in this
appendix authorizes an experiment; it defines what would be run if one were
authorized.

Numbering matches the plan's own sections so the two documents can be read side
by side.

---

## A.0 A prerequisite the plan does not state: no frozen constant transfers

The plan's §1, §3.2 and §7.1 speak of retaining "the existing raw correction"
and "the existing spectral anchor." Read as a reuse of B1's and B2's frozen
numerical constants, that is wrong, and the error would be silent.

| constant | value | why it does not transfer |
|---|---:|---|
| B2 Laplace bandwidth τ | 7.085388360479058 | calibrated on all-class CIFAR-32 raw pixels, median pairwise distance 37.49. Automobile-only is a tighter cloud with a different median. |
| B2 event weight λ | 1.9294302093274076e-4 | calibrated to hit event gradient ratio 0.25 against the **F3B bridge's** flow-matching loss. EMF's loss has an unrelated gradient scale. |
| B1 event weight | 0.9310125645774651 | same reason. |
| B1 projected scale | 0.4299860893300136 | derived from bridge-scale activations. |

**What transfers is the *form* of each term and the *calibration procedure*,
never the numbers.** Every coefficient, bandwidth, and scale in ASFD must be
re-derived outcome-blind against CAP-EMF-1 on automobile-only data, and the
artifact must record that it did so rather than loading a B1/B2 freeze.

This also means the ASFD arms are a **new development configuration**, not a
continuation of a confirmed one. §12's claim ledger should say so.

---

## A.1 Replaces §5.2 — feature levels

### A.1.1 Naming

Delete the term `bottleneck`. The trunk is a U-ViT with long skips
([`stage_pmf_r/model.py:43–50, 123–130`](encoder_independent_drifting/stage_pmf_r/model.py#L43));
every block runs at the same width and the same token count, and there is no
spatially compressed stage. Calling block 6 a bottleneck imports an intuition
the architecture does not supply.

### A.1.2 Levels

Predeclare **four** levels spanning the trunk, mirroring TFD's verified
encoder/mid/decoder spread rather than the self-perceptual paper's single
midblock:

| label | tap point (depth 12) |
|---|---|
| `enc_mid` | output of encoder block 3 |
| `enc_final` | output of encoder block 6 |
| `dec_mid` | output of decoder block 9, **after** its long-skip fusion |
| `dec_final` | output of decoder block 12, **before** `final_norm` and `pixel_head` |

Rationale, recorded so the choice is falsifiable: TFD Fig. 4b finds that
combining intermediate and deep representations matters and that deep-only is
worse; five levels beat three. A distributional field over many levels does not
over-constrain the way a per-sample regression does, because each level
contributes an independent nonnegative discrepancy whose zero set contains the
true one — extra levels tighten an intersection rather than over-determining a
point. This is the same argument the plan already uses for separate squares.

The final level is taken **before** `final_norm`/`pixel_head` so the descriptor
is not a near-affine image of the output pixels, which would make it redundant
with the raw branch by construction.

### A.1.3 Per-level descriptor

Unchanged from the plan: reshape the 16×16 image tokens, 2×2 non-overlapping
average pool to 8×8 = 64 local vectors, plus one global channel-mean and one
global channel-standard-deviation vector. **66 vectors per level.**

### A.1.4 Token-grid extraction

The current trunk prepends two conditioning tokens
(`torch.cat((time_token, interval_token, patches), dim=1)`,
[`model.py:121`](encoder_independent_drifting/stage_pmf_r/model.py#L121)), so
the image grid is `tokens[:, 2:]`. CAP-EMF-1 §10.2 proposes AdaLN-Zero
conditioning instead, in which case there are exactly 256 tokens and no slice.

**Write the extraction against whichever conditioning CAP-EMF-1 freezes, and
assert it.** §6.1.1's parity test cannot catch an off-by-two slice, because it
only checks that ordinary model outputs are unchanged — and they are, since
extraction is a read-only hook. Add the assertions in A.10.

### A.1.5 Cost of four levels

Kernel work scales with levels; the trunk forward does not. At `n_z = 64`
probes, `n_p = 256` positives, `n_q = 64` negatives, `C = 512`:

- per level per location: `64×256×512 + 64×64×512 ≈ 10.5M` MACs
- × 66 locations × 4 levels ≈ **5.5 GFLOP**
- radii share the distance matrices; only the softmax temperature changes

Against a forward+backward of the frozen trunk on 64 images at 256 tokens,
depth 12, width 512 (**≈ 2 TFLOP**), the kernel work is **≈ 0.3%**. Four levels
and three radii are free relative to the extraction they ride on.

Peak activation memory is dominated by the frozen trunk's backward graph, not
the kernels: generated-side feature tensors are `4 × 64 × 66 × 512 × 4 B ≈
35 MB` and the distance matrices `≈ 4 MB` per level.

---

## A.2 Replaces §5.3 — feature noising and `t_f` selection

The construction stands: `x_{t_f} = (1−t_f)x + t_f ξ`, evaluated at absolute
time `t_f` and interval 0, with independent noise streams per role and no
positive/negative noise pairing. This is the in-distribution corruption for a
time-conditioned flow trunk, and up to the `(1−t_f)` signal scaling it matches
TFD's ablated `σ_tf = 0.1`.

**Change: do not freeze `t_f = 0.10` by analogy. Select it.**

Run §6.2's geometry audit — as amended in A.5 — over the grid

    t_f ∈ {0.05, 0.10, 0.20, 0.35, 0.50}

on **target-training images only**, and select the value maximizing a
predeclared composite of the audit statistics (benign-vs-random AUC, the
per-band sensitivity profile of A.5.2, and inter-level non-redundancy),
subject to every hard threshold passing. Freeze one value for the run and
record the full profile across the grid.

This is calibration, not tuning: it is the same status as the ESS bandwidth
calibration the plan already performs, it touches no held-out data, and it
does not create the interpretability hazard §5.3 is guarding against — which is
about varying `t_f`, layer set, and kernel *together inside the training
experiment*.

**Record the tradeoff being navigated.** Along `t_f`, injectivity and semantic
abstraction pull in opposite directions: at small `t_f` a direct-`x` trunk's
task is near-identity, so hidden states are near-invertible codes — maximal
injectivity, minimal abstraction; at large `t_f` the trunk must use global
context — more abstraction, more collision risk. ASFD wants abstraction from
the semantic branch and relies on the **raw** branch for correctness, so it can
afford to sit further toward abstraction than an injectivity-seeking design
would. Naming this makes the selection principled rather than inherited.

---

## A.3 Replaces §5.4 — feature normalization

The plan's single scalar per level, `S_j`, is retained as the second stage, but
is preceded by a conditional per-channel stage with a **predeclared trigger**,
so that outlier-channel domination cannot silently reduce the semantic metric
to a one- or two-dimensional statistic.

**Stage 1 — conditional per-channel scaling.** On the target-only calibration
allocation, compute the token-level principal-component variance share of level
`j`. If `PC1 > 0.35`, apply frozen per-channel scales

    s_{j,c} = max( σ_{j,c},  0.1 · median_c σ_{j,c} )

and divide channel `c` by `s_{j,c}`. The floor bounds amplification of
low-variance noise channels at 10×. If `PC1 ≤ 0.35`, this stage is the identity.
Record which branch fired.

**Stage 2 — scalar level scale.** As the plan specifies:

    S_j = (1/√C_j) · mean_{a≠b, ℓ} ‖ h_j(x_a)_ℓ − h_j(x_b)_ℓ ‖₂

computed **after** stage 1, then frozen. Reject zero, non-finite, or poorly
resolved scales.

Both stages are frozen from target-only data before any correction event, so
the metric cannot move in response to model failure — the property the plan's
§5.4 was protecting, now protected against the channel-domination failure too.

---

## A.4 Replaces §5.5–§5.6 — fields, radii, and health gates

### A.4.1 Field and energy — unchanged in form

The sample-split normalized Laplace mean-shift difference, and the
mean-of-squared-norms energy over `(j, r, ℓ, z)`, stand exactly as written.
Separate squares before averaging is correct and must not be relaxed. Positives
and probes detached; generated negatives differentiable through the frozen
trunk.

### A.4.2 Batch shapes — the highest-value change in this appendix

| role | plan | **revised** | why |
|---|---:|---:|---|
| positives per event | 64 | **256** | cached; costs storage, not compute |
| probes per event | 64 | 64 | unchanged |
| negatives per event | 64 | 64 | each costs a generator forward |

At 64-vs-64 the real-versus-real floor is **54–67% of total measured energy**
(§5.4 of the audit). The floor falls roughly as `1/n`, so 64 → 256 positives
should cut the positive side's contribution ≈ 4× and lift signal-to-floor from
≈ 0.85 toward ≈ 3. **Most of what the correction currently differentiates is
sampling noise.**

Asymmetric roles are legitimate here and should be used: the positive bank is
precomputed, so a larger positive side is nearly free, while the negative side
is bounded by generator-forward cost.

**Bank storage arithmetic**, so the decision is made on numbers: 66 vectors ×
512 channels × 4 levels × 2 views = 270 336 values per image ≈ **0.52 MB per
image in fp16**; × 5 000 automobile training images ≈ **2.6 GB**. This fits in
host RAM, pinned, paged to the GPU per event at 256 images × 0.52 MB ≈ 132 MB.
Store fp16; accumulate kernels and energies in fp32.

### A.4.3 Radii — span a real range

Replace `R = {0.35, 0.60, 0.85}` with

    R = {0.10, 0.35, 0.85}

as target **median off-diagonal** ESS fractions, calibrated per feature level by
bisection exactly as
[`stage_b2/core.py:244`](encoder_independent_drifting/stage_b2/core.py#L244)
does for the raw branch.

The plan's set spans under 2× in τ and contains no local regime; all three
fields are therefore blind to sub-bandwidth structure in the same way and
averaging them does not repair it. TFD's ablated set spans 10×. At `n_p = 256`,
ESS fraction 0.10 is **25.6 effective neighbours** — comfortably clear of the
one-to-five-neighbour tail that motivated the plan's health floors, which is
precisely what the enlarged bank in A.4.2 buys.

**Fallback ladder.** If the smallest radius fails the health floors at the
frozen batch size, step it along the predeclared ladder `{0.10, 0.15, 0.20}` and
record which rung was used. Do not silently widen it.

### A.4.4 The raw branch is multi-radius too

The rank collapse measured in B2 is a property of **broad-bandwidth barycenter
matching**, not of pixel geometry. A single raw radius at ESS 0.60 reproduces
it. Apply the same three-radius construction, with separate squares averaged, to
`E_raw`.

This is one reason A.0 matters: the raw branch is no longer B2's frozen
single-τ configuration, so nothing is being reused numerically and the
incumbent arm is a new configuration too. Say so in the artifact.

### A.4.5 Health gates — now two-sided and applied to both roles

Retain the plan's prospective limits (5th-percentile ESS ≥ 0.10, 95th-percentile
maximum weight ≤ 0.50) as **calibration-time** rejection criteria on the target
side, and add **runtime** gates the plan lacks:

| gate | applies to | threshold | rationale |
|---|---|---|---|
| negative-side median ESS | every level × radius, every logged event | **≤ 0.90** | if it approaches 1 the negative barycenter approaches a plain batch mean and the energy degenerates to first-moment matching |
| raw energy vs real-real floor | raw branch, every checkpoint | **≥ floor** | B2 scored 13.933 against a floor of 14.103; a correctly distributed sample cannot beat the floor, so undershooting is estimator exploitation |

[`stage_b2/core.py:214`](encoder_independent_drifting/stage_b2/core.py#L214)
already records `positive` and `negative` weight health separately — the
plumbing exists; only the thresholds are missing. B2's negative-side ESS ran
0.666 → 0.702 across audits with nothing watching it.

### A.4.6 Logging — add the location split

Retain the plan's per-level, per-radius log list, and add:

- **energy split between the 2 global and 64 local vectors.** 97% of `E_self`
  is position-locked: location `ℓ` of a generated image is only ever compared
  with location `ℓ` of a target image. Under the horizontal flip recorded in
  §5.7's augmentation bits, token `(i,j)` maps to `(i, 15−j)`, so a flipped car
  reads as far from an unflipped one at every local location. A large fraction
  of the energy may be measuring pose rather than semantics, and the split is
  the only way to see it.
- **negative-side ESS and maximum-weight summaries**, matching the positive side.
- **train-bank minus fresh-bank energy gap** at every checkpoint (see A.7).

Note the audit/deployment mismatch explicitly in the artifact: §6.2.1 undoes
known flips and translations *for the qualification audit*, but the training
field gets no registration. The audit measures the feature map under conditions
the objective never enjoys.

---

## A.5 Replaces §6.2 — feature qualification gate

### A.5.1 Fail-fast ordering

Run **G7 and G8 first**. Both are cheap, target-only, and test the two
assumptions the entire branch rests on. If either fails, §11's "cancel the ASFD
arm" branch fires before any bank construction or coefficient calibration is
spent.

### A.5.2 G7 (new) — two-sided per-band sensitivity

The plan's §6.2.5 is an **upper** bound only: *"the response to small
high-frequency noise is not more than four times the response of a normalized
raw-pixel control."* A feature map with **zero** high-frequency sensitivity
passes it trivially — and that is the failure this architecture is most likely
to have, because the trunk allocates capacity under an MSE objective that
underweights exactly the bands the foundation is missing. S3R's EMF arm measured
final raw Haar variance ratios LL 0.509, LH 0.410, HL 0.512, **HH 0.159**.

Replace with a two-sided, per-band test. Inject fixed-energy perturbations into
each orthonormal Haar band and measure the feature-distance response relative to
a normalized raw-pixel control:

    ρ_b = Δ_feature(band b) / Δ_raw(band b),   b ∈ {LL, LH, HL, HH}

**Require `ρ_b ∈ [0.25, 4.0]` in every band, for every level.** Report the full
profile alongside S3R's Haar table so the trunk's blind spots are visible before
any training is spent.

### A.5.3 G8 (new) — inter-level non-redundancy

§6.2.6 checks the semantic field against the **raw** field but nothing checks
the levels against **each other**. Under the plan's original two adjacent levels
this was the gap most likely to make the branch a silent no-op.

Require, on target-only data, for every pair of levels:

- median field cosine `< 0.90` at the audit events, **and**
- linear CKA between level descriptors `< 0.95`.

Levels failing the pair test are dropped, not tuned; record which survive.

### A.5.4 G9 (new) — local-token rank and concentration

The plan's §6.2.3 checks effective rank ≥ 16 and PC1 ≤ 50% on the **global**
mean/std descriptor, but 64 of 66 vectors per level are local tokens. Apply the
same two checks to the local-token descriptor. If token-level PC1 exceeds 0.35,
A.3's per-channel stage fires; if it still exceeds 0.50 after that stage, the
level fails.

### A.5.5 Tighten G6

`|cosine| < 0.995` against the raw field permits ~99% shared variance. Require
**`|cosine| < 0.90`** at the median audit event. A semantic branch that is
90%-aligned with the raw branch is not adding geometry, it is adding weight.

### A.5.6 Unchanged

G1 (benign vs random AUC ≥ 0.80, computed both on the invariant global
descriptor and on registered local tokens), G2 (patch-shuffle/phase-scramble
farther in ≥ 80% of paired cases), G4 (pairwise-distance CV ≥ 0.05), and the
uncurated nearest-neighbour grids as sanity checks rather than
threshold-selection devices.

---

## A.6 Replaces §7 — gradient integration

### A.6.1 No projection

Delete the first-order projection entirely. The applied update is

    g_total = g₀ + λ₁g₁ + λ_r g_r + λ_s g_s

which **is** the gradient of

    L = L_EMF + λ₁L_B1 + λ_r E_raw + λ_s E_self,

so §3.1's implication applies to the objective the optimizer actually descends,
and `λ_raw > 0` does the work the plan claims for it. Ordinary global gradient
clipping — as already used by the foundation — is applied afterwards to the
summed gradient, exactly as B2.5 does.

Projection would make the update non-conservative, leaving no potential whose
stationary points the dynamics seek, and would delete precisely the component of
the raw anchor that opposes `g₀` — which is the only component that does any
work, since an anchor that agrees with the primary gradient changes nothing.

### A.6.2 Per-component caps, not a shared total

| component | cap on `‖λᵢgᵢ‖ / ‖g₀‖` |
|---|---:|
| B1 spectral | 0.15 |
| raw Laplace | 0.10 |
| self-feature | 0.10 |

Each cap is applied to its own weighted gradient norm, independently, after AMP
unscaling. The total auxiliary norm is then permitted to reach 0.35 in the ASFD
arm and 0.25 in the raw arm.

This is the point. Under the plan's shared 0.25 cap the ASFD arm runs at
**≈ 29% less raw anchor and 29% less B1** than the arm it is compared against,
so any Stage-D difference confounds "added a semantic term" with "cut the
protection by a third" — and the confound points the wrong way, since the arm
carrying the new pressure is the one with weakened protection.

B2.5's protocol §2 already settled the principle: *"'B1 present' and 'B2
present' have the same treatment level in the single and combined cells...
Consequently the full combined cell can have greater correction compute and a
larger total correction gradient than either single arm. That is not a confound
in the factorial interaction; it is the defined joint treatment."* Report the
realized total as a Pareto cost, as B2.5 does.

Cadence remains one correction event in ten.

### A.6.3 Abort criteria — outcomes, not cosines

This refines the audit's own §3.4 recommendation, which said to abort on
sustained negative `cos(g₀, gᵢ)`. That is too aggressive: **a correction that
never opposes the primary gradient is useless**, so mild opposition is the
working regime, not a fault. Cosines are diagnostics with one pathological
threshold; the real aborts are outcome-based.

| abort | threshold |
|---|---|
| non-finite gradient, broken bank hash, zero generated-input Jacobian | any occurrence (as the plan already specifies) |
| **feature-space effective rank** vs the arm's own step-0 value | `< 0.70`, sustained over two logged checkpoints |
| **raw-pixel effective rank** vs the paired control | `< 0.70`, sustained over two logged checkpoints |
| **raw energy below the real-real floor** | any checkpoint |
| **negative-side median ESS** | `> 0.90` sustained (A.4.5) |
| anti-parallel auxiliary | `cos(g₀, gᵢ) < −0.8` sustained over a 200-event window — the term is negating training, not correcting it |

Cosines outside the last row are logged and interpreted, never acted on.

### A.6.4 Mandatory diagnostics — additions to §7.2

Retain the plan's list and add: negative-side ESS and maximum weight per level
and radius; the global-vs-local energy split; the **train-bank minus fresh-bank
energy gap**; raw energy expressed as excess over the real-real floor rather
than as a raw number; and realized per-component post-cap ratios for **both**
arms so the dose match in A.6.2 is auditable rather than asserted.

---

## A.7 Replaces §5.7 — target feature banks

The construction stands. Two changes.

**Views per image: 2 → 4.** Over ≈ 16 000 correction events the positive
barycenter built from 2 frozen views is a *fixed* random function whose
deviation from the population barycenter is a fixed bias the generator can
partly learn to match. Four views halves that bias at 2× the storage — ≈ 5.2 GB
fp16 for four levels, still host-RAM resident. If storage binds, prefer more
views over more levels.

**Log the bias directly.** §7.2 must record the **train-bank minus fresh-bank
energy gap** at every checkpoint. The Stage-D gate already evaluates on a fresh
bank, so the failure is *caught*; the gap is what makes it *visible* early
enough to act on.

Evaluation continues to use a separately seeded fresh bank, as the plan
specifies.

---

## A.8 Replaces §8 Stage D — paired development fork

### A.8.1 Arms

Unchanged in identity — `EMF-control`, `EMF-raw`, `EMF-ASFD` — with A.6.2's
per-component caps so that "raw present" and "B1 present" mean the same thing
in every arm. All shared streams as the plan specifies.

### A.8.2 Revised advancement gate

`EMF-ASFD` advances only if, relative to `EMF-raw`:

1. it reduces **fresh-bank** semantic energy;
2. its fresh-bank raw energy is not worse by more than 5% **and is not below the
   real-versus-real floor** *(new: two-sided)*;
3. it retains ≥ 90% of the incumbent's **raw-pixel** effective rank;
4. it retains ≥ 90% of the incumbent's **feature-space** effective rank *(new)*;
5. it does not increase **within-class maximum pairwise SSIM** *(new — this is
   TFD's own missing-mode diagnostic, and the failure its coverage ablation
   describes)*;
6. precision and recall stay within prospective uncertainty margins;
7. duplicates and nearest-training-image concentration do not increase;
8. the semantic post-cap gradient share is nonzero; and
9. projected cost is compatible with the remaining budget.

Conditions 4 and 5 are the ones the plan's gate could not see. Its rank
condition is raw-pixel only, and a within-class semantic collapse — samples
varying in colour, background and texture while repeating one car pose — can
preserve raw-pixel rank and a raw-pixel characteristic-function anchor while
being exactly the failure TFD reports.

### A.8.3 Coverage protection

B1 is a random-Fourier-feature characteristic-function criterion in **raw pixel
space** ([`b1.py:3`](encoder_independent_drifting/b1.py#L3)), and its own
docstring records that the finite V-statistic *"is not itself measure
determining."* It has never been tested against a feature-space pressure. TFD
reports its feature-space coverage term is **essential**.

Adopt option (b) of the audit's §8: keep the clean single factor, but make
conditions 4 and 5 **monitored abort criteria** during training (A.6.3), not
merely end-of-run report items, and **predeclare the anchor-margin arm as the
immediate follow-up** rather than "a later experiment."

Recorded counterargument, since it is real: TFD's drifting loss is the
**primary** objective while ASFD's semantic term is auxiliary and
cadence-averaged near 2.5%. But B2's dosing was comparably light — λ = 1.9×10⁻⁴,
one event in ten — and the rank loss was 38–40% anyway. Light dosing is not the
protection it appears to be.

---

## A.9 Replaces §8 Stage E — confirmation

### A.9.1 The missing decision rule

Stage D has a nine-condition gate; Stage E, as written, says only *"Report:"*.
Every confirmation in this repository has had a preregistered k-of-n rule — B0
3/3, B1 3/3, B2 2/3.

Nine simultaneous conditions across two units is a rule a genuinely better
method can easily fail. **Designate two primary endpoints and require both in
both units:**

| primary endpoint | rule |
|---|---|
| KID against `EMF-raw` on the sealed split | lower in **2 of 2** units |
| recall non-inferiority against `EMF-raw` | within a prospectively declared margin in **2 of 2** units |

Everything else in §8's report list is **report-only** and cannot promote or
demote the result.

### A.9.2 Two limits that must be stated, not discovered

**The two units share one foundation.** CAP-EMF-1 trains a single capability
unit (S3R §10.4: *"One strong unit is more informative for this proof of
concept"*), and Stage E clones it into two continuations with new stochastic
streams. Those units therefore measure **continuation variance, not foundation
variance**. No Stage-E result can claim replication across foundations. This
belongs in §12's "may not claim" list.

**The sealed split is 1 000 images.** B0/B1/B2 used 2 048-sample references and
three units and were still described as *"coarse consistency, not high-powered
inference."* Report KID with paired without-replacement subsampling over common
indices; report FID as **indicative only** — this repository has already
measured FID's small-sample bias, with a floor near 70 at n = 512.

---

## A.10 Additions to §10 — required tests

Retain the plan's list and add:

| test | catches |
|---|---|
| extracted grid reshapes to exactly 16×16, and its spatial autocorrelation is nontrivial | the conditioning-token off-by-two slice, which §6.1.1 cannot see |
| summed-gradient regression: the applied update equals `∇(L_EMF + λ₁L_B1 + λ_rE_raw + λ_sE_self)` to float tolerance | reintroduction of projection |
| per-component cap correctness on **weighted** norms, **after** AMP unscale | caps silently applied to unscaled or unweighted gradients |
| inter-level field cosine and CKA on a synthetic two-level fixture | G8 plumbing |
| per-band Haar sensitivity on a fixture with a known band response | G7 plumbing |
| negative-side ESS gate fires on a synthetic degenerate negative cloud | A.4.5 |
| real-real floor is deterministic given the allocation, and the sub-floor abort fires | A.6.3 |
| ESS calibration converges for every level × radius, including the fallback ladder | A.4.3 |
| bank hash reproduces under replay at 4 views | A.7 |
| no frozen B1/B2 constant is loaded anywhere in the ASFD path | A.0 |

---

## A.11 Replaces §9 — cost, and what this actually costs

The plan does not state a budget for Stages D and E. It should, because the
answer is large and the user is renting the GPU.

Correction events cost ≈ 2× an ordinary update (one differentiable frozen-trunk
forward/backward on the generated batch, plus kernels at ≈ 0.3%) at one event in
ten, so a corrected arm costs **≈ 1.1× per update**.

Expressed in foundation-update-equivalents against CAP-EMF-1's 160 k:

| stage | arms × units × updates | equivalents | fraction of foundation |
|---|---|---:|---:|
| Stage D | 3 × 1 × 20 k | ≈ 66 k | ≈ 41% |
| Stage E (3 arms) | 3 × 2 × 20 k | ≈ 132 k | ≈ 83% |
| Stage E (**2 arms** — recommended) | 2 × 2 × 20 k | ≈ 88 k | ≈ 55% |

**ASFD as specified roughly doubles the total budget.** Dropping `EMF-control`
from Stage E — it is established in Stage D and Stage E's primary comparison is
against `EMF-raw` — saves ≈ 28% of the foundation cost for no loss of primary
inference.

If the budget will not carry both stages, **run Stage D only and report it as a
development-scope result.** That is consistent with how this repository has
reported B2.5 and S3R, and it is far better than shortening both stages until
neither can resolve anything. §9's existing rule — *"If compute runs short,
shorten all matched continuation arms equally. Never protect the candidate's
update count by shortening only its control."* — is correct and stands.

---

## A.12 Replaces §11 — amended decision tree

The plan's tree is sound. Three amendments.

**"If the feature audit fails"** — the audit now fails earlier and for sharper
reasons. Record *which* gate failed, because the remedies differ: a G7 band
failure argues for a different `t_f` or a decorrelated trunk (A.13); a G8
redundancy failure argues for wider level spacing; a G9 concentration failure
argues that A.3's per-channel stage was insufficient.

**"If semantic energy falls but rank/coverage falls"** — this is no longer only
a post-hoc branch. A.6.3 makes it a runtime abort, so the arm stops instead of
burning the full continuation to reconfirm a published ablation.

**New branch — "if raw energy falls below the floor."** Stop the arm. This is
estimator exploitation, not distributional improvement, and it is the signature
that produced B2's rank collapse. Do not record it as a drift-reduction success.

---

## A.13 Fallback if the trunk is blind

If G7's band profile comes back degenerate — most likely `ρ_HH < 0.25`, since
the trunk allocates capacity under an MSE objective that underweights exactly
the band CAP-EMF-1 is expected to be weakest in — the plan's §11 cancels the
arm. Two intermediate options are worth preregistering before that:

1. **Decorrelate the feature map from the generator.** Extract from a trunk
   checkpoint that is *not* the fork point — an earlier checkpoint, or an
   independently seeded foundation. The trunk's blind spots are correlated with
   the generator's precisely because they are the same weights at the moment the
   branch switches on. Cost: a stored checkpoint, or a second foundation run.
2. **Re-select `t_f`** toward the abstraction end of A.2's grid and re-run G7.
   The band profile is a function of `t_f`, and this costs one audit pass.

Only if both fail should the scattering or random-convolution branch be
designed, as §11 specifies.

Partial mitigation worth recording: the trunk is trained on **real** noised
images, so its features do encode real-image structure, not merely the
generator's output manifold. The self-referential problem is weaker than
"the student teaching itself" — but the band argument survives it, because it
concerns capacity allocation under MSE, not which images were seen.

---

## A.14 Amended claim ledger (§12)

Add to **"may not claim"**:

- replication across foundations — Stage E's units share one foundation
  checkpoint and measure continuation variance only (A.9.2);
- that any B1 or B2 confirmation transfers — every constant is re-derived and
  the raw branch is a new multi-radius configuration (A.0, A.4.4);
- that reduced raw drift energy is itself an improvement, unless it approaches
  the real-versus-real floor from above (A.4.5).

The plan's existing "may claim" paragraph stands, with one insertion: the
improvement is *"on a single foundation, at development scope"* unless Stage E
runs and both primary endpoints hold in both units.

---

## A.15 Ordering

1. **Finish B2.5 units 501 and 502** (~11 h local, already implemented and
   paused). It is the only measurement of whether the spectral anchor and the
   raw drift term compound, which is exactly ASFD's §3.2 premise extended to a
   third term. Unit 500 fails two of four preregistered conditions despite
   `B1B2` leading on recall, KID and FID — that ambiguity should be resolved
   before a third correction term is designed on top of it.
2. **CAP-EMF-1 foundation**, unchanged, with A.1.4's conditioning decision
   recorded so extraction can be written against it.
3. **Stage B qualification, G7 and G8 first** (A.5.1), over A.2's `t_f` grid.
4. **Stage C preflight**, with A.10's added tests.
5. **Stage D**, with A.6's gradient scheme and A.8's gate.
6. **Stage E only if budget remains** (A.11), two arms, two primary endpoints.

Steps 3 and 4 are cheap and target-only. They are where this plan should be
allowed to fail.

