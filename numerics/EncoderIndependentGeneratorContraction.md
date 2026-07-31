# Encoder-Independent Kernel Drifting — why the generator contracts

## It is the objective's optimum, not a failure to reach it

*In-depth pass on the question Phase 7 sharpened. Code: `diagnose_phase8.py`.
Artifact: `phase8_probe.json` (+ `.sha256`), stdout alongside. Development
seeds (`MASTER_SEED + 13000..`), 3 seeds, 600 steps, CIFAR-16, **at the
Phase-7C bandwidth optimum (target ESS 0.9)** so nothing here is confounded
by the badly-set kernel of Phases 2–6. Nothing feeds a gate.*

---

## 0. The finding

Phase 7 left one question: the generator's second-moment deficit is not the
optimizer (6A), not the bandwidth (7A), not the cloud size (7A), not the
field's fixed point (7B/7C), and five direct hypotheses were refuted in
Phases 3–5. Four measurements later:

> **The generator is not failing to expand. It is sitting exactly where its
> objective's optimum is, and that optimum is a shrunken cloud.** The cause
> is least-squares shrinkage: point-target regression reproduces its targets
> faithfully when it can represent them and contracts smoothly when it
> cannot.

---

## 1. N2 — a free dilation parameter does not rescue it

The decisive test. Add **one** learnable scalar output gain, `f(z) = g·net(z)`,
putting pure dilation explicitly into the model's tangent space at a cost of
a single parameter. If expansion were unrealizable, or were being
out-competed inside the shared convolutional parameters, the gain would grow.

| variant | 2nd moment | ED² | radial component | outward frac | **gain** |
|---|---:|---:|---:|---:|---:|
| plain | 0.462 | 0.8354 | **+0.0004** | 0.551 | — |
| + gain parameter | 0.476 | 0.7973 | +0.0009 | 0.559 | **0.829** |
| plain + R11 | **0.926** | **0.2027** | **−0.1260** | 0.379 | — |

Three things, each decisive:

1. **The gain went *down*, to 0.829**, and the second moment did not move
   (0.462 → 0.476). Dilation is realizable and the model declines it. The
   deficit is not an optimization failure.
2. **At the converged plain generator the field's mean radial component is
   +0.0004 — indistinguishable from zero.** The field is *not* straining
   outward against a stuck model; the generator sits at a genuine radial
   equilibrium. This is exactly what the fixed-point condition predicts: the
   generator equilibrates where `E_z[(∂f/∂θ)ᵀV] = 0`, and since the head is a
   plain convolution, dilation is in the tangent space, so the radial
   component *must* vanish there.
3. **R11 holds the cloud past that equilibrium against the field's own
   restoring force.** At R11's operating point the radial component is
   −0.1260 and only 38% of samples are pushed outward — the field is pulling
   back in. R11 is active counter-pressure, confirmed directly rather than
   inferred.

---

## 2. N3 — the mechanism: least-squares shrinkage

Fit the generator by least squares to the converged free-particle cloud, and
sweep the ratio of **parameters to target values**. Each fit is compared
against *the very particles it was asked to reproduce*, so "regression costs
something" is never confounded with "a smaller cloud is a worse sample".

| arm | params | **params/values** | fit 2nd | control 2nd | fit ED² | control ED² |
|---|---:|---:|---:|---:|---:|---:|
| w=64, n=64 | 109 635 | **2.23** | **0.933** | 0.939 | 0.2919 | 0.2989 |
| w=128, n=256 | 366 723 | 1.87 | 0.843 | 0.963 | 0.1325 | 0.1031 |
| w=64, n=128 | 109 635 | 1.12 | 0.902 | 0.956 | 0.1543 | 0.1541 |
| w=64, n=256 | 109 635 | 0.56 | 0.733 | 0.963 | 0.1753 | 0.1031 |
| w=64, n=512 | 109 635 | **0.28** | **0.602** | 0.993 | 0.3894 | 0.0690 |

Within the fixed-width family the second moment is **monotone in
params/values**: 0.933 → 0.902 → 0.733 → 0.602 as the ratio falls 2.23 →
0.28, while the particle control stays at 0.94–0.99 throughout.

At p/v = 2.23 the fit reproduces its targets essentially exactly — second
moment 0.933 against 0.939, ED² 0.2919 against 0.2989. **Point-target
regression costs nothing when it has the capacity.** It costs a factor of
1.65 when it is short by 3.6×, which is precisely the regime the real recipe
runs in.

### This corrects two entries in the record

**Phase 4's `fit_to_free` figure was the artifact it flagged.** It reported
that "least-squares point-target regression costs ~2×" and noted the probe
was underparameterized by 3.6×. Crossing the parameters/values line shows
the cost is *entirely* that: it vanishes at p/v ≥ 1. The ~2× was
underparameterization, not regression.

**Refuted hypothesis 1 was right in kind and tested against the wrong noise.**
Phase 3 refuted "minibatch-noise regression attenuation" by measuring the
*field's* sampling noise at a fraction ≈0.001 and concluding attenuation was
orders of magnitude too small. Correct — but least-squares shrinkage has two
sources, and that measured only one:

- **estimation variance** — finite batch. Small, as Phase 3 found, and the
  axis Phase 7A moved by enlarging the field cloud (0.27 → 0.38, real but
  minor);
- **approximation error** — the model cannot represent its target. Never
  measured until now, and it is the dominant term.

Hypothesis 1 named the right phenomenon and was dismissed on the strength of
the smaller of its two terms.

---

## 3. N4 — it is not the latent dimension (H10 refuted)

The natural competing account was that the generator's cloud is a
low-dimensional manifold (latent 32 into 768 pixels) missing the data's
spectral tail, and that a cloud without a tail balances at a smaller radius.
Sweeping the latent dimension at the good bandwidth:

| latent | 8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|
| 2nd moment | 0.400 | 0.462 | 0.479 | 0.440 |
| tail fraction | 0.0015 | 0.0032 | 0.0042 | 0.0058 |
| ED² | 1.0440 | 0.8354 | 0.7401 | 0.8864 |

Sixteen times the latent room buys nothing: the second moment never exceeds
0.479 and turns over after 128. **H10 is refuted**, and R17's original
negative — recorded at ESS 0.5, before the bandwidth optimum was known — is
confirmed at the good bandwidth.

This also usefully constrains the capacity story. Latent 512 carries *more*
parameters than latent 32 (the input projection grows to 512×1024) and is
still at 0.440, so it is **not** raw parameter count that matters but the
capacity of the map where the target actually varies. That is the conv
stack's width, not the latent's dimension — and it is exactly the axis Phase
8 must sweep.

---

## 4. The spectral tail: R11 does more than rescale

| | real data | plain | + gain | **+ R11** |
|---|---:|---:|---:|---:|
| variance beyond top 32 directions | **0.1375** | 0.0032 | 0.0029 | **0.0474** |

R11 restores the tail **15×** (0.0032 → 0.0474) while remaining 2.9× short
of the data. So it is not a pure rescaling — holding the teacher inflated
every step keeps pushing in directions that would otherwise be squeezed out.
This is a partial answer to why the corrected generator still loses to free
particles at the good bandwidth (ED² 0.167 against 0.075, Phase 7B): it
recovers the scale and only a fraction of the shape.

---

## 5. What this explains

| observation | phase | explained by least-squares shrinkage |
|---|---|---|
| deficit is **bandwidth-independent** | 7A | shrinkage depends on model-vs-target capacity, not on the field |
| deficit is **optimizer-independent** | 6A | it is a property of the optimum, not the path |
| bigger field cloud helps **a little** | 7A | it reduces the estimation-variance term only |
| **η is inert**, step caps never bind | R21, R24 | scale-based interventions cannot move an optimum |
| R11 works, and is **isotropic** | 3, 5, 6 | shrinkage is a scalar contraction; a scalar inflation inverts it |
| richer corrections **do not beat** the scalar | 6B | nothing anisotropic to correct |
| over-correction (gain 1.2) is **worse** | 6B | the right inflation is exactly 1/shrinkage; more overshoots |
| free particles are **fine** | 6C, 7B | no regression, no shrinkage |

That is nine standing observations, including four previously unexplained
negatives, under one mechanism.

---

## 6. Phase 8 — the experiment this implies

**Sweep the conv stack's width in the real recipe, at the good bandwidth,
crossed with R11.** This is the axis N4 isolated as the relevant one and
which the program has never varied — `width = 64` has been fixed since Phase
1 as part of the matched-capacity rule.

| axis | values |
|---|---|
| width | 32, 64 *(incumbent)*, 128, 256 |
| R11 | off, on |
| bandwidth | target ESS 0.9 (Phase-7C optimum), held fixed |

**The prediction, stated sharply so it can fail:** the uncorrected
generator's second moment should rise monotonically with width, and **R11's
advantage should shrink as width grows** — capacity and R11 are substitutes,
not complements, because they address the same shrinkage.

**Proposed gate:** if any width without R11 reaches a second-moment ratio in
`[0.7, 1.3]` with ED² within 25% of the best R11 cell, **capacity supersedes
R11** and the program's positive result is re-scoped as a
matched-capacity artifact.

**What refutes the account:** R11 helping equally at every width, or the
second moment staying flat as width grows. Either would show that
representational shrinkage is not the operative term in the real recipe,
where the target is a moving distribution rather than N fixed points — a gap
between N3's setting and the recipe's that this pass has not closed.

Cost note: width 256 at a 512-sample cloud is ~290 GFLOP per step, hours of
CPU per fit. Phase 8 should hold the field cloud at 128–256 and budget
accordingly, or move to GPU first (see the deferred GPU decision).

---

## 7. What this pass does not establish

- **N3's setting is not the recipe's.** It fits a *fixed* cloud with fixed
  latents; the recipe regresses onto a *moving distribution* with fresh
  latents every step. The shrinkage mechanism is demonstrated in the former
  and *inferred* for the latter. Phase 8 is what would close that gap, and
  N4's flatness in the real recipe is a live warning that it might not.
- No width sweep in the real recipe was run here.
- The radial and gain measurements are at one bandwidth, one architecture,
  3 seeds.
- Nothing here concerns ImageNet, FID, or the paper's trained model. The
  geometry thread stays closed; the anchor remains unexamined since Phase 2.

## 8. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.diagnose_phase8 `
  --seeds 3 --steps 600 --resolution 16
```
