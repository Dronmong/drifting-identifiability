# Stage F3B protocol — prescribed bridge with source-space authority

**Status: REVISED IMPLEMENTATION PROTOCOL. B0 development is runnable; no
confirmatory outcome exists until a freeze artifact is created and the frozen
runner is executed.**

This stage implements §19.8 of `EncoderIndependentDriftFlowResearch.md`. It is
selected by the corrected F1 K=200 confirmation, not by the superseded
long-horizon `f1.json` interpretation.

---

## 1. What F1 established, and what it did not

`EncoderIndependentF1K200ConfirmationResults.md` established the following
narrow result:

> Under the frozen raw-pixel, smooth-Laplace, RMS-normalized, R11-corrected
> update, random-generator starts did not acquire detected CIFAR-10 coverage
> by K=200 in either regime, while all paired real-data controls remained
> valid.

The exploratory K=20,000 study separately observed a similar low-rank collapse
phenotype from several starts. Neither experiment proves a unique or global
attractor, rules out every drift design, or shows that no encoder can alter the
dynamics.

The result selects the prescribed-bridge branch because the current autonomous
teacher did not demonstrate basin reachability from the deployed start. F3B
asks whether a non-self-referential path objective can provide that
reachability without a pretrained feature encoder.

---

## 2. Exact mathematical construction

Draw independently

```text
X0 ~ N(0, I) in normalized image space,
X1 ~ the CIFAR-10 training distribution,
t  ~ Uniform(0, 1).
```

For the sampled conditional path,

```text
Xt = (1 - t) X0 + t X1,
conditional path velocity = X1 - X0.
```

Train an image-conditioned velocity network with

```text
L_FM(phi) = E ||u_phi(Xt, t) - (X1 - X0)||^2.
```

The distinction between a sampled target and the population vector field is
load-bearing. `X1 - X0` is the target attached to a sampled pair. The
deterministic squared-loss minimizer is

```text
u*(x,t) = E[X1 - X0 | Xt = x].
```

The baseline uses independent endpoint pairing. Minibatch OT or another
coupling is a later, separately labelled variation.

Sampling starts from a fresh `N(0,I)` image and integrates
`dx/dt = u_phi(x,t)` from zero to one by explicit Euler. The state is not
clipped during integration. Inception preprocessing clips only its own report
input; raw range violations and clipping fractions are reported separately.

This prescribed target removes the particular self-referential teacher
mechanism diagnosed in F1. It does **not** prove that an approximate learned
ODE cannot contract, collapse, or develop numerical pathologies.

---

## 3. Two B0 configurations with different purposes

The old design tried to use `OneStepGenerator` as the velocity model. That
module is a latent-to-image upsampler and cannot consume `Xt`; it is not used.
Both B0 configurations use the new `TimeConditionedUNet`, an image-to-image
network with downsampling, skip connections, time-conditioned residual blocks,
and optional low-resolution self-attention.

### B0-compact

A resource-realistic U-Net used to determine whether this program can reach
detectable coverage at an affordable budget. It is not described as a
published CIFAR recipe merely because its objective is standard.

### B0-reference-scale

A larger CIFAR-scale U-Net and longer development budget, intended as a
stronger positive-control attempt. It follows the standard architectural
family used by public flow-matching image examples, but local results are not
called a reproduction of published FID unless the complete external recipe,
data processing, training work, and evaluation sample count are matched.

Parameter count, training examples, optimizer updates, wall time, peak device
memory, and inference NFE are reported for both. Capacity matching is based on
counts and work, not the word “width.”

---

## 4. Development and confirmation are separate experiments

### B0 development

Development may measure:

- a declared training-budget ladder;
- a declared Euler-NFE ladder;
- compact and reference-scale profiles;
- validation loss, recall, precision, KID, indicative small-sample FID,
  collapse diagnostics, range violations, wall time, and memory.

Model selection uses development unit IDs and the development reference only.
Choosing a budget or NFE after observing this ladder is permitted because the
result is explicitly exploratory. The final selection artifact must contain
all three declared development units and the selected `(steps, NFE)` cell for
each; subset runs may debug cost but cannot cross the freeze boundary.

### B0 confirmation

Before confirmation, `freeze_f3b.py` records:

- the selected model, training, evaluation, and solver configuration;
- the development artifact that justified the selection;
- the protocol hash and all executable source hashes;
- three new confirmation unit IDs and every seed role;
- the compatible calibration/veto artifact hash.

The confirmation runner refuses changed code, protocol, configuration,
calibration, unit IDs, or artifact hashes. Confirmation uses no checkpoint
selection and no budget/NFE ladder.

---

## 5. B0 gate and permitted conclusion

The primary gate remains the previously calibrated **recall > 0.05** test,
with 512 generated samples against a fixed 2,048-image evaluation reference.
It is a *detectable-coverage* gate, not a high-quality-generation threshold.

For each confirmation unit:

```text
unit_pass :=
    recall > 0.05
    and every calibrated collapse/diversity veto passes
    and the full-training-pool memorization veto passes
    and the matched real-vs-real metric control is valid.
```

The stage passes when at least two of three new units pass. A failed metric
control voids the affected run according to the frozen adjudication rule.

Permitted readings are deliberately narrow:

- **PASS:** this frozen bridge model achieves detected fresh-sample CIFAR-10
  coverage above the calibrated null while passing the declared vetoes.
- **FAIL:** the selected bridge architecture/training budget did not establish
  detected coverage. This does not refute flow matching or the whole harness.
- **VOID:** the evaluator/calibration control was invalid.

A recall of 0.051 is not called good generation. Precision, KID, indicative
FID, visual samples, and recall relative to matched real-vs-real are mandatory
descriptors of quality even though they do not retroactively change this
categorical reachability gate.

The F3B calibration artifact is generated or adopted only after checking the
exact upstream artifact hashes and the hashes of every metric/veto source it
depends on. Merely finding an old `f1_calibration.json` or `f1_vetoes.json`
does not satisfy confirmation compatibility.

---

## 6. Mandatory preflight checks

Before development or confirmation:

1. **Bridge identity:** verify `Xt` and `X1-X0` exactly satisfy the declared
   interpolation.
2. **Oracle integration:** a constant oracle velocity for a fixed endpoint
   pair must reconstruct `X1` under Euler integration at every tested NFE.
   This checks signs and time direction only; it does not predict learned-model
   quality.
3. **Architecture:** verify image and scalar-time inputs, image-shaped output,
   nonzero dependence on both inputs, finite gradients, and reachable spatial
   resolutions.
4. **Reproducibility:** initialization, data order, endpoint pairing, time
   draws, augmentation, inference prior, and metric streams have distinct,
   recorded deterministic seeds.
5. **Cost:** measure seconds per optimizer step, peak memory, parameter count,
   and projected training/sampling work. Do not infer these from F1 rollout
   cost.

---

## 7. Later arms — gated and not yet confirmatory

B1–B4 are not run until B0 passes. They are design targets, not frozen
experiments:

| arm | required comparison |
|---|---|
| B1 `bridge+anchor` | the identical frozen bridge plus a separately nonnegative fixed source-space anchor loss |
| B2 `bridge+drift` | the identical frozen bridge plus an explicitly normalized drift correction whose equation, coefficient, gradient path, and work are frozen in advance |
| B3 `drift_only` | the frozen encoder-free drifting reference under matched reporting |
| B4 `paper_encoder` | an actual implementation of the paper method, not a generic encoder proxy |

For B1, a neutral recall result alone does not establish E3. The held-out
source-anchor audit must remain active. The now-frozen B1 design is specified
in `EncoderIndependentB1Protocol.md`: its correctness authority is the ideal
full-support characteristic-function discrepancy, while a finite feature bank
is only a stochastic surrogate. For B2, degradation is evidence against that
correction, not against encoder-independent generation.

Exact B1/B2 coefficients and schedules will be frozen only after B0 licenses
those experiments; leaving them unspecified now prevents an unrun arm from
masquerading as a confirmatory design.

---

## 8. Scope

- CIFAR-10 at 32×32 is the scientific target; reduced resolutions are smoke
  tests only.
- Training uses the 40,000-image train partition. Model selection uses a
  development reference; confirmation uses the untouched frozen evaluation
  reference.
- Inception features are report-only evaluation tools and never define the
  training objective.
- Three confirmation units support a categorical gate, not precise method
  ranking.
- Every result reports NFE and training work.
- B0 is standard flow matching, not a new drifting result and not the paper’s
  method.

---

## 9. Executable order

Run from the repository root. The exact Torch installation command may differ
between CPU and CUDA environments; the repository’s recorded CUDA convention
uses matching Torch and Torchvision wheels.

```powershell
# Mechanics, determinism, bridge identities, and all repository regressions
uv run --python 3.12 --with torch==2.7.1 --with torchvision==0.22.1 `
  --with numpy --with scipy `
  python -m numerics.encoder_independent_drifting.tests.run_all

# Measure the selected architecture before committing the development budget
uv run ... python -m numerics.encoder_independent_drifting.preflight_f3b `
  --profile compact --device cuda

# Exploratory budget/NFE ladder; this is allowed to inform selection
uv run ... python -m numerics.encoder_independent_drifting.run_f3b_development `
  --profile compact --device cuda `
  --out numerics/encoder_independent_drifting/f3b_development.json

# Freeze exactly one measured (steps, NFE) choice
uv run ... python -m numerics.encoder_independent_drifting.freeze_f3b `
  --profile compact --steps 30000 --nfe 32 `
  --development numerics/encoder_independent_drifting/f3b_development.json

# Fresh null, health, and full-training-pool memorization calibration
uv run ... python -m numerics.encoder_independent_drifting.calibrate_f3b `
  --device cuda

# Only after the previous artifact says GO
uv run ... python -m numerics.encoder_independent_drifting.run_f3b_confirmation `
  --device cuda
```

The ellipses above mean “reuse the complete matching dependency prefix from
the first command,” not an executable literal. `f3b_development_smoke.json`
and `f3b_preflight_smoke.json` are mechanical checks only and are refused by
the confirmation freeze.
