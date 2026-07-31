# Encoder-Independent Kernel Drifting — Phase 9 results

## The deficit survives all the way down to `f(z) = Az + b`

*Protocol: `EncoderIndependentPhase9Protocol.md`, frozen before the run.
Code: `run_phase9.py`. Artifact: `phase9.json` (+ `.sha256`), stdout in
`phase9.stdout.txt`. 69.1 minutes, 3 fresh seeds (`MASTER_SEED + 16000..`),
2000 steps, target-ESS 0.9 bandwidth solved per cell, audited
`lowdim_drift.drift_paper` field throughout.*

---

## 0. Summary

**Prediction P-lin — that a linear generator on Gaussian data has no
second-moment deficit — is refuted.** Every cell of the ladder shows one, and
the linear generator at CIFAR lands within 6% of the full convolutional
recipe.

| cell | per-seed | median 2nd | ED² | window growth |
|---|---|---:|---:|---:|
| Gaussian isotropic, **linear** | 0.215, 0.208, 0.211 | **0.211** | 1.103 | +0.0006 |
| Gaussian isotropic, MLP | 0.055, 0.062, 0.060 | 0.060 | 2.423 | +0.0083 |
| Gaussian decaying, **linear** | 0.665, 0.648, 0.633 | **0.648** | 0.090 | −0.0026 |
| Gaussian decaying, MLP | 0.546, 0.480, 0.473 | 0.480 | 0.269 | −0.0046 |
| CIFAR-16, **linear** | 0.502, 0.509, 0.490 | **0.502** | 0.553 | −0.0006 |
| CIFAR-16, MLP | 0.357, 0.365, 0.351 | 0.357 | 1.251 | +0.0162 |
| *(reference: conv recipe, Phase 8)* | | *0.472 converged* | | |

All six cells are converged (|growth| ≤ 0.017) and every seed agrees to
within a few percent.

**The three findings:**

1. **The architecture is irrelevant.** A linear map at CIFAR reaches 0.502
   against the convolutional recipe's 0.472 — a 6% difference across a model
   class that differs by a conv stack, GroupNorm, upsampling and 110k
   parameters. This closes the question Phase 8 left open when capacity
   proved inert.
2. **The nonlinearity modulates but does not cause.** The MLP is consistently
   worse than linear — 0.211 → 0.060, 0.648 → 0.480, 0.502 → 0.357 — a factor
   of 1.3–3.5, in every data law. It deepens the deficit; it does not create
   it.
3. **The deficit is present in the simplest case there is.** Declared outcome
   2 of the protocol: *the mechanism lives in the regression structure
   itself, and every object in it is now closed-form or nearly so.*

---

## 1. Why the derivation failed — two premise errors, both instructive

The protocol's §2 derived that for `f(z) = Az + b` the fixed point
`E[V zᵀ] = c(σ)·A = 0` forces `σ = s`. The measurement says otherwise, and
the derivation was wrong for two separate reasons.

### (a) It used the wrong field

The closed form `V(x) = c(σ)x` is the **SNIS mean-shift** field. The recipe
uses Algorithm 2's **bi-softmax**, which this repository explicitly labels a
different object (`lowdim_drift` marks the SNIS field "DIAGNOSTIC ONLY"). The
bi-softmax has the `sqrt(row ⊙ col)` reweighting and cross-scaling, and no
reason to share the mean-shift's zero. The `c(σ)` column in the artifact is
retained as a *diagnostic* — it reads 9.8e-2 for the isotropic cell against
1.3e-3 for CIFAR — but it does not describe the field the recipe uses.

### (b) The self-term, and a reduction artifact I nearly reported as the mechanism

Measuring the bi-softmax field's raw radial component for a Gaussian cloud
against Gaussian data, the zero sits at **σ ≈ 0.49**, not at σ = 1, and at
the correct scale the field pulls inward with magnitude 1.58. Decomposing it:

| quantity (64-D isotropic Gaussian) | value |
|---|---:|
| self weight `Wn[i,i]` | 0.2057 |
| total negative row mass | 0.9661 |
| **self share of the repulsion** | **21.3%** |
| \|V\| unmasked → masked | 1.631 → **0.439** |
| \|V_unmasked − V_masked\| | 1.587 |
| \|self-term `Wn[i,i]·x_i`\| | 1.641 *(residual 0.082)* |

The self-term — each cloud point repelling from itself at distance zero, kept
because `FieldConfig.self_mask` defaults to **False** — is a pure inward pull
**3.7× larger than the entire masked field**. In 64-dimensional isotropic
geometry every pair sits at distance ≈ 11.3 while the self-distance is 0, so
the self-affinity is ~77× a typical neighbour.

**That looked like the mechanism. The control says it is not.** Repeating the
measurement in the actual CIFAR-16 setting at the same target ESS:

| setting | k(x,x)/k(x,y) | \|V\| unmasked → masked | self effect |
|---|---:|---|---:|
| 64-D isotropic Gaussian | **77** | 1.631 → 0.439 | 21% of repulsion |
| **CIFAR-16 (the real recipe)** | **4.5** | 0.670 → **0.672** | **7%, negligible** |

Natural images are concentrated, so typical pairwise distances are small
relative to the bandwidth and the self-term is ~1.7% of the negative mass.
**The self-term dominance is an artifact my own reduction introduced**, and it
is why the isotropic cell collapses to 0.211 while the decaying-spectrum and
CIFAR cells sit at 0.648 and 0.502.

Two consequences, stated plainly:

- **The isotropic-Gaussian cell is not a valid proxy for the recipe.** Its
  geometry manufactures a mechanism the real system does not have. The
  decaying-spectrum and CIFAR cells are the informative ones, and the ladder
  should be read from those.
- Phase 4's finding that "the self-mask is protective" now has a mechanism
  *and* a scope: it matters enormously in isotropic high-dimensional
  geometry and little on concentrated natural-image data.

Without the CIFAR control I would have reported the self-term as the answer.
It is the third time in this program that a control has overturned a
conclusion I was about to draw, and the reason the control was run is that a
21% weight change producing a 3.7× field change did not look credible.

---

## 2. What this establishes

**Established:**

- the deficit does **not** require the convolutional architecture, the
  nonlinearity, or non-Gaussian data — it is present for `f(z) = Az + b` on
  Gaussian data, and the linear model at CIFAR is within 6% of the full
  recipe;
- the nonlinearity contributes a consistent additional factor of 1.3–3.5 in
  every data law;
- the field's self-term is dominant in isotropic high-dimensional geometry
  (21% of the repulsion, 3.7× the masked field) and negligible on CIFAR (7%),
  which scopes Phase 4's self-mask observation and invalidates the isotropic
  reduction;
- Algorithm 2's bi-softmax is **not** the SNIS mean-shift field, so the
  closed-form Gaussian mean-shift coefficient does not describe it.

**The object of study is now much smaller.** The remaining question is a
statement about a linear map: for which `A` does

```
E_z[V(Az + b)] = 0    and    E_z[V(Az + b) zᵀ] = 0
```

hold, when `V` is the bi-softmax field? The cloud is exactly Gaussian, so this
is a fixed-point equation in `AAᵀ` alone — a `d × d` symmetric matrix, with no
network, no optimizer, and no architecture in the way. That is a far more
tractable object than anything the previous eight phases were working with,
and it is directly solvable numerically to high precision.

**Still unknown:** why that fixed point sits at roughly half the data's
variance. Eleven hypotheses have been refuted; this phase does not add a
twelfth, it removes the scaffolding around the question.

---

## 3. Scope and caveats

- The isotropic-Gaussian cell is reported but should **not** be used as a
  proxy for the recipe (§1b).
- The CIFAR cells use a linear map of rank ≤ 64 into 768 dimensions, so the
  linear arm there is also spectrally confined; the Phase-8 follow-up showed
  confinement contributes part of the gap.
- `c(σ)` in the artifact is a diagnostic against the wrong field and is
  labelled as such.
- 3 seeds, 2000 steps, one bandwidth rule (target ESS 0.9), one η.
- Nothing here concerns ImageNet, FID, or the paper's trained model. The
  anchor stays disabled; the geometry thread stays closed.

## 4. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.run_phase9 `
  --seeds 3 --steps 2000 --dim 64 --latent 64 --cloud 256
```
