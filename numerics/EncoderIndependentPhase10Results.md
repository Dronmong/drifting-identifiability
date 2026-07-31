# Encoder-Independent Kernel Drifting — Phase 10 results

## The shape law is quantitative, and R11 is the only thing that breaks it

*Protocol: `EncoderIndependentPhase10Protocol.md`, frozen before the run.
Code: `run_phase10.py`. Artifact: `phase10.json` (+ `.sha256`), stdout in
`phase10.stdout.txt`. 25.6 minutes, 3 fresh seeds (`MASTER_SEED + 18000..`),
CIFAR-16, target ESS 0.9, field cloud 256, 600 steps.*

---

## 0. Summary

| stage | question | outcome |
|---|---|---|
| **10A** | is it the spectrum or the packing? | **the spectrum**, by 9× |
| **10A** | is there a law? | **yes, and it predicts out-of-sample** |
| **10B** | can acting on shape supersede R11? | **gate not passed** — and the law says exactly why |

The finding is that the mechanism is now **quantitative**: given a cloud's
spectral tail, the second moment it equilibrates at is predictable to within
~0.1 — for every arm measured **except R11**, which misses by 0.51.

---

## 1. 10A — the spectrum is the variable, not the packing

Two families, each scanned for the scale at which the field's radial
component vanishes.

**Family S — vary the spectrum** (`S^β`, renormalized to fixed variance):

| β | tail | nn spacing | nn CV | **radial zero** |
|---|---:|---:|---:|---:|
| 0.5 | 0.5045 | 14.15 | 0.149 | **1.0749** |
| 1.0 | 0.1235 | 10.37 | 0.215 | **0.7999** |
| 1.5 | 0.0157 | 6.73 | 0.256 | **0.3886** |
| 2.0 | 0.0016 | 4.36 | 0.300 | **0.2370** |
| 3.0 | 0.0000 | 2.09 | 0.391 | **0.1790** |
| *0.0 (degenerate)* | *0.8745* | *18.61* | *0.000* | *0.1797* |

Monotone across a **6× span** of the equilibrium radius as the tail falls.

*β = 0 is excluded from the law and reported:* it sets every singular value
equal, giving a perfectly regular white cloud (nn CV exactly 0.000) that is
not a rescaling of anything realistic. It breaks monotonicity and is why the
family's rank correlation reads only +0.429 — the coefficient is dragged by
one degenerate point, so the span is the honest summary, not the Spearman.

**Family P — hold the covariance fixed, vary only the packing:**

| relaxation steps | tail | nn spacing | nn CV | radial zero |
|---|---:|---:|---:|---:|
| 0 | 0.1235 | 10.37 | 0.215 | 0.8288 |
| 5 | 0.1235 | 10.45 | 0.202 | 0.8414 |
| 20 | 0.1233 | 10.66 | 0.168 | 0.7643 |
| 80 | 0.1216 | 11.21 | 0.124 | 0.7432 |

The construction works — the tail is held to within 1.5% while nn CV falls
by 42%. And the equilibrium barely moves: **span 0.098 against the spectrum
family's 0.896, a factor of 9.**

> **The spectral tail is the variable. Packing regularity, at fixed
> spectrum, is nearly inert.** This is the separation the mechanism pass
> flagged as necessary and could not make, since every cloud measured until
> now confounded the two.

---

## 2. The law, and its out-of-sample test

Interpolating the equilibrium radius against the tail over the
non-degenerate spectrum arms gives a predictor. Applied to every 10B arm:

| arm | tail | **law predicts** | **measured** | error |
|---|---:|---:|---:|---:|
| E0 none | 0.0050 | 0.274 | 0.321 | +0.046 |
| E2 repulsion 0.1 | 0.0052 | 0.276 | 0.332 | +0.055 |
| E3 tail 0.1 | 0.0053 | 0.278 | 0.320 | +0.042 |
| E2 repulsion 1.0 | 0.0075 | 0.301 | 0.410 | +0.108 |
| E3 tail 1.0 | 0.0113 | 0.342 | 0.335 | **−0.006** |
| E2+E3 both 1.0 | 0.0199 | 0.405 | 0.431 | **+0.026** |
| **E1 R11** | 0.0496 | 0.518 | **1.029** | **+0.511** |

**Held-out points, never used to build the law:**

| | tail | law | measured | error |
|---|---:|---:|---:|---:|
| free particles | 0.4150 | **1.010** | **0.995** | **−0.015** |
| generator (mechanism pass) | 0.0034 | 0.257 | 0.417 | +0.160 |

The free-particle point — a completely different system, measured two passes
ago — is predicted to within **1.5%** by a law fitted on rescaled data
clouds. That is a real out-of-sample validation.

A second consistency check: real data has tail 0.1376, and the law predicts
its cloud equilibrates at **0.810**. The mechanism pass measured the radial
zero for data-shaped clouds directly at **0.844**. Two independent routes,
3% apart — and both explain the ~15% inward bias as the self-term.

### R11 is the sole outlier, and that is informative

Every arm sits within 0.11 of the law. R11 misses by **0.511**, an order of
magnitude worse. So **R11 does not work by opening the tail** — it reaches a
second moment twice what its shape warrants.

That is exactly consistent with what the mechanism pass measured directly:
at R11's operating point the field's radial component is **−0.126** and only
38% of samples are pushed outward. R11 holds the cloud *past* the field's
equilibrium against a restoring force. **It is an override, not a shape fix**
— and the law is what makes that statement quantitative rather than
rhetorical.

---

## 3. 10B — the gate does not fire, and the law says why

| arm | 2nd moment | ED² | score | tail | tail vs baseline |
|---|---:|---:|---:|---:|---:|
| E0 none | 0.321 | 1.3470 | 5.787 | 0.0050 | 1.00× |
| **E1 R11** | **1.029** | **0.1964** | **1.944** | 0.0496 | — |
| E2 repulsion 0.1 | 0.332 | 1.2792 | 5.642 | 0.0052 | 1.04× |
| E3 tail 0.1 | 0.320 | 1.3408 | 5.771 | 0.0053 | 1.06× |
| E2 repulsion 1.0 | 0.410 | 0.8367 | 4.424 | 0.0075 | 1.50× |
| E3 tail 1.0 | 0.335 | 1.2817 | 5.615 | 0.0113 | 2.24× |
| E2+E3 both 1.0 | 0.431 | 0.7664 | 4.242 | 0.0199 | **3.95×** |

**This is not the protocol's refutation branch.** That branch was "the
interventions move the shape but not the second moment", which would have
made shape a correlate rather than a cause. What happened instead is that
they moved the shape **and moved the second moment by precisely the amount
the law predicts** (errors +0.026 to +0.108). The mechanism is confirmed; the
interventions are simply too weak.

The law makes the shortfall computable rather than mysterious. The best arm
reaches tail 0.0199 — still **7× short of real data's 0.1376** — and the law
says that buys 0.405. To reach the band a generator needs a tail near
**0.12**, roughly β = 1.0, which is 6× further than the strongest declared
weight achieved.

So the honest status of the shape route: **directionally correct, verified
quantitatively, and under-powered at the weights declared.** Whether a much
stronger penalty reaches tail ~0.12 without destroying sample quality is
open — E2+E3 already cost nothing (ED² 0.767 against the baseline's 1.347,
i.e. it *improved* quality while opening shape).

---

## 4. What Phase 10 establishes

**Established:**

- **the spectral tail, not packing regularity, sets the field's equilibrium
  radius** — 9× separation in a family that holds the covariance fixed;
- **the relationship is a usable law**, predicting six of seven arms to
  within 0.11 and the held-out free-particle system to within 0.015;
- **R11 is an override, not a shape fix** — it is the single arm the law
  fails on, by 10× the typical error, in the direction the earlier radial
  measurement predicted;
- shape interventions work in the predicted direction and improve quality
  (ED² 1.347 → 0.767) but reach only 14% of the data's tail at the declared
  weights, which the law says is worth 0.405 — and it was.

**The mechanism now predicts rather than merely explains.** That is the first
time in this program that a proposed cause has produced a quantitative,
out-of-sample-validated prediction instead of a post-hoc account.

**Open:** whether a stronger shape intervention reaches tail ≈ 0.12 and so
supersedes R11; and why a smooth pushforward's tail is so hard to raise
(latent dimension does not do it — N4 — and neither do these penalties at
these weights).

---

## 5. Scope and caveats

- The law is fitted on clouds that are *rescalings and spectral reweightings
  of real data*; generators differ in more ways than that, and the generator
  point itself is the worst-predicted of the held-out pair (+0.160).
- β = 0 is degenerate and excluded from the law; the family's Spearman
  (+0.429) is dominated by it and the span is the honest statistic.
- Family P holds the covariance fixed only up to per-coordinate rescaling in
  the principal basis; higher moments are uncontrolled.
- Declared weights only; no search. One dataset, one geometry, 3 seeds.
- Nothing here concerns ImageNet, FID, or the paper's trained model. The
  anchor stays disabled; the geometry thread stays closed.

## 6. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.run_phase10 `
  --stage all --seeds 3 --steps 600
```
