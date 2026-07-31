# Encoder-Independent Kernel Drifting — the metric audit

## ED² is saturated by a moment-matched Gaussian, and that is what the program has been optimizing

*Exploratory pass on Phase 14A. Code: `diagnose_phase15.py`. Artifact:
`phase15_probe.json` (+ `.sha256`). 3 seeds, CIFAR-10 at 32×32, GPU.
Nothing feeds a gate.*

---

## 1. E1 — what FID means, in interpretable degradations

| degradation | FID | **ED²** | 2nd moment |
|---|---:|---:|---:|
| real vs real *(floor)* | 71.7 | **0.1118** | 0.993 |
| **Gaussian, exact mean + covariance** | **240.6** | **0.1079** | 1.006 |
| blur 2× | 178.3 | 0.1325 | 0.925 |
| blur 4× | 310.1 | 0.2623 | 0.807 |
| noise σ = 0.1 | 132.0 | 0.1186 | 1.034 |
| noise σ = 0.3 | 234.7 | 0.4591 | 1.357 |
| **shuffled pixels** | **403.5** | **0.9602** | 0.993 |
| pure noise *(ceiling)* | 426.5 | 1.2233 | 1.012 |

Two rows carry the whole result.

**A Gaussian with the data's exact mean and covariance — containing no image
structure whatever — scores ED² 0.1079, *better* than a second real sample's
0.1118.** By energy distance it is indistinguishable from CIFAR. By FID it
sits 49% of the way from real data to pure noise.

**Shuffled pixels** — every spatial relationship destroyed, marginals
preserved — reads FID 403.5, essentially the noise ceiling, at ED² 0.96.

> **Energy distance in pixel space is saturated by matching the first two
> moments. It is close to blind to image structure.**

---

## 2. E2 — the program's two headline reforms, scored both ways

| arm | FID | ED² | tail | 2nd moment |
|---|---:|---:|---:|---:|
| generator, ESS 0.5, no R11 | 260.3 | 1.8985 | 0.0104 | 0.417 |
| **generator, ESS 0.5, + R11** | **244.0** | **0.3054** | 0.0767 | 0.929 |
| generator, ESS 0.9, no R11 | 266.8 | 2.2746 | 0.0129 | 0.359 |
| generator, ESS 0.9, + R11 | 258.8 | 0.4224 | 0.1046 | 1.014 |
| **free particles, ESS 0.9** | **388.1** | 0.2242 | 0.6111 | 0.993 |

- **R11 is worth 6.2× on ED² and 6% on FID** (1.899 → 0.305 against 260.3 →
  244.0). At ESS 0.9 it is 5.4× and 3%.
- **The ESS-0.9 bandwidth — Phase 7's 4.9× result — makes FID *worse***, both
  with R11 (244.0 → 258.8) and without (260.3 → 266.8).
- **Free particles, the reference the last seven phases measured the
  generator against, sit at FID 388.1** — past shuffled pixels (403 is the
  same neighbourhood) and close to the 426.5 ceiling.

And the comparison that states it most plainly:

> **The best configuration this program has produced (FID 244.0) is not
> better than a Gaussian fitted to the data's mean and covariance
> (FID 240.6).**

**E3**: the FID ordering is unchanged between 512 and 2048 samples, so none
of this is a small-sample artifact.

---

## 3. What this means, stated carefully

The account is coherent and it is not flattering:

1. ED² in pixel space is maximized by matching the first two moments (E1).
2. R11 *is* a second-moment match, by construction.
3. So R11 improves precisely the axis the metric measures and the paper's
   metric ignores — 6.2× against 6%.
4. The mechanism work — the shape law, the self-reference finding, the
   spectral-tail equilibrium — is all built on ED² and the second moment.

**What survives.** Every mechanism result remains true *as a statement about
the second moment and about ED²*. The shape law predicts equilibrium radius
from spectral tail and was validated out-of-sample; R11 replicates across
five settings; the self-reference finding is a real property of the
stop-gradient recipe. None of that is retracted.

**What does not survive.** The inference from those results to *better
generation*. Thirteen phases improved a statistic that a structureless
Gaussian already saturates.

**And it reframes Phase 14A.** The inverted encoder ladder is no longer
puzzling: at FID 244–302 every arm is in the region where a moment-matched
Gaussian also lives, so the ladder was ranking geometries by how well they
match moments, which is not what an encoder is for.

---

## 4. The definitive next experiment

Everything above points to one question, and it has an unambiguous baseline
for the first time in this program.

> **Can drifting on raw pixels produce anything a moment-matched Gaussian
> cannot?**
>
> The bar is **FID 240.6**. The floor is 71.7.

That bar is the right one because beating it requires producing *structure*,
not moments — exactly the thing ED² cannot see and the encoder is supposed to
supply.

### Design

**Primary readout: FID.** ED² and the second moment are reported as
diagnostics only, never as the outcome. This inverts the program's practice
and is the point.

| axis | values |
|---|---|
| budget | 600 → 3k → 10k → 30k steps *(the scaling curve)* |
| generator width | 64, 256 |
| field cloud | 256, 1024 |
| geometry | raw pixels *(encoder-free)*, pretrained ResNet18 |
| R11 | on *(it costs nothing and fixes the second moment)* |

Fixed baselines drawn on every plot: **real 71.7**, **moment-matched Gaussian
240.6**, **noise 426.5**.

### The declared outcomes

- **FID falls clearly below 240.6 and keeps falling with budget** → drifting
  produces structure; the encoder ladder becomes meaningful; 14B is finally
  worth running, and the program has a real result.
- **FID plateaus at or above 240.6 regardless of budget, for raw pixels but
  *not* for the pretrained encoder** → this is the paper's encoder dependence,
  reproduced and quantified in our own harness. It would be a clean negative
  answer to the program's thesis and the most valuable outcome short of
  success.
- **FID plateaus near 240.6 for *both*** → the harness, the generator, or the
  recipe cannot produce structure at this scale at all, and no conclusion
  about encoders can be drawn from it. Then the honest move is to stop and
  write up the mechanism results for what they are.

Each outcome ends an open question rather than opening a new one, which no
phase since Phase 3 has managed.

### Cost

At the measured GPU rates, the 30k-step arms dominate: roughly 3–5 hours for
the full grid. That is the largest single run this program has attempted and
it is affordable only because of the GPU work.

---

## 5. Recommended immediate action, before the big run

**Re-score the sealed record.** A short pass computing FID for the arms of
Phases 3, 7, 10 and 13 would say which of this program's recorded results
survive a semantic metric. It is cheap, it is the honest thing to do before
any write-up, and it determines what the final document can claim.

---

## 6. Scope and caveats

- FID at 512 samples has a floor of **71.7**, not 0. Comparisons between arms
  measured identically are sound; absolute values are not comparable with
  published FIDs. E3 confirms the *ordering* is stable at 2048.
- The moment-matched Gaussian is sampled in the data's own principal basis,
  so it reproduces the empirical covariance exactly on its support.
- Inception-V3/ImageNet is a specific semantic prior. It is the field's
  standard and the paper's metric, but "FID cannot see it" is not the same
  as "a human cannot see it".
- One dataset, one resolution, 3 seeds, 600 steps for the E2 arms — the
  reforms are compared at the budget every earlier phase used, not at
  convergence.
- Run entirely on GPU (arithmetic validated to 9.4e-7, `device_check.json`).
