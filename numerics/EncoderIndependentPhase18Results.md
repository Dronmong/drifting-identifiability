# Encoder-Independent Kernel Drifting — Phase 18 results

## Depth does not rescue the encoder, and the screen's own control shows why reduced-budget screening cannot be trusted

*Runners: `diagnose_phase18.py` (invariance), `run_phase18b.py` (depth
sweep). Artifacts: `phase18_probe.json`, `phase18b.json` (+ `.sha256`).
6 arms × 2 seeds × 5 000 steps, CIFAR-32, target ESS 0.9, R11 throughout,
paired within seed, GPU.*

---

## 1. The depth sweep

| arm | FID | ED² | tail | 2nd | **paired vs raw** |
|---|---:|---:|---:|---:|---:|
| **raw** | **275.87** | 0.2406 | 0.1430 | 0.992 | — |
| rand_layer3 *(untrained)* | 299.85 | 0.4313 | 0.1102 | 0.953 | **+23.97** |
| pre_layer1 | 335.68 | 0.4844 | 0.2020 | 0.935 | **+59.80** |
| pre_layer3 | 344.83 | 0.6680 | 0.1947 | 1.037 | +68.96 |
| pre_layer2 | 346.88 | 0.6947 | 0.2150 | 1.053 | +71.01 |
| pre_layer4 | 357.59 | 0.7528 | 0.1828 | 1.018 | +81.72 |

*(bar 246.28; real data tail ≈ 0.13)*

**The hoped-for result does not hold. A shallow pretrained encoder does not
rescue the geometry** — `pre_layer1` is +59.80 FID worse than raw pixels, and
every one of the eight individual within-seed pretrained-vs-raw differences
is positive. That part is robust to the noise.

This is the middle branch declared in advance: depth is associated with the
failure but does not fix it, and **encoder-free remains the best geometry in
this harness.**

---

## 2. The screen's own control invalidates its finer readings

`rand_layer3` was included as a control, because Phase 17 measured the
untrained network at **−8.55 FID better than raw** at 30 000 steps. Here, at
5 000 steps, the same arm reads **+23.97 worse**.

**The control reverses sign between budgets.** I flagged this check before
the run precisely so it could not be waved through, and it fires: the
5 000-step screen is trustworthy only for the coarse split (pretrained ≫ raw,
a 60–82 FID effect) and **not** for anything finer.

Two consequences:

- **The depth ordering is not established.** FID by depth reads 335.7 →
  346.9 → 344.8 → 357.6 — weakly rising but non-monotone, with layer3 below
  layer2. Against seed spreads exceeding 100 FID at this budget (layer1:
  283.68 vs 387.67), no depth ranking within the pretrained arms is
  supported.
- **The sensitivity → depth → FID chain is unconfirmed.** The invariance
  probe's monotone sensitivity ladder (4.3× → 9.5× → 12.6× → 14.2×) does not
  have a matching monotone FID ladder to justify it. It remains a plausible
  mechanism for *pretrained versus untrained*, not a demonstrated one for
  *depth*.

**This is a design error worth naming.** I chose 5 000 steps because the
pretrained penalty was visible at 600 and at 30 000, and inferred the
ordering would be stable in between. It is stable for the large effect and
not for the small one, and I should have anticipated that a reduced budget
buys speed with variance rather than with bias.

---

## 3. What Phases 17 and 18 together establish

**Robust:**

- **A pretrained semantic encoder is a bad kernel for drifting.** +138 FID
  over raw at 30 000 steps, +60 to +82 across all four depths at 5 000, every
  within-seed comparison agreeing in sign.
- **Encoder-free (raw pixels) is the best or equal-best geometry at every
  budget tested.**
- **Pretraining, not architecture, is what hurts.** The untrained ResNet18 —
  same architecture, same L2 normalization, same calibration, differing only
  in weights — is within noise of raw at both budgets (−8.55 at 30 000,
  +23.97 at 5 000).
- **Pretrained features are hypersensitive, not invariant**: 12.6× the
  untrained network's response to high-frequency noise at layer3, and the
  failing arm's spectral tail is 3.7× real data's.

**Not established:** the depth ordering; the sensitivity→FID causal chain;
anything about the paper's method.

---

## 4. The caveat that bounds every claim above

**This is not the paper's algorithm.** The paper drifts *within* an encoder's
latent space. This harness computes the drift in **pixel space using
feature-space kernel weights**. A feature map that fails as a *kernel over
pixels* says nothing directly about the same encoder used as a *space to
drift inside*.

So Phase 17/18 do **not** refute the paper's encoder ablation. What they
establish is a statement about kernel drifting as this program defines it —
which is the thing the program set out to build.

---

## 5. Where this leaves the target, and what I recommend

The program's question was whether drifting can be made encoder-independent.
**Within kernel drifting, the answer is yes, and more strongly than hoped:**
no encoder is needed, and a pretrained one actively hurts. That is a
defensible, scoped finding backed by paired measurements at two budgets under
a semantic metric.

Two things bound it, and both should be stated in any write-up rather than
worked around:

1. **Absolute quality is poor.** The best configuration reaches FID ~232
   against a floor of 71. The comparison between geometries is sound; the
   claim "this is a good generative model" is not available.
2. **The method is not the paper's**, so the result is about *kernel*
   drifting, not about drifting in general.

**Recommendation: consolidate rather than continue.** The mechanism work is
finished (fourteen refuted hypotheses, one shape law, one metric audit that
reframed all of it), and the thesis now has an answer at the scope the
harness supports. Further phases inside this design will refine numbers that
are already bounded by (1).

**The one experiment that would genuinely extend the claim** is to implement
the paper's actual method — drifting *within* a learned latent space, with a
decoder — and re-run the encoder ladder there. That is the only route from
"a pretrained encoder is a bad kernel" to a statement about the paper's
encoder dependence. It is a substantial new implementation, not a variation
on the existing one, and it should be costed as such before starting.

---

## 6. Scope

- CIFAR-10 at 32×32, FID at 512 samples with a floor of 70.9; only arms
  measured identically are comparable, none comparable with published FIDs.
- 2 seeds throughout; the 5 000-step screen has seed spreads over 100 FID and
  its finer readings are not usable (§2).
- ResNet18/ImageNet is supervised, not the paper's self-supervised encoder;
  one input size (128px), one normalization (L2), neither swept.
- R11 on every arm, so this measures the corrected recipe.
- GPU throughout; arithmetic validated against CPU to 9.4e-7.
