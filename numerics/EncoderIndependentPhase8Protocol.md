# Encoder-Independent Kernel Drifting — Phase 8 protocol

## Are capacity and R11 substitutes?

*Frozen pre-outcome design. Every threshold was fixed before any Phase-8 run.
Source: `EncoderIndependentGeneratorContraction.md`. Results go to
`EncoderIndependentPhase8Results.md`.*

---

## 1. Why this phase exists

The generator's second-moment deficit has been shown **not** to be the
optimizer (6A), the kernel bandwidth (7A), the field cloud size (7A), the
field's own fixed point (7B/7C), or the latent dimension (N4). The
generator-contraction pass then identified it as **least-squares shrinkage**:

- adding one free dilation parameter leaves the deficit unmoved and the gain
  converges *downward* (0.829), so the generator is at its objective's
  optimum rather than failing to reach it;
- the field's mean radial component at that optimum is +0.0004, i.e. zero;
- fitting the converged particle cloud by least squares gives second moment
  **0.933 / 0.902 / 0.733 / 0.602** as parameters-per-value falls
  **2.23 / 1.12 / 0.56 / 0.28**, with the particle control flat at 0.94–0.99.
  Point-target regression is faithful when it has the capacity and shrinks
  when it does not.

If that mechanism operates in the real recipe, **capacity and R11 address the
same thing** and must be substitutes. The conv stack's width has been fixed
at 64 since Phase 1 by the matched-capacity rule and has never been varied.

**This protocol can retire R11**, as Phases 6 and 7 could. It is the third
such test and the first aimed at the mechanism rather than at a confound.

---

## 2. Phase 8A — the width sweep

| axis | values |
|---|---|
| conv width | 32, **64** *(incumbent)*, 128, 256 |
| R11 | off, on |
| bandwidth | target ESS 0.9 (the Phase-7C optimum), **held fixed** |
| field cloud | 256, held fixed |

Data: CIFAR-10 at 16×16, raw pixel geometry, paper Algorithm-2 field.
Seeds: 3 fresh, `MASTER_SEED + 14000..`. Budget 600 steps, target batch 64,
Adam / 2e-3 (6A established the optimizer does not matter; holding it fixed
keeps this a one-axis test).

Approximate parameter counts: 35k / 110k / 367k / 1.32M — a 38× span.

Reported per cell: ED², the second-moment ratio, the spectral tail fraction
beyond the top 32 directions, parameter count, R15 admissibility, and the
paired R11-versus-plain ratio.

### 8A gate — the decisive one

> **If any width *without* R11 reaches a second-moment ratio inside
> `[0.7, 1.3]` and an ED² within 25% of the best R11 cell, then capacity
> supersedes R11**, and the program's positive result is re-scoped as a
> matched-capacity artifact.

### 8A secondary prediction — declared, and separately scored

> **R11's advantage shrinks monotonically as width grows.** Scored as the
> paired ED² ratio (R11 ÷ plain) per width: the mechanism predicts this ratio
> rises toward 1 with width. A flat ratio across a 38× parameter span
> **refutes** least-squares shrinkage as the operative term in the real
> recipe.

This second condition matters as much as the gate. The gate can fail while
the prediction holds (capacity helps but not enough at these widths), and the
prediction can fail while the gate fails too (capacity is simply irrelevant)
— those are different findings and are reported separately.

---

## 3. Phase 8B — where the shrinkage actually binds

The generator-contraction pass demonstrated shrinkage on a **fixed** cloud
with fixed latents. The real recipe regresses onto a **moving distribution**
with fresh latents every step, and N4 found the latent dimension flat there.
That gap is the pass's main stated weakness, and 8B measures it directly.

At each width, with the field frozen, measure how much of its own teacher the
generator actually realizes in one step:

```
realized = <f_after − f_before, T − f_before> / ‖T − f_before‖²
```

Reported, not gated. The shrinkage account predicts `realized` rises toward 1
with width. If it stays pinned while width grows 38×, the recipe's target is
not being under-fitted for capacity reasons and the account does not transfer
from 8's fixed-cloud setting.

---

## 4. Declared failure branches

- **8A gate fires** → capacity supersedes R11; re-scope Phases 3/5/6/7.
- **Gate fails, prediction holds** → shrinkage is the mechanism but these
  widths are insufficient; report the trend and the extrapolated width.
- **Gate fails, prediction fails** → least-squares shrinkage does **not**
  transfer to the moving-target recipe. Say so plainly; the
  generator-contraction pass's §7 already flags this as live.
- **Wide cells become R15-inadmissible or diverge** → report with health
  numbers, do not score.
- **No tuning.** The grid is declared above and is swept in full.

## 5. What Phase 8 cannot conclude

- Nothing about ImageNet, FID, or the paper's trained model.
- Widths beyond 256 are untested; a null result is scoped to a 38× span.
- The anchor stays disabled and the geometry thread stays closed.
- A positive result is scoped to: *CIFAR-10 at 16×16, raw pixel geometry,
  one generator family, target ESS 0.9, field cloud 256, 3 fresh seeds,
  600 steps.*
