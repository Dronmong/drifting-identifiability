# Encoder-Independent Kernel Drifting — Phase 19 results

## The best configuration yet, and a screen too underpowered to prove it

*Protocol: `EncoderIndependentPhase19Protocol.md` (frozen before the run).
Runner: `run_phase19.py`. Artifact: `phase19.json` (+ `.sha256`), stdout in
`phase19.stdout.txt`. 4 arms × 4 seeds × 15 000 steps, raw geometry, R11,
paired within seed, GPU, 2.72 h.*

---

## 1. Results

| arm | ESS | LR | s0 | s1 | s2 | s3 | **median** | sd |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `A_ess9_const` *(as-is)* | 0.9 | const | 258.13 | 204.89 | 245.41 | 264.18 | **251.77** | 26.68 |
| `B_ess5_const` | 0.5 | const | 236.25 | 205.33 | 235.59 | 234.52 | **235.06** | 15.08 |
| `C_ess9_cos` | 0.9 | cosine | 248.43 | 214.22 | 235.73 | 238.38 | **237.06** | 14.39 |
| `D_ess5_cos` | 0.5 | cosine | 255.85 | 209.82 | 225.12 | 221.67 | **223.39** | 19.62 |

*floor (real) 70.26 · bar (moment-matched Gaussian) 248.31*

**The as-is configuration does not beat the bar at this budget (251.77 vs
248.31). `D` beats it by 25 points.** The median improves 28.4 FID from the
configuration every previous phase used.

---

## 2. The declared rule fired "neither helps". It is not usable.

| contrast | mean | sem | t | p | per-seed |
|---|---:|---:|---:|---:|---|
| `ess_at_constant` | **−15.23** | 6.63 | −2.30 | 0.105 | −21.9 **+0.4** −9.8 −29.7 |
| `cosine_at_ess9` | −8.96 | 7.18 | −1.25 | 0.301 | −9.7 **+9.3** −9.7 −25.8 |
| `ess_at_cosine` | −6.08 | 5.15 | −1.18 | 0.323 | **+7.4** −4.4 −10.6 −16.7 |
| `cosine_at_ess5` | +0.19 | 7.52 | +0.03 | 0.981 | +19.6 +4.5 −10.5 −12.8 |
| `best_vs_baseline` | **−15.04** | 10.58 | −1.42 | 0.250 | −2.3 **+4.9** −20.3 −42.5 |

Every contrast returned **unresolved**, so the runner emitted the
`(False, False)` branch: *"neither helps — the recipe is at its ceiling
here; run the long budget as-is and write up."*

**That verdict is an artifact of my rule, and reporting it at face value
would be wrong.** The medians move 28 FID. What actually happened is that
the screen could not have detected these effects, for two separate reasons.

**The sign test was a bad statistic.** Under the null, 4 seeds agree in sign
with probability 2·(½)⁴ = **12.5%** — so it is *weaker* than a 0.05 test —
while a single reversal destroys it. It is simultaneously too permissive and
too brittle. I chose it in §5 of the protocol to avoid Phase 18B's
over-reading, and over-corrected into a rule that cannot pass.

**But the run is underpowered regardless of the statistic.** No contrast
reaches p < 0.05 by a paired t-test either; the best is `ess_at_constant` at
p = 0.105. With paired sds of 13–21 against effects near 15, resolving these
needs roughly **6–8 seeds for the bandwidth contrast and 12–16 for
`best_vs_baseline`** — not 4. That calculation takes a minute and I did not
do it before committing 2.7 h of GPU.

**This is the second underpowered screen in a row.** Phase 18B failed at 2
seeds × 5 000 steps; I raised it to 4 × 15 000 and it failed again. The
correct standing rule for this program is now explicit: **no comparison of
~15 FID effects on fewer than 8 seeds.**

### One seed drives every failure

`A`'s seed1 is an outlier low — **204.89**, the best number in the run — and
every intervention on that seed lands slightly above it (B 205.33, C 214.22,
D 209.82). The reversals in `ess_at_constant`, `cosine_at_ess9` and
`best_vs_baseline` are all the same seed. They are not three independent
pieces of counter-evidence; they are one lucky baseline draw that nothing
could improve on.

That is an explanation, not a licence. The rule stands as declared and every
contrast above is reported unresolved.

---

## 3. What *is* established

**EMA is refuted — cleanly, in the harmful direction.**

| arm | mean | per-seed |
|---|---:|---|
| `A_ess9_const` | +0.48 | −0.25 −1.56 +2.69 +1.06 |
| `B_ess5_const` | **+2.64** | +4.51 +2.87 +2.66 +0.51 |
| `C_ess9_cos` | **+0.53** | +0.57 +0.43 +0.43 +0.71 |
| `D_ess5_cos` | **+0.48** | +0.98 +0.28 +0.37 +0.28 |

Three of four arms are sign-consistent, and arms C and D have spreads of
**0.3 and 0.7 FID** — tight enough that these are real measurements, not
noise. Weight EMA does not help this recipe; it costs a fraction of a point.

This is the one place the screen had power, because EMA is evaluated on the
*same trajectory* as the live weights and so carries no seed variance at all.
The design choice that made EMA an evaluation factor rather than a training
arm is what produced the only resolved result in the run.

**§F2 splits.** The constant-step diagnosis proposed two remedies. EMA is
refuted. LR decay is not — cosine carries a −8.96 mean at ESS 0.9 — but it is
unresolved, and at ESS 0.5 it is worth nothing (+0.19). The diagnosis that
the recipe has no annealing path remains a correct description of the
dynamics; only one of its two remedies is now dead.

**The bandwidth finding replicates independently.** `ess_at_constant` reads
**−15.23** here at 15 000 steps. `EncoderIndependentMetricAudit.md` measured
**−14.8** (244.0 vs 258.8) in a separate run at a much shorter budget. Two
independent measurements agreeing to within half a point is stronger evidence
than either p-value, and it is the reason I would act on this despite the
sign test.

**Every intervention reduces seed variance** — sd 26.68 → 15.08 / 14.39 /
19.62. Not predicted, and worth more than the mean shift: it is why every
future comparison in this program will be cheaper to resolve.

---

## 4. The interaction is real but inconsistent

The fixes compose on seeds 2–3 and conflict on seed 0. `cosine_at_ess5` is
+0.19 — once the bandwidth is fixed, the schedule buys nothing on average —
which suggests both act on the same underlying problem rather than adding.
With 4 seeds this cannot be separated from noise and is recorded as an
observation, not a finding.

---

## 5. What to do

**Do not report "the recipe is at its ceiling."** That branch fired
mechanically and its premise is false.

**Drop EMA.** It is the one thing this run settled.

**Take `D` (ESS 0.5 + cosine, live weights) to the long budget**, on the
evidence of: best median (223.39), best paired mean (−15.04), a bandwidth
component that independently replicates a prior measurement, and no evidence
of harm anywhere. This is a decision under uncertainty, not a proven
improvement, and it should be described that way.

**Run `A` and `D` together at 100 000 steps, 3 seeds each** (~6 h). That
resolves the comparison at the budget that actually matters instead of
spending more compute resolving a 15 000-step screen nobody cares about, and
it extends the scaling curve that was still falling at 30 000. Noise also
falls with budget — sd 26.7 at 15 k against Phase 16's 12.8 at 30 k — so the
long run is better-powered per seed than this screen was.

---

## 6. Scope

- CIFAR-10 32×32, FID at 512 samples, floor 70.26, bar 248.31. Only arms
  measured identically are comparable; none comparable with published FIDs.
- 4 seeds; §2 is explicit that this is too few. Every number in §1 carries
  that caveat.
- Fresh seed block (`SEED_OFFSET = 32000`), disjoint from Phases 16/17/18B.
- `ema9999` was computed but is barred by protocol §3 from selecting the
  long-run configuration; it tracks `ema999` and changes nothing.
- Raw geometry only. This run licenses **no** claim about encoder dependence.
- Still not the paper's method: pixel-space drift with feature-space kernel
  weights.
