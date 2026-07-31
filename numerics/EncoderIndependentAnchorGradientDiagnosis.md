# Encoder-Independent Kernel Drifting — anchor gradient diagnosis

*Diagnostic companion to `EncoderIndependentPhase1Results.md`. These
experiments were run to explain a Phase-1 observation, not to select an arm;
none of them feeds a gate. Code: `numerics/encoder_independent_drifting/`.*

## The observation

Every structured-geometry arm in the Phase-1 screen (A3–A7) and the
anchor-only arm A2 reach **zero calibrated support coverage** on targets
where raw pixel drifting (A0/A1) reaches full coverage. The generator does
not collapse — its output RMS (0.94–0.97) matches the raw arms' (0.98) — it
simply converges somewhere that is not the target support.

Three candidate explanations were separated by measurement.

## 1. Not a step-budget artifact

The kernel-gradient field under a structured kernel is nearly orthogonal to
raw displacement (Phase-0 G0.4, cosine −0.002), so a fixed step budget could
in principle favour the direct raw field. It does not explain this:

| arm | steps | geometry score | ED² | coverage | precision |
|---|---:|---:|---:|---:|---:|
| A1 (raw) | 300 | 3.648 | 0.848 | 1.000 | 0.652 |
| A1 (raw) | 1200 | 2.341 | 0.243 | 1.000 | 0.852 |
| A4 (wavelet) | 300 | 18.663 | 7.361 | 0.000 | 0.000 |
| A4 (wavelet) | 1200 | 19.844 | 8.436 | 0.000 | 0.000 |

Four times the budget makes A4 *slightly worse*, not better. A4 has
converged — to a law that is not the target.

This is the empirical face of
`DriftingIdentifiability/FeatureSpaceIdentifiability.lean`. The
stop-gradient regression drives the generator to a stationary point of the
geometry loss, i.e. to `V_j = 0`. For a kernel that is not
measure-determining, `V = 0` does not imply `q = p` — and the Phase-0
collision suite independently measured this geometry as non-determining
(the wavelet branch is blind to `color_swap`, p = 0.74).

## 2. Not an absent anchor

The anchor is numerically present and its weight does move the optimization,
so this is not the plan's "rhetorically present but practically absent"
failure (section 10.2):

| λ_A | geometry score | anchor gradient share | coverage |
|---:|---:|---:|---:|
| 1 | 16.557 | 0.257 | 0.000 |
| 10 | 9.825 | 0.781 | 0.000 |
| 100 | 27.130 | 0.974 | 0.000 |
| 1000 | 47.893 | 0.997 | 0.000 |
| 10000 | 50.794 | 1.000 | 0.000 |

Raising the anchor weight helps a little and then hurts, and never restores
coverage. Turning the anchor up is not the repair.

## 3. The actual cause: the anchor loss never descends

Anchor-only training (A2), 600 steps, logged on both the training bank and
the independent audit bank:

```
step   1  train 0.2799  audit 0.2685   grad norm 6.7e-2
step 300  train 0.2064  audit 0.2405   grad norm 5.7e-2
step 600  train 0.2240  audit 0.2573   grad norm 5.7e-2
```

The loss fluctuates around its initial value with no trend, while the
gradient norm stays constant. The training and audit banks track each other
throughout, which **rules out finite-bank overfitting** — the generator is
not quietly matching 512 memorized moments; it is not descending at all.

### Why: high-frequency features dominate the gradient with noise

The anchor gradient is

    dL/dy_i = (2 / (L n)) * sum_l [ -dc_l sin<w_l,y_i> + ds_l cos<w_l,y_i> ] w_l

— **each term is proportional to `w_l`**. So a band's contribution to the
gradient *magnitude* grows with its frequency, whether or not that band
carries usable signal. Measured on the checkerboard target at the declared
calibration (projected scale s = 0.2075):

| band | median ‖ω‖ | phase spread std(ω·x) | fraction of a period | band gradient norm |
|---|---:|---:|---:|---:|
| low | 0.967 | 0.493 rad | 0.079 | 1.68e-2 |
| mid | 3.454 | 1.910 rad | 0.304 | 3.63e-2 |
| high | 16.167 | 7.185 rad | **1.143** | **5.67e-2** |

The high band's phase varies by more than a **full period** across samples,
so its cosine/sine features are decorrelated — pure noise for descent — yet
it contributes the *largest* gradient norm of the three, 3.4× the low band's.
Two thirds of the declared bank actively drowns the informative third.

### Confirmation by band ablation

Anchor-only training, 300 steps, audit-bank loss start → end:

| anchor bands | audit anchor loss | reduction | geometry score | coverage |
|---|---|---:|---:|---:|
| declared low+mid+high | 0.2561 → 0.2408 | 6% | 47.9 | 0.000 |
| low+mid (no high) | 0.3166 → 0.2602 | 18% | 42.7 | 0.000 |
| low only | 0.1954 → 0.0472 | **76%** | 11.4 | 0.000 |
| very coarse only | 0.1041 → 0.0042 | **96%** | 13.7 | 0.000 |

Dropping the high band alone is not enough; the anchor only descends when
it is essentially coarse-only.

## The structural tension this exposes

The two things the anchor is asked to do are in direct conflict at fixed
bands:

- **Detection** wants high frequencies. The full three-band bank detects
  6/6 source collisions at the smallest attainable p-value (Phase-0 G0.2),
  including a 5% rare-mode drop and a phase scramble.
- **Optimization** wants low frequencies. Only a coarse-only bank descends,
  and a coarse-only bank is not sensitive enough to be the correctness
  authority — it matches coarse structure and stops.

Neither fixed choice satisfies both. Note that the coarse-only anchor drives
its own loss to 0.004 and *still* leaves coverage at zero: descending a
coarse discrepancy is not sufficient to place mass on the support.

This does not contradict the plan's population theory. The ideal statement
`L_anchor(p,q) = 0 ⟹ p = q` is about the expectation over a full-support
spectral measure and is untouched. What is measured here is that the finite
random-feature *estimator* of that ideal object has a gradient whose
signal-to-noise ratio degrades with frequency exactly where its
discriminating power comes from. Plan section 15.2 anticipated the general
risk — "characteristic population theory does not guarantee acceptable
minibatch conditioning" — and this is a concrete instance with a measured
mechanism.

## Indicated repair (untested)

The plan's own literature survey already names it. Section 5.1 records that
[Generative Drifting is Secretly Score Matching](https://arxiv.org/abs/2603.09936)
"identifies the Gaussian high-frequency bottleneck and motivates
heavy-tailed kernels and **coarse-to-fine bandwidth schedules**", and the
repository's own Phase-C bandwidth result (`DesignRules.md`, C1) is that one
must not begin at a bandwidth far below the relevant separation scale.

The indicated design is therefore a **frequency schedule**, not a fixed
bank: begin coarse so the gradient is informative, and anneal the band
weights toward high frequencies as the coarse discrepancy is exhausted —
with the *audit* bank kept at full three-band width throughout, so that
detection power is reported honestly at every step even while the training
bank is deliberately narrow.

Three cautions before implementing it:

1. This is a hypothesis with a measured motivation, not a result. It has not
   been run.
2. A schedule reintroduces a tuning surface (the annealing rate) that must be
   frozen from target-only criteria, not from a gate.
3. The schedule fixes the *anchor*. It does not address finding 1 — the fixed
   wavelet geometry converging to a wrong law — which is a separate defect
   in a separate branch.

## Reproduce

Each table above is produced by a short script against the package's public
API (`train.train_arm`, `evaluate.evaluate_arm`,
`spectral_anchor.build_bank/phases/anchor_gradient`) with
`torch.set_num_threads(3)`; the band-scale table uses
`AnchorConfig(features=512)` on `datasets.checkerboard()` at seed
`MASTER_SEED`.
