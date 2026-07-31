# Encoder-Independent Kernel Drifting — reformed screen protocol

*Successor design after the failed Phase-1 exit gate. Implements the seven
reforms of `EncoderIndependentPhase1Diagnosis.md`. Frozen before any
successor screen is run; results will go to a separate results document.*

This protocol replaces `EncoderIndependentPhase1Protocol.md`. That document
remains as the audit trail of what was actually run in Phase 1.

---

## 0. Why the previous screen is not simply repeated

Phase 1 passed a four-condition Phase-0 gate and still produced an
uninterpretable result. Three defects made that possible, and each has a
matching reform:

| Defect | Reform |
|---|---|
| No arm reached the target; the budget was frozen from the baseline's *coverage*, which saturates at 1.0 while precision is 0.199 | **R1** skyline arm, admissibility, oracle-derived budget |
| The geometry loss was pinned at exactly η², so the screen ran with no convergence signal and the exact-zero argument was unreachable | **R2** honest unnormalized loss |
| Kernel health was certified without ever asking whether the field's zero-set contains the target | **R5** zero-set gate condition |

---

## 1. Implemented reforms

All seven are implemented and unit-tested (86 tests, all passing).

### R1 — skyline arm and admissibility *(`oracle.py`)*

- `train_skyline` trains the **same** `OneStepGenerator` with a
  sliced-Wasserstein-2 objective: no kernel, no bandwidth, no feature map, no
  field normalization, so it shares none of the suspected defects.
- `admissibility` declares a target admissible only if the skyline reaches
  `ADMISSIBLE_PRECISION_FRACTION = 0.5` of the target-vs-target **null
  precision**. Precision, not coverage — coverage saturates early and is what
  produced the Phase-1 under-budget.
- `sufficient_budget` sweeps `(300, 600, 1200, 2400)` and returns the
  smallest budget at which the skyline clears the bar. **The budget is
  derived from the skyline, never from a candidate or a baseline arm.**
- A target that never clears is **dropped**. The bar is not lowered.

### R2 — the objective's value means something *(`objectives.py`)*

`reported_geometry_loss(raw_drift_rms, eta) = eta² · rms²` recomputes the
loss from the field magnitude measured *before* normalization. RMS
normalization is retained for step-size control on the **update**; every
report, stopping rule and gate reads the unnormalized value. Logged per
branch as `loss_geometry_unnormalized` and as a mixture-weighted total.

Three tests pin this down, including one that reads an actual training log
and asserts the reported loss varies while the trained loss stays constant —
the old test used a literal zero drift and so passed against the defect.

### R3 — constrained movement *(`kernel_gradient.py`)*

New direction mode `projected_kernel_gradient`: the kernel-gradient field is
projected onto `span{Y_j − x_i}`, the subspace the standard displacement rule
is confined to by construction. The structured kernel still decides which
neighbours matter and how much; the off-manifold freedom is removed.

Implementation note: `Y_j − x_i = (Y_j − Ȳ) + (Ȳ − x_i)`, so the span is a
shared subspace plus one probe-dependent direction. One SVD per call suffices
(QR is wrong here — centring drops the rank, and a reduced QR returns a
spurious ordering-dependent column, which made the projector depend on the
batch permutation until it was caught by test 4). Reports
`projection_retained_fraction`.

### R4 — re-specified composite *(`metrics.py`)*

`normalized_geometry_score_v2` over `(ed2, sw1, patch_ed2, nearest_real)`:

- **`spectral_l1` dropped from scoring.** Its null is ≈6e-4, three orders of
  magnitude below every other component, so its ratio ran to 48–346 and
  dominated the geometric mean. Still reported.
- **`off_support` replaced by `nearest_real`** — median distance to the
  nearest real sample in target-NN units, null 1.0 by construction.
  `off_support` pins at `1/null` for every arm that has left the support and
  stops discriminating, which is the regime Phase 1 actually operated in.
- Any component whose null falls below `MIN_TRUSTWORTHY_NULL = 1e-3` is
  excluded and named in `untrustworthy_nulls`, rather than silently emitting
  a huge ratio.
- `component_verdicts` reports the per-component win/loss so a gate can
  require the sign to hold on a majority individually.

v1 is retained unchanged so `phase1_screen.json` stays reproducible.

### R5 — zero-set gate condition *(`run_phase0_gate.py`, G0.5)*

Optimizes a **free particle cloud** on each candidate field — no generator,
no optimizer state — with a linearly decaying step, fresh target batches each
step, and the residual read on a **held-out** batch. Reports
`residual / (q=p floor)`.

Two construction details were forced by measurement: a constant step on a
unit-RMS field cannot settle (the R2 defect again), and a fixed positive
batch lets the cloud memorize 64 points and drive the residual *below* the
floor. Both are fixed.

**This condition reproduces the Phase-1 ranking before any generator is
trained**, at a cost of one short particle run per configuration. Measured
over 4 targets × 3 seeds at 300 steps (`phase0_gate.json`, G0.5); threshold
2.00, i.e. twice the raw kernel-gradient ratio:

| configuration | residual / floor | verdict | Phase-1 outcome |
|---|---:|---|---|
| `raw::kernel_gradient` | 0.84 | reaches | A1 — best arm overall |
| `raw::standard` | 1.02 | reaches | A0 — second |
| `randconv::standard` | 0.85 | reaches | — |
| `randconv::kernel_gradient` | 1.14 | reaches | A6 — best structured + anchor |
| `randconv::projected_kernel_gradient` | 1.16 | reaches | *(new)* |
| `wavelet::standard` | 0.68 | reaches | A3 — beat A4 by 1.86× |
| `wavelet::kernel_gradient` | **2.39** | **plateaus** | **A4 — failed the gate** |
| `wavelet::projected_kernel_gradient` | **2.21** | **plateaus** | *(new)* |

Every configuration that plateaus corresponds to a Phase-1 arm that failed,
and every configuration that reaches corresponds to one that did comparatively
well. The Phase-1 screen cost four hours; this predicts its ordering in
minutes.

### R6 — coarse-to-fine anchor schedule *(`spectral_anchor.py`)*

`band_weights(config, progress)` holds the coarsest band at full weight from
step 1 and ramps each finer band in over its own slice of a declared warmup
window, never below `schedule_floor`. Weights are normalized to mean one so
`lambda_anchor` keeps its meaning as the schedule opens. The **audit bank
always runs at full declared width** (`progress=None`), and
`loss_anchor_full_band` is reported alongside the scheduled loss, so a
narrower training bank can never make the anchor look better than it is.
`band_schedule="fixed"` reproduces Phase-1 behaviour exactly.

### R7 — target suite

Handled operationally by R1: the suite is whatever survives admissibility at
the oracle-derived budget. No target is retained on the strength of being
interesting.

---

## 2. Frozen design for the successor screen

| Item | Value |
|---|---|
| Targets | those passing R1 admissibility, from `datasets.suite()` |
| Budget | smallest of (300, 600, 1200, 2400) at which the skyline clears the bar, per target |
| Seeds | 3 |
| Score | `normalized_geometry_score_v2` |
| Batch / controller / audit | 64 / 32 / 32, disjoint every step |
| Generator | one `OneStepGenerator`, identical init across arms in a cell |
| Reported | scheduled *and* full-band anchor loss; unnormalized geometry loss; `projection_retained_fraction`; `collapsed_row_fraction` |

### Arms

| ID | Anchor | Geometry | Direction | Note |
|---|---|---|---|---|
| S0 | — | raw pixel | standard | Phase-1 A0 |
| S1 | — | raw pixel | kernel gradient | Phase-1 A1, the arm to beat |
| S2 | scheduled | — | anchor gradient | tests R6 in isolation |
| S3 | — | wavelet | standard | Phase-1 A3 |
| S4 | — | wavelet | **projected** kernel gradient | tests R3 |
| S5 | scheduled | wavelet | **projected** kernel gradient | R3 + R6 |
| S6 | scheduled | random conv | **projected** kernel gradient | best structured family in Phase 1 |
| **SKY** | — | — | sliced Wasserstein | **skyline, not an arm; never in a gate** |

`kernel_gradient` (unprojected) is retained as the R3 ablation in place of
`standard` wherever G0.5 says it reaches the floor.

---

## 3. Entry conditions

The successor screen may not start until **all** hold:

1. The Phase-0 gate passes including **G0.5**.
2. At least three targets are admissible at a budget ≤ 2400 steps.
3. For every arm to be run, G0.5 reports its `(geometry, direction)` pair as
   reaching the floor. **An arm whose field provably cannot reach the floor
   is not run** — that is the whole point of R5, and A4 would have been
   excluded by it.

Condition 3 is the sharpest change: Phase 1 spent four hours measuring an
arm that a twenty-second test predicts cannot work.

---

## 4. Exit gate

| ID | Condition | Threshold |
|---|---|---|
| **S.1** | a structured arm beats S1 | paired v2 ratio ≤ 0.90, bootstrap upper bound < 1, **and** a majority of components individually |
| **S.2** | it stays within the skyline | v2 ratio vs SKY ≤ 1.25 |
| **S.3** | the anchor covers geometry blindness | audit-bank anchor 6/6 where the geometry is blind on ≥ 1 |
| **S.4** | the anchor is present and descending | gradient share ≥ 0.05 in ≥ half of cells **and** full-band anchor loss falls ≥ 25% |
| **S.5** | robustness | ratio < 1 on a majority of targets and on every seed |
| **S.6** | the objective is not vacuous | unnormalized geometry loss falls ≥ 25% from its first logged value |

S.4's descent requirement and S.6 are new; both are direct consequences of
defects that Phase 1 could not see.

---

## 5. What this protocol still cannot conclude

Unchanged from Phase 1, and worth restating because the reforms make the
screen *sounder*, not *bigger*:

- nothing about CIFAR-10, ImageNet, FID, natural images or the paper's model;
- nothing about the exactness of the finite random-feature anchor;
- nothing about cost advantage — the structured arms remain 4–20× the
  wall-clock of raw pixel drifting at this scale, and the ledger is reported
  with every result.

One limitation is new and should be stated plainly: the reforms were derived
from a diagnosis run on **three targets at single seeds**. They are
well-motivated and unit-tested, but their effect on generation *quality* is
untested — no successor screen has been run.

The clearest instance is R3. Its projection does **not** rescue the wavelet
zero-set: G0.5 puts `projected_kernel_gradient` at 2.21 against
`kernel_gradient`'s 2.39, both plateauing well above the threshold.
Constraining the update *direction* was never expected to repair a wrong
*objective*, and the measurement confirms it does not. R3 is therefore
retained for what it does do — it removes a measured adversarial channel and
is the honest form of the rule — not as a fix for the wavelet family.

The consequence for the arm table above is concrete: under entry condition 3,
**S4 and S5 as written are not runnable**, because their `(wavelet,
projected_kernel_gradient)` pair plateaus. Either they use
`wavelet::standard` (which reaches the floor, and which beat the
kernel-gradient rule in Phase 1), or the wavelet family is dropped in favour
of `randconv`, whose every direction rule reaches the floor. That decision
belongs to whoever runs the successor screen, and it must be made from G0.5
before the screen starts — not after seeing scores.
