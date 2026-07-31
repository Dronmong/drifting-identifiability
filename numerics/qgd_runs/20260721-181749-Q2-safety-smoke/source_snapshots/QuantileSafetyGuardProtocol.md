# Quantile Safety Guard development protocol

**Protocol ID:** `QGD-development-Q2-safety-v1`  
**Scope:** post-Q1 mechanism repair on the already-open QGD development registry  
**Status:** exploratory development, not sealed confirmation

## Motivation

QGD-v1 failed its registered advancement gate.  The audit found four coupled
problems: its fallback omitted the planned step-size calibration; the quantile
Adam proposal was commonly 4--13 times larger than the paper proposal; two
independent Adam histories manufactured proposal conflict despite mostly
aligned current gradients; and the empirical rank field remained much noisier
than the paper field near equilibrium.

Q2 replaces the persistent competing optimizer with a one-sided safety check.
It asks only whether the ordinary paper-Adam update is robustly predicted to
increase empirical quantile mismatch.  It never forces positive quantile
progress.

## Frozen mechanism

At each suffix update:

1. Compute the ordinary paper field and its Adam proposal `delta_D` on the
   training batch.
2. Compute current stop-gradient rank-surrogate gradients `g_Q1` and `g_Q2`
   on two independent generator/target batches.  There is no quantile Adam
   state or quantile momentum.
3. Compute a target-only finite-sample noise floor by rank-matching the two
   independent target batches.  The guard is eligible only when the RMS
   generator-to-target rank mismatch exceeds a declared multiple of this
   target-to-target null mismatch.
4. Require the two quantile gradients to agree in the local Adam dual metric,
   and require both to classify `delta_D` as quantile ascent by a declared
   metric cosine margin.
5. If eligible, solve the exact projection

   ```text
   minimize  ||delta - delta_D||^2_(M_D^-1)
   subject to g_Q1 dot delta <= 0
              g_Q2 dot delta <= 0.
   ```

6. Apply the projected point only if its correction is no more than the
   declared fraction of `||delta_D||_(M_D^-1)` and it does not turn a locally
   descending update into local ascent.  Otherwise execute the historical
   paper Adam operation exactly.
7. The local Adam moments advance from the local gradient regardless of an
   accepted small projection.  No second optimizer state exists.

The target-vs-target noise test is the training-only stopping/annealing rule:
as generator mismatch reaches finite-batch resolution, the safety mechanism
becomes ineligible rather than continuing to chase empirical rank noise.

## Registered arms

- `lbqcd`: frozen LB-QCD prefix followed by the historical paper suffix.
- `qsafe-balanced`: agreement `0.25`, ascent cosine `0.05`, signal/null `1.50`,
  correction cap `0.25`.
- `qsafe-permissive`: agreement `0.10`, ascent cosine `0.02`, signal/null
  `1.25`, correction cap `0.25`.
- `qsafe-strict`: agreement `0.50`, ascent cosine `0.05`, signal/null `2.00`,
  correction cap `0.15`.

All arms share the exact LB-QCD handoff, suffix training stream, evaluation
stream, and two-bank checkpoint selector.  Safety arms also share the same
independent diagnostic stream.  Their extra generator evaluation, target
sampling, rank sorting, and backward-equivalent work are recorded explicitly.

## Advancement gate

The best registered safety arm by target-balanced selected ED2 must satisfy:

- selected ED2/LB-QCD at most `0.98`;
- selected SW1/LB-QCD at most `0.99`;
- every target-family selected-ED2 ratio at most `1.05`;
- both initialization selected-ED2 ratios at most `1.02`;
- no divergence;
- intervention frequency below `0.20`;
- cap-rejection frequency below `0.10`.

The first smoke run is a plumbing and mechanism check.  The screen profile is
the decision run.  Because the QGD registry was inspected during the Q1 audit,
even a passing Q2 screen would justify a fresh-registry confirmation, not a
general performance claim.

## Required diagnostics

Record endpoint and independently selected ED2, SW1, and W2; family and
initialization splits; robust-ascent, intervention, cap-rejection, and
local-rejection frequencies; active-action counts; correction norms and
ratios; gradient agreement; ascent cosine; signal-to-null ratio; divergence;
and the complete compute ledger.

