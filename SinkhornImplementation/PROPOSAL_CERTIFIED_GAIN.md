# Proposal: certificate-scheduled gain ("theorem-in-the-loop" drifting)

**EXTENSION — not part of the paper.**  Status: proposed 2026-07-07,
**implemented and evaluated 2026-07-08** (S7 in `run_sinkhorn.py` /
`RESULTS.md`; library support in `sinkhorn_drift.py`).  This file is the
resumable spec; see the "Actual results" section below for what was
verified versus what the original prediction got wrong, and the checklist
at the bottom for what remains.

## The idea in one line

The audited factorization `algorithm2Drift_eq_massProduct_centroidDiff`
splits Algorithm 2's drift into a **signal** (the self-normalized centroid
difference, which carries *all* of the identifiability and has certified
finite-sample robustness) times a **gain** (the raw affinity mass product
`P·Q`, which carries *none* of it and collapses exponentially off-support):

```
V(x_i)  =  (P_i * Q_i) . (C+_i - C-_i)          [Lean: Algorithm2Estimator.lean:346]
```

Replace the paper's implicit gain `P·Q` with a gain scheduled by the
*computable finite-sample certificate* for the signal — the empirical shadow
of the deviation bounds proved in `BalancedSampling.lean` /
`DenominatorTail.lean`.  Move fast where the theorems say the estimate is
trustworthy; move slowly (but never zero) where they do not.

## Formal provenance (which insights, from which theorems)

1. `algorithm2Drift_eq_massProduct_centroidDiff`
   ([Algorithm2Estimator.lean](../DriftingIdentifiability/Algorithm2Estimator.lean)):
   the exact gain × signal factorization.  All identifiability theorems in
   the repo factor through `C+ = C-`; the `P·Q` prefactor is inert for
   identifiability (it is branch-symmetric — after row normalization it is
   `P(1-P)`, blind even to the *sign* of local over/under-representation).
2. `algorithm2Drift_norm_le_affinityMass`: `‖V_i‖ ≤ 2·P_i·Q_i·R` — the
   formal statement that the paper's drift magnitude is *capped by the mass
   product*.  Kernel masses decay like `exp(-dist/τ)`, so the cap — hence
   the paper's effective step — collapses **exponentially** in the distance
   from support.  This is a proved freeze mechanism, not a conjecture.
3. `selfNormalizedCentroid_relative_perturbation` (B1,
   [BalancedSampling.lean](../DriftingIdentifiability/BalancedSampling.lean)):
   self-normalized centroids are the robust object — relative weight error η
   moves them ≤ ηD/(1-η) *independently of the absolute mass scale*, and they
   live in the convex hull, so their error is bounded by the diameter D no
   matter how small the masses are.  The signal never degenerates; only the
   gain does.
4. `selfNormalizedIndexed_deviation_prob_le`, `weightSum_deviation_prob_le`,
   `balancedTwoStepCentroid_deviation_prob_le`: the per-query concentration
   certificates.  Their hypotheses name exactly the empirical quantities
   (weight variance / mass floor ≈ ESS, denominator mass, hull radius) that
   the plug-in gain uses.  The gain is the theorem's own reliability dial.
5. `interactionFrameBound_of_probeScaling`
   ([SinkhornBalanced.lean](../DriftingIdentifiability/SinkhornBalanced.lean)):
   any per-probe positive rescaling of the field preserves the frame bound
   with constant `ρmin·c`.  The modification is literally a per-query
   positive scaling `ρ(i) = g_i/(P_i·Q_i) > 0`, so **identifiability of the
   modified estimator is already certified by an existing audited theorem**
   — no new Lean work is required for the headline transfer (a dedicated
   corollary would still be nice polish).
6. S2/S3 numerics: balancing depth makes masses deterministic (certificates
   tight), and S3 documents the failure this proposal attacks: from the far
   initialization both `t=1` and `t=3` sit at mode-mass error 0.300 for 300
   steps.

## The measured smoking gun (far init, S3 configuration, step 0)

Measured 2026-07-07 (seed 20260707, 400 particles at (5,5)+noise, τ = 0.2):

| t | median P·Q | median ‖C+−C−‖ | cosine(C+−C−, toward data) | median ESS+ | median ‖V_paper‖ |
|---|-----------|----------------|-----------------------------|-------------|-------------------|
| 1 | 8.8e-07 | 5.29 | 0.987 (min 0.968) | 77 / 400 | 4.7e-06 |
| 3 | 8.5e-03 | 5.88 | 0.993 (min 0.970) | 287 / 400 | 5.1e-02 |

The signal is nearly perfect (right direction, high ESS, certified-robust);
the paper multiplies it by 1e-6.  With η = 0.5 the paper moves 2e-6 per step
— frozen.  Balancing to t = 3 lifts the gain 10⁴× (that is *why* t = 3
crawls while t = 1 is stuck) but is still ~30× short at this budget.  Note
the certificate at far init is actually *tight* (high ESS, bounded hull), so
a certificate-scheduled gain would grant near-full speed exactly here.

## Estimator spec (v1)

```
V_cert(x_i) = g_i . (C+_i - C-_i),   with branches from the jointly
              balanced affinity at depth t (unchanged machinery)

g_i = gmax / (1 + lam * Ehat_i)
Ehat_i = Dhat * ( 1/sqrt(ESS+_i) + 1/sqrt(ESS-_i) )      # plug-in certificate
ESS±_i = (sum_j A±_ij)^2 / sum_j (A±_ij)^2               # per-branch, per-query
Dhat   = robust diameter of the y_pos ∪ y_neg batch (e.g. max coordinate
         range); the hull bound that makes Ehat bounded
```

- `Ehat` is the empirical shadow of the certified deviation bounds: an
  ESS-type variance term for each self-normalized branch, capped by the hull
  diameter (B1).  Because `Ehat ≤ Dhat·2`, the gain has an automatic floor
  `gmax/(1+2·lam·Dhat) > 0` — the paper's gain has **no floor**.  Exponential
  freeze becomes polynomial caution.
- Calibration: pick `gmax` so that `V_cert ≈ V_paper` on a healthy matched
  configuration (e.g. `gmax = median(P·Q)` on the `between` init at step 0),
  so near-support behavior is paper-like by construction; `lam = 1/Dhat`
  as default.
- Ablation ladder (each is a positive per-probe gain, all pre-certified):
  (a) paper `P·Q`; (b) `(P·Q)^γ`, γ ∈ {1/2, 1/4} — the cheapest interpolation;
  (c) `min(P,Q)`; (d) constant `gmax` (pure centroid-difference flow);
  (e) the certificate gain above.  (b) and (d) bracket (e).

## Implementation plan

1. [x] `sinkhorn_drift.py`: add `gain: str = "paper"` parameter to
   `compute_v_sinkhorn` (values `paper | power | min | const | cert`, plus
   `gamma`, `gmax`, `lam`).  Compute `P, Q, C+, C-, ESS±` from the
   already-built balanced matrix `A`; keep `paper` bit-identical to the
   pre-extension code by default.  Done exactly as specified, plus a
   `calibrate_gmax` helper (needed in practice: `const`/`cert` require a
   *fixed* reference scale from a healthy batch, not the current batch's
   own median, or the far init's own collapse just reintroduces itself —
   see "Actual results").
2. [x] `run_sinkhorn.py`: **S7** — the S3 particle-descent harness swept over
   gain modes × t ∈ {1, 3} × inits {between, far, collapsed}, same seed,
   same budget; an equilibrium-noise check (particles drawn from the target
   itself, all five gain modes on the identical batch); PLUS (added during
   verification, not originally planned) a step-by-step diagnosis trace at
   the far init recording gain magnitude, swarm mean position, and mode-
   nearest fraction, needed to explain why the mode-mass-error gains were
   modest rather than dramatic.
3. [x] `RESULTS.md`: S7 tables + honest reading (including the corrected
   prediction — see "Actual results" above).
4. [ ] Optional Lean polish (small, NOT done this round): a named corollary
   specializing `interactionFrameBound_of_probeScaling` to the
   gain-scheduled field, so the crosswalk table can cite one line instead
   of relying on the general lemma by inspection.  Extension-marked,
   audited tree.  Deferred for tokens/time, not difficulty — the transfer
   is a direct instantiation, no new proof idea needed.

## Success criteria / predictions (as originally written, 2026-07-07)

- **far init**: cert-gain (e) reaches mode-mass error < 0.10 by step 300 at
  t ∈ {1, 3} (paper: 0.300 flat).  Power-gain (b, γ = 1/4) also unfreezes
  but slower/mistuned; (d) unfreezes fastest but may overshoot near support.
- **between / collapsed**: no regression beyond +10% vs paper at matched
  budget (calibrated gmax should make near-support behavior comparable).
- **equilibrium**: matched-batch drift stays ~0 (`algorithm2Drift_matched_zero`
  covers every gain mode — the centroid difference vanishes identically);
  near-matched noise level comparable to paper.

## Actual results (S7, verified 2026-07-08) — what held up and what didn't

**Confirmed exactly as predicted:**

- Every alternative gain is a strictly positive per-query rescaling of the
  signal; identifiability needs no new theorem
  (`interactionFrameBound_of_probeScaling` already covers the class).
- Matched-batch drift is exactly zero for every gain mode (verified
  numerically to machine precision before S7 was even run).
- The mass-product collapse is real and the fix works: traced directly at
  the far init (`t = 1`), the paper's gain is `~1e-6`-scale (frozen — S3's
  error sits at exactly `0.300` for the full 300 steps); `cert`'s gain is
  `~0.1–0.2` throughout, five to six orders of magnitude larger, and the
  swarm's mean position genuinely travels from `(5, 5)` to `(1.94, 0.01)` —
  essentially mode 1's center `(2, 0)` — within 300 steps.  This is the
  headline mechanism the proposal predicted, and it is unambiguous.

**Did NOT hold up — corrected here rather than glossed over:**

- The prediction "`cert`-gain reaches mode-mass error < 0.10 by step 300 at
  the far init" was **wrong**.  Measured: far/t=1 `cert` error is `0.287`
  (mean, last 100 steps); far/t=3 `cert` is `0.272` — real improvement over
  the paper's `0.300`/`0.297`, but nowhere near `< 0.10`.
- **Why**: a second, independent failure mode that the proposal did not
  anticipate.  The far/collapsed initializations start with almost no
  spread across particles, so every particle computes nearly the same
  local drift and the *whole swarm moves as one coherent body* toward its
  nearest/dominant mode (here mode 1, since `(5,5)` is Euclidean-closer to
  `(2,0)` than to `(-2,0)`) instead of splitting into the correct 30/70
  proportions.  A per-particle gain multiplies every particle's signal by
  a comparable factor, so it cannot break this symmetry by itself — it
  fixes *how far* the swarm travels, not *how it partitions* once there.
  This is visible directly in the diagnosis trace in `RESULTS.md` (`frac.
  nearest mode 1` stays at `0.985–1.000` throughout, even as the mean
  position fully converges).
- **Unpredicted ranking surprise**: `cert` (the sophisticated,
  finite-sample-certificate-shadowed gain) is the *least* reliable of the
  four alternatives — it gives zero improvement at collapsed/`t=1`, ties
  for best at far/`t=3` and collapsed/`t=3`, and is *worse than the paper*
  at between/`t=3` (`0.112` vs `0.092`).  `const` — a single fixed
  reference gain with no adaptivity at all, i.e. plain constant-speed
  centroid-difference flow — is the most consistently strong performer.
  At this toy scale, dropping the mass-product attenuation mattered far
  more than the particular schedule chosen to replace it.  `min(P,Q)` is
  the weakest alternative, consistent with it still shrinking alongside
  whichever branch's own mass is smaller.
- **Equilibrium check**: no mode amplifies noise pathologically; `cert`
  actually has the *lowest* median equilibrium residual of all five modes
  (`~3–4x` below the paper's), which is a genuine point in its favor even
  though it underperforms on mode-mass recovery.

**Isolated follow-up this suggests** (not yet attempted): the homogeneity
failure is orthogonal to the gain question.  Injecting per-particle
diversity — a small repulsive term between particles, or per-step
diffusion noise, both standard in particle/mean-shift samplers — should
break the "moves as one body" symmetry and let a gain fix compound with it.
This is a distinct experiment (new mechanism, not a new gain mode) and has
not been implemented.

## Risks and honesty

- The paper's `P·Q` damping is an implicit "no evidence, no move" guard.  At
  *training* time queries are generator samples, so the self-branch mass `Q`
  is always healthy and `P·Q` small ⟺ data is far ⟺ exactly the must-move
  case — the guard only ever suppresses the escape.  Still, the collapsed
  init is the stress test where local crowding could interact badly with a
  gain floor; that is what S7's regression criterion is for.
- Constant-gain centroid-difference flow is mean-shift-classical; the claimed
  novelty is (i) deriving the surgery from a *certified* factorization that
  proves the discarded factor is identifiability-inert, (ii) scheduling the
  gain by the finite-sample certificate itself, (iii) inheriting the frame
  bound through an already-audited scaling theorem.
- Toy scale only; no FID/at-scale claim (explicitly deferred by the user).

## Runner-up idea (recorded for later)

**Cross-fitted / EMA-frozen balancing**: the entire difficulty of
`BalancedSampling.lean` (4δ → 16δR at t = 2, 32δ → 128δR at t = 3, open
level-1 tails) exists because the algorithm recomputes scalings on the same
batch it estimates with.  Estimate the scaling *functions* on an independent
half-batch (or as EMAs across steps) and apply them to the other half: the
weights become fixed functions conditionally, the existing fixed-weight SNIS
theorems apply *verbatim at every depth t*, the open t = 3 tail item
dissolves for the modified algorithm, and the exponential-in-depth constants
never arise.  "Change the algorithm so the theorem's hard part is about
nothing."  Composes with the certificate gain (frozen scalings ⟹
deterministic masses ⟹ tight certificates).  Not started.

## Checklist (tick as completed)

- [x] PROPOSAL_CERTIFIED_GAIN.md committed (this file)
- [x] `gain_schedule` / `calibrate_gmax` / `gain=` param in `sinkhorn_drift.py`
- [x] Regression check: `gain="paper"` bit-identical to pre-extension code
      (max err `1.53e-14` over 300 configs, same tolerance as S0)
- [x] Correctness check: `gain="power", gamma=1` reproduces the paper's
      `P*Q` gain exactly (identity check, max err `8.9e-16`) — confirms the
      signal/gain decomposition itself, not just the new code paths
- [x] Correctness check: matched-batch drift is exactly zero for every gain
      mode (machine precision)
- [x] S7 harness in `run_sinkhorn.py` (mode-mass table + equilibrium-noise
      table + far-init diagnosis trace)
- [x] `RESULTS.md` regenerated with S7; `paper` rows cross-checked to
      reproduce S3's original table exactly (built into the harness)
- [x] Honest reconciliation of predictions vs measurements (this file,
      "Actual results" section) — do not skip this step for future
      ablations; the discrepancy was the most useful output of S7
- [x] README.md / ResearchStatus.md / memory updated
- [ ] Optional named Lean corollary (see Implementation plan item 4)
- [ ] Follow-up experiment: per-particle diversity/repulsion to break
      far/collapsed swarm homogeneity (new mechanism; the natural next
      step this proposal's own results point to — NOT started)
- [ ] Runner-up idea (cross-fitted/EMA-frozen balancing) — NOT started,
      recorded above for later; independent of the gain question

## Resumption notes

- Everything needed to rerun is deterministic: `uv run --with numpy python
  SinkhornImplementation/run_sinkhorn.py` regenerates `RESULTS.md` in full
  (~225s; S7 is the majority of that cost — a `--quick`/reduced-grid flag
  would be a reasonable infra improvement if iteration speed matters before
  the next result is needed).
- If picking up the per-particle-diversity follow-up: the natural place to
  add it is `particle_descent` in `sinkhorn_drift.py` (add a small isotropic
  noise term or pairwise repulsion to the position update, gated by a new
  parameter with a zero default so existing behavior is unchanged), then a
  new S8 section in `run_sinkhorn.py` re-using the far/collapsed harness
  from S7 with `gain="const"` (the strongest performer found here) as the
  baseline to beat.
- If picking up the optional Lean corollary: instantiate
  `interactionFrameBound_of_probeScaling` (SinkhornBalanced.lean) with
  `rho(n) := g(anchors n) / (P n * Q n)` for a fixed positive reference
  function `g`; the positivity side-condition is exactly "gain is strictly
  positive", already guaranteed by construction for every mode in
  `gain_schedule`.

Run: `uv run --with numpy python SinkhornImplementation/run_sinkhorn.py`
(uv-managed CPython; no system python on this machine).
