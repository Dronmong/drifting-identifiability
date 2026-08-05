# Pre-run audit — CAP-EMF-1 foundation and the ASFD correction

**Date:** 2026-08-04
**Scope:** both packages, together, before any GPU budget is committed
**Status: six defects found and fixed. 47 ASFD tests + 42 CAP tests pass; preflight GO.**

Artifacts: `stage_cap/cap_preflight.json` (sha256 `438b88f84472b0db…`), 17 hashed
sources.

---

## 1. Why both arms were audited together

They now share code. `stage_asfd` hashes `stage_cap/model.py`, `config.py`,
`objective.py` and `diagnostics.py` as dependencies, because a different trunk
is a different feature geometry. A defect in the foundation is a defect in the
correction, and one of the six below was found in ASFD and fixed in CAP.

---

## 2. The six defects

Ordered by what they would have cost.

### 2.1 The band probes were not band-limited — G7 was measuring nothing

**Severity: highest.** G7 decides whether the ASFD arm proceeds at all, and the
audit that specified it argued the *lower* bound is the point.

`_inverse_haar` recovered the analysis matrix by pushing the standard basis
through `haar_transform`. Row `i` of that result is `H e_i` — column `i` of `H` —
so the assembled matrix is `H^T`. Synthesising a coefficient row vector `c`
needs `c @ H`, i.e. `c @ transformed.T`. The code wrote `c @ transformed`.

Measured before the fix:

| probe | result |
|---|---|
| Haar round-trip error | **4.31** (should be ~5e-7) |
| LL band energy landing outside LL | **74.0%** |
| HH band energy landing outside HH | **70.9%** |

So the gate would have produced four numbers that look like per-band
sensitivities and are not — and it would have *passed*, because four
roughly-equal wrong numbers sit comfortably inside `[0.25, 4.0]`. A silent pass
is worse than a failure here: it licenses the arm.

Fixed, and pinned by two tests — round-trip inversion, and band leakage below
1% in every band.

### 2.2 Normalization calibration could not run at production shape

`calibrate_normalization` computed matched-location distances by broadcasting
`[N, 1, L, C] − [1, N, L, C]`. At the production shape (256 images, 66
locations, 384 channels) that intermediate is **1.66×10⁹ elements = 12.4 GB in
float64**. It would have OOM'd on first contact with real data.

Replaced with per-location `cdist`: `[L, N, N]` = **33 MB**, a 375× reduction.

### 2.3 Bandwidth calibration used the wrong distance distribution

`calibrate_bandwidth` flattened `[N, L, C]` to `[N*L, C]` and took the first
`ess_samples` rows. That has two consequences, both wrong:

- it calibrates on distances **between locations of the same few images**,
  when the field only ever compares location `ℓ` of one image with location `ℓ`
  of another;
- at 256 samples over 66 locations it silently uses **fewer than four images**.

Replaced with per-location `cdist` pooled across locations, which is both the
correct distribution and the same tensor the fix in §2.2 needed. A test builds
locations with large mutual offsets and asserts the calibrated scale tracks the
within-location spread rather than the between-location one.

### 2.4 Every checkpoint would have recorded zero parameters

In `run_unit.py` the checkpoint callback read `parameters["count"]` from a
closure that was assigned **after** `train_cap_unit` returned. Every checkpoint
written during a 40-hour run would have carried `parameter_count: 0`.

The count is now taken from the state dict being written — exact, and it cannot
desynchronise from the thing it describes.

### 2.5 A resumed run would have lost its earlier checkpoints

The callback stored records in a dict local to `run_unit`. The recovery file
carries `outcome.checkpoints`, not that dict, so after any interruption the
final artifact would have listed only the checkpoints written *after* the
resume.

This matters more than it sounds: a resume is the expected case at pessimistic
GPU scaling, and the sealed evaluator indexes `unit["checkpoints"][step]`.
The callback now returns its record and the training loop stores it on the
outcome. `outcome.snapshots` is also carried in the recovery payload, which it
was not.

Pinned by a test that trains, interrupts, resumes, and asserts both checkpoints
survive.

### 2.6 The evaluator was frozen but its arithmetic was not

`stage_cap/evaluation.py` was hashed into the manifest — deliberately, so it
cannot be authored while looking at training curves. But the modules it calls
to compute FID, KID, precision/recall and the nearest-reference audit
(`appearance.py`, `fid.py`, `stage_b2/metrics.py`, `stage_b25/evaluation.py`)
were **not** hashed. The reported numbers could have changed without the
preflight noticing.

All four added. `stage_asfd` was missing `stage_cap/diagnostics.py` for the same
reason and now has it.

A new test walks the AST of every hashed module, resolves its
`from ..x import` statements, and fails if any resolved dependency is absent
from the manifest — so this class of gap cannot silently return.

---

## 3. Smaller items fixed

- **Abort reasons were not deduplicated.** A sustained condition appends on
  every correction event; over ~5 000 events the artifact would fill with copies
  of one sentence. Now one entry per distinct cause.
- **A scripted rewrite mangled UTF-8** in `gradients.py` during this audit
  (`λ` → `Î»`). Repaired by moving the docstring to ASCII, and a test now
  scans every hashed ASFD module for mojibake.

---

## 4. What the audit checked and found sound

- **`combine` is a plain sum.** A test feeds a component that is exactly
  anti-parallel to the primary gradient and asserts it survives at full
  magnitude. If projection ever creeps back, that test fails.
- **Caps are independent.** Two components both far above their caps land at
  exactly 0.10 each, and the realized total exceeds any single cap — the
  defined joint treatment, per B2.5's factorial argument.
- **Separate squares block cancellation.** The counterexample is explicit:
  two opposite fields whose mean is exactly zero still produce nonzero averaged
  energy, asserted in both directions.
- **Negatives keep their graph** and the frozen trunk passes input gradient
  while its parameters receive none — the `no_grad` trap.
- **Mild opposition does not abort; negation does.** 200 events at cosine
  −0.3 pass; at −0.95 abort.
- **ESS excludes the self-match**, so a nominal target cannot be met through
  the zero-distance diagonal as it was in the Phase-25 legacy calibration.
- **Descriptor globals are permutation-invariant** and the local 64 are not,
  verified by a token shuffle.
- **The EMF derivative** still converges to the float64 JVP at first order
  (rates 9.8–11.2 across runs).
- **Restart determinism** reproduces `raw_mse` to 12 significant figures.

---

## 5. Two design constraints surfaced, not fixed

Neither is a defect; both are properties the design should own explicitly.

**The smallest admissible radius is bounded by the cloud size.** At an ESS
fraction of 0.10 the effective neighbourhood is `0.1n`, so a small cloud puts
almost all the weight on one neighbour and the max-weight ceiling refuses it.
This is why the positive side is 256 and asymmetric with the negatives. A test
asserts a 16-sample cloud cannot support a local radius and that the ladder
reports every rung it tried.

**An absolute tail floor is incompatible with a radius set spanning an order of
magnitude.** The floor inherited from the earlier audit was `p05 ≥ 0.10`, which
against a radius set starting at 0.10 demands the 5th percentile reach the
median. No distribution satisfies that, so **every local radius would have been
rejected and the multi-radius set would have silently collapsed back into three
broad fields** — the exact failure it exists to prevent, wearing the costume of
health gates working correctly. The floor is now a fraction of the requested
radius.

---

## 6. Known limits carried into the run

1. **`development.py` does not exist.** The three-arm Stage D fork is the thin
   orchestration layer over everything audited here, and it cannot be
   meaningfully tested until a foundation checkpoint exists. Writing it now
   would mean shipping untested code; it is deliberately deferred.
2. **The qualification gate has never run against a real trunk.** Every
   component is tested, and the gate has been exercised on synthetic and
   randomly-initialised trunks, but its thresholds are prospective. G7's
   `[0.25, 4.0]` band and G8's CKA ceiling are declared, not calibrated.
3. **Manifold statistics at 10k×10k** allocate roughly 800 MB per distance
   matrix in the sealed evaluation. Comfortable on a rented box, tight on a
   laptop.
4. **The audit found six defects in code that already passed 80 tests.** That is
   the honest base rate to carry into reading the run's results: a green suite
   is evidence, not proof.

---

## 7. Verdict

**GO for the $1 cloud benchmark.** The training run remains blocked behind
`--i-have-authorized-the-budget-run`, and that opt-in should follow the
benchmark's measured cost, not this audit.

Two of the six defects (§2.1, §2.2) would have been discovered only *after*
spending GPU budget — one as an OOM on first contact with real data, the other
never, because it would have passed.
