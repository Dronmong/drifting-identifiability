# B2.5 resume — pre-run audit

**Date:** 2026-08-03
**Scope:** units 501 and 502, then aggregation. Unit 500 is complete and sealed.
**Verdict: GO on the mechanics. Two interpretive findings must be recorded
before the run, not after.**

Build order item 1 of
[`AnchoredSelfFeatureDriftingSpecification.md`](AnchoredSelfFeatureDriftingSpecification.md)
§13. Nothing has been run; this is the audit that precedes it.

---

## 1. Mechanical verification — all green

`numerics/encoder_independent_drifting/stage_b25/verify_resume.py`, read-only:

```
[PASS] preflight artifact          b25_preflight.json 9eedf9580a730455
[PASS] preflight verdict           GO
[PASS] hashed executable sources   27 files match
[PASS] protocol document           EncoderIndependentB25Protocol.md
[PASS] B1 freeze                   b1_freeze.json
[PASS] B2 freeze                   b2_freeze.json
[PASS] shifted data pool           cinic10-imagenet-only-b25-disjoint-balanced-600-seed-20260801
[PASS] unit 500 artifact           b25_unit_500.json 229dff09ddba1753
[PASS] unit 500 preflight binding  bound to this preflight
[PASS] unit 500 factorial          B0/B1/B1B2/B2
[PASS] units pending               completed [500]; pending [501, 502]
[PASS] orphaned checkpoints        none
[PASS] aggregate not yet written   b25_development.json
[PASS] disk headroom               186.6 GiB free
[PASS] cuda device                 RTX 4050 Laptop, 6.00 GiB total, unit 500 peaked at 4.97 GiB
```

**All 27 hashed executable sources are byte-identical to what unit 500 ran
against**, as are the protocol, both inherited freezes, and the shifted data
pool. The frozen constants match the preflight exactly:

| constant | value |
|---|---:|
| `b1_scale` | 0.4299860893300136 |
| `lambda_b1` | 0.9310125645774651 |
| `lambda_b2` | 0.00019294302093274076 |
| `tau_b2` | 7.085388360479058 |

The B2.5 regression suite passes 6/6: full-dose event counts, exact flow
pairing, component-gradient diagnostics, allocation disjointness and capacity,
metric sanity, decision logic, and exclusion-pool binding.

**Interruption state is clean.** `b25_driver.txt` records that the driver
printed `=== B2.5 unit 501 ===` before the pause, so unit 501 *started*. It was
killed before its first checkpoint at step 10 000, and no `b25_u501_*` file
exists. There is nothing to clean up.

---

## 2. Finding 1 — the realized correction dose is 2–4×, not the protocol's 0.25

The protocol §2 states: *"The two single corrections each have calibrated
event-gradient ratio 0.25."* Unit 500's own diagnostics, at the three declared
checkpoints, in-domain:

| arm | step 10 000 | step 20 000 | step 30 000 |
|---|---:|---:|---:|
| B1 — `‖λ₁g₁‖/‖g₀‖` | 2.77 | 1.83 | 2.91 |
| B2 — `‖λ₂g₂‖/‖g₀‖` | 3.06 | 2.65 | 3.19 |
| B1B2 — B1 share | 2.75 | 2.05 | 3.92 |
| B1B2 — B2 share | 2.49 | 3.21 | 3.50 |

**The realized ratio is roughly an order of magnitude above the calibrated
0.25.**

The mechanism is visible in the same artifact: the flow gradient decays over
training while λ is fixed. B0's flow-gradient norm runs **0.718 → 0.253 →
0.146** across the three checkpoints, a 4.9× decay. A ratio calibrated early
therefore inflates by roughly that factor by step 30 000. Measured decay
accounts for much of the gap; it is not claimed to account for all of it, since
the calibration units (410–412 for B1, 420–422 for B2) were separate runs.

**This does not license changing anything.** The same λ ran in unit 500, will
run in 501/502, and is identical across all four cells, so every within-B2.5
comparison is unaffected. Changing a coefficient now would break comparability
with a sealed unit and would violate the protocol's rule that *"a crash never
licenses changing a coefficient, bandwidth, threshold, source, arm, or
checkpoint."*

**What it changes is language.** The B2.5 results must not describe the
treatment as a light 0.25 event-gradient touch. The corrections dominate the
flow gradient at events. This also retires the same description in
[`EncoderIndependentB0B1B2Results.md`](EncoderIndependentB0B1B2Results.md) §6
— *"B2's weight is a light touch"* — which was already flagged there as an
untested characterization and is now measured to be wrong at the event level.

---

## 3. Finding 2 — the global clip binds mainly in the combined cell, confounding the interaction

Gradient clipping is a global norm clip at **1.0**
(`clip_factor = min(1.0, clip / total_norm)`,
[`stage_b25/core.py:221`](encoder_independent_drifting/stage_b25/core.py#L221)).
History rows are logged every 100 steps, and corrections fire every 10, so
**every logged row is a correction event**. Over unit 500's 301 logged events:

| arm | mean pre-clip norm | events clipped |
|---|---:|---:|
| B0 | 0.370 | **2.0%** |
| B1 | 1.078 | 41.2% |
| B2 | 0.950 | 29.9% |
| **B1B2** | **1.702** | **65.4%** |

At the declared checkpoints the combined cell's clip factor was **0.516 / 0.999
/ 0.906**, while B0, B1 and B2 each read exactly 1.000 at all three.

### Why this matters for the factorial

Protocol §2 anticipated a larger combined gradient and argued it was the defined
treatment, not a confound:

> Consequently the full combined cell can have greater correction compute and a
> larger total correction gradient than either single arm. That is not a
> confound in the factorial interaction; it is the defined joint treatment.

The global clip silently converts that into something else. It does not scale
the *corrections* down — it scales the **whole update**, flow included, and it
does so on 65% of the combined cell's events against 2% of B0's. So the combined
cell does not receive `L_flow + λ₁L_B1 + λ₂L_B2`; it receives a
state-dependently shrunk multiple of it, shrunk hardest exactly where the joint
treatment is strongest.

The interaction

```
I_Y = Y_B1B2 − Y_B1 − Y_B2 + Y_B0
```

presumes the joint cell delivers the sum of the two single treatments. It
delivers a scaled version of that sum.

### Direction of the bias, and what it does to the headline result

Unit 500's interactions are `I_rank = +4.18`, `I_recall = +0.0449`,
`I_rawE = +6.45`. The rank interaction is the headline: B1 appears to rescue
most of B2's geometry loss.

**Part of that could be dosing rather than complementarity.** B2's rank damage
scales with how much B2 pressure lands; the combined cell is clipped harder than
the B2 cell (65.4% vs 29.9% of events), so it absorbs *less* B2 pressure per
event. A cell that receives less of the harmful treatment will show better rank
without any genuine complementarity.

This is the same failure mode this repository has just flagged elsewhere: the
ASFD audit's Defect 2 — a shared cap on a combined treatment dilutes the
components and confounds the arm contrast. There, the shared cap was explicit.
Here, the global clip is a shared cap nobody labelled as one.

### What to do about it — nothing, this run

Changing the clip would break comparability with unit 500. Run 501/502 as
specified. What changes is the **reading of the aggregate**:

1. Report the per-arm clip rate and mean pre-clip norm alongside every
   interaction. They are already in `history`; they were simply never surfaced.
2. State the interaction as **confounded with clip incidence**, not as a clean
   estimate of complementarity.
3. If B1B2's rank advantage replicates in 501/502 **and** its clip rate stays
   far above the singles', the honest reading is that complementarity is *not
   separated from dose reduction* by this design, and a clip-matched or
   per-component-capped follow-up is required to separate them.

That follow-up is already specified for a different stage: ASFD's per-component
caps (`AnchoredSelfFeatureDriftingSpecification.md` §6.2) are exactly the repair.

---

## 4. Finding 3 — no within-unit recovery

B3 saved optimizer, model, RNG, history and timing recovery state every 1 000
updates and could continue a crashed unit from the last verified point. **B2.5
has none of that.** `core.py` invokes the checkpoint callback only at steps
10 000 / 20 000 / 30 000, and those save EMA weights for evaluation, not
training state.

Consequences for an 11-hour unattended run:

- a crash at update 25 000 of the B1B2 arm loses the entire unit (up to ~5.5 h);
- the surviving checkpoints then trip `run_unit.py`'s *"a planned B2.5
  checkpoint path already exists"* guard and **block the restart**.

`verify_resume.py` detects exactly that state and prints the removal commands,
so the recovery is mechanical rather than a diagnosis at 3 a.m. It is a
deliberate manual step: deleting checkpoints discards real work and should not
happen automatically.

**Memory headroom is the most likely crash cause.** Unit 500's B1B2 arm peaked
at 4.97 GiB reserved on a 6.00 GiB card — about 1 GiB of headroom. The GPU
currently reports 0 MiB in use. **Nothing else should touch the GPU during the
run.**

---

## 5. Unit 500, replayed through the real adjudicator

Not a new measurement — a confirmation that the hand-read of unit 500 matches
what `aggregate.py` will compute. Real `adjudicate_development`, real unit-500
rows, in-domain, step 30 000:

| condition | threshold | result |
|---|---|---|
| `drift_effect_retained` | ≥ 80% of B2's raw-drift reduction | **False** (75.6%) |
| `rank_restored` | rank/B0 ≥ 0.85 **and** > B2's | **False** (0.7938) |
| `precision_retained` | ≥ 90% of better(B0, B1) | True |
| `recall_retained` | ≥ 90% of better(B0, B1) | True |
| **`unit_promising`** | all four | **False** |

`B1B2` leads on recall (0.2148), KID (0.0529) and FID (110.7) and still fails
the preregistered rule. The aggregate requires 2 of 3 units promising
(`unit_wins_required`), so **B2.5 cannot be promising unless both 501 and 502
pass all four conditions** — a demanding bar given unit 500 missed two.

That is the correct prospective position and must not be relaxed after seeing
501/502.

---

## 6. What was built for this resume

| file | purpose |
|---|---|
| `numerics/b25_verify_resume.py` | new; read-only fail-fast verification of manifest equality, artifacts, freezes, data binding, orphaned checkpoints, disk and GPU headroom |
| `stage_b25/run_all.sh` | amended to invoke the verifier before any training and to timestamp each unit |

`run_all.sh` remains resumable: it skips completed units, refuses half-written
artifacts, and refuses to overwrite the aggregate. **No training code was
modified.** The 27 hashed sources are untouched, which is what makes 501/502
comparable to 500 at all.

### 6.1 A correction — the first placement of the verifier broke the run

This section originally read *"Neither file is in the preflight's hashed source
manifest, so neither invalidates the hash-bound preflight."* **That was wrong**,
and the first launch aborted on it:

```
RuntimeError: B2.5 executable sources changed after preflight
```

`source_manifest()` does not read the recorded keys — it *rebuilds* the manifest
with `paths.extend(sorted(HERE.rglob("*.py")))`
([`stage_b25/artifacts.py:62`](encoder_independent_drifting/stage_b25/artifacts.py#L62)).
The verifier was placed in `stage_b25/`, so it became a 28th entry in a manifest
recorded with 27, and `load_preflight`'s dict comparison failed. Every recorded
file still matched; the manifest simply gained a member.

The same trap exists one level up: `diagnostics.py:190`, `b1_freeze.py:38`,
`f3b_freeze.py:44` and `f1_k200.py:88` all glob `PACKAGE.glob("*.py")` over
`encoder_independent_drifting/`. **Neither that package nor `stage_b25/` can
accept a new module without invalidating a hash-bound preflight.** The verifier
now lives at `numerics/b25_verify_resume.py`, outside every such glob.

Two consequences worth keeping:

1. **The guard worked exactly as designed.** Nothing trained, no artifact was
   written, no checkpoint was created, and no state needed cleaning. The failure
   was immediate and legible.
2. **The verifier had the wrong check.** It verified that every *recorded* path
   still hashes correctly — necessary, but blind to an *added* file, which is
   precisely what happened. It now performs whole-dict equality against a live
   `source_manifest()`, the same comparison `load_preflight` makes, and reports
   added / removed / changed separately. That check is what turned green before
   the relaunch:

   ```
   [PASS] source manifest equality    27 files, manifest identical
   ```

A second process error, unrelated to the manifest: the first launch piped the
driver through `tee` without `pipefail`, so the failing run reported **exit 0**.
The relaunch sets `pipefail` so a failure inside the driver surfaces as a
failure.

---

## 7. Run command

```bash
bash numerics/encoder_independent_drifting/stage_b25/run_all.sh
```

Skips unit 500, runs 501 then 502, then aggregates. Projected **~11.0 h** at
unit 500's measured 5.49 h per unit, plus aggregation.

---

## 7.1 BLOCKED — the B1 freeze binding was broken by unrelated work

The relaunch cleared `load_preflight` and then failed one call later:

```
File "numerics/encoder_independent_drifting/b1_freeze.py", line 175, in load_freeze
  raise RuntimeError("B1 executable sources changed after freeze")
```

### What drifted

The **B1 freeze carries its own source manifest**, independent of B2.5's, built
by `sorted(PACKAGE.glob("*.py")) + sorted((PACKAGE / "tests").glob("*.py"))`
([`b1_freeze.py:38`](encoder_independent_drifting/b1_freeze.py#L38)) over the
whole `encoder_independent_drifting` package **including `tests/`**. Recorded
113 files; live 115:

| | path |
|---|---|
| ADDED | `encoder_independent_drifting/tests/test_pmf_s3.py` |
| ADDED | `encoder_independent_drifting/tests/test_pmf_s3r.py` |
| CHANGED | `encoder_independent_drifting/tests/run_all.py` |

All three are the S3/S3R work of 2026-08-03 — **after** unit 500 ran on
2026-08-01/02, which is why unit 500 succeeded and 501 cannot. The `run_all.py`
change is a 4-line registration of the two new test modules.

The B2 freeze is unaffected: 21 files, manifest identical.

### The recorded bytes are unrecoverable

| candidate | sha256 | |
|---|---|---|
| B1 recorded | `876d49e2c71fef61` | target |
| worktree now | `726a79dac591a302` | differs |
| worktree minus the 4 registration lines | `7ca9f4759b2b0b68` | differs |
| `HEAD:` blob, LF | `511d945c3ba681db` | differs |
| `HEAD:` blob, CRLF | `fe04c7bcf3e4b1fe` | differs |

`tests/run_all.py` has been committed exactly once, at `b89936a` (2026-07-31),
and that version is not what the B1 freeze recorded — so the file carried
uncommitted edits before the freeze and has carried more since. A search of all
**392 dangling and unreachable git blobs** found no object that reproduces it.

**The B1-era bytes cannot be restored from this repository.**

### Why no fix is local

Every repair path cascades:

- **Editing `b1_freeze.py`** to narrow its manifest self-invalidates — that file
  is itself inside `PACKAGE.glob("*.py")`.
- **Editing `stage_b25/artifacts.py`** to bypass `load_b1_freeze` changes a
  stage_b25 source, invalidating the B2.5 preflight. Regenerating the preflight
  changes its SHA-256, and `aggregate.py` requires **every** unit to carry a
  matching `preflight_sha256`
  ([`aggregate.py:38`](encoder_independent_drifting/stage_b25/aggregate.py#L38)).
  **Unit 500 would be orphaned**, so this costs the 5.49 h it already bought
  *and* compromises a freeze guard. Strictly worse than simply restarting.

### The structural lesson

`encoder_independent_drifting/tests/` is hash-bound by a completed confirmation.
**No test can ever be added to it without invalidating the B1 freeze**, and
nothing warns the author at the time — the breakage surfaces only when some
later stage tries to load the freeze. The same is true of the package root
(`diagnostics.py:190`, `f3b_freeze.py:44`, `f1_k200.py:88`) and of
`stage_b25/` (`rglob`).

Future stages should hash an **explicit dependency list** — as
`stage_b25/artifacts.py:_DEPENDENCIES` already does for the parent package —
rather than globbing directories that other work will legitimately grow into.

### State after two failed launches

Both failures were before any training. `assert_result_path_unused` and the
planned-checkpoint guard both passed, no `b25_unit_501.json` exists, and no
`b25_u501_*` checkpoint was written. **Nothing needs cleaning.**

### The decision

| option | cost | integrity |
|---|---|---|
| **A. Re-run B2.5 from scratch** under a fresh preflight bound to current sources | preflight (minutes) + **3 units ≈ 16.5 h**; discards unit 500's 5.49 h | clean |
| **B. Drop B2.5**, proceed to CAP-EMF-1 | none | clean, but precondition P2 of the ASFD specification goes unanswered |
| ~~C. Bypass the B1 guard~~ | same cascade as A *plus* a compromised freeze | rejected |

**Recommendation: A.** It is one night, it resolves the premise ASFD's §3.2
rests on, and unit 500's sealed artifact — while no longer aggregatable —
remains a valid informal fourth replicate to check the new units against.

The rerun should keep the design **identical**, not "fixed". The clip confound
of §3 is a real limitation, but changing the clip would make it a different
experiment and would forfeit that cross-check against unit 500.

---

## 7.2 How close was unit 500, and could 501/502 change the verdict?

Asked before committing 16.5 h to a restart. The margins look small; the
structure behind them does not.

### The two failing margins

| condition | unit 500 | threshold | shortfall |
|---|---:|---:|---|
| drift effect retained | 0.7564 | 0.80 | B1B2 energy must drop 0.304 (**1.9%**) |
| rank restored | 0.7938 | 0.85 | B1B2 rank must rise 0.787 (**7.1%**) |
| precision retained | 0.7930 | 0.7523 | passes, +5.4% |
| recall retained | 0.2148 | 0.1811 | passes, +18.7% |

Both misses are small. Taken alone, that reads like a coin flip.

### But no checkpoint of unit 500 passes both at once

The same trained models, evaluated at three checkpoints, in-domain:

| step | drift retained (≥0.80) | rank restored (≥0.85) |
|---:|---:|---:|
| 10 000 | 0.7865 ✗ | **0.8728 ✓** |
| 20 000 | **0.9504 ✓** | 0.7705 ✗ |
| 30 000 | 0.7564 ✗ | 0.7938 ✗ |

**Each condition is cleared at some checkpoint. Neither is cleared at the same
checkpoint as the other.** Three independent looks at one run, and every one
fails at least one condition — usually the one the previous checkpoint passed.

### The reason: the two conditions pull in opposite directions

Effective rank is computed on generated raw pixels and is **identical across
both evaluation instruments** (13.998 / 12.856 / 8.072 / 11.111 in-domain and
shifted). It carries no evaluation noise — it is a pure property of the trained
model. So the gate is asking the mechanism itself to move, and the mechanism has
essentially one axis: how much B1 neutralises B2.

Normalising unit 500 at step 30 000, with `t = 0` at pure B2 and `t = 1` at
pure B1:

```
drift fraction(t) = 1.0000 − 0.3182·t          (B1 retains 0.6818 of B2's reduction)
rank ratio(t)     = 0.5766 + 0.3418·t          (B2 0.5766, B1 0.9184)
```

The gate requires

```
drift ≥ 0.80  ⟹  t ≤ 0.629
rank  ≥ 0.85  ⟹  t ≥ 0.799
```

**These are inconsistent. No blend of B1 and B2 satisfies both.** The
interpolation line never enters the pass region — it misses by about 0.055 in
rank ratio at the best available point.

Passing therefore requires B1B2 to sit **strictly above** the B1–B2
interpolation on rank while holding its drift position — genuine selective
complementarity, cancelling B2's geometry damage but not its drift reduction.
Unit 500 sat **0.044 below** that line. The required swing is ≈ +0.10 in rank
ratio.

For scale, the checkpoint-to-checkpoint standard deviations within unit 500 are
**0.054** (rank retention) and **0.104** (drift retention). The required move is
roughly two standard deviations in a specific direction, and it is needed in
**both** remaining units, because a restart re-runs unit 500 under the same
seeds and the same 27 hashed sources with `use_deterministic_algorithms(True)` —
it will reproduce and fail again, so 2-of-3 means 501 **and** 502.

### The drift condition is also measuring the wrong thing

From §3: B2's raw energy is **13.933 against a real-versus-real floor of
14.103** — below the floor a correctly distributed sample cannot beat. So
condition 1 asks B1B2 to retain 80% of a reduction that is itself partly
estimator exploitation.

On a floor-relative reading, which is the defensible one:

| arm | energy | excess over floor | share of B0's excess removed |
|---|---:|---:|---:|
| B0 | 20.904 | +6.801 | — |
| B1 | 16.151 | +2.048 | 69.9% |
| B2 | 13.933 | **−0.170** | 102.5% — overshoots the floor |
| **B1B2** | 15.631 | **+1.528** | **77.5%** |

**B1B2 is the best arm that approaches the floor from above.** The gate marks it
down for not matching an arm that went through the floor.

### Answer

Unit 500 is close on the numbers and far on the structure. The two failing
conditions are in direct tension along the only axis this mechanism can move,
no blend point satisfies both, and one of the two is calibrated against a
partly-artifactual reference.

**Re-running is unlikely to change the verdict**, and if it did, the verdict
would not mean what it appears to mean. The information B2.5 was built to
produce — *do the spectral anchor and the raw drift term compound?* — is
already visible in unit 500: **they do, substantially and super-additively on
rank (`I_rank = +4.18`), but not selectively**, and B1B2 leads on recall, KID
and FID while being the best-behaved arm on the floor-relative drift axis.

That is a usable answer to ASFD's premise. It is not a `promising` verdict under
a heuristic the protocol itself calls *"a prospective development heuristic, not
a calibrated test."*

---

## 8. Unrelated finding: uncommitted work

`git status` shows a substantial body of uncommitted work, including the plan
this program is currently building on:

- **untracked:** `AnchoredSelfFeatureDriftingResearchPlan.md`, five
  S3/PMF/Sinkhorn documents, and the entire `stage_pmf/`, `stage_pmf_r/` and
  `stage_sinkhorn/` packages — the S3R implementation and results;
- **modified:** `MeanFlowDriftingMechanismAnalysis.md`,
  `encoder_independent_drifting/tests/run_all.py`.

None of it affects this run — no B2.5-hashed source is among them, which
`b25_verify_resume.py` confirms. But wipe protection was an explicit concern
earlier in this program, and the S3R package is the foundation the whole ASFD
sequence depends on. It should be committed.

---

## 9. Resolution — B2.5 is closed as answered-with-qualification

**Decision: do not restart. B2.5 is closed on unit 500.**

### What B2.5 was for

One question: *do the B1 spectral anchor and the B2 raw drift correction solve
complementary problems, or do they trade off?* It was queued ahead of ASFD
because ASFD proposes stacking a **third** correction term on that same
mechanism.

### What unit 500 answers

**They compound, substantially and super-additively — but not selectively.**

| outcome | interaction `I = B1B2 − B1 − B2 + B0` |
|---|---:|
| effective rank | **+4.18** |
| recall | **+0.0449** |
| raw drift energy | +6.45 |
| precision | −0.0078 |

B1 recovers most of B2's geometry loss (rank/B0 from 0.577 to 0.794) and B1B2
leads every appearance metric — recall 0.2148, KID 0.0529, FID 110.7, all best
of the four cells. It is also the best-behaved arm on the floor-relative drift
axis (§7.2), removing 77.5% of B0's legitimate excess while remaining above the
floor that B2 goes through.

What it does **not** do is separate the two effects: the same B1 pressure that
restores rank also gives back a quarter of B2's drift reduction, roughly in
proportion (§7.2). There is no blend point that satisfies both preregistered
thresholds.

### Why this is enough, and why more units would not help

1. **The verdict cannot realistically change.** No blend of B1 and B2 satisfies
   both conditions; the required super-additive selectivity is ≈ +0.10 in rank
   ratio against checkpoint standard deviations of 0.054–0.104, and it is needed
   in both remaining units because a restart reproduces unit 500
   deterministically.
2. **One of the two conditions is miscalibrated.** Condition 1 benchmarks
   against B2's reduction, and B2's energy sits *below* the real-versus-real
   floor. It rewards estimator exploitation.
3. **The interaction is confounded with clip incidence** (§3). Three units of a
   confounded interaction is not three times the information; the confound is
   systematic and would replicate.

### What this licenses, and what it does not

**Licensed:**

- ASFD precondition **P2 is satisfied with qualification** — stacked corrections
  compound rather than cancel, at development scope, on one unit.
- The floor-relative reading of drift energy is adopted as the correct axis.

**Not licensed:**

- no `promising` verdict; `aggregate.py` was never run and no
  `b25_development.json` exists;
- no claim of replication — this is one training unit;
- no promotion of `B1B2` to anything.

The aggregate is deliberately left unwritten. Unit 500's artifact remains sealed
and hash-verified as the record.

### Consequences carried into ASFD

| finding | change to the ASFD specification |
|---|---|
| condition 1 rewarded going below the floor | Stage D's raw-energy condition is stated as **floor-relative excess**, two-sided |
| the global clip acted as an unlabelled shared cap (§3) | per-component caps, already specified in §6.2, are confirmed as necessary rather than merely preferable |
| B1 restores rank but not selectively | ASFD's third term must be justified on evidence that it is *not* simply another point on the same tradeoff line |

### The structural repair worth making regardless

`encoder_independent_drifting/tests/` is hash-bound by a completed
confirmation, so **no test can be added to it without invalidating B1** (§7.1),
and nothing warns the author at the time. Any future stage should hash an
explicit dependency list, as `stage_b25/artifacts.py:_DEPENDENCIES` already does
for the parent package, rather than globbing directories that later work will
legitimately grow into. `stage_cap` follows that rule from the start.
