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

## 8. Unrelated finding: uncommitted work

`git status` shows a substantial body of uncommitted work, including the plan
this program is currently building on:

- **untracked:** `AnchoredSelfFeatureDriftingResearchPlan.md`, five
  S3/PMF/Sinkhorn documents, and the entire `stage_pmf/`, `stage_pmf_r/` and
  `stage_sinkhorn/` packages — the S3R implementation and results;
- **modified:** `MeanFlowDriftingMechanismAnalysis.md`,
  `encoder_independent_drifting/tests/run_all.py`.

None of it affects this run — no B2.5-hashed source is among them, which
`verify_resume.py` confirms. But wipe protection was an explicit concern
earlier in this program, and the S3R package is the foundation the whole ASFD
sequence depends on. It should be committed.
