# Stage B2.5 protocol — true B0 / B1 / B2 / B1+B2 development factorial

**Status: IMPLEMENTED, AUDITED SETUP; NOT EXECUTED. DEVELOPMENT ONLY.**

This stage asks whether the B1 spectral anchor and B2 normalized-Laplace
correction solve complementary problems inside the working bridge generator:

- B1 retained global geometry better;
- B2 reduced the theory-aligned raw drift energy; but
- B2 reduced effective rank by roughly 38–40% in all three confirmation units.

B2.5 does not reconfirm B1 or B2, promote a model, or make an identifiability
claim. It is deliberately prepared now and deferred until after B3/B4.

## 1. Primary design: a genuine factorial

All four cells are retrained on each new paired unit.

| arm | objective at a scheduled correction step |
|---|---|
| `B0` | flow matching |
| `B1` | flow + complete frozen B1 anchor |
| `B2` | flow + complete frozen B2 correction |
| `B1B2` | flow + complete frozen B1 anchor + complete frozen B2 correction |

The combined cell is **not interleaved**. Both corrections fire every ten
steps, exactly as in their single-correction cells. This is required for a
true `2 × 2` factorial: “B1 present” and “B2 present” have the same treatment
level in the single and combined cells.

At a combined event, the implementation backpropagates flow, weighted B1, and
weighted B2 sequentially before one global clip and optimizer step. Gradient
linearity makes this the gradient of

\[
L = L_{\mathrm{flow}} + \lambda_1 L_{B1} + \lambda_2 L_{B2},
\]

while avoiding simultaneous retention of both trajectory graphs. An optional
compute-matched interleaved arm is a different experiment and is not included
in this factorial.

## 2. Frozen inherited interventions

The runner loads exact binary floating-point values from the hash-verified B1
and B2 freeze artifacts; it does not retype rounded constants.

| item | recorded value for readability |
|---|---:|
| B1 event weight | `0.9310125645774651` |
| B1 projected scale | `0.4299860893300136` |
| B2 event weight | `0.00019294302093274076` |
| B2 Laplace bandwidth | `7.085388360479058` |
| B1 and B2 cadence | every 10 steps |
| B1 and B2 trajectory NFE | 8 |
| training steps | 30,000 |
| evaluation NFE | 32 |
| paired units | 500, 501, 502 |
| checkpoints | 10,000; 20,000; 30,000 |

The two single corrections each have calibrated event-gradient ratio 0.25.
Consequently the full combined cell can have greater correction compute and a
larger total correction gradient than either single arm. That is not a
confound in the factorial interaction; it is the defined joint treatment.
Runtime and memory remain part of the Pareto report.

## 3. Pairing and stochastic roles

Within a unit, every arm replays exactly the same:

- model initialization;
- flow data order and horizontal flips;
- endpoint noise;
- bridge time draws; and
- evaluation priors.

B1 banks/data/priors and B2 probes/positives/negative priors have independent,
stage-local streams under the namespace `b25-development`. The runner refuses
units outside `{500,501,502}` and verifies exact step-one flow-loss equality
across all four cells.

The three units are the independent training replicates. Bootstrap or
subsampling intervals over generated examples quantify conditional evaluation
uncertainty; they are not substitutes for training-unit replication.

## 4. Evaluation sources

Two instruments are reported separately.

### 4.1 In-domain development instrument

The CIFAR-10 test set measures closeness to the actual target law. It was
adaptively consumed by earlier stages, so it is explicitly labelled
`in_domain_development_reused`. It may guide this development experiment but
cannot support a fresh confirmation.

### 4.2 Disjoint shifted instrument

A second balanced 6,000-image pool is derived from CINIC-10's ImageNet
contribution. The builder excludes **both** every source path and every decoded
pixel hash in the consumed B2 pool, in addition to excluding all CIFAR-10
pixels and within-pool duplicates. A different seed by itself is not accepted
as evidence of disjointness.

This source measures class-aligned distribution-shift robustness. It is not an
in-domain replacement for CIFAR-10. Results from the two instruments must not
be pooled into one number.

All reference, matched-real-control, probe-centre, positive, floor-negative,
and unused roles are disjoint within each source and hash-bound by preflight.

## 5. Quantities and uncertainty

Each arm × unit × checkpoint reports:

- raw normalized-Laplace drift energy and the real-real floor separately;
- the floor-relative excess as secondary context;
- raw-pixel effective rank and paired rank retention relative to B0;
- Inception-space precision, recall, KID, density, and coverage;
- indicative, fixed-sample FID;
- lower-tail event quantiles of B2 ESS minima and medians;
- maximum-weight summaries;
- clip factor, weighted component-gradient norms, and pairwise gradient cosines
  at the three declared checkpoints;
- wall time, model-forward counts, and peak memory.

Precision/recall intervals bootstrap the paired fixed-manifold membership
indicators. This avoids refitting a kNN manifold on duplicate-heavy bootstrap
samples. KID uncertainty uses paired without-replacement subsampling with
common generated and reference indices. Both procedures are conditional on
the trained models.

Checkpoint 30,000 is the only decision checkpoint. Checkpoints 10,000 and
20,000 diagnose whether rank compression occurs early or accumulates; they may
not be selected post hoc as the B2.5 winner.

## 6. Exact development decision heuristic

The combined arm is “promising” in a unit only if all four conditions hold at
step 30,000 on the in-domain development instrument:

1. it retains at least 80% of B2's nonnegative raw-drift reduction from B0;
2. `rank(B1B2) / rank(B0) ≥ 0.85` and its rank exceeds B2's;
3. its precision is at least 90% of the better B0/B1 precision; and
4. its recall is at least 90% of the better B0/B1 recall.

The stage is promising only if at least two of three units satisfy all four.
The threshold is a prospective development heuristic, not a calibrated test.
The artifact also reports the factorial interaction for each outcome,

\[
I_Y = Y_{B1B2} - Y_{B1} - Y_{B2} + Y_{B0}.
\]

A promising result authorizes designing a new confirmation on untouched data;
it is not itself a promotion.

## 7. Artifact and failure discipline

- B1 and B2 freezes and sidecars must verify before preflight.
- The protocol and explicit executable dependency manifest are hashed.
- The disjoint data bytes and provenance are hashed.
- Preflight binds both evaluation allocations and exact inherited constants.
- Unit runners refuse to overwrite checkpoints or result artifacts.
- The aggregate requires all three units and one common preflight.
- A crash never licenses changing a coefficient, bandwidth, threshold, source,
  arm, or checkpoint. Any such change creates a new development design.

The regression suite checks full-dose event counts, exact flow pairing,
component-gradient diagnostics, allocation disjointness/capacity, metric
sanity, decision logic, and exclusion-pool binding.

## 8. Resource expectation

Historical timings imply approximately 13.5 hours for the old interleaved
design. The corrected full-dose combined arm is expected to require about
1.75–1.9 hours per unit, giving approximately 15–15.5 training hours overall.
Checkpoint evaluation and uncertainty add roughly 1–2.5 hours.

**Expected complete runtime: approximately 16–18 hours on the RTX 4050 laptop
GPU**, subject to thermal throttling. The mandatory full-shape throughput and
memory preflight replaces this estimate before the long run.

Units run independently so the experiment can be split across sessions while
retaining complete paired four-arm units.

## 9. Reproducible execution order — deferred

These commands document the eventual run. Do not execute the three unit
commands until B3/B4 have been completed and B2.5 is deliberately resumed.

```powershell
# 1. Build the second, explicitly disjoint shifted pool.
uv run --python 3.12 --with torch==2.7.1 --with numpy --with pillow `
  python -m numerics.encoder_independent_drifting.stage_b25.source_pool `
  --archive "$HOME/.cache/cinic-10/CINIC-10.tar.gz" `
  --output numerics/encoder_independent_drifting/stage_b25/data/cinic10_imagenet_only_b25_disjoint.npz `
  --provenance numerics/encoder_independent_drifting/stage_b25/data/cinic10_imagenet_only_b25_disjoint.provenance.json `
  --exclude-pool numerics/encoder_independent_drifting/stage_b2/data/cinic10_imagenet_only_b2_seed20260731.npz

# 2. Run tests, then the full-shape mechanics/memory/throughput preflight.
python -m numerics.encoder_independent_drifting.stage_b25.tests.test_b25
python -m numerics.encoder_independent_drifting.stage_b25.preflight

# 3. Later: one complete paired unit per invocation.
python -m numerics.encoder_independent_drifting.stage_b25.run_unit --unit 500
python -m numerics.encoder_independent_drifting.stage_b25.run_unit --unit 501
python -m numerics.encoder_independent_drifting.stage_b25.run_unit --unit 502

# 4. Aggregate only after all three unit artifacts exist.
python -m numerics.encoder_independent_drifting.stage_b25.aggregate
```

Use the repository's pinned CUDA Torch/Torchvision environment for steps 2–4;
an ambient CPU-only `uv run` is not an equivalent performance environment.

## 10. Interpretation after eventual execution

- **Drift retained and rank restored:** B1 and B2 are complementary; design a
  fresh confirmation.
- **Rank restored but drift lost:** the objectives trade off rather than
  complement at the frozen strengths.
- **Drift retained but rank remains compressed:** the B2 geometry pressure is
  not repaired by B1; a separate ESS/multiscale development study is justified.
- **Neither retained:** do not compound these mechanisms in B3 or a promoted
  model.

B3 remains the next experiment because it tests the independent, higher-value
question of drift-only training under matched architecture and reporting.
B2.5 is now a correct, preserved branch to revisit afterward.
