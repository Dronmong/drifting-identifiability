# Objective 7 numerics

Numerical evaluation of the machine-checked identifiability conditions at the
paper's actual operating point (arXiv:2602.04770v2, Table 8 and A.6).

Run (no install needed beyond [uv](https://docs.astral.sh/uv/)):

```
uv run --with numpy python numerics/run_all.py
```

Output: `numerics/RESULTS.md` (deterministic, seed 20260707).

Real encoder features:

```
uv run --with numpy python numerics/real_feature_diagnostics.py \
  --features path/to/features.npy \
  --m 8 --num-probes 64 --taus 0.02 0.05 0.2
```

The feature file should be a `.npy` array of shape
`[num_samples, feature_dim]`, or a `.npz` archive containing such an array
(select with `--key`; otherwise the first array is used). The runner centers
and rescales features by default so the sampled mean pairwise distance is one,
matching the normalized units used elsewhere in this numerical ledger. It
reports softmax effective sample sizes, bare interaction-matrix conditioning,
and column-reweighted interaction-matrix conditioning. Large feature tensors
and generated `REAL_FEATURES*.md` reports are ignored by Git by default; commit
only small sanitized summaries intentionally.

## What this is and is not

This suite does **not** prove anything.  It evaluates the *proved* formulas at
concrete numbers, audits the formalization against the paper's own Algorithm-2
pseudo-code, and answers the Objective-7 question: which certified conditions
are realistic enough to matter, and where do the certified constants leave
practice behind.

## Dynamics Phase C validation

The audit-corrected finite-particle dynamics benchmark is separate from the
earlier identifiability ledger:

* `PhaseCValidation.md` specifies the controlled C1--C3 experiments;
* `driftbench_v2.py` is the exact executable harness;
* `DesignRules.md` records the qualified conclusions;
* `bench_runs_v2/` contains standard-profile manifests, source snapshots,
  per-seed data, and trajectories;
* `bench_figs_v2/` contains the corresponding uncertainty plots.

The old `PhaseC_DesignRules.md`, `driftbench.py`, and `bench_runs/` are retained
as an explicitly superseded first pass. The V2 results are synthetic particle
experiments; they do not claim trained-model or real-encoder performance.

## Low-dimensional base-versus-modified study

The next-stage matched Algorithm-2 study is documented in:

* `LowDimPerformanceRoadmap.md` and `LowDimPerformanceResults.md` for the
  historical D0--D3 negative result;
* `LowDimAttributionProtocol.md` and `lowdim_attribution.py` for the
  audit-corrected fresh bandwidth × mask × step experiment;
* `LowDimAttributionResults.md` for the fresh held-out result.

The fresh global gate also failed, so no aggregate outperformance or learned-
generator claim is made. The surviving empirical finding is conditional: with
bandwidth and step fixed, disabling the eye mask substantially improves ED² on
fresh curved ring/circle/moon targets, while a cluster-count trigger can
misfire on a finite Gaussian mixture.

The operating point is taken from the paper, not chosen for convenience:

- kernel `exp(-dist/tau_tilde)` with `tau_tilde = tau*sqrt(C)` on features
  normalized so that the mean pairwise distance is `sqrt(C)` (A.6, eqs. 18-22);
  in normalized units the kernel is `exp(-u/tau)` with typical `u ~ 1`;
- temperature grid `tau in {0.02, 0.05, 0.2}` (Table 8);
- per-class batch `N = Npos = Nneg = 64`, negatives = generated samples reused
  with the `eye(N)*1e6` self-mask (Algorithm 2);
- CFG scale `alpha in [1, 4]` (Table 8).

## Quantile-to-Laplace learned-generator program

The later one-dimensional learned-generator program is recorded separately:

* `QLDNextGenerationResearch.md` develops the large-batch successor to the
  original quantile-to-Laplace candidate;
* `LBQCDDevelopmentResults.md` records the development selection;
* `LBQCDConfirmatoryProtocol.md` and `LBQCDConfirmatoryResults.md` contain the
  frozen confirmatory protocol and its scoped 17.8% ED2 improvement;
* `OccupancyAdaptiveQuantileResearch.md` audits the hypothesis that
  large-batch transport could improve coarse mass allocation more reliably
  than final error and specifies the state-aware, stratified successor;
* `OASQDDevelopmentProtocol.md`, `oasqd.py`, and
  `run_oasqd_development.py` implement that fresh staged campaign;
* `OASQDDevelopmentResults.md` records the negative O5 decision: the atlas and
  unbiased stratified estimator worked, but the frozen OA-SQD candidate tied
  QLD rather than surpassing it, so no confirmation was launched.

The earlier cross-arm LB-QCD coverage-time comparison used unequal event
sample sizes and is withdrawn. The OA-SQD runner uses the same independent
event-probe size for every arm; endpoint LB-QCD results are unaffected by this
correction.

Neither frozen LB-QCD registry may be reused to tune the successor. The result
is limited to the documented one-dimensional synthetic generator benchmark
and does not assert ImageNet, real-feature, or multidimensional superiority.

### Transport-aligned generator successor

The architecture-level successor is documented in:

* `TransportAlignedGeneratorResearchPlan.md` for the full attempt audit,
  literature cross-check, candidate definition, and falsification gates;
* `persistent_quantile_transport.py` for the monotone running-quantile map and
  deterministic invariants;
* `run_transport_aligned_development.py` for paired execution, exact work
  accounting, target-level bootstrap inference, and artifact snapshots;
* `TransportAlignedGeneratorDevelopmentResults.md` for the completed
  development and cross-registry replication analysis;
* `PQTConfirmatoryProtocol.md`, `pqt_confirmatory_registry.json`, and
  `PQTConfirmatoryResults.md` for the subsequent fresh frozen test;
* `transport_aligned_runs/` for raw candidate rows and source snapshots.

The budget-matched 128-knot candidate first obtained ED2 ratios `.6231` and
`.6427` against LB-QCD on two development/replication registries. It then
passed all 13 gates on a new untouched registry with ED2 ratio `.4230` versus
LB-QCD (95% interval `[.3258,.5344]`) and `.3487` versus selected paper. This is
the strongest low-dimensional result in the repository, but it remains a
nonparametric one-dimensional architecture result rather than a claim about
the paper's high-dimensional image generator.

### Two-dimensional PSQT accumulator confirmation

The pooled-rank repair of persistent sliced quantile transport is documented
in `PSQTAccumulatorConfirmatoryRoadmap.md`,
`PSQTAccumulatorConfirmatoryProtocol.md`, and
`PSQTAccumulatorConfirmatoryResults.md`. The quality arm uses independently
audited Apache DataSketches KLL summaries; the efficiency arm uses a bounded
reservoir of raw 2D samples.

On one sealed registry of 64 fresh 2D targets, KLL-k128 passed every
preregistered gate. Its geometric-mean ED2 ratio was `.3370` versus historical
online PSQT (95% target-bootstrap interval `[.3111,.3669]`) and `.0762` versus
the registered paper `tau = 1` arm (`[.0634,.0927]`). It also improved held-out
SW1 in every target and had no family-level ED2 regression. Reservoir-1024
passed its separate efficiency gates at essentially the same persistent-state
size as historical PSQT. The full immutable output is under
`psqt_accumulator_confirmatory_runs/20260722-005506-confirmatory/`.

This supports the protocol's scoped general low-dimensional improvement claim
for the nonparametric 2D particle testbed. It does not establish superiority on
images, learned encoder features, neural generators, or arbitrary dimension.

### Neural conditioned transport development

The first neural pooled-rank attempt and its repaired successor are documented
in `KLLPSQTNeuralAmortizationResearch.md`,
`ConditionedTransportAmortizationResearch.md`, and
`ConditionedTransportAmortizationResults.md`. The implementation adds a
quadratic (covariance-sensing) frame audit, coherent PSQT particle teachers,
compute-matched microbatch amortization, an RMS-normalized paper-field term,
and an optional protected-tail selector.

On the consumed 16-target synthetic matrix, KLL conditioned-plus-local beat
the paired paper neural port on ED2 and held-out SW1 in all 16 single-replication
cells and removed the earlier 16D reversal. This is post-hoc development
evidence, not confirmation. The candidate must next pass a fresh frozen
registry before any general neural superiority claim or frozen-feature work.

That fresh confirmation has now been completed. The frozen design is in
`NeuralConditionedTransportConfirmatoryProtocol.md`, infrastructure failures
are retained in `NeuralConditionedTransportFailures.md`, and the successful
v3 result is in `NeuralConditionedTransportConfirmatoryResults.md`. Exact
conditioned transport passed all 9 primary gates; frozen KLL retention passed
all 5 retention gates. The scoped result covers new 2D/4D/8D/16D synthetic
neural-generator targets at matched generator-example count, not image
benchmarks or total compute superiority.

Reproduce the repaired regression suite and development runner with:

```powershell
uv run --python 3.12 --with torch==2.7.1 --with numpy `
  --with datasketches==5.2.0 python numerics/neural_pooled_rank_tests.py

uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy `
  --with datasketches==5.2.0 `
  python numerics/run_conditioned_transport_development.py --profile smoke
```

The cost-development runner also accepts `--active-directions`,
`--local-field-calls`, and `--local-representatives`.  For example, the
selected K2 development setting uses all local calls, 32 active registered
directions, and 128 weighted representatives per support:

```powershell
uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy `
  --with datasketches==5.2.0 `
  python numerics/run_conditioned_transport_development.py --profile consumed `
    --active-directions 32 --local-representatives 128
```

`--representative-audit-calls 2` additionally compares two compressed fields
per local arm with the dense field and records field/mass errors.  Those dense
audit calls are diagnostic overhead and are reported separately from model
kernel pairs.

The subsequent persistent-teacher repair is now freshly confirmed. The
dimension-adaptive arm preserves the prior model exactly in 2D/4D, uses two
reranked free-particle transport steps in 8D and four in 16D, and passed both
target-level ED2/SW1 gates against the prior control on 32 unseen targets under
two initializations. See:

- `AdaptiveRolloutConfirmationProtocol.md` for the pre-outcome design;
- `AdaptiveRolloutConfirmationResults.md` for the scoped interpretation;
- `adaptive_rollout_confirmation_analysis.json` for exact paired statistics;
- `adaptive_rollout_component_moments.json` for post-hoc
  component-conditioned diagnostics; and
- `adaptive_rollout_confirmation_figures/` for presentation-ready summaries.

The quality gain costs more non-neural projection/sort/kernel work but no
additional generator forward calls. It must not be described as an
image-scale or universal compute-superiority result.

The subsequent safety/tail audit repair is recorded in
`TailSafetyRepairExperiment.md` and `TailSafetyRepairResults.md`. Projected-rank
tail balancing, corrected tail/bulk losses, same-cohort retention, and complete
cost accounting were implemented. Neither safety nor rank-aware ordering
improved the confirmed adaptive rollout on the consumed factorial screen, so
they remain unpromoted ablations.

The projection/kernel cost audit has now been repaired and confirmed on a
fresh frozen 16-target registry. See:

- `ProjectionKernelOptimizationConfirmationProtocol.md` for the pre-outcome
  design;
- `ProjectionKernelOptimizationConfirmationResults.md` for the scoped result;
- `projection_kernel_confirmation_analysis.json` for exact paired statistics;
  and
- `analyze_projection_kernel_confirmation.py` for the deterministic analysis.

The frozen active-32, exact-atlas, `M=128` arm passed both primary quality
gates against a matched current paper-neural comparator: geometric ED2/SW1
ratios were `0.3305 / 0.5711` with bootstrap upper bounds below one and
`15/16 / 16/16` paired wins. At equal generator-example and kernel-pair count,
the median CPU online/setup-inclusive wall ratios were `0.6150 / 0.7636`.
This is a synthetic 2D/4D/8D/16D result, not an image benchmark, total-
operation equivalence, peak-memory result, or universal superiority claim.

## Calibrated conservative bridge experiment

The later QLD-to-sharp-to-paper mechanism experiment is documented in:

* `CalibratedConservativeBridgePlan.md` for the gated implementation order;
* `CalibratedBridgeProtocol.md` for the pre-outcome frozen arms and thresholds;
* `calibrated_bridge.py` and `run_calibrated_bridge_development.py` for the
  configurable Adam, optimizer-state isolation, ESS diagnostics, and runner;
* `CalibratedBridgeResults.md` for the completed Registry-A decision;
* `bridge_runs/` for immutable smoke and screen artifacts.

The calibrated primary was numerically safe but failed its registered ED2,
SW1, and median-step gates. Consequently no scale-normalized or multiscale
candidate was implemented and no fresh registry was opened under that plan.
The short sharp bridge improved mean ED2 transiently, but the final schedule
did not establish general endpoint superiority over QLD-v1.

## Quantile-Guarded Drifting development screen

The optimizer-space protected-update experiment is documented in:

* `QuantileGuardedDriftingResearchPlan.md` for the mathematical mechanism and
  staged execution plan;
* `QuantileGuardedDriftingProtocol.md` and
  `qgd_development_registry.json` for the registered Q1 screen;
* `quantile_guarded_drifting.py` and
  `run_quantile_guarded_development.py` for the audited projection, dual-Adam
  proposals, independent checkpoint selector, and paired runner;
* `QuantileGuardedDriftingResults.md` for the completed decision;
* `qgd_runs/20260721-172034-Q1-screen/` for full run artifacts.

QGD-v1 was numerically stable and its projection was genuinely active, but it
failed the registered ED2, SW1, family-robustness, and safe-proposal gates. No
candidate advanced to replication or sealed confirmation. In particular, the
screen is evidence about a failed low-dimensional development mechanism, not
a claim of improvement over the paper's reported image-generation results.

The post-audit QGD-v2 safety repair is documented in
`QuantileSafetyGuardProtocol.md` and `QuantileSafetyGuardResults.md`, with the
full screen under `qgd_runs/20260721-182056-Q2-safety-screen/`.  It removes the
second Adam state, uses independent current-gradient agreement and a
target-to-target rank-noise floor, and caps consensus non-ascent corrections.
The repair eliminated QGD-v1's damage but intervened on only 0.42% of suffix
updates.  Its best selected ED2 ratio was `0.999776`, so it is a safe
higher-cost tie rather than an improvement and did not advance.

## Encoder-independent kernel drifting (SACKGD)

The attempt to remove the pretrained feature encoder from the *training
objective* is documented in:

* `EncoderIndependentKernelDriftingResearchPlan.md` for the mechanism,
  literature cross-check, and phased falsification gates;
* `numerics/encoder_independent_drifting/` for the isolated implementation
  (spectral source anchor, fixed compositional geometry, kernel-gradient
  drift, cross-fit adaptive mixture, separately nonnegative objectives);
* `EncoderIndependentPhase0Results.md` and `phase0_gate.json` for the passed
  Phase-0 exit gate;
* `EncoderIndependentPhase1Protocol.md` for the frozen pre-outcome Phase-1
  design and its five exit conditions;
* `EncoderIndependentPhase1Results.md` and `phase1_screen.json` for the
  structured-image mechanism screen;
* `EncoderIndependentAnchorGradientDiagnosis.md` for the anchor's
  high-frequency gradient bottleneck;
* `EncoderIndependentPhase1Diagnosis.md` with `phase1_diagnosis.json` and
  `phase1_capacity.json` for the deep failure investigation and the seven
  reforms it recommends;
* `EncoderIndependentReformedScreenProtocol.md` for the frozen successor
  design, with all seven reforms implemented and unit-tested;
* `EncoderIndependentSecondPassAudit.md` with `phase2_cifar.json` for the
  implementation audit, the regime search, and the CIFAR-10 Phase-2 design
  (nine reforms, 90 tests).

No pretrained encoder appears in any training objective. Arm A8 is a locally
trained autoencoder stand-in, not the paper's pretrained encoder, and is
excluded from every gate.

Phase 0 passed all four conditions on synthetic 16x16 structured images: 64
unit tests; the spectral anchor detects 6/6 source-collision pairs; fixed
geometry branches beat raw pixel drifting on affinity ESS and cross-fit drift
SNR; and the kernel-gradient field becomes orthogonal to standard
displacement under a structured kernel (cosine -0.002) while reproducing it
exactly for a raw Gaussian kernel (cosine 1.0000, positive control).

Two results there are load-bearing for interpretation. The orthonormal Haar
control reproduces raw-pixel kernel health to four significant figures, which
confirms that measured gains come from the modulus, pooling and multiscale
structure rather than from "using wavelets". And the fixed geometries are
measurably *not* measure-determining -- the wavelet branch is blind to a
colour-channel swap, the random convolutional branch to a 5% rare mode --
which is precisely why the plan forbids them from being the correctness
authority and keeps the source anchor as a separate nonnegative loss.

Everything about the anchor is an ideal-expectation statement approximated by
a finite random-feature bank; the bank is refreshed on a declared schedule and
an independent audit bank is used for reporting. These are synthetic mechanism
results and assert nothing about CIFAR-10, ImageNet, FID, or the paper's
trained model.

**Phase 1 FAILED its exit gate, so the program does not proceed to CIFAR-10.**
On 9 structured targets x 3 seeds, fixed wavelet geometry with kernel-gradient
movement is 4.5x worse than raw pixel drifting on the pre-registered
normalized geometry score and loses all 27 paired cells, at 49x the kernel
work. Two further results are worth carrying forward:

* the plan's section-6.3 prediction is **falsified** -- kernel-gradient
  movement is mildly better than standard displacement for the raw kernel
  (ratio .981) but 1.86x *worse* for the structured kernel, the opposite of
  the design rationale. The required standard-displacement ablation is what
  caught this;
* every claim made for the **anchor held**. It improves fixed geometry by 30%
  (A5/A4 = .701, 24/27 wins), stays practically present (gradient share .311
  in 27/27 cells), and its independent audit bank detects 6/6 source
  collisions where the wavelet branches repeatedly miss colour swaps and a 5%
  rare mode.

A deeper investigation of the failure is recorded in
`EncoderIndependentPhase1Diagnosis.md` (artifacts `phase1_diagnosis.json`,
`phase1_capacity.json`). It overturns part of the first reading:

* **the screen ran below the budget at which anything works.** A
  sliced-Wasserstein oracle on the same generator -- no kernel, no features,
  no normalization -- scores worse than A1 at 300 steps and never reaches
  target-level precision; two of three probed targets are unsolvable even at
  4x budget. The gate's ranking survives, its interpretation does not;
* **A4 did not converge to a wrong law** (the first pass's claim, withdrawn).
  It plateaus at 3x the q=p field floor while A1 reaches 1.26x -- yet A4
  attains a *lower* wavelet-field residual than A1 and is still 4.5x worse.
  Each arm descends its own discrepancy; the wavelet discrepancy is the wrong
  one;
* **the mechanism is off-manifold drift.** Kernel-gradient movement through a
  non-injective feature map exploits the map's null directions like an
  adversarial attack, leaving output 2-3.4x further from real data than the
  raw arms; the anchor measurably pulls it back (3.42 -> 2.91), which explains
  why it helps every geometry arm;
* **the geometry loss is pinned at exactly eta^2** in all 243 rows, so the
  plan's exact-zero argument is vacuous as implemented and the screen ran
  without a convergence signal;
* bandwidth (64x sweep) and field normalization were tested and are **not**
  the cause. An ESS diagnostic that reported 1e10 under kernel collapse was
  found and fixed, with a regression test.

The limiting caveat from the screen itself also stands: the locally trained
encoder stand-in fails to beat raw pixels too (A8/A1 = 1.322, interval
crossing one), so this testbed cannot separate "fixed geometry is inadequate"
from "feature geometry is unnecessary at 16x16".

All seven reforms are now implemented and unit-tested, and the Phase-0 gate
has been re-run and passes all five conditions. The load-bearing addition is
**G0.5, a zero-set reachability condition**: it optimizes a free particle
cloud on each candidate field with fresh target batches and a decaying step,
and reports the held-out residual against the `q=p` floor. Over 4 targets x 3
seeds it reproduces the entire Phase-1 ranking in minutes rather than hours --
`wavelet::kernel_gradient` plateaus at 2.39x the floor (arm A4, which failed
the gate) while `wavelet::standard` reaches it at 0.68 (arm A3, which beat A4)
and the raw kernel reaches it at 0.84 (arm A1, the best arm). An arm whose
field provably cannot reach the floor is now refused entry to a screen.

The reforms also record an honest negative: the R3 data-span projection does
**not** rescue the wavelet family (2.21 vs 2.39, both plateauing). Constraining
the update direction does not repair a wrong objective.

A second research pass then audited the implementation again and searched for
the regime the program had never actually tested. Two more reforms came out of
it (R8 an averaged null reference, after finding a fresh real sample scored
2.78 rather than the nominal 1.0; R9 a shared projection factorization, making
the projection free), one standing question was closed by a clean negative (the
unmasked self-term biases the field by under 2.2%), and the regime search
succeeded on real data:

* **no synthetic testbed makes pixel geometry fail** -- across resolutions
  16-32 and translations up to +/-4, pixel content-grouping never becomes
  misleading, which is why every earlier phase found fixed geometry useless;
* **on CIFAR-10 it does fail, and fixed geometry fixes much of it.** Pixel
  k-NN content accuracy is .267 (chance .100); fixed wavelet reaches .390 and
  scattering .386-.389, a 36-50% relative gain, matching a small supervised
  control (.389-.406) -- while a locally trained denoising autoencoder reaches
  only .282, barely above pixels;
* a distance-ratio statistic **saturates** on real data (pixels .956 vs a
  100%-train-accuracy supervised encoder .922) and would have produced a
  confident false negative without the learned control -- recorded as reform
  R10;
* **CIFAR-10 at 16x16 is admissible at 300 steps** (skyline precision 1.000
  against a .482 bar), a gate four of nine synthetic targets never cleared.

Phase 2 was therefore run on CIFAR-16 as a four-arm comparison of raw pixel
against fixed wavelet and scattering geometry, all with standard displacement,
plus a skyline. See `EncoderIndependentPhase2Protocol.md` (frozen pre-outcome),
`EncoderIndependentPhase2Results.md`, `phase2_entry.json` and
`phase2_screen.json`.

**Phase 2A entry gate PASSED all four conditions** -- the first testbed in the
program that was solvable (skyline precision 1.000 at 300 steps), discriminating
(pixel k-NN .245 vs scattering .383) and free of the Phase-1 zero-set failure
(all fields reach the q=p floor at .42-.49).

**Phase 2B exit gate FAILED.** Fixed compositional geometry is **37% worse**
than raw pixel drifting (ratio 1.366, 0/3 seeds) at 49x the kernel cost -- the
third consecutive negative, reported without retuning per the protocol's
declared failure branch. The instrumentation localizes it:

* fixed geometry **wins** `nearest_real` (.500 vs .490 vs raw .528) and loses
  every distributional component, with coverage falling .965 -> .926 -> .910.
  A better neighbour ranking converts into density-seeking, not distribution
  matching;
* it is **not** an optimization failure -- every arm drives its unnormalized
  geometry loss down 97-98% (visible only because of reform R2), and every
  field reaches the zero-set floor. Both surviving Phase-1 explanations are
  excluded by measurement;
* **the anchor result replicates on real data** (B3/B1 = .906, 3/3 wins),
  having now helped in every configuration it has been placed in.

Phase 2 also reported the skyline as 4x better than every drifting arm and
called that the phase's biggest finding. A follow-up investigation
(`EncoderIndependentPhase2Diagnosis.md`) shows **that was an implementation
defect, now fixed**:

* the *same* field on free particles scores 1.89 against the skyline's 1.84 --
  the field was never the problem;
* the drifting generator had collapsed to **effective dimension 2.34** against
  the data's 8.32. The stop-gradient target moves each sample toward its
  neighbourhood barycentre, so the teacher cloud is narrower than the data,
  and least-squares regression onto it narrows the output again -- a double
  contraction that free particles and the sliced-Wasserstein skyline both
  escape;
* **reform R11**, a single scalar rescaling the teacher to carry the real
  batch's second moment, takes the score from 6.69/7.05 to **1.92/1.30** --
  matching or beating the skyline -- and restores effective dimension to ~6.7;
* **reform R12** implements the paper's real Algorithm-2 bi-softmax field.
  Every phase had used the row-normalized SNIS mean shift that `lowdim_drift`
  labels "DIAGNOSTIC ONLY"; the omitted column reweighting is worth ~13% on
  raw and does not change any ranking;
* **the geometry verdict survives**: R11 helps raw and wavelet almost equally
  (3.9x vs 3.8x) and raw still wins by 1.8x, which the free-particle result
  confirms with no generator involved.

The geometry thread stays closed. What re-opened was the baseline, and
**Phase 3 confirmed it on fresh seeds** (`EncoderIndependentPhase3Protocol.md`,
`EncoderIndependentPhase3Results.md`, `phase3_confirmation.json`): across two
resolutions, three budgets and three unseen seeds, R11 improves the
encoder-free baseline by **3.1x** (paired ratio .319 [.292,.351], **18/18
wins**), restores effective dimension from .264 to .867 of the data's, and
closes the generator-versus-free-particle gap from 3.44 to 1.09. All seven
exit conditions passed. Corrected encoder-free drifting reaches **parity** with
the sliced-Wasserstein skyline (1.140 [.999,1.307]) -- parity, not superiority.
This is the program's first confirmed positive result. R12 turns out not to be
needed once R11 is applied (C2/C1 = 1.014, a tie).

A design investigation for the next phase (`EncoderIndependentPhase4Design.md`)
then measured two things rather than assuming them. The anchor's benefit is
**not** a disguised variance fix -- it survives R11 unchanged (.962 with, .966
without, 3/3 wins each) and the two effects are orthogonal -- but on raw pixel
geometry it is worth only ~3.5%, so the 30-45% figures from earlier phases
belong to structured-geometry arms the program has abandoned. And the
contraction R11 repairs is **universal**: effective dimension collapses to
.218-.308 in every configuration tested (batch 32/64/128, raw and wavelet) and
R11 recovers .25-.33x in all of them. Phase 4 ran that study
(`EncoderIndependentPhase4Protocol.md`, `EncoderIndependentPhase4Results.md`,
`phase4.json`, `phase4_ad.json`) and **failed all three of its gates,
informatively**:

* the contraction **does** reproduce in `lowdim_drift.py` -- this repository's
  independently audited verbatim Algorithm-2 port, imported unmodified -- which
  rules out an artifact of the new package. But it is strongly
  **dimension-dependent**: effective-dimension ratio .998 at d=2 falling to
  .096 at d=64, with a crossover near d=4 below which R11 is unnecessary and
  mildly harmful and above which it is worth 7-14x on ED2;
* **R11 fails at the paper's declared temperature grid.** Across tau in
  {.02, .05, .2} the ratio degrades to .86 / 1.12 / .47, and at tau = .05 the
  correction actively hurts and does not restore effective dimension at all.
  Every earlier R11 result used an ESS-calibrated bandwidth, which is not the
  paper's rule;
* **Algorithm 2's self-mask is itself protective** (uncorrected effective
  dimension .521 with it against .303 without), so the regime R11 repairs is
  one the paper's own recipe already partly avoids;
* **both proposed mechanisms were refuted.** Minibatch-noise regression
  attenuation is three orders of magnitude too small (noise fraction ~.001),
  and the teacher map is isotropic and preserves effective dimension to four
  significant figures in one application. The cause is unknown;
* the residual skyline gap looks like a **missing spectral tail** (real data
  carries 13.4% of its variance beyond the top 32 directions, the corrected
  generator 4.7%), consistent with the 32-dimensional latent rather than a
  second defect in the objective.

**Phase 6 put R11 through the sharpest test aimed at it and it survived**
(`EncoderIndependentPhase6Protocol.md` frozen pre-outcome,
`EncoderIndependentPhase6Results.md`, `phase6.json`). The prompt was the
discovery that the objective's declared step `step_eta` is **inert under an
adaptive optimizer** -- it enters the stop-gradient loss only as a constant
gradient multiplier, and Adam normalizes that away (a 100x change in eta moves
the output by 2.8e-5 under Adam against 1.34 under SGD). That retroactively
explains five failed reforms (R16, RMS normalization, R21, the eta^2-pinned
loss, and five magnitude-based mechanism hypotheses), and it meant the *real*
step control -- the optimizer learning rate, fixed at Adam/2e-3 since Phase 1
-- had never been swept. Phase 6 swept it, and:

* **the deficit is not a learning-rate artifact.** Across 3 optimizers x 4
  learning rates x 3 seeds, **0 of 12** uncorrected cells reach the
  `[0.7, 1.3]` second-moment band (range .148-.369); R11 reaches .853-1.195.
  The best uncorrected cell misses the supersession ceiling by 9x;
* **the deficit is not a neural-port artifact either.** Under a matched
  comparison the **particle algorithm carries it too** (.594 constant step,
  .627 decaying) -- milder than the generator's .269, but real on every seed.
  It belongs to drifting's dynamics, not to the port. This refutes the
  hypothesis stated in `EncoderIndependentOpenQuestions.md` Q3;
* **richer direction changes do not beat one scalar** (per-coordinate 1.095,
  eigendirection 1.130, both intervals straddling 1), and a declared 1.2x
  over-correction is decisively worse (1.975) -- so R11's undershoot to
  .81-1.08 is not a gain deficit waiting to be closed;
* **R11 needs a magnitude-normalizing optimizer**: it diverges under plain SGD
  at lr >= 2e-3 in every cell tested, because a corrected teacher presents a
  large gradient. The same blindness that makes Adam ignore `step_eta` is what
  lets Adam use R11.

R11 after Phase 6 is: a real, cross-harness-reproducible repair to
stop-gradient regression in higher dimensions at ESS-calibrated bandwidths, of
unknown mechanism (six hypotheses refuted), that does not transfer to the
paper's declared operating point, is not explained by optimizer settings, and
corrects a deficit the particle algorithm already has.

**The Phase-6 follow-up (`EncoderIndependentPhase6Followup.md`,
`phase7_probe.json`, `phase7_cifar.json`) then found the deficit is a
BANDWIDTH artifact, and that this program has used the wrong bandwidth since
Phase 2.** 6C's .594 is a genuine attractor (.600 at 2400 steps, window growth
-.003), but across every R15-admissible bandwidth at CIFAR-16 the fixed-point
second moment and the quality are **monotone in the kernel's realized
neighbour count, with no crossings**:

| realized ESS | .146 | .743 | .777 | **.903 (incumbent)** | .971 | **.978** |
|---|---:|---:|---:|---:|---:|---:|
| 2nd moment | .304 | .469 | .503 | **.600** | .754 | **.856** |
| ED2 | 1.669 | .632 | .529 | **.349** | .127 | **.071** |

A **4.9x quality gain from the bandwidth alone**, not memorization
(nearest-real ratio .944 against the incumbent's .739, every component ratio
near 1). **Free particles at tau = 1 beat the R11-corrected generator, .071
against .160** -- so once the kernel is set properly the generator, not the
field, is the bottleneck. Cloud size compounds it: 64 particles (the
generator's training batch) costs an order of magnitude against 512 in every
arm. Two further mechanism hypotheses died: the analytic blur factor (predicts
contraction at every bandwidth; small tau *expands*, 8.9x in low dimension)
and balancing depth (converged Sinkhorn detonates the cloud, 1.27 -> 248,
which **explains** Algorithm 2's `sqrt(row*col)` as a deliberate half-strength
density correction keeping attraction and repulsion in balance). R15 earned
its keep: tau = .02 posted the sweep's best second moment (1.092, in band,
converged) while being a fully collapsed kernel (100% dead rows, affinity
1.9e-23). **Phase 7A must ask of the kernel what 6A asked of the optimizer**:
R11 has never been tested against a properly set bandwidth, and free particles
already clear both of that gate's conditions there.

**Phase 7 asked it, and found there are TWO deficits, not one**
(`EncoderIndependentPhase7Protocol.md` frozen pre-outcome,
`EncoderIndependentPhase7Results.md`, `phase7.json`). The follow-up's
hypothesis was right about the particle flow and **wrong about the
generator**:

* **7A -- the generator's deficit is bandwidth-INDEPENDENT.** Across four
  bandwidths (realized ESS .82-.99) x three field-cloud sizes x 3 seeds, the
  uncorrected second moment spans **.101-.443 in all 36 runs** and **0 of 12
  cells** reach the band; with R11, .874-1.159. The best uncorrected cell
  misses the supersession ceiling by 5.5x, so **R11 survives a second sharp
  test**. Bandwidth moves the *particle* fixed point .23 -> .99 and the
  *generator's* only .34 -> .38 -- they are different defects;
* **7B -- the amortization ordering inverts, and it is a crossing not a
  property.** The corrected generator is flat at ED2 .16-.18 across the whole
  bandwidth axis while particles swing 3.3x through it: at ess = .5 the
  generator wins .65x, at ess = .9 particles win 2.24x. The six-phase
  impression that amortization beat particles was an artifact of a badly set
  kernel;
* **7C -- the optimum is interior and located.** Quality is U-shaped, best at
  ess = .9 (**ED2 .0729**, second moment .989 -- the moment crosses 1 almost
  exactly at the quality optimum), 3.2x worse at the incumbent ess = .5 and
  3.9x worse at tau = 8. **Free particles at ess = .9 beat every generator
  configuration measured, corrected or not.** The rule check reported "no
  rule" (Spearman +.733) using a monotonicity test that cannot detect a U;
  the data support a *candidate* target-only calibration at ESS ~ .93, left
  unvalidated because it was read off the same sweep.

R27 (`TrainConfig.field_cloud`, decoupling the field's cloud from the target
batch) landed with a real but insufficient effect. **The open question is now
sharp: why does the *generator* contract?** Not the optimizer (6A), not the
bandwidth or cloud size (7A), not the field's own fixed point (7B/7C), and
five direct hypotheses refuted in Phases 3-5.
`EncoderIndependentMechanismSynthesis.md` proposes H9 -- drifting as an
MMD-flow-type interacting particle system inheriting the known mode-collapse
of blurring mean-shift and MMD gradient flows -- but Phase 7 scopes it to the
*particle* flow, which is the deficit that is **not** the generator's problem.

**`EncoderIndependentGeneratorContraction.md` then answered the generator
question: it is least-squares shrinkage -- the objective's OPTIMUM, not a
failure to reach it** (`phase8_probe.json`, 3 seeds, measured at the Phase-7C
bandwidth optimum):

* **a free dilation parameter declines to grow.** Adding one learnable output
  gain leaves the second moment unmoved (.462 -> .476) and the gain converges
  *downward*, to **.829**;
* **the field is not straining outward.** At the converged plain generator its
  mean radial component is **+.0004** -- exactly what the fixed-point
  condition `E_z[(df/dtheta)^T V] = 0` requires, since the head is a plain
  conv and dilation is therefore in the tangent space. R11 by contrast holds
  the cloud *past* that equilibrium against a restoring force (radial -.126,
  only 38% of samples pushed outward);
* **the mechanism.** Fitting the converged particle cloud by least squares
  while sweeping parameters/values: second moment **.933 / .902 / .733 /
  .602** at p/v **2.23 / 1.12 / .56 / .28**, with the particle control flat at
  .94-.99. Point-target regression costs *nothing* when overparameterized
  (.933 against its control's .939) and shrinks smoothly below p/v ~ 1;
* this **corrects Phase 4's "regression costs ~2x"** -- entirely the
  underparameterization artifact it flagged -- and shows **refuted hypothesis
  1 was right in kind but tested against the wrong noise**: Phase 3 measured
  the estimation-variance term (~.001, correctly small) and never measured the
  approximation-error term, which dominates;
* **latent dimension is not it** (8 -> 512 gives .400/.462/.479/.440), so H10
  is refuted and R17's negative is confirmed at the good bandwidth;
* R11 restores **15x of the spectral tail** (.0032 -> .0474 against real
  data's .1375), so it repairs scale and only part of shape.

Nine standing observations -- including four previously unexplained negatives
(eta's inertness, the never-binding step cap, richer corrections failing to
beat the scalar, over-correction hurting) -- fall under that mechanism, which
made a sharp prediction: widen the generator and R11's advantage should
shrink.

**Phase 8 tested it and the prediction failed**
(`EncoderIndependentPhase8Protocol.md` frozen pre-outcome,
`EncoderIndependentPhase8Results.md`, `phase8.json`,
`phase8_longrun.json`). Sweeping the conv width 32 -> 256 (36x parameters,
36k -> 1.32M) at the good bandwidth:

| width | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|
| uncorrected 2nd moment | .354 | .401 | .349 | **.331** |
| uncorrected ED2 | 1.169 | 1.166 | 1.082 | 1.198 |
| **realized fraction of teacher** | .212 | .313 | .358 | **.493** |

**The wide model tracks its teacher 2.3x better and produces an identical
cloud.** Under-fitting is therefore not the cause -- the deficit is in *what
the teacher asks for*, not in the ability to deliver it. That independently
reproduces Phase 5's inner-steps result by a different route, and it refutes
the inference from the contraction pass's fixed-cloud probe to the
moving-target recipe (a gap that pass flagged as live). 0 of 4 widths reach
the band and the best plain cell misses the supersession ceiling by 6.2x, so
**R11 survives a third sharp test**.

The secondary prediction is refuted and the trend flag reported the opposite:
the R11/plain ratio "rose" (.165 -> .388) only because **R11 got worse**
(.139 -> .296) while plain did not move (1.166 -> 1.198), and the width-256
interval [.193, 1.226] contains every other width. Convergence was also
checked at 6000 steps (10x budget): the uncorrected arm goes .433 -> .472
with late-window growth **-.034**, so the equilibrium reading stands and 600
steps understates by ~18%.

**The generator's deficit is now invariant to optimizer, learning rate, eta,
kernel bandwidth, field cloud size, latent dimension, model capacity and
teacher-fitting quality. The only intervention that has ever moved it is
changing what the teacher asks for.**

The follow-up pass (`EncoderIndependentPhase8Followup.md`,
`phase9_probe.json`, `phase9_subspace.json`, `phase9_genavg.json`) added four
more negatives, one partial mechanism, and a methodological correction:

* **Phase 8's width "trend" is statistically empty** -- between-width sd of
  medians .0256 against within-width seed sd **.0855**, Spearman +.098;
* **fluctuation refuted**: averaging the particle field over 64 independent
  batches moves the equilibrium 1.012 -> 1.006 (positives), 1.022 -> 1.023
  (negatives);
* **the noise budget inverts by three orders of magnitude along the
  trajectory** -- noise/signal **.007 at step 0, 4.459 at step 599**. Phase
  3's refutation of minibatch-noise attenuation rested on a ~.001 figure that
  corresponds to *initialization*, so that refutation should no longer be
  cited (the hypothesis still fails on direct test);
* **target noise refuted, and an apparent positive explained**: 16x field
  averaging raises the generator's 600-step second moment .372 -> .474, which
  matches the **.472** that plain training reaches at 6000 steps. Averaging
  *accelerates convergence* -- a 10x shorter run for a 16x cleaner field --
  and does not move the fixed point;
* **spectral confinement is the first intervention ever to move a particle
  equilibrium toward the generator's**: confining particles to the data's
  top-k directions gives in-subspace second moment 1.012 (unconfined) -> .917
  (k=511) -> .700 (k=128) -> .664 (k=32). It **saturates** near .7 and never
  reaches .47, so it is a partial contributor, not the mechanism.

**Recommended next step: stop testing interventions and solve the
linear-generator case.** With `f(z) = Az + b`, Gaussian latent and Gaussian
data, the pushforward is exactly Gaussian, the mean-shift field is
closed-form, and the stop-gradient fixed point `E_z[(df/dtheta)^T V] = 0`
becomes a matrix equation in `AA^T`. It is the only remaining route that can
produce a derivation rather than a twelfth elimination, and it follows the
two passes in this program that actually worked (R23's drop to 8-dimensional
mixtures, and the subspace framing above). In parallel, **no configuration
has ever combined the ESS-.9 bandwidth, R11 and the anchor** -- that package
is a concrete deliverable needing no new mechanism.

**Phase 9 ran that reduction and the deficit survives all the way down to
`f(z) = Az + b`** (`EncoderIndependentPhase9Protocol.md` frozen pre-outcome,
`EncoderIndependentPhase9Results.md`, `phase9.json`). The declared prediction
P-lin -- no deficit for a linear generator on Gaussian data -- is **refuted**.
The 2x2 ladder, 3 seeds, 2000 steps, all converged:

| data law | linear generator | MLP generator |
|---|---:|---:|
| Gaussian isotropic | **.211** | .060 |
| Gaussian decaying | **.648** | .480 |
| CIFAR-16 | **.502** | .357 |
| *(conv recipe reference)* | *.472* | |

* **the architecture is irrelevant** -- a linear map at CIFAR is within 6% of
  the full convolutional recipe, closing the question Phase 8 left open when
  capacity proved inert;
* the **nonlinearity deepens** the deficit by 1.3-3.5x in every data law but
  does not create it;
* **two premise errors in the derivation, both recorded.** The closed form
  `V = c(sigma) x` is the *SNIS mean-shift* field, not Algorithm 2's
  bi-softmax. And the **self-term** (`self_mask` defaults to False) is a pure
  inward pull carrying **21.3%** of the repulsion and 3.7x the masked field in
  64-D isotropic geometry, where every pair sits at distance 11.3 and the
  self-affinity is 77x a neighbour. **That looked like the mechanism and a
  control refuted it**: at CIFAR-16 the ratio is 4.5 and masking moves |V| by
  7% (.670 -> .672), so self-term dominance is an artifact the isotropic
  reduction *introduced*. The isotropic cell is therefore not a valid proxy,
  and Phase 4's "self-mask is protective" gains both a mechanism and a scope.

**The object of study is now a fixed-point equation in `AA^T` alone** --
Gaussian cloud, no network, no optimizer, no architecture.

### The mechanism

`EncoderIndependentDeficitMechanism.md` (`phase10_probe.json`) then closed
Phase 9's contradiction -- in the Gaussian cells `q = p` is reachable and IS a
fixed point, yet the generator converges elsewhere. Two measurements:

* **the field is nearly unbiased.** Along data-shaped clouds its radial
  component crosses zero at alpha = **.844** (64 positives) and **.830**
  (2048) -- a ~15% inward bias from the self-term, stable across a 32x batch
  change and far too small to explain a generator at .42;
* **the generator's cloud is not data-shaped.** At *matched second moment*, so
  every difference is shape:

| cloud | 2nd moment | radial | nn spacing | nn CV | spectral tail |
|---|---:|---:|---:|---:|---:|
| free particles | .995 | -.0020 | 13.45 | **.023** | **.415** |
| data-shaped | .995 | -.0019 | 9.93 | .245 | .142 |
| **generator** | **.417** | **+.0006** | **3.47** | .337 | **.0034** |
| **data-shaped, same scale** | **.417** | **+.0264** | 6.36 | .245 | .142 |

**At the generator's own scale its cloud is radially balanced while a cloud of
the data's shape at the identical scale is pushed outward 43x harder.** The
generator is not failing to reach equilibrium -- it is at a *different* one,
belonging to its cloud's shape.

**Why.** Algorithm 2 balances attraction to the data against repulsion from
the cloud's own neighbours, and repulsion is set by **packing**. The
generator's cloud is clumped: nearest-neighbour spacing 3.47 against 6.36 at
equal variance, achieved by concentrating energy in few directions (tail
**.0034** against **.142**, a factor of 42). A clumped cloud reaches the
repulsion it needs at a small radius. Free particles do the opposite --
unconstrained points under a repulsive interaction settle into a near-regular
packing (nn CV **.023** against an i.i.d. sample's .245) and spread across
*more* directions than the data (tail .415), landing at .995. **A smooth
pushforward of a low-dimensional Gaussian cannot pack regularly in 768
dimensions**, which is why the deficit survives to the linear case and why
nine axes are inert: none of them changes packing.

This accounts for twelve standing observations including seven previously
unexplained negatives, and it predicts the *sign* of both things that ever
moved the deficit (R11 opens spacing and raised the tail 15x; spectral
confinement reduces spread and hurts). **Next: (10A)** measure the law --
equilibrium radius as a function of tail fraction, with three recorded points
already lying on it in the right order (.0034 -> .42, .047 -> .95, .415 ->
1.0); **(10B)** intervene on shape directly (repulsion penalty or spectral-tail
floor) under the same supersession gate -- the first intervention in nine
phases derived from a measured mechanism rather than guessed.

### The law

**Phase 10 turned the mechanism into a quantitative, out-of-sample-validated
predictor** (`EncoderIndependentPhase10Protocol.md` frozen pre-outcome,
`EncoderIndependentPhase10Results.md`, `phase10.json`).

**The spectral tail is the variable, not the packing.** Varying the spectrum
(`S^beta`) moves the field's radial zero across a **6x span** (1.075 at tail
.505 down to .179 at tail .000). Holding the covariance *fixed* and
regularizing only the packing -- nn CV .215 -> .124 with the tail held to
within 1.5% -- moves it by **.098**, a factor of **9** less. That is the
separation the mechanism pass flagged as necessary and could not make.

**It is a usable law.** Interpolating equilibrium radius against tail predicts
every Phase-10B arm to within .11, and out of sample predicts the
free-particle system measured two passes ago to within **.015** (law 1.010,
measured .995). An independent check: real data's tail .1376 gives law .810
against a directly measured data-shaped radial zero of .844 -- 3% apart.

| arm | tail | law predicts | measured | error |
|---|---:|---:|---:|---:|
| E0 none | .0050 | .274 | .321 | +.046 |
| E2+E3 both | .0199 | .405 | .431 | +.026 |
| **E1 R11** | .0496 | .518 | **1.029** | **+.511** |
| *free particles (held out)* | *.4150* | *1.010* | *.995* | *-.015* |

**R11 is the sole outlier**, missing by an order of magnitude more than any
other arm -- exactly matching the earlier direct measurement that at R11's
operating point the field's radial component is -.126 with only 38% of samples
pushed outward. **R11 holds the cloud past the field's equilibrium against a
restoring force: an override, not a shape fix**, now stated quantitatively.

The 10B gate did **not** fire, and this is *not* the declared refutation
branch (which was "shape moves but the second moment does not"). The
interventions moved the shape -- tail up to **3.95x** baseline -- and moved
the second moment by precisely the predicted amount, while *improving* quality
(ED2 1.347 -> .767). They are under-powered: the best arm reaches tail .0199,
still 7x short of data's .1376, which the law says buys .405 -- and it did.
**A generator needs tail ~ .12 to reach the band.**

### The tail is destroyed, not absent

`EncoderIndependentTailDestruction.md` (`phase11_probe.json`) answered the
puzzle Phase 10 left -- why the tail is so hard to raise -- and **refuted my
own hypothesis in the opposite direction**. The field is not tail-blind; it is
generous:

| cloud | cloud tail | **field tail** | ratio |
|---|---:|---:|---:|
| **generator** | .0182 | **.2893** | **15.9x** |
| particles | .2627 | .1948 | .74x |
| real data | .1637 | .1649 | **1.01x** |

At real data the field carries *exactly* the data's own tail fraction; at the
generator's cloud it offers **16x more tail than the cloud has**. The teacher
asks for tail every step. **The generator starts with a healthy tail -- .221,
comparable to real data's .164 -- and training destroys it ~29x within the
first 100 steps** (per-step retention ~.96), flat thereafter at ~.004.

**The chain is now complete, every link measured:** least-squares regression
onto a smooth map discards trailing directions ~4%/step -> the RMS-normalized
field (norm ~.5 against an output norm ~9) cannot compensate -> equilibrium
tail ~.004 -> the Phase-10 law maps that to second moment .27-.32 -> **the
generator sits at .32**. This says *why* the standing negatives are negative:
capacity and latent dimension are inert because spectral bias is not a
capacity limit, and fitting the teacher *better* (8B) fits the *dominant*
directions better.

**Causal confirmation**: free particles started from a zero-tail cloud never
rebuild one (.0000 -> .0001 over 600 steps) and land at .559/.710 instead of
.995. **Honest limit**: the Phase-10 law predicts .18 for those, and the two
zero-tail starts differ by .15 despite equal tail -- so tail alone is not
sufficient outside the family the law was fitted on.

**Next: whiten the regression metric**, `||Sigma^(-1/2)(f - T)||^2` with
detached shrinkage-regularized Sigma. Every intervention so far acted on the
*target*; the measurement says the target is fine and the *metric* is the
problem, since `||f - T||^2` weights each direction by its own variance and
drops the trailing ones first. Primary readout is the tail trace, not the
score; refuted if the tail rises and the second moment does not follow.

### It is the self-referential teacher, not the metric

**Phase 11 ran that intervention and it failed its primary prediction**
(`EncoderIndependentPhase11Protocol.md` frozen pre-outcome,
`EncoderIndependentPhase11Results.md`, `phase11.json`). With the metric
conditioned up to **137x** in favour of the trailing directions, the tail
still collapsed ~20x in the first 100 steps (.0090-.0109 against the
baseline's .0095). At gamma >= .9 whitening actively *hurt* -- second moment
.233/.231 against .418, ED2 roughly doubled. (A 120-step smoke had shown
gamma=.99 improving; at 3 seeds that reverses, so the smoke was a one-seed
artifact.) R11 remains the only arm preserving tail -- **.0665 against .0050,
13x** -- and survives a fifth supersession gate.

**The discriminating measurement** -- same architecture, same optimizer, same
ordinary loss, differing only in the target:

| target | tail @0 | @300 | @600 |
|---|---:|---:|---:|
| **drifting** (`T = f + eta*V`, recomputed each step) | .216 | .0049 | **.0037** |
| a **fixed** particle cloud (tail .275) | .217 | .0406 | **.0960** *(still rising)* |
| a **fixed** real-data cloud | .217 | .0633 | **.0897** |

**A 26x difference with everything else identical.** The generator is fully
capable of building and holding a tail; it does not under drifting because of
*what the teacher asks for*. `T = f + eta*V` anchors the target to the
generator's own current cloud, and `||eta*V||` is ~6% of the output norm,
recomputed from a fresh batch every step -- so it never accumulates into a
persistent demand for **shape**, only for a small displacement. A fixed target
demands the shape every step and the generator builds it.

Chain link 1 (the metric) is replaced; links 2-5 stand. **Next: make the shape
demand persist** -- rollout teachers (apply the field K times before
regressing; Phase 2 and `AdaptiveRolloutConfirmationProtocol.md` have the
machinery but never evaluated it against the *tail*), or an EMA/persistent
target cloud. *Caveat: the discriminating measurement is one seed and needs
replication at three.*

### Formal audit: it is self-reference itself

`EncoderIndependentProgramAudit.md` (`phase12_probe.json`) audits all eleven
phases and adds three measurements. **It refutes the hypothesis above.**

The coherence claim is descriptively true -- at the generator's cloud the
field's bulk demand is reproducible across disjoint batches while its tail
demand is not (.375 vs .066, **5.7x**). But it is not causal:

| arm | tail end | 2nd moment | ED2 |
|---|---:|---:|---:|
| moving K=1 | .0048 | .547 | .8145 |
| moving K=16 *(large + coherent)* | .0035 | .489 | .7738 |
| moving avg=16 *(coherent, fixed size)* | .0035 | .404 | 1.0487 |
| **fixed particle cloud** | **.1590** | .572 | **.4658** |
| **fixed data cloud** | **.0948** | **1.165** | **.3595** |

Sixteen committed rollout steps -- large *and* coherent -- leave the tail at
.0035. **What distinguishes the arms that work is self-reference**: every
failing arm has the form `T = f + Delta`, anchored to the generator's own
output, and rollout does not change that. The two working arms are the two
whose target never references `f`.

Also recorded: a confound in the earlier Phase-11 probe (it measured the fixed
arms' tail on their *training* latents) was checked and the finding survived
on a fresh probe -- the 20-45x gap is in the learned map, not memorized
points. And a claim made in the previous pass was wrong: Phase 2 carries a
rollout *probe* but it was **never executed**; rollout ran for the first time
here. Verified from sealed artifacts: **five supersession gates, all
`passed=false`** -- R11 has never been superseded by anything tried.

**Proposal: replace the self-referential teacher with an external one** --
have the generator chase an *evolving particle cloud* with a transport
assignment (nearest-neighbour or Sinkhorn). A fixed particle cloud already
beats the moving teacher on every metric with an *arbitrary* pairing, so the
assignment is the missing piece. This meets the repository's existing
`coherent_transport` / `ConditionedTransportAmortization` track on evidence
rather than analogy, and must carry a cost ledger against the paper's NFE-1
claim.

### Phase 12: the external target works, the assignment does not

(`EncoderIndependentPhase12Protocol.md` frozen pre-outcome,
`EncoderIndependentPhase12Results.md`, `phase12.json`.)

| arm | latents | tail | 2nd moment | ED2 | cost |
|---|---|---:|---:|---:|---:|
| A0 self-reference | fresh | .0041 | .331 | 1.2996 | 1.0x |
| **A1 self + R11** | fresh | .0523 | .873 | **.1112** | 1.0x |
| **A2 particles, index pairing** | **fixed** | **.3243** | **.922** | **.2091** | 3.6x |
| A3 particles, nearest | fresh | .3616 | **.000** | 19.51 | 3.6x |
| A4 particles, Sinkhorn | fresh | **.0007** | .343 | 1.3710 | 3.6x |

**The gate does not fire -- R11 survives a sixth supersession test -- but the
hypothesis is confirmed.** Removing the self-referential anchor (A2) gives
**tail x79** (and 2.5x real data's own .132), second moment .331 -> **.922,
inside the band**, and **ED2 6.2x better**, per-seed .946/.910/.922. All on a
*fresh probe*, so the learned map carries the tail, not just the fitted
points. It misses the gate by 1.5x at 3.6x training cost; inference is
unchanged at NFE = 1.

**Both amortizing assignments fail, for two distinct reasons.**
*Nearest-neighbour* collapses to second moment **.000** on every seed -- a
greedy match lets many samples claim one particle. (Its tail reads *high*,
.36-.47, while the generator is destroyed: tail is a means, ED2 is the
outcome.) *Sinkhorn's barycentric projection* gives tail **.0007**, a fifth of
the self-referential baseline: a barycentric projection is a conditional
expectation, so by the law of total variance it **contracts** -- reintroducing
the very pathology the phase was built to remove, and more severely. That was
a foreseeable design error given this program's own findings.

### The assignment was never the problem

`EncoderIndependentPhase12Investigation.md` (`phase13_probe.json`) ran the
missing control and the proposed fix, and **redirected the line**.

* **the Phase-12 attribution holds.** A2 differed from the baseline in two
  ways -- external target *and* fixed latents. Self-reference with **fixed**
  latents gives ED2 **1.0079** against fresh latents' .9908: **fixed latents
  alone do nothing**, so the external target is load-bearing;
* **both failures diagnosed directly**, by measuring the target before
  training: greedy nearest claims **1.0 distinct particles** (target tail
  .0000, second moment .000 -- a single point from step one), and the
  barycentric projection gives target tail **.0037** against the cloud's
  .4059, a factor of 110;
* **the Hungarian assignment fixes exactly that** -- target tail .327-.367,
  second moment .976-1.105, essentially perfect -- **and still fails**:

| Hungarian assignment | target tail | generator tail | ED2 |
|---|---:|---:|---:|
| **fixed** latents | .3669 | **.3430** | .3014 |
| **fresh** latents | .3267 | **.0009** | .8238 |

Same assignment, same target quality, and the tail transfers or is destroyed
by a factor of **380** depending only on whether the correspondence is
stable. **So "a balanced and hard assignment" was the wrong diagnosis** -- an
exactly optimal one does not help. What fixed latents supply is a *stable
correspondence*: the same latent maps to the same particle every step, so the
generator learns one consistent function, where a per-step re-assignment is
matched against a different sample each time and the generator averages the
inconsistency away.

### Phase 13: the amortizer works, and still loses to R11

(`EncoderIndependentPhase13Protocol.md` frozen pre-outcome,
`EncoderIndependentPhase13Results.md`, `phase13.json`.)

| arm | bank | visits/pair | ED2 | / particles | 2nd moment | tail | band |
|---|---:|---:|---:|---:|---:|---:|---|
| C0 self-referential | -- | -- | .9624 | 5.05 | .380 | .0059 | out |
| **C1 self + R11** | -- | -- | **.0955** | .44 | .972 | .0589 | in |
| C2 bank 256 | 256 | 1000 | .6184 | 3.00 | 1.353 | .4200 | out |
| **C3 bank 512** | 512 | 500 | **.1391** | **.73** | **.989** | .1607 | in |
| **C4 bank 1024** | 1024 | 250 | **.1449** | **.70** | .771 | .0442 | in |
| C5 bank 2048 | 2048 | 125 | .3835 | 1.84 | .573 | .0248 | out |

*(particle cloud ED2 .2089; gate ceiling .1193)*

**Gate not passed** -- the best bank misses R11's ceiling by **1.17x**, the
closest any alternative has come, and **R11 survives a seventh supersession
test**. But the separately declared **trend criterion passed**: the amortizer
reaches **.70-.73x the particle cloud's own ED2** -- *better than the system
it amortizes* -- with the second moment in band and **the tail restored 27x**
(.0059 -> .1607, against real data's ~.13, which R11 never produces at .0589).
Per-seed .157/.139/.127, from a baseline of .9624: a factor of **6.9**.

**The sweep is not monotone, and the `visits/pair` column says why.** At a
fixed 1000-step budget, bank 256 gets 1000 visits (over-fitted: second moment
**1.353**, out of band) and bank 2048 gets 125 (under-fitted, tail decaying
back toward baseline). The U-shape is the budget, not the method, so no
optimal bank size is claimed. Jitter hurts (.5868 vs .3835 on the same bank):
correspondence stability wants the *same* latents, not merely frequent ones.
Cost is **14.4x** the baseline in kernel pairs, all of it the particle cloud;
inference is unchanged at NFE = 1.

**Honest position: on the program's own gate this is not an improvement.**
R11 is a free scalar rescale; this is a second particle system at 14.4x
training cost, landing 1.17x behind. What it *is*: the first **derived**
structural account reproducing most of R11's effect, and independent
confirmation that the self-reference diagnosis is right.

**The one settling experiment**: hold visits/pair constant (bank
512/1024/2048/4096 at 1000/2000/4000/8000 steps). Below .0955 and the
structural route wins on quality; a plateau near .13-.14 means 512-1024 pairs
is the ceiling at this model size, R11 stands, and the line closes as a
*mechanism* result rather than a method. `EncoderIndependentAnchorGradientDiagnosis.md` traces the anchor's
optimization failure to a measured high-frequency gradient bottleneck: the
anchor gradient scales with the frequency norm, so the high band supplies the
largest gradient while carrying pure noise, and only a coarse-only bank
descends. Detection power and optimization tractability are in direct conflict
at fixed bands.

### GPU

Enabled and validated (`device.py`, `check_device.py`, `device_check.json`).
The default `--with torch==2.7.1` resolves to the **CPU-only** wheel on
Windows; the CUDA build needs the PyTorch index:

```powershell
uv run --python 3.12 `
  --extra-index-url https://download.pytorch.org/whl/cu126 `
  --index-strategy unsafe-best-match `
  --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 `
  --with numpy --with scipy python -m <module>
```

| check | result |
|---|---|
| **arithmetic** (one forward pass) | **9.4e-7** -- float32 round-off |
| trajectory (after 40 training steps) | ~1.7e-4 -- chaotic amplification, not bias |
| speed at field cloud 64 / 256 / 512 | **7.4x / 8.7x / 11.6x** |

Judge equivalence on the **arithmetic** check; the trained comparison measures
trajectory chaos (0 steps 9.4e-7 -> 20 steps 2.6e-4 on either device), which is
what multiple seeds are for. Second moments agree to ~1e-4, far below any
effect size in this program.

Three rules are built into `device.py`: **the device is declared, never
auto-detected** (default CPU, since every sealed artifact is a CPU result);
**random draws are made on CPU and moved**, so the sample path is identical
across devices and any difference is arithmetic alone; and **TF32 is off by
default**, since it drops matmul mantissa to ~10 bits.

```powershell
uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy `
  python -m numerics.encoder_independent_drifting.tests.run_all

uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy `
  python -m numerics.encoder_independent_drifting.run_phase0_gate

uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy `
  python -m numerics.encoder_independent_drifting.run_phase1_screen
```

## Formula crosswalk (Python -> Lean)

| `driftlab.py` | Lean declaration |
|---|---|
| `gaussian_kernel` | `Paper.gaussianKernel` |
| `alg2_kernel` | `Algorithm2.algorithm2Kernel` (= `Paper.laplaceKernel`) |
| `interaction_matrix` | `inducedInteractionVector` at the empirical point basis (`basisInteraction_empirical2` / `EmpiricalFrameBound`) |
| `certified_frame_constant` | `gaussianEmpiricalPointCertifiedFrameConstant` |
| `frame_ceiling` | `gaussianEmpiricalPoint_frameConstant_le` |
| `frame_violation` | `InteractionFrameBound` (sup norm on the probe axis) |
| `u01_bare` | two-atom `U_01` via `basisInteraction_empirical2` (`empirical01Laplace` family) |
| `col_reweight_scale` | `inducedInteractionVector_columnReweighted01_eq_smul` |
| `column_reweighted_weight` | `algorithm2ColumnReweightedWeight` |
| `modified_field_two_atom` | limiting field of `ColumnReweightedMeanShiftFiniteSetup` |
| `bare_field_two_atom` | `Paper.meanShiftDrift` (eqs. 8/10) |
| `compute_v_paper` | paper Algorithm 2 pseudo-code (verbatim port) |
| `compute_v_lean` | `Paper.algorithm2Drift` (literal `finiteSoftmax` pipeline) |
| `centroid_diff` | `Algorithm2.noMaskCentroidDrift` / masked analogue |
| E3b bias | `deletedNegativeColumnWeight`, hypotheses of `deletedNegativeCentroid_meanSquare_le` |
| E3c delta | `maskPenaltyFactor`, `algorithm2Drift_sub_deletedDrift_norm_le` |
| E5 chain | `columnReweighted01_coefficientStability` + `estimate_failure_le_meanSquare` + `selfNormalizedIndexed_meanSquare_le` + `selfNormalizedIndexed_deviation_prob_le` |
| E6 gate | `CFGTargetNonnegative` / `cfgTargetCoefficients` (`CFGAffine.lean`) |
| `real_feature_diagnostics.py` | empirical finite matrix for `InteractionFrameBound`; column-reweighted matrix for `ColumnReweightedMeanShiftFiniteSetup` |

## Experiments

- **E0** — transcription audit: two independent Algorithm-2 implementations
  agree to 1e-16; spot-checks of proved identities (matched-batch zero, the
  mass-product centroid form, the drift bound, frame-certificate validity).
- **E1** — certified frame constant vs basis size `m`, support window, and
  probe placement; the sqrt(2/e) separation sweet spot.
- **E2** — exact column-reweighting scale vs the proved `1/N` floor.
- **E3/E3b/E3c** — Monte-Carlo: `1/N` SNIS rate; the estimator's bias floor
  against the *bare* field (it targets the column-reweighted field); eye-mask
  leave-out bias `O(1/N)`; the self-mask perturbation scale.
- **E4** — softmax effective sample size per temperature at `N = 64` under a
  normalized-distance feature-geometry model: what each `tau` "sees".
- **E5** — the conditioning ledger: the full certified chain multiplied out
  into a sample-complexity number, comparing the old deterministic denominator
  floor, the new high-probability denominator-tail refinement, the LLN-typical
  benchmark, and observed Monte-Carlo scaling.
- **E6** — how often the CFG affine target is an actual probability vector at
  the paper's guidance scales.
- **Real features** — `real_feature_diagnostics.py` runs the same frame and
  Algorithm-2 column-reweighting diagnostics on externally supplied encoder
  features.
