# Stage B1 protocol: paired bridge plus an encoder-free spectral anchor

**Status: repaired implementation, not yet executed.**

B1 is licensed by the final F3B/B0 result (`f3b_confirmatory.json`), which
passed in all three units. B1 asks a narrower follow-up question:

> Can an encoder-free source-law regularizer reduce a held-out distributional
> discrepancy without materially reducing the detectable coverage already
> achieved by B0?

This is a paired finite-sample experiment. It is not a new identifiability
theorem and it is not a compute-matched efficiency comparison.

## 1. Mathematical object and honest claim

For generated law \(p\), target law \(q\), and a full-support spectral
distribution \(\rho\), the ideal population anchor is

\[
  A_\rho(p,q)
  = \mathbb E_{\omega\sim\rho}
      \left|\varphi_p(\omega)-\varphi_q(\omega)\right|^2 ,
\]

where \(\varphi_p\) is the characteristic function of \(p\). Characteristic
function uniqueness makes the ideal object measure determining:

\[
  A_\rho(p,q)=0 \quad\Longrightarrow\quad p=q.
\]

The implemented loss is a finite random-feature V-statistic. It is a
nonnegative stochastic surrogate for \(A_\rho\), but a fixed finite bank is
not measure determining. Its observed value, including zero, does not by
itself imply equality of laws. Training banks are partially refreshed to
reduce overfitting to one finite witness, and all scientific reporting uses
independent audit banks.

The flow-matching MSE also has irreducible conditional variance in general.
Consequently, B1 makes no argument of the form
`observed total loss = 0 -> p = q`. The ideal characteristic-function anchor,
not the finite training loss and not the formal Laplace mean-shift converse,
is the correctness authority for this experiment.

## 2. What changes relative to B0

B1 uses

\[
  L_{\mathrm{B1}}
  = L_{\mathrm{flow}}
    + \lambda_{\mathrm{event}} L_{\mathrm{anchor}}
\]

on anchor events. The anchor is evaluated on actual model samples obtained by
integrating from the prior, and gradients pass through the entire Euler
trajectory.

For units 300, 301, and 302, B1 reuses exactly B0's:

- model initialization;
- data-order stream;
- endpoint-noise stream;
- bridge-time stream;
- horizontal-flip stream;
- architecture, optimizer, EMA, training steps, batch size, and final NFE.

B1-only anchor data batches, anchor flips, anchor priors, inference priors,
training banks, and refresh operations have separate deterministic streams.
Anchor targets use the same identity/horizontal-flip endpoint law as B0. This
makes the comparison paired at the initialization and flow-batch level while
keeping new stochastic roles independent.

## 3. Frozen B1 configuration

| Item | Value |
|---|---:|
| paired units | 300, 301, 302 |
| anchor cadence | every 10 optimizer steps |
| anchor batch | 64 |
| anchor trajectory | 8 Euler steps |
| training features | 256 |
| audit features | 512 |
| audit replicates | 6 |
| training estimator | biased, nonnegative V-statistic |
| report-only estimator | unbiased U-statistic |
| band schedule | coarse to fine |
| bank refresh | 25% every 25 anchor events |
| event gradient target | 0.25 of flow-gradient norm |
| calibration tolerance | frozen ratio within 0.125 to 0.5 in every calibration unit |
| average-event interpretation | 0.025 because cadence is 1 in 10 |
| recall non-inferiority margin | 0.05 absolute, per unit |
| anchor reduction target | at most 0.5 times paired B0 excess |
| required audit wins | 5 of 6, per unit |
| final rule | at least 2 of 3 units pass |

The numerical event weight is not guessed. Before any B1 model is trained,
three outcome-blind calibration models measure flow and anchor parameter
gradient norms. The frozen weight is the median value satisfying

\[
  \lambda_{\mathrm{event}}
  \frac{\|\nabla L_{\mathrm{anchor}}\|}
       {\|\nabla L_{\mathrm{flow}}\|}
  = 0.25.
\]

This preserves the meaning of the intervention despite the one-in-ten
cadence. The single median weight must leave every calibration unit within a
factor of two of the target ratio; otherwise preflight returns `NO-GO`.

## 4. Data separation

The 40,000-image CIFAR training split remains the only endpoint-training pool.

Calibration consumes only 2,048 of the 2,832 indices left unused by B0's
evaluation allocation:

- 512 calibration reference images;
- three disjoint 512-image matched-real controls.

Confirmation uses the previously untouched official CIFAR-10 test set:

- 2,048 reference images;
- three disjoint 512-image metric controls;
- six pairs of disjoint 512-image target/control batches for the anchor audit.

This consumes 9,728 test images and leaves 272 unused. All roles and index
digests are allocated before a B1 outcome exists.

## 5. Mandatory pre-B1 stages

### 5.1 Outcome-blind preflight and calibration

`preflight_b1.py` must return `GO`. It checks:

- finite, nonzero anchor gradients through the 8-step integration path;
- calibrated event-gradient ratios;
- peak memory below 90% of device memory;
- valid matched-real metric controls;
- positive collapse and memorization veto thresholds;
- memorization against both original and horizontally flipped training images;
- finite-bank sensitivity to a moment-matched Gaussian and adopted B0 samples.

The last check is only an instrument check. It does not select a B1 result.

### 5.2 Fresh paired B0 baseline

`measure_b1_baseline.py` re-evaluates immutable B0 EMA checkpoints with the
exact priors, audit banks, target batches, official-test reference set, and
controls that B1 will use. This removes the invalid comparison between old B0
units and fresh, independently seeded B1 units.

The baseline must return `GO`: every assigned metric control and B0 veto must
be valid, and the B0 audit excess above its matched-real floor must be
resolvable. If the denominator is already at the finite-sample floor, the
declared 50% reduction is not testable and B1 does not run.

### 5.3 Source freeze

`freeze_b1.py` hashes the protocol, every executable package/test source, the
adopted B0 artifact and checkpoints, calibration, paired baseline, allocation,
configuration, scale, and calibrated event weight. Any later source or
protocol change invalidates the freeze.

## 6. Confirmation gate

For audit replicate \(r\), define the biased excess over its matched-real
finite-sample floor:

\[
  E_r(\text{model})
  = \widehat A_{\mathrm{biased}}(\text{model},q)
    - \widehat A_{\mathrm{biased}}(q',q).
\]

Candidate and B0 use the same bank, target batches, real-real floor, and
inference prior in each replicate. The unbiased U-statistic is reported by
band but never used for the gate because it can be negative.

A unit passes only if all of the following hold:

1. B1 recall is at least that unit's fresh B0 recall minus 0.05.
2. Median B1 biased excess is at most 0.5 times median B0 biased excess.
3. B1 improves on B0 in at least five of six paired audit replicates.
4. Collapse and augmentation-aware memorization vetoes pass.
5. That unit's assigned matched-real metric control has recall above 0.5.

At least two of three units must pass. Any invalid assigned control makes the
whole experiment `VOID`, not `FAIL`.

## 7. Cost and interpretation

B1 adds one 8-step differentiable trajectory every 10 training updates. The
runner records anchor events, refreshes, extra forward equivalents, wall time,
peak memory, and the ratio to historical B0 wall time. This is extra compute,
so a pass establishes an accuracy/discrepancy trade-off, not improved
efficiency. A compute-matched B0 frontier would be a separate experiment.

FID at 512 generated samples remains indicative and is comparable only between
these identically measured arms. Recall, precision, KID, collapse diagnostics,
and augmentation-aware memorization checks remain part of the report.

## 8. Execution order

After the source tree is stable:

```powershell
python -m numerics.encoder_independent_drifting.preflight_b1
python -m numerics.encoder_independent_drifting.measure_b1_baseline
python -m numerics.encoder_independent_drifting.freeze_b1
python -m numerics.encoder_independent_drifting.run_b1_confirmation
```

Use the repository's pinned CUDA `uv run` environment rather than an ambient
Python installation. Do not alter code, protocol text, constants, artifacts,
or hashes between freezing and confirmation.

## 9. Licensed conclusions

- **PASS:** under this frozen compact CIFAR setting, the encoder-free spectral
  regularizer materially reduced held-out characteristic-function discrepancy
  above the matched-real floor while retaining B0 recall within the declared
  margin in at least two paired units.
- **FAIL:** this frozen weight/cadence/bank did not establish that joint result.
- **VOID:** the measurement or prerequisite controls were invalid.

No outcome licenses a claim that the finite bank proves \(p=q\), that B1 is
compute efficient, that it matches the paper's method, or that it removes the
need for representation learning on arbitrary high-dimensional data.
