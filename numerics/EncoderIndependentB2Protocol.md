# Stage B2 protocol: paired bridge plus normalized Laplace drift energy

**Status: PREFLIGHT GO. A genuinely fresh external pool has been sourced and
validated. No B2 baseline or candidate has been trained. The next action is the
fresh-source paired B0 baseline, followed by the immutable B2 freeze.**

B2 is licensed by the passing B0 and B1 artifacts. It asks one narrow question:

> Does a differentiable, encoder-free, normalized Laplace mean-shift correction
> reduce its held-out drift-energy discrepancy while retaining the coverage of
> the paired B0 flow-matching bridge?

A degradation is evidence against this correction, not against
encoder-independent generation generally. A pass is a finite-sample mechanism
result, not an empirical proof of identifiability.

---

## 1. Why this is not the original proposed B2

The audited first design used `kernel_gradient.field` with
`direction_mode="paper"`. That cannot be used:

1. the API returns a detached stop-gradient field, so the proposed loss has no
   parameter gradient;
2. Algorithm 2's finite bi-softmax field is not the normalized population field
   consumed by the proved arbitrary-target Laplace converse;
3. its raw cross-mass factor can vanish when target affinity underflows, giving
   almost zero loss to a catastrophically remote model;
4. the design reused generated probes as negatives without specifying the
   dependence or gradient semantics.

B2 therefore has its own implementation under
`encoder_independent_drifting/stage_b2/`. It does not call the detached field
API and does not use Algorithm 2's column normalization or cross-mass factor.

---

## 2. Population object and theoretical scope

For the exact data-space Laplace kernel

\[
k_\tau(x,y)=\exp(-\|x-y\|_2/\tau), \qquad \tau>0,
\]

define

\[
V_{p,q}(x)=
\frac{\int k_\tau(x,y)y\,dp(y)}{\int k_\tau(x,y)\,dp(y)}-
\frac{\int k_\tau(x,z)z\,dq(z)}{\int k_\tau(x,z)\,dq(z)}.
\]

This is exactly the difference of normalized Laplace mean shifts: the two
displacement terms `-x` cancel. The verified Euclidean converse states that
`V_{p,q}(x)=0` for every `x` implies `p=q`.

The population energy used to motivate B2 is

\[
\mathcal E_\rho(p,q)=\mathbb E_{X\sim\rho}\|V_{p,q}(X)\|_2^2.
\]

The probe law `rho` is a target sample plus independent Gaussian noise with
positive standard deviation. Its population ideal has full Euclidean support.
Together with continuity of the normalized Laplace field, zero population
energy then upgrades to pointwise zero drift before the converse is applied.

This support step is load-bearing. Sampling probes only from the current model
would allow the model to avoid regions where its field is nonzero.

The finite minibatch loss below is biased and noisy. It is a stochastic
surrogate for this energy; zero finite loss is not claimed to identify a law.

---

## 3. Differentiable sample-split loss

At a correction event draw three independent roles:

1. `X_i = C_i + sigma_probe * epsilon_i`, where `C_i` is an augmented target
   sample and `epsilon_i` is standard Gaussian noise;
2. positives `Y_j` from a separate augmented target stream;
3. negatives `Z_l` by integrating a separate prior batch through the model.

Then compute stable normalized weights

\[
a_{ij}=\operatorname{softmax}_j(-\|X_i-Y_j\|_2/\tau),\qquad
b_{i\ell}=\operatorname{softmax}_\ell(-\|X_i-Z_\ell\|_2/\tau)
\]

and

\[
\widehat V_i=\sum_j a_{ij}Y_j-\sum_\ell b_{i\ell}Z_\ell,
\qquad
L_{\mathrm{drift}}=\frac1{N_X}\sum_i\|\widehat V_i\|_2^2.
\]

`softmax` evaluates the analytically normalized Laplace weights after a rowwise
logit shift, so a remote finite row cannot underflow to an all-zero affinity
row. There is no denominator floor and no cross-mass factor.

The probes and positives are detached. Gradients flow through every generated
negative sample and its Euler trajectory. A separate generated negative batch
prevents self-affinity and reused-sample bias.

The training objective is

\[
L_{B2}=L_{\mathrm{flow}}+\lambda_{\mathrm{event}}L_{\mathrm{drift}}
\]

at the frozen correction cadence. Squaring the mean field is forbidden because
opposite directions can cancel. RMS-normalizing each field is forbidden because
it makes the squared energy constant.

---

## 4. Frozen configuration

| item | value |
|---|---:|
| paired units | 300, 301, 302 |
| calibration units | 420, 421, 422 |
| correction cadence | every 10 optimizer steps |
| probe / positive / negative batch | 64 / 64 / 64 |
| generated-negative trajectory | 8 Euler steps |
| probe noise standard deviation | 0.05 in normalized pixel units |
| geometry | raw data space |
| kernel | exact Laplace `exp(-norm/tau)` |
| bandwidth quantile | 0.5 |
| ESS calibration | off-diagonal, self excluded |
| target realized ESS fraction | 0.60 |
| ESS samples / iterations | 128 / 24 |
| event gradient target | 0.25 of flow-gradient norm |
| gradient tolerance | 0.125–0.5 in every calibration unit |
| maximum preflight GPU-memory fraction | 0.95 |
| audit batch / replicates | 128 / 6 |
| B0 baseline recall floor | 0.05 per unit |
| recall non-inferiority margin | 0.025 absolute per unit |
| drift reduction target | at most 0.5 times paired B0 excess |
| required paired audit wins | 5 of 6 per unit |
| final rule | at least 2 of 3 units pass |

The smaller recall margin replaces the original 0.05 margin, which permitted a
large relative coverage loss at B0's observed recall.

`tau` is calibrated exactly once from target-only training data. The calibration
uses distinct-neighbour ESS directly (`exclude_self=True`); it does not reuse
the legacy diagonal-dominated label where nominal 0.05 meant roughly 0.6 in the
actual field. Kernel health is recorded during correction events but never used
to adapt `tau` after candidate outcomes exist.

`lambda_event` is calibrated rather than guessed. Three outcome-blind units
measure flow and drift parameter-gradient norms. The median weight targeting
0.25 is accepted only when that one frozen weight places every unit within a
factor of two.

---

## 5. Pairing and computation

B2 reuses B0's model initialization, flow batches, endpoint noise, bridge times,
and augmentation streams. Only the declared correction streams are new. Thus
B0/B2 first differ at the objective, not at initialization or flow data order.

The preflight records peak memory and event timing. Confirmation reports actual
wall time relative to each historical B0 unit. Forward counts are diagnostic
only because they omit pairwise-kernel FLOPs and backward cost.

B1 remains a report-only incumbent. The causal B2 comparison is against B0,
while B0/B1/B2 quality metrics are shown together. B2 is not permitted to claim
that it improves on B1 merely because it optimizes a different discrepancy.

---

## 6. Held-out drift quantity

For audit replicate `r`, use disjoint fresh probe centres, positives, and
real-floor negatives:

\[
E_r(q)=\widehat{\mathcal E}_r(p,q)-
       \widehat{\mathcal E}_r(p,p_{\mathrm{independent}}).
\]

Candidate and B0 use the same:

- probe centres and Gaussian probe noise;
- target positives;
- real-real floor batch;
- bandwidth;
- generated-negative prior seed.

The floor is a matched operational correction for finite-sample noise. It is
not asserted to be an unbiased removal of every distribution-dependent
variance term. Therefore the reported claim is reduction of this frozen
held-out discrepancy, not recovery of the exact population energy.

If every B0 unit does not have positive median excess, recall above 0.05, and
valid collapse/memorization vetoes, the paired baseline is not a sound B2
reference and confirmation does not run.

---

## 7. Fresh-data boundary

B1 consumed 9,728 of the 10,000 official CIFAR-10 test images. B2 was designed
after that result, so those data are no longer a clean independent confirmation
source. The implementation refuses source IDs naming the official CIFAR-10 test
set.

Before the paired baseline can run, the operator must supply a `.npy` or `.npz`
file containing a genuinely unused external image pool, give it a stable source
ID, and pass `--attest-unused`. Integer images are mapped from `[0,255]` to
`[-1,1]`; floating input is rejected unless its `[0,1]` or `[-1,1]` convention
is declared explicitly. The loader records its absolute path, byte hash, shape,
dtype, applied normalization, and normalized range. The allocation uses
disjoint reference, control, probe, positive, and floor-negative roles and is
frozen by digest.

An external dataset may introduce distribution shift. That limitation must be
reported. It is preferable to adaptive reuse of B1's test images.

### Sourced pool (2026-07-31)

The external pool is a balanced subset of the ImageNet-derived contribution to
[CINIC-10](https://doi.org/10.7488/ds/2448), distributed by the University of
Edinburgh under CC BY 4.0. CINIC-10 has 210,000 ImageNet-derived images and
60,000 embedded CIFAR-10 images. Its official filename convention exposes the
origin of every image, so the sourcing utility rejects every CIFAR-style name
before decoding.

`source_cinic10_pool.py` reproducibly selected 600 images from each of the ten
CIFAR-compatible classes using seed `20260731`. It then compared decoded-pixel
hashes against the complete official 60,000-image CIFAR-10 corpus and rejected
duplicates within the proposed pool. The accepted artifact has:

- source ID
  `cinic10-imagenet-only-balanced-600-per-class-seed-20260731`;
- shape `6000 x 32 x 32 x 3`, dtype `uint8`;
- artifact SHA-256
  `89cb9f8157f053f6343de4690379cbf32d815482a1b45f0ac69906c98a0d8728`;
- zero decoded-pixel matches to CIFAR-10;
- zero retained decoded-pixel duplicates (eight duplicate candidates were
  skipped);
- exactly 600 retained images per class.

The generated pool is
`stage_b2/data/cinic10_imagenet_only_b2_seed20260731.npz`; its tracked adjacent
provenance record binds the official archive hash, source member paths,
selection rule, and output hash. The binary is intentionally ignored by git.
The B2 loader accepted it as `6000 x 3 x 32 x 32` in `[-1,1]`; all frozen roles
consume 5,888 disjoint indices and leave 112 unused.

This pool is class-aligned but not in-domain CIFAR-10. A B2 result on it tests
whether the mechanism survives a documented ImageNet-to-CIFAR distribution
shift. It must not be reported as a replacement estimate of CIFAR-10 sample
quality.

---

## 8. Execution chain

```text
python -m numerics.encoder_independent_drifting.stage_b2.preflight
    verifies passing B0/B1 artifacts
    calibrates target-only tau and gradient weight
    checks remote-target and equal-law behavior
    records resource use                                      -> GO / NO-GO

python -m numerics.encoder_independent_drifting.stage_b2.baseline
    --fresh-data <external.npy-or-npz>
    --fresh-source-id <stable-id>
    [--fresh-float-encoding minus-one-one|zero-one]
    --attest-unused
    measures B0 on the exact fresh B2 instrument              -> GO / NO-GO

python -m numerics.encoder_independent_drifting.stage_b2.freeze
    binds protocol, code, prerequisites, fresh-data hash,
    allocation, tau, lambda, and B0 baseline

python -m numerics.encoder_independent_drifting.stage_b2.confirmation
    refuses every changed hash and runs the three paired units
```

Adding a future experimental stage does not invalidate B2: its manifest hashes
an explicit dependency set rather than every future package file. The existing
B1 freeze remains loadable because B2 lives in an isolated subpackage.

---

## 9. Gate and interpretation

A unit passes only when all hold:

1. B2 recall is at least paired B0 recall minus 0.025;
2. median B2 drift-energy excess is at most half of paired B0 excess;
3. B2 beats B0 in at least five of six paired audits;
4. collapse and augmentation-aware memorization vetoes pass;
5. the assigned matched-real metric control has recall above 0.5.

At least two of three units must pass. An invalid control makes the experiment
`VOID`, not `FAIL`.

- **PASS:** this differentiable encoder-free correction reduces its frozen
  fresh-source drift discrepancy while retaining B0-detectable coverage.
- **FAIL, recall:** the correction harms coverage at the calibrated strength.
- **FAIL, discrepancy:** the model does not reduce the chosen held-out energy.
- **NO-GO:** preflight or the B0 baseline cannot resolve the test.
- **VOID:** a confirmation control or immutable boundary failed.

None of these outcomes proves or refutes the exact Laplace converse. The
experiment tests whether a finite, theory-aligned surrogate is useful inside
this particular generator and compute regime.

---

## 10. Implemented regression requirements

The B2 test suite checks:

- identical empirical laws give exactly zero field;
- swapping positive and negative laws negates the field;
- autograd agrees with a central finite difference;
- the loss reaches model parameters through the Euler-generated negative law;
- a target shifted by 100 units gives a large finite loss rather than a false
  zero from affinity underflow;
- off-diagonal bandwidth calibration reaches the declared ESS;
- B0 and B2 share initialization and the first flow batch;
- stochastic roles replay independently;
- the official CIFAR-10 test source is rejected;
- fresh allocation roles are disjoint;
- paired audits share noise, target roles, bandwidth, and real-real floor.

---

## 11. Outcome-blind preflight result

The CUDA 12.6 preflight passed on 2026-07-31:

- exact-Laplace bandwidth `tau = 7.085388`, with achieved off-diagonal ESS
  fraction `0.59999984` for the declared target `0.60`;
- calibrated event weight `lambda_event = 0.000192943`;
- frozen weighted gradient ratios `0.20385`, `0.36377`, and `0.25000`, all
  inside the declared `[0.125, 0.5]` interval;
- maximum calibration memory `5,621,757,952` bytes, or about `87.31%` of the
  6 GiB GPU, below the frozen `95%` ceiling;
- equal empirical laws produced energy exactly `0`;
- the remote-target regression produced energy `30,555,420`, versus `137.70`
  for the near comparison, so the old affinity-underflow false zero is absent.

The memory result is a GO but leaves limited headroom. Actual confirmation wall
time and peak memory remain mandatory report fields. This preflight inspected no
B2 candidate or confirmation outcome.
