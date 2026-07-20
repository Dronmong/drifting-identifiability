# Phase C validation pass

*Audit-corrected procedure implemented by `driftbench_v2.py`.  This pass
preserves the original `driftbench.py` runs as historical evidence and tests
the causal and mathematical claims that the first pass could not isolate.*

## Scope

The experiments remain synthetic finite-particle tests.  They use normalized
multimodal Gaussian targets, exact sample-based diagnostics, a finite target
minibatch, and either:

* the bare self-normalized mean-shift estimator; or
* a dimension-general transcription of the paper's row/column bi-softmax
  Algorithm-2 drift, cross-checked exactly against
  `driftlab.compute_v_paper` in one dimension.

They do not test learned encoder features or a neural generator.

## Reproducibility

Each run writes a directory under `numerics/bench_runs_v2/` containing:

* a manifest with commit, dirty status, diff and source hashes, command line,
  complete configuration, versions, wall time, and kernel-pair count;
* per-seed CSV data rather than aggregate medians only;
* compressed metric trajectories;
* the exact source snapshot used for the run;
* a generated textual summary.

The committed standard-profile runs are:

* `20260719-155529-C1-validated-bandwidth`;
* `20260719-160032-C2-validated-generator`;
* `20260719-160230-C3-validated-mask`.

They were run from a dirty tree while this validation pass was being created,
but the source snapshot and SHA-256 hash remove the first pass's source
ambiguity.  A post-commit rerun can provide a clean-commit duplicate without
changing the protocol.

## C1: separate bandwidth from step size

Targets are normalized so their minimum mode separation is exactly `L = 1`.
The fixed paper-temperature comparison therefore uses the literal normalized
set `{0.02, 0.05, 0.2}`.  All runs start from a central Gaussian cloud of width
`0.25 L`, which initially misses outer modes.

Two comparisons are kept separate:

1. **Bandwidth-only:** every policy uses the same conservative step
   `eta = 0.01 L`.  Multi-bandwidth policies receive proportionally fewer
   updates so all arms use essentially the same number of kernel matrix
   entries.
2. **Joint practical policy:** fixed bandwidth uses `eta = 0.1 tau`, while the
   scheduled policy uses `eta_t = 0.1 tau_t`.  This measures the combined
   procedure but is not used to attribute the gain to bandwidth alone.

The target-geometry schedule is run twice: once with oracle `L,sigma` and once
with values estimated by unsupervised k-means from target samples.  The oracle
fixed-bandwidth baseline is selected on separate tuning seeds from a
nine-point logarithmic grid, and its selection cost is recorded.

Energy-distance threshold crossing is checked at every update.  Censored runs
remain censored; summaries use a Kaplan--Meier median rather than a sentinel.

## C2: differentiate the coupled particle field

At an empirical truth `q = p`, the target atoms are frozen and the complete
field `F(q) in R^(N*d)` is differentiated by central finite differences with
respect to every coordinate of the flattened particle cloud.  This includes
all probe/support and particle/particle interactions omitted by the first
pass's singleton derivative.

For generator eigenvalues `lambda`, the explicit-Euler boundary is reported
only if every real part is strictly negative:

```text
eta* = min_lambda -2 Re(lambda) / |lambda|^2.
```

The implementation verifies `rho(I + eta* J) = 1` numerically and benchmarks
fixed fractions of both `tau` and `eta*`.  Fine (`tau = sigma`) and coarse
(`tau = L`) regimes are tested separately.

## C3: mask policy under both estimators

C3 crosses:

* estimator in `{SNIS, paper bi-softmax}`;
* equal and unequal target mode weights;
* particles per mode in `{1,2,4,8,16}`;
* target batch size in `{16,64}`;
* eye mask on/off;
* six paired seeds.

Endpoints with missing coverage or mass error above `0.4` are continued for
1,600 additional steps.  At low particle counts the final state is also
checked using a large frozen target sample for:

* RMS on-support residual;
* maximum real part of the full local generator;
* contraction after a small perturbation.

A state is called a `stable_wrong_candidate` only if it is wrong, has residual
below `1e-3 L`, has strictly contracting local spectrum, and returns after the
perturbation.  A dropped mode or bad finite-horizon endpoint alone is not
called an equilibrium.

## Commands

```powershell
uv run --with numpy --with scipy python numerics/driftbench_v2.py C1 --profile standard
uv run --with numpy --with scipy python numerics/driftbench_v2.py C2 --profile standard
uv run --with numpy --with scipy python numerics/driftbench_v2.py C3 --profile standard
uv run --with matplotlib --with numpy python numerics/phase_c_v2_figs.py
```

The `smoke` profile checks wiring.  The `full` profile expands seed count,
horizon, reference sample size, the `N/K = 32` cell, and `B = 256`.
