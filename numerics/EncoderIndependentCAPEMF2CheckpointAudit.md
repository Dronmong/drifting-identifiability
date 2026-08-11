# CAP-EMF-2 checkpoint audit

**Audit scope:** logic, implementation, methodology, artifact integrity,
recovery safety, cost control, and the failure modes observed in CAP-EMF-1.

**Current verdict:** the code and methodology repair pass is complete and the
local release candidate is mechanically green. This document still does not
authorize the paid run. A production `GO` exists only after fresh source-bound
evidence, the production-GPU admission/benchmark, and the aggregate-budget
preflight all pass.

The operator subsequently raised the hard all-in ceiling to **USD 50**. The
frozen protocol reserves USD 5 for non-training work and applies 15%
contingency to the benchmark's conservative training projection. This raises
the affordability ceiling only; it does not enlarge the model, horizons, arm
count, or scientific claim.

## 1. Release-blocking findings and repairs

1. **The old inference-corner gate was not comparable across arms.** Natural
   occupancy of a tiny corner is common for ordered uniform, rare for legacy,
   and essentially absent for ordered logit-normal. A shared minimum count was
   therefore impossible for one declared arm. Training now evaluates a fixed,
   exact `(t,r,h)=(1,0,1)` probe using all 2,048 sealed rows at 100k and 150k,
   under both raw and EMA weights. Natural occupancy remains a diagnostic.
2. **The benchmark called a resume path without proving continuation.** It now
   stops before update 2,000, reloads model, optimizer, EMA, RNGs, ledgers and
   counters from disk, performs update 2,000, and verifies continuity.
3. **Rolling recovery was vulnerable to instance loss and partial overwrite.**
   A run now requires a pre-existing, operator-attested, instance-independent
   mirror root. Recovery versions are content-addressed and immutable; a numeric
   commit is published only after the payload/sidecar pair is complete. Restore
   ignores incomplete newer work and fails closed on committed corruption.
4. **There was no aggregate dollar ceiling.** Preflight now prices the worst
   declared schedule, `C150 + 2*C300`, then adds contingency and a non-training
   reserve. It fails if that upper envelope exceeds the operator's ceiling.
   Each paid arm also has a conservative cumulative wall stop, enforced only
   after a recoverable checkpoint is durably committed.
5. **The CleanFID decision margin used the wrong difference.** The gate now uses
   the absolute direct disjoint real/real CleanFID and CleanKID observation. It
   is explicitly a deterministic finite-sample discrepancy, not a confidence
   interval.
6. **Noisy 2,048-sample auxiliary metrics were overused.** Repository KID and
   relative precision/recall are report-only. The standard 50k CleanFID and
   CleanKID scores select arms; absolute precision, recall, F1, duplicate, and
   exact-copy floors remain collapse vetoes.
7. **The concurrent legacy control could be selected but not loaded.** Its two
   historical-quality failures are now separated from binding failures. A
   legacy quality `NO_GO` may continue only if the independently recomputed
   control-validity record is `GO`; no safety or integrity check is exempt.
8. **One-step inference was asserted rather than executed.** The training
   runner now hooks the actual inference module and records the measured call
   count from a real `one_step_sample` execution.
9. **Checkpoint previews and feature evidence were incomplete.** Every
   checkpoint now publishes fixed, uncurated raw/EMA preview rows. Development
   evaluation preserves the exact 50k PNG manifest, full generated
   clean-Inception feature archive, and exact KID reference archive.
10. **The terminal 300k result was too weakly bound.** The final verifier is
    required to recompute the capability gate, bind 150k authorization to the
    300k ladder, load strict recovery state, reconcile final raw/EMA tensors,
    verify the durable recovery commit, revalidate the metric populations, and
    require both a concurrent margin-separated win and absolute quality
    retention.
11. **Per-arm recovery did not preserve the cross-arm authorization graph.** A
    second, attested common workspace now retains preflight evidence, all three
    150k arms, promotion leaves, selection, and both 300k arms in their original
    relative layout. Every paid runner rejects paths outside that workspace.
12. **A crash during terminal publication could strand a completed run.** A
    completed recovery can now re-enter finalization without another optimizer
    update. The existing result is excluded from generic mirroring until its
    full history/health/counter/checkpoint/snapshot/gate/durability ledger is
    reconstructed from the committed recovery. Incomplete local or remote
    pairs are quarantined and repaired; complete conflicting pairs fail closed.
13. **The documented terminal arm and batch were partly hard-coded.** Selection
    is now revalidated before compute and drives both the ordered arm and its
    concurrent legacy control. Generation, metric, and feature batches are
    explicitly frozen at 128 for the baseline and every 150k/300k candidate.
14. **Recorded metric scalars were accepted too early.** Preflight now reloads
    the full generated/reference clean-Inception archives and recomputes the
    baseline, positive-control, and real/real calibration decisions before any
    paid screen. The terminal verifier repeats the relevant baseline,
    calibration, and candidate recomputations.
15. **Immutable recovery history was absent from the resource budget.** The
    measured recovery, raw/EMA checkpoint, and snapshot sizes now drive a full
    campaign storage projection. It derives 150 recovery commits, 15 checkpoint
    pairs, and 30 snapshots from benchmark event counts, includes mirrored
    copies, a 20 GiB evaluation reserve, and 20% contingency, and requires both
    total and currently free durable capacity to pass. Each continuation also
    checks one-transaction free-space headroom.

## 2. CAP-EMF-1 failure mapping

| Prior failure or ambiguity | CAP-EMF-2 safeguard |
|---|---|
| finite-difference/JVP mismatch | full production-GPU quotient, target, and parameter-gradient admission matrix |
| sparse/incorrect endpoint evidence | fixed exact 2,048-row endpoint probe, raw and EMA |
| resumed run broke state/device continuity | real disk reload/next-update rehearsal; strict model/optimizer/EMA/RNG accounting |
| provider instance disappeared with artifacts | common layout-preserving workspace plus per-arm versioned recovery mirrors |
| long run could exceed spend or storage | aggregate compute ceiling, measured storage-capacity gate, and per-arm hard wall after safe recovery |
| one-call property was a literal field | measured hook around the executable sampler |
| metric environments/reference populations drifted | exact local environment and sealed full-train KID population bindings |
| relative auxiliary noise could choose a winner | 50k standard metrics select; small auxiliary metrics cannot rescue or rank |
| one 300k score could hide joint deterioration | concurrent legacy comparison plus historical/150k retention gates |

## 3. What remains before any paid training

The code audit cannot manufacture production evidence. Because the source
manifest and protocol changed during this repair, every earlier source-bound
sampler audit, gate calibration, baseline/control evaluation, metric
calibration, numerical admission, forensics record, benchmark, and preflight is
stale for authorization purposes. Regenerate them under fresh filenames.

The chronological gate is:

1. freeze the repaired code and protocol;
2. regenerate local sampler/gate/metric evidence and the historical baseline
   in the exact declared local evaluation environment;
3. provision and probe a shared durable filesystem (practically 200 GiB for
   the current measured plan), the common workspace, and fresh arm namespaces;
4. run production numerical admission, checkpoint forensics, and the genuine
   resume benchmark;
5. build the aggregate compute-and-storage preflight and require `GO`;
6. run the three matched arms only to 50k, re-admit every raw checkpoint, and
   stop immediately on a failed early certificate;
7. continue valid arms to 150k, evaluate locally using the exact frozen metric
   environment, re-admit raw weights, and build the concurrent selection;
8. continue only the selected ordered arm and mechanically valid legacy control
   to 300k;
9. run fresh raw readmission and local full evaluation for both, then build and
   revalidate the terminal paired verdict.

No command in the readiness tool launches training, and no development result
opens the CIFAR-10 test split. A single matched seed supports only the phrase
"margin-separated on the recorded fixed-seed development scores." Replication,
test-set evidence, encoder-independent competitiveness, and broad image-model
claims remain outside this screen.

## 4. Final local verification

The 2026-08-08 checkpoint was verified after the code and recovery repairs.
The full suite ran before the final protocol-only restore-order clarification;
a focused 55-test source-manifest, artifact-integrity, preflight, recovery, and
durable-mirror suite then passed against that exact final document:

- all **170** CAP/CAP2 tests passed, including adversarial artifact mutation,
  incomplete recovery commits, stale-local/newer-durable recovery, filtered
  restore after an interrupted future step, genuine disk resume followed by
  the next optimizer/EMA update, exact inference-corner probes, promotion and
  legacy-control selection, retained-feature score recomputation, and terminal
  300k verdict revalidation and idempotent post-crash finalization;
- Ruff lint and format checks passed for `stage_cap/` and `stage_cap2/`;
- Python byte compilation, every CAP2 CLI `--help` path, and `git diff --check`
  passed;
- an AST closure test now proves that every local Python module reached through
  a relative import from the source boundary is itself included in the source
  manifest. The fixed-grid preview helper no longer imports the much larger,
  previously unhashed historical evaluation module;
- the standard-metric revalidator decodes and hashes the exact sequential PNG
  population, strict-loads the full generated and reference clean-Inception
  feature archives, checks their source-population bindings, and recomputes
  CleanFID and CleanKID. It deliberately does not re-extract features from PNG
  bytes; this remaining distinction is recorded in every revalidation result;
- metric package versions, device/numerical provenance, image quantization,
  deterministic setting, metric batch, workers, KID feature population, and
  KID seed are matched before comparisons are admitted.

The EMF implementation was also checked against the numbered direct-`x`
equation under this repository's reversed clock: the future state advances by
the stopped boundary field, both time coordinates decrease by `delta`, the
correction is `(t-r-delta)_+ * t / r`, only the divisor is numerically clamped,
and the loss carries the corresponding `1/t^2` weight. The production
admission matrix remains the decisive numerical check on the actual rented GPU
and checkpoint; a source inspection is not a substitute for that gate.

## 5. Honest readiness boundary

The previous budgeted run's **measurement and execution ambiguities** are now
either repaired or made fail-closed:

- finite-difference/JVP and gradient fidelity must pass on the actual hardware;
- sampler support and the exact inference endpoint are measured separately;
- raw, EMA, optimizer, RNG, counters, and continuation authority survive a
  verified disk/mirror recovery;
- the full worst-case successive-halving schedule is priced and its immutable
  storage footprint capacity-checked before training;
- standard evidence can no longer be reduced to an editable JSON score;
- 300k is judged only as a paired ordered-versus-concurrent-legacy result with
  historical and 150k retention checks.

The prior run's **scientific model defects** are not assumed fixed. In
particular, the extreme patch-head high-frequency output, fragile refiner
cancellation, late optimization noise, low recall, and poor semantic image
quality may recur. CAP2 measures those trajectories at fixed checkpoints,
preserves raw/EMA evidence, and stops promotion when they fail; it does not
claim that changing the numerical difference and time sampler necessarily
repairs the decoder. If no ordered arm clears the matched 150k comparison, the
correct result is `NO_GO` and the next intervention is a decoder or stability
factor—not reinterpretation of auxiliary metrics.

One narrower evidence boundary remains explicit: the 2,048-sample repository
precision/recall and memorization diagnostics are source-bound and used only as
absolute collapse vetoes, but their intermediate torchvision-feature arrays
are not independently replayed from retained arrays by the final command. They
cannot select or rescue an arm. The primary 50k CleanFID/KID populations are
retained and recomputed. This is acceptable for the developmental screen, but
a later confirmatory protocol should retain/replay those auxiliary leaves too.

Two operational controls remain outside Python's ability to prove and are
mandatory before rental: the durable root must truly survive instance deletion,
and the provider account/project must have a hard spend or runtime cap. The
in-process wall stop is checked immediately after a durable recovery and has a
declared one-recovery-interval detection envelope; it is not a provider billing
guarantee.
