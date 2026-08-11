# CAP-EMF-2 -> ASFD paid-run release audit

**Audit date:** 2026-08-10<br>
**Canonical executable run card:**
[`EncoderIndependentCAPEMF2ASFDProtocol.md`](EncoderIndependentCAPEMF2ASFDProtocol.md)<br>
**Campaign:** one ordered-uniform model, not a multi-arm screen

## Release verdict

The pipeline is a **conditional release candidate**. There is no remaining
known code or protocol change to make before provider setup. The next action is
to rent one production GPU and execute the canonical run card from source
freeze through the no-training admission gates. Paid training starts only if
those gates return `GO`.

This is a high-risk, carefully gated proof of concept. Literature supports its
ingredients, not this exact combination. A `GO` means that the measured
hardware, storage, budget, foundation mechanism and feature geometry satisfy
the prospectively frozen checks; it is not a prediction that image quality will
improve.

## The one-model experiment

```text
one source commit
  -> regenerate source-bound evidence
  -> production GPU admission + 2,000-update full-loop benchmark
  -> initialize one ordered-uniform direct-x MeanFlow model
  -> train that same model to 50k and pause
  -> raw numerical readmission of the actual trained weights
  -> resume exact optimizer/EMA/RNG state to 750k
  -> evaluate fixed 650k and 750k EMA checkpoints
  -> require quantitative, diversity, memorization and uncurated-grid gates
  -> freeze the 750k EMA as the self-feature teacher
  -> qualify its target-only feature geometry and build bound feature banks
  -> calibrate ASFD coefficients on independent draws
  -> execute a 500-update exact prefix of the 50k correction schedule
  -> resume the same online/Adam/EMA foundation state from 750k to 800k
  -> fixed 50k-sample evaluation + noncollapse + visual veto report
```

The 50k pause does **not** spend budget on another candidate. The recovery was
planned for 750k before update one, and Phase B reloads the same model,
optimizer, EMA and random streams. Its purpose is to avoid paying for 700k more
updates if the newly trained raw weights already fail the numerical mechanism.

At inference, the final sampler is still one generator call. The frozen
teacher, cached features, raw/kernel fields and spectral bank are training-only.

## Why the mechanism is reasonable

1. **One-step foundation.** MeanFlow derives a learned average-velocity field
   whose endpoint sampler uses one network evaluation, and demonstrates
   from-scratch one-step generation. Our direct-x finite-difference variant is
   in that family, but is not the published recipe. See
   [MeanFlow](https://arxiv.org/html/2505.13447) and
   [Euler Mean Flows](https://arxiv.org/abs/2602.02571).
2. **Frozen internal geometry.** Teacher-Feature Drifting provides the closest
   precedent for multiple frozen hidden levels, moderate feature noise and
   multiple drift scales. Self-perceptual diffusion independently supports
   using a converged model's frozen internal features rather than a moving
   perceptual network. See
   [Teacher-Feature Drifting](https://arxiv.org/html/2605.07327) and
   [Diffusion Model with Perceptual Loss](https://arxiv.org/html/2401.00110v7).
3. **Raw distribution anchor.** Characteristic-kernel/MMD theory supports the
   population motivation for a source-law anchor; multi-radius Laplace fields
   reduce dependence on one bandwidth. See
   [Gretton et al.](https://www.jmlr.org/papers/v13/gretton12a.html) and
   [Sriperumbudur et al.](https://www.jmlr.org/papers/v11/sriperumbudur10a.html).
   The implemented finite random-feature bank is only an approximation and is
   not claimed to be globally measure determining.
4. **Multi-loss stabilization.** The three auxiliary terms are explicit
   nonnegative scalar losses before capping. Their independent live norm caps
   limit destructive updates. Because those multipliers depend on the current
   batch and primary gradient, the realized update is not generally the
   gradient of one fixed objective. It is an empirical stabilization heuristic,
   not a corollary of the formal identifiability theorem.

## Exact extrapolations and hard aborts

The two largest scientific extrapolations are explicit:

- the width-384, batch-64, ordered-uniform, direct-x U-ViT foundation differs
  from published MeanFlow CIFAR settings;
- the frozen teacher is the same from-scratch one-step model, whereas the
  closest teacher-feature evidence used a strong pretrained multi-step teacher.

Accordingly, ASFD is forbidden unless the 750k foundation is quantitatively
capable, visibly recognizable/diverse on a fixed uncurated grid, and its frozen
features pass sensitivity, rank, spread, benign-pair, scramble,
inter-level-nonredundancy and raw-vs-feature-gradient gates. Qualification
rejects unhealthy geometry; it does not prove semantic correctness.

## Previous-failure ledger and repairs

| Previous failure mode | Release safeguard |
|---|---|
| Proxy metric improved while target geometry looked worse | Both fixed uncurated grids are retained; final human review is veto-only; precision, recall, duplicate and exact-copy checks cannot be rescued by FID/KID. |
| Tiny metric noise was called a win | `PROMISING_IMPROVEMENT` requires at least one primary change beyond the frozen real/real discrepancy margin and noninferiority on both FID and KID. |
| A weak foundation was given a sophisticated correction | 50k raw readmission, 650k/750k quantitative stability, fixed-grid capability review and target-only feature qualification all fail closed before ASFD. |
| Feature cache silently referred to different pixels | Bank metadata hashes the exact CIFAR train tensor and labels; every correction construction rehashes and compares them. |
| Calibration accidentally previewed production draws | Calibration has a separate RNG namespace. The 500-update smoke and production share the production namespace so the smoke is an exact replayable prefix. |
| Short smoke compressed a long curriculum | Smoke uses the full 50k ASFD progress horizon while stopping training after 500 updates. |
| Source was editable after foundation | CAP2's source manifest includes the entire ASFD continuation/evaluation stack before update one. |
| Crash/restart could fork or strand the paid model | Numeric immutable recovery commits, exact optimizer/EMA/RNG extension state, terminal artifact reopening, attested off-instance storage, live probes and idempotent restart. |
| Runtime estimate was transferred across hardware | Foundation benchmark, ASFD preflight and ASFD continuation require the same recorded GPU/software identity; a measured hard wall stops only after a mirrored recovery. |
| Evaluation used only FID | CleanFID and unbiased-sample KID are primary, with separate precision/recall and pixel-copy/duplicate diagnostics. FID finite-sample bias remains a claim limitation; see [Chong and Forsyth](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html). |

## Operational release boundary

The source tree cannot prove two provider facts: that the selected volume truly
survives instance deletion, and that the provider-side billing cap works. The
operator must provision and probe the exact workspace and mirror namespaces,
set the provider cap, and keep them on the same measured durable volume.

The historical 650k admission checkpoint is not stored in Git. Upload it to the
path in the run card and require SHA-256
`b55b2a62bfc44e546f347cb348b8e7e63aef6686d8a97527f6d4d232a5023f49`
before any evidence or GPU gate consumes it.

The declared USD 75 ceiling is a cap, not a target. A genuine benchmark may
reject it before update one. If the budget is raised, change both the declared
ceiling and provider hard cap before training; do not weaken the model or
silently reduce evaluation. The ASFD reserve cannot be enlarged after the
foundation because the foundation is bound to its original preflight.

## Claim boundary

If successful, the honest claim is:

> In one all-class CIFAR-10 development run, a one-call generator trained with
> no external or separately trained encoder produced a final 800k package that
> improved its prospectively frozen 750k foundation beyond the recorded metric
> noise margin without triggering the declared coverage, memorization or visual
> collapse vetoes.

Do not call the method representation-independent: it deliberately depends on
its own learned hidden geometry during training. Do not attribute the 750k to
800k change causally to ASFD without a matched raw-only continuation. Do not
claim replication, sealed-test generalization, ImageNet parity, or a theorem
about the dynamically capped finite-batch optimizer.

## Verification record

- End-to-end synthetic ASFD correction graph: generator + B1 + raw Laplace +
  frozen self-feature Laplace + generated-input Jacobian + gradient combination
  + state replay.
- AST source-manifest closure audits for CAP2 and ASFD.
- Frozen-tree CAP/CAP2/ASFD test suite: **244/244 passed** on CPython 3.11.15
  (`159.00s`, 10 August 2026).
- Ruff lint and format: green over all three stages (74 files).
- Python bytecode compilation: green over all three stages.
- Every executable named in the run card: **15/15** CLI `--help` checks green.
- Every PowerShell block in the run card: **15/15** parsed without an error.
- Git whitespace check: green.

No paid training was launched during verification. The immutable source commit
is the scoped release commit that contains this record; its hash is reported in
the handoff after Git creates it.
