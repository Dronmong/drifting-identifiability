# Encoder-Independent Kernel Drifting

## Research analysis and implementation roadmap

**Status:** executed through Phase 4. **Phase 0 passed; Phase 1 FAILED; Phase
2A passed; Phase 2B FAILED on CIFAR-10; Phase 3 PASSED; Phase 4 FAILED all
three gates.** Fixed compositional geometry returned three consecutive
negatives, the last in the regime the program spent two phases searching for.
**The geometry thread is closed.** The spectral anchor passed every condition
ever asked of it, though on raw geometry it is worth only ~3.5%. Reform R11
(teacher variance match) was confirmed on fresh seeds in Phase 3 (3.1×, 18/18
wins) and then **narrowed by Phase 4**: the contraction reproduces in this
repository's own audited Algorithm-2 harness with a clean dimension law, but
R11 fails at the paper's declared temperature grid, Algorithm 2's self-mask
already mitigates the problem, and two proposed mechanisms were refuted.  
**Last reviewed:** 2026-07-26  
**Working name:** Spectrally Anchored Compositional Kernel-Gradient Drifting
(`SACKGD`)

### Execution record

| Document | Contents |
|---|---|
| `numerics/encoder_independent_drifting/` | the implementation (plan section 8 layout) |
| `EncoderIndependentPhase0Results.md` | Phase-0 exit gate: **PASSED** all four conditions |
| `EncoderIndependentPhase1Protocol.md` | frozen pre-outcome Phase-1 design |
| `EncoderIndependentPhase1Results.md` | Phase-1 exit gate: **FAILED** G1.1 and G1.5 |
| `EncoderIndependentAnchorGradientDiagnosis.md` | why the anchor loss does not descend |
| `EncoderIndependentPhase1Diagnosis.md` | deep failure investigation (D1–D8) and seven reforms |
| `EncoderIndependentReformedScreenProtocol.md` | frozen successor design; all seven reforms implemented and unit-tested |
| `EncoderIndependentSecondPassAudit.md` | second audit (A1–A5), reforms R8–R10, and the CIFAR-10 Phase-2 design |
| `EncoderIndependentPhase2Protocol.md` | frozen pre-outcome CIFAR-10 design |
| `EncoderIndependentPhase2Results.md` | Phase-2A **PASSED**, Phase-2B **FAILED**; geometry thread closed |
| `EncoderIndependentPhase2Diagnosis.md` | the skyline gap was a variance-collapse defect (R11); geometry verdict reconfirmed |
| `EncoderIndependentPhase3Protocol.md` | frozen pre-outcome corrected-baseline confirmation |
| `EncoderIndependentPhase3Results.md` | Phase 3 **PASSED** all seven conditions; R11 confirmed on fresh seeds |
| `EncoderIndependentPhase4Design.md` | measured design investigation (G1, G2) and the Phase-4 proposal |
| `EncoderIndependentPhase4Protocol.md` | frozen pre-outcome generality and mechanism study |
| `EncoderIndependentPhase4Results.md` | Phase 4 **FAILED** all three gates; R11 narrowed, mechanism unknown |
| `EncoderIndependentPhase4Diagnosis.md` | corrects the "paper's operating point" claim; reforms R15–R19 |
| `EncoderIndependentPhase5Protocol.md` | frozen pre-outcome design for R15–R19 |
| `EncoderIndependentPhase5Results.md` | Phase 5 **FAILED** 4/5; R16 withdrawn, third mechanism refuted, R11 replicated again |
| `EncoderIndependentRootCauseAnalysis.md` | fourth mechanism refuted; **corrects the "variance collapse" framing**; reforms R20–R23 |
| `EncoderIndependentReformsR20R23Results.md` | R20–R23 implemented; **the controlling variable is the second moment**, not effective dimension |
| `EncoderIndependentOpenQuestions.md` | **`step_eta` is inert under Adam**; explains five failed reforms; Phase-6 design |
| `EncoderIndependentPhase6Protocol.md` | frozen pre-outcome; declared in advance that 6A could retire R11 |
| `EncoderIndependentPhase6Results.md` | **R11 survives**; the deficit is not a learning-rate artifact and **not a port artifact** — particles have it too |
| `EncoderIndependentPhase6Followup.md` | **the deficit is a BANDWIDTH artifact**; ESS-0.5 is badly suboptimal; free particles at τ=1 beat the corrected generator; Phase-7 design |
| `EncoderIndependentMechanismSynthesis.md` | H9: drifting is an MMD-flow-type system inheriting known mode-collapse; **scoped down to the particle flow by Phase 7** |
| `EncoderIndependentPhase7Protocol.md` | frozen pre-outcome; declared in advance that 7A could retire R11 |
| `EncoderIndependentPhase7Results.md` | **TWO deficits, not one**: the field's is bandwidth-controlled, the generator's is bandwidth-INDEPENDENT; R11 survives again; bandwidth optimum located at ess≈0.9 |
| `EncoderIndependentGeneratorContraction.md` | the generator sits at its objective's **optimum** (free dilation parameter declines); least-squares shrinkage proposed — **refuted for the recipe by Phase 8** |
| `EncoderIndependentPhase8Protocol.md` | frozen pre-outcome; declared in advance that 8A could retire R11 |
| `EncoderIndependentPhase8Results.md` | **capacity is not it either**: 36× parameters fits the teacher 2.3× better and moves the cloud not at all; R11 survives a third time; convergence confirmed at 6000 steps |
| `EncoderIndependentPhase8Followup.md` | fluctuation and target noise both refuted; **spectral confinement is a real partial contributor**; Phase-1's noise refutation invalidated; proposes the **linear-generator solvable case** |
| `EncoderIndependentPhase9Protocol.md` | frozen pre-outcome; declared P-lin (no deficit for linear+Gaussian) and the 2×2 ladder |
| `EncoderIndependentPhase9Results.md` | **P-lin refuted — the deficit survives down to `f(z)=Az+b`**; architecture irrelevant (linear 0.502 vs conv 0.472 at CIFAR); the self-term is a reduction artifact, not the mechanism |
| `EncoderIndependentDeficitMechanism.md` | **★ THE MECHANISM: the deficit is a SHAPE phenomenon** — a clumped, low-tail cloud balances the field's repulsion at a smaller radius; explains twelve standing observations including seven unexplained negatives |
| `EncoderIndependentPhase10Protocol.md` | frozen pre-outcome; separates spectrum from packing, and declares the shape-supersession gate |
| `EncoderIndependentPhase10Results.md` | **★ THE LAW: the spectral tail sets the equilibrium radius** (9× over packing), predicting 6/7 arms to <0.11 and free particles out-of-sample to 0.015 — **R11 is the sole outlier and is an override, not a shape fix** |
| `EncoderIndependentTailDestruction.md` | **★ the tail is DESTROYED, not absent** — the field supplies 16× the cloud's tail, the generator starts at 0.221 and loses it 29× in 100 steps; implies whitening the regression metric |
| `EncoderIndependentPhase11Protocol.md` | frozen pre-outcome; whitened metric, tail trace as primary readout |
| `EncoderIndependentPhase11Results.md` | **whitening refuted (137× conditioning, no effect)** — and the discriminating test localizes the cause to the **self-referential moving teacher**, not the metric/optimizer/architecture |
| `EncoderIndependentProgramAudit.md` | **★ formal audit of all 11 phases** — 5 gates verified from artifacts, 14 refuted hypotheses, 5 instrument defects, 8 self-corrections; **rollout and averaging both fail**, so the cause is *self-reference itself*; proposes transport-based particle amortization |
| `EncoderIndependentPhase12Protocol.md` | frozen pre-outcome (one amendment, recorded, after smoke and before any run) |
| `EncoderIndependentPhase12Results.md` | **★ the external target works — tail ×79, second moment 0.331→0.922, ED² 6.2× better** — but **the assignment fails**: greedy collides, barycentric contracts. Gate misses by 1.5× at 3.6× cost |
| `EncoderIndependentPhase12Investigation.md` | **★ the assignment was never the problem** — an exact Hungarian assignment preserves the target perfectly and still fails on fresh latents (tail 0.343 fixed vs **0.0009** fresh). The requirement is **correspondence stability**. Next: sweep the pair bank |
| `EncoderIndependentPhase13Protocol.md` | frozen pre-outcome; pair-bank sweep with a separately declared trend criterion |
| `EncoderIndependentPhase13Results.md` | **★ the amortizer works and beats the particle cloud it fits (0.70–0.73×)** — ED² 0.962 → **0.139**, second moment in band, tail restored 27× — **but loses to R11 (0.0955) by 1.17× at 14.4× cost** |

Findings that revise this document:

1. **Section 5.3's novelty claim was downgraded** — fixed compositional
   features for generation are already established (section 5.1b).
2. **Section 6.3's central prediction is falsified.** Kernel-gradient
   movement is *worse* than standard displacement under a structured kernel
   (1.86×, interval excluding 1), while being mildly better under the raw
   kernel. The plan's own required ablation caught this.
3. **Section 6.2's fixed geometry is not competitive** with raw pixel
   drifting at 16×16 — 4.5× worse at 49× the kernel cost — but the testbed
   also fails to reward a learned encoder, so the result is scoped to
   "geometry is unnecessary here", not "fixed geometry cannot work".
4. **Section 6.1's anchor is vindicated as a correctness mechanism and
   indicted as an optimizer.** It improves every geometry arm by 30–45% and
   detects 6/6 collisions from an independent audit bank, but its loss barely
   descends: the gradient is proportional to ‖ω‖, so the high band dominates
   the gradient magnitude while carrying pure noise. A coarse-to-fine
   frequency schedule is the indicated repair, and is untested.
5. **The plan's declared sum combination beat the product alternative**, and a
   bare median-heuristic bandwidth left every kernel flat; the repair is a
   target-only ESS-targeted calibration.
6. **The Phase-1 screen itself was under-specified** (deep diagnosis). It ran
   at a budget where a well-posed oracle objective also fails, so its
   mechanism conclusions are provisional. Section 9 needs a **skyline/oracle
   arm** and an oracle-derived budget before any successor screen is run.
7. **Section 6.5's exact-zero argument is not instantiated by the
   implementation.** Under RMS field normalization the geometry loss is
   identically η², so `L_total = 0` is unreachable. The population argument
   is untouched; the code must be changed to realize it.
8. **Section 6.3's kernel-gradient rule has a named failure mode.** Through a
   non-injective feature map it behaves as an adversarial attack on that map,
   driving the output off the data manifold. Standard displacement is immune
   because it can only interpolate real data. The repair — projecting the
   kernel-gradient direction onto `span{Y_j − x}` — is implemented as
   `projected_kernel_gradient` and removes the adversarial channel, but
   **measurement shows it does not rescue the wavelet family**: its zero-set
   residual is 2.21 against 2.39 unprojected, both plateauing. A constrained
   direction does not repair a wrong objective.
9. **Section 9's phase structure needs a zero-set condition.** All four
   original Phase-0 conditions passed and Phase 1 still failed. The added
   **G0.5** measures whether minimizing a candidate field actually reaches
   the `q = p` floor, and it reproduces the entire Phase-1 ranking before any
   generator is trained. Any successor screen must refuse arms whose field
   plateaus.
10. **Section 2.2's premise is correct, but only on real images.** No
    synthetic testbed reachable here makes pixel geometry fail: across
    resolutions 16–32 and translations to ±4, pixel content-grouping never
    becomes misleading, which is why Phases 0–1 could not reward fixed
    geometry. On CIFAR-10 the premise holds — pixel k-NN content accuracy
    .267 against chance .100 — and **fixed wavelet/scattering geometry
    recovers 36–50% of the gap with no training**, matching a small
    supervised control while a learned autoencoder does not. Section 9's
    Phase-2 target should therefore be CIFAR-10 at 16×16, which is also
    admissible at 300 steps.
11. **Section 9 needs Phase-2 to be four arms, not nine.** G0.5 rejects the
    kernel-gradient rule for structured kernels in seconds, so the screen is
    raw-pixel against wavelet and scattering under standard displacement,
    plus the anchor and a skyline.
12. **Section 1's executive decision is answered, negatively.** Phase 2 ran
    that design on CIFAR-16 with every precondition satisfied — solvable
    testbed, live geometry question, all fields reaching the zero-set floor,
    objectives descending 97–98% — and fixed compositional geometry was
    **37–44% worse** than a raw pixel kernel at 49–81× the kernel cost, on
    every seed. The measured mechanism is that a better neighbour ranking
    produces *density-seeking* rather than distribution matching: the
    structured arms sit closer to individual real images while covering the
    distribution worse.
13. **The skyline gap was an implementation defect, not a property of
    drifting.** The same field on free particles scores 1.89 against the
    skyline's 1.84; the generator had collapsed to effective dimension 2.34
    against the data's 8.32, because the stop-gradient teacher contracts the
    cloud toward local barycentres and least-squares regression contracts it
    again. **Reform R11** — rescaling the teacher about its own mean to
    carry the real batch's second moment — takes the generator to 1.92,
    parity with the skyline, and restores effective dimension to ~6.7.
    Section 6.5 should state that the stop-gradient regression is
    variance-contracting and must be corrected.
14. **Section 6.3 quotes the wrong field.** What it calls "the paper's
    standard normalized displacement field" is the row-normalized SNIS mean
    shift, which this repository labels DIAGNOSTIC ONLY; the paper's
    Algorithm 2 normalizes the affinity matrix along both axes. **Reform
    R12** implements the real one as `direction_mode="paper"`, verified
    against `lowdim_drift.drift_paper`. It is worth ~13% on raw geometry and
    changes no ranking, but the plan's text must be corrected.
15. **The corrected baseline is confirmed.** Phase 3 tested R11 on three
    fresh seeds, two resolutions and three budgets: paired ratio 0.319
    [0.292, 0.351] with 18/18 wins, effective dimension restored from 0.264
    to 0.867 of the data's, and the generator-versus-free-particle gap closed
    from 3.44 to 1.09. Corrected encoder-free drifting reaches parity with a
    sliced-Wasserstein skyline (1.140 [0.999, 1.307]) — parity, not
    superiority. R12 turns out not to be needed once R11 is applied (C2/C1 =
    1.014, a tie), though it remains the correct implementation of the
    paper's estimator.
16. **Phase 4 narrowed R11 and refuted its mechanism.** The contraction
    reproduces in `lowdim_drift.py` (imported unmodified) with a tight
    dimension law — effective-dimension ratio .998 at d = 2 falling to .096
    at d = 64, crossover near d = 4 below which the correction is
    unnecessary and mildly harmful, above which it is worth 7–14× on ED².
    But R11 **fails at the paper's declared temperatures** (τ = .02/.05/.2 →
    ratios .86/1.12/.47, with effective dimension *not* restored at .05 or
    .2), and Algorithm 2's **self-mask is itself protective** (uncorrected
    effective dimension .521 with it against .303 without). Both proposed
    mechanisms — minibatch-noise regression attenuation, and anisotropic
    contraction of the teacher map — were refuted by direct measurement.
18. **Phase 4's temperature claim was wrong, and the mechanism is now
    partly understood** (Phase-4 diagnosis). The paper's τ values applied to
    raw 768-dimensional pixels *collapse the kernel* — 93.8% of rows dead at
    τ = 0.02, median affinity 4e-20 — so R11 was being asked to rescale a
    teacher built from no information. The paper's grid is calibrated for
    normalized encoder features; **the paper's real operating point remains
    untested.** Both earlier mechanism refutations probed the teacher *at the
    target law*, where the field is zero by construction; along the
    trajectory the teacher loses **2–8% of effective dimension per
    application** at constant variance, worst mid-way, which compounds. Free
    particles escape it by taking a decaying fractional step where the
    regression applies the map at full strength.
19. **That mechanism was refuted too, and so was the reform built on it**
    (Phase 5). Measured on the *real* trajectory the teacher's dimension
    ratio is **0.997** — no contraction — against the 0.92–0.98 a synthetic
    interpolation had suggested; those clouds do not occur in training. The
    decaying-step reform R16 is slightly **worse** than a constant step
    (1.097, 1/9 wins) and leaves the collapse untouched. Latent dimension
    (R17) also changes nothing across 32/64/128.
    **Three mechanisms have now been proposed and refuted.** The
    phenomenology that survives: the generator starts at effective dimension
    ≈13.9 and ends at ≈2.2, *no single step contracts*, and R11 moves the
    endpoint to ≈5.9 without changing the per-step ratio — so the loss is a
    property of where the iteration converges, not of any step in it. R11
    should be described as an empirical correction of **unknown mechanism**;
    it has now replicated three times (Phase 3: 18/18; Phase 5: 9/9).
20. **The "variance collapse" framing carried since Phase 2 is wrong, and a
    fourth mechanism is refuted** (root-cause analysis). The collapse is a
    *fast transition to an attractor* — effective dimension falls 34.7 → 1.8
    within 70 of 600 steps and is then flat — which is why every per-step
    compounding hypothesis failed. The generator closes only 13% of the gap
    to its own teacher per step, and the realized output change is
    uncontrolled (42× the requested displacement at initialization). But
    fixing that (32 inner steps, gap closure 68%) **overshoots**: effective
    dimension reaches 1.4–3.3× the data's, coverage falls from .97 to
    .23–.66, and the score does not improve. **Effective dimension has an
    optimum at ≈1, not a monotone benefit**, so R11 works by *matching* the
    data's second moment, not by raising dimension. Sections describing it as
    repairing a collapse must be restated, and any gate on effective
    dimension needs an upper bound (reform R20) — Phase-3 C.5 and Phase-5
    G5.2 would currently pass a configuration now known to be bad.
21. **The controlling variable is the attractor's second moment** (R20–R23
    implementation). Run in the repository's audited low-dimensional harness,
    effective dimension is 0.43–0.53 in *every* configuration — best and
    worst alike — while the second-moment ratio tracks ED² exactly: 0.39–0.42
    gives ED² 0.10–0.18, and 0.84–0.89 gives 0.016–0.026. So effective
    dimension was a proxy that happened to correlate at CIFAR-16 and does not
    generalize; **reports and gates should use the second-moment ratio**,
    which is precisely what R11 is defined to control. R21 (capping the
    realized output step) has **no measurable effect** — the cap never binds
    below 32 dimensions, so the 42× overshoot seen at CIFAR initialization is
    high-dimensional or architecture-specific and does not drive the
    attractor. R22 (steps per teacher) does not help and mildly hurts,
    confirming the CIFAR result. R11 remains the only intervention that
    works, in a third independent setting; it reaches 0.84–0.89 rather than
    1.0, and closing that undershoot is the natural next question.
22. **Section 6.5's `step_eta` is inert under Adam, and that explains five
    failed reforms.** The stop-gradient loss has gradient `−2ηV/n`, so η
    enters only as a constant multiplier — and Adam is invariant to constant
    gradient rescaling (up to its `ε`). Measured: η = 0.05/0.5/5.0 give
    results identical to every reported digit under Adam (0.6404 / 0.0301
    throughout) and a large effect under SGD (ED² 0.219 / 0.036 / 0.121); at
    CIFAR-16 a 100× change in η moves the output by a relative 2.8e-5 under
    Adam against 1.34 under SGD. This retroactively explains R16 (η
    schedule), the inertness of RMS normalization, R21 (step cap), the
    pinned η² loss, and why every magnitude-based mechanism hypothesis
    failed: **under Adam only interventions that change the gradient's
    *direction* can matter, and R11 is the only one tested that does.** The
    real step control is the optimizer's learning rate, fixed at 2e-3 since
    Phase 1 and never swept — so Phase 6A must check whether the
    second-moment deficit is simply a mis-set learning rate, which would
    supersede R11. Note this concerns the *neural-generator port*; the
    paper's particle algorithm has no optimizer and is untouched by it.
23. **Phase 6 answered both of those and refuted a hypothesis of mine.**
    (a) The deficit is *not* a learning-rate artifact: across three
    optimizers × four learning rates × 3 seeds, **0 of 12 uncorrected cells**
    reach the `[0.7, 1.3]` second-moment band (range 0.148–0.369), while R11
    reaches 0.853–1.195. The best uncorrected cell misses the supersession
    ceiling by 9×, so **R11 survives its sharpest test**. (b) The deficit is
    *not* a neural-port artifact either — under a matched comparison the
    **particle algorithm carries it too** (0.594 constant step, 0.627
    decaying, no seed overlapping the band), roughly half the generator's
    0.269. It belongs to the drifting dynamics, not the port. (c) Richer
    direction changes do **not** beat the single scalar (per-coordinate
    1.095, eigendirection 1.130, intervals straddling 1), and a declared
    1.2× over-correction is **decisively worse** (1.975, interval excluding
    1) — so R11's undershoot is not a gain deficit to be closed. (d) A new
    constraint: **R11 diverges under plain SGD at lr ≥ 2e-3** in every cell
    tested, because a corrected teacher presents a large gradient that only
    a magnitude-normalizing optimizer can absorb — the same blindness that
    makes Adam ignore `step_eta` is what lets it use R11.
24. **The deficit is a bandwidth artifact, and the operating point has been
    wrong since Phase 2.** 6C's 0.594 is a genuine attractor (0.600 at 2400
    steps, window growth −0.003), but it is a property of the *kernel
    bandwidth*, not of drifting. Across every R15-admissible bandwidth at
    CIFAR-16 the fixed-point second moment and the quality are **monotone in
    the kernel's realized neighbour count, with no crossings**: ESS 0.146 →
    2nd moment 0.304, ED² 1.669; ESS 0.903 (**the program's ESS-0.5
    setting**) → 0.600, 0.349; ESS 0.978 (τ=1.0) → **0.856, ED² 0.0708**.
    That is a **4.9× quality gain** from the bandwidth alone, and it is not
    memorization (nearest-real ratio 0.944 against the incumbent's 0.739,
    every component ratio near 1). **Free particles at τ=1.0 beat the
    R11-corrected generator 0.071 vs 0.160.** A second axis compounds it:
    the field cloud size, where 64 (the generator's training batch) costs an
    order of magnitude against 512 in every arm. Two more mechanism
    hypotheses died — the analytic blur factor (predicts contraction
    everywhere; small τ *expands*, 8.9× in low dimension) and balancing
    depth (converged Sinkhorn detonates the cloud, 1.27 → 248, which
    *explains* the paper's `sqrt(row⊙col)` as a deliberate half-strength
    density correction that keeps attraction and repulsion in balance).
    **R15 earned its keep**: τ=0.02 posted the best second moment in the
    sweep (1.092, in band, converged) while being a fully collapsed kernel
    (100% dead rows, affinity 1.9e-23). **Phase 7A must now ask of the
    kernel what 6A asked of the optimizer** — R11 has never been tested
    against a properly set bandwidth, and free particles already clear both
    gate conditions there.
25. **Phase 7: there are TWO deficits, and only one of them is the
    bandwidth's.** The follow-up's hypothesis (item 24) was right about the
    particle flow and **wrong about the generator**. Across four bandwidths
    (realized ESS 0.82–0.99) × three field-cloud sizes × 3 seeds, the
    uncorrected generator's second moment spans **0.101–0.443 in all 36
    runs** and **0 of 12 cells** reach the band; with R11, 0.874–1.159. The
    best uncorrected cell misses the supersession ceiling by 5.5×, so **R11
    survives a second sharp test**. Bandwidth moves the *particle* fixed
    point 0.23 → 0.99 and barely moves the *generator's* (0.34 → 0.38), so
    the field's deficit and the generator's are different objects. 7B: the
    corrected generator is flat at ED² 0.16–0.18 across the whole bandwidth
    axis while particles swing 3.3× through it — at ess=0.5 the generator
    wins 0.65×, at ess=0.9 particles win 2.24×, so the six-phase impression
    that amortization beat particles was an artifact of a badly set kernel.
    7C locates an **interior optimum**: quality is U-shaped, best at
    ess=0.9 (ED² 0.0729, second moment 0.989 — the moment crosses 1 almost
    exactly at the quality optimum), 3.2× worse at the incumbent ess=0.5 and
    3.9× worse at τ=8. **Free particles at ess=0.9 beat every generator
    configuration measured, corrected or not.** The `_rule_7c` monotonicity
    check reported "no rule" (Spearman +0.733) — the wrong test for a
    unimodal relationship; the data support a *candidate* target-only
    calibration at ESS ≈ 0.93, unvalidated because it was read off the same
    sweep. R27 (`field_cloud`) landed: real effect, insufficient alone.
    **The open question is now sharp — why does the *generator* contract?**
    Not the optimizer (6A), not the bandwidth or cloud (7A), not the field's
    fixed point (7B/7C), and five direct hypotheses refuted in Phases 3–5.
26. **The generator's contraction is least-squares shrinkage — its
    objective's optimum, not a failure to reach it.** Measured at the
    Phase-7C bandwidth optimum, 3 seeds. (a) **A free dilation parameter
    declines to grow**: adding one learnable output gain leaves the second
    moment unmoved (0.462 → 0.476) and the gain converges to **0.829**, i.e.
    *downward*. (b) **At the converged plain generator the field's mean
    radial component is +0.0004** — indistinguishable from zero, exactly as
    the fixed-point condition `E_z[(∂f/∂θ)ᵀV] = 0` requires once one notes
    that dilation is in the tangent space (the head is a plain conv). The
    generator is not straining outward against a stuck model; it is at
    equilibrium. (c) **R11 holds the cloud past that equilibrium against the
    field's restoring force** — at R11's operating point the radial
    component is −0.126 and only 38% of samples are pushed outward.
    (d) **The mechanism**: fitting the converged particle cloud by least
    squares while sweeping parameters/values gives second moment 0.933 →
    0.902 → 0.733 → 0.602 as the ratio falls 2.23 → 1.12 → 0.56 → 0.28,
    with the particle control flat at 0.94–0.99. **Point-target regression
    costs nothing when overparameterized** (0.933 vs its control's 0.939,
    ED² 0.2919 vs 0.2989) and shrinks smoothly below p/v ≈ 1. This
    **corrects Phase 4's "regression costs ~2×"** (entirely the
    underparameterization artifact it flagged) and shows **refuted
    hypothesis 1 was right in kind, tested against the wrong noise** — Phase
    3 measured the *estimation-variance* term (~0.001, correctly small) and
    never measured the *approximation-error* term, which dominates.
    (e) **Latent dimension is not it (H10 refuted)**: 8→512 gives second
    moment 0.400/0.462/0.479/0.440, and R17's negative is confirmed at the
    good bandwidth. (f) R11 restores **15× of the spectral tail** (0.0032 →
    0.0474 against the data's 0.1375), so it repairs scale and only part of
    shape — a partial account of why the corrected generator still loses to
    free particles (0.167 vs 0.075). **Phase 8 must sweep the conv stack's
    WIDTH in the real recipe** — fixed at 64 since Phase 1 by the
    matched-capacity rule, and the axis N4 isolates as the relevant one —
    predicting that R11's advantage shrinks as width grows.
27. **Phase 8: capacity is not it either, and least-squares shrinkage does
    not transfer to the recipe.** Width 32→256 (36× parameters, 36k→1.32M)
    at the good bandwidth: the uncorrected second moment is **0.354 / 0.401 /
    0.349 / 0.331** — flat, if anything falling — with ED² 1.17/1.17/1.08/1.20
    and **0 of 4 widths** in the band; the best plain cell misses the
    supersession ceiling by 6.2×. **R11 survives a third sharp test.** The
    decisive pairing is with 8B: the realized fraction of the teacher rises
    **0.212 → 0.493** with width, so the wide model tracks its teacher
    **2.3× better and produces an identical cloud**. Under-fitting is
    therefore not the cause — the deficit is in **what the teacher asks
    for**, not in the ability to deliver it. This independently reproduces
    Phase 5's inner-steps result by a different route, and it refutes the
    inference from the contraction pass's fixed-cloud N3 to the moving-target
    recipe (a gap that pass flagged as live). **The secondary prediction is
    refuted and my trend flag reported the opposite**: the R11/plain ratio
    "rose" (0.165→0.388) only because R11 got *worse* (0.139→0.296) while
    plain did not move (1.166→1.198), and the width-256 interval
    [0.193, 1.226] contains every other width. **Convergence checked**: at
    6000 steps (10× budget) the uncorrected arm goes 0.433 → 0.472 with
    late-window growth −0.034, so the equilibrium reading stands and 600
    steps understates by ~18%. The deficit is now invariant to optimizer,
    learning rate, η, bandwidth, cloud size, latent dimension, capacity and
    teacher-fitting quality; **the only thing that has ever moved it is
    changing what the teacher asks for.**
28. **Phase-8 follow-up: four more negatives, one partial mechanism, and an
    old refutation invalidated.** (a) Phase 8's width "trend" is statistically
    empty — between-width sd of medians 0.0256 against within-width seed sd
    **0.0855**, Spearman +0.098. Capacity does nothing; read it as flat.
    (b) **Fluctuation refuted**: averaging the particle field over 64
    independent batches moves the equilibrium 1.012 → 1.006 (positives) and
    1.022 → 1.023 (negatives). (c) **The noise budget inverts by three orders
    of magnitude along the trajectory** — noise/signal 0.007 at step 0,
    **4.459** at step 599. Phase 3's refutation of minibatch-noise
    attenuation rested on a ≈0.001 figure that corresponds to
    *initialization*; that refutation must no longer be cited, though the
    hypothesis still fails on direct test. (d) **Target noise refuted**: 16×
    field averaging for the generator raises the 600-step second moment
    0.372 → 0.474, which looks positive until compared with N6's 6000-step
    value of **0.472** — averaging *accelerates convergence* (10× shorter
    run) and does not move the fixed point. (e) **Spectral confinement is
    the first intervention ever to move a particle equilibrium toward the
    generator's**: confining particles to the data's top-k directions gives
    in-subspace second moment 1.012 (unconfined) → 0.917 (k=511) → 0.700
    (k=128) → 0.664 (k=32), but it **saturates** near 0.7 and never reaches
    0.47, so it is a partial contributor, not the mechanism. **Next: stop
    testing interventions and solve the linear-generator case** — `f(z)=Az+b`
    with Gaussian latent and data makes the pushforward exactly Gaussian, the
    mean-shift field closed-form, and the fixed point a matrix equation in
    `AAᵀ`. It is the only remaining route that can produce a derivation
    rather than a twelfth elimination. In parallel, **no configuration has
    ever combined the ESS-0.9 bandwidth, R11 and the anchor** — that package
    is a deliverable needing no new mechanism.
29. **Phase 9: the deficit survives all the way down to `f(z) = Az + b`.**
    The declared prediction P-lin — that a linear generator on Gaussian data
    has no deficit — is **refuted**. The 2×2 ladder (3 seeds, 2000 steps, all
    converged): isotropic Gaussian **0.211** linear / 0.060 MLP; decaying
    spectrum **0.648** / 0.480; CIFAR-16 **0.502** / 0.357, against the conv
    recipe's 0.472. **The architecture is irrelevant** — a linear map at
    CIFAR is within 6% of the full convolutional recipe, which closes the
    question Phase 8 left open when capacity proved inert. The nonlinearity
    consistently *deepens* the deficit (factor 1.3–3.5 in every data law) but
    does not create it. **Two premise errors in my own derivation, both
    recorded**: (a) the closed form `V = c(σ)x` is the *SNIS mean-shift*
    field, not Algorithm 2's bi-softmax, which the repo explicitly labels a
    different object; (b) the **self-term** (`self_mask` defaults to False)
    is a pure inward pull carrying 21.3% of the repulsion and 3.7× the masked
    field in 64-D isotropic geometry — where every pair sits at distance 11.3
    and the self-affinity is 77× a neighbour. **That looked like the
    mechanism and a control refuted it**: at CIFAR-16 the same ratio is 4.5
    and masking changes |V| by 7% (0.670 → 0.672), so self-term dominance is
    an artifact the isotropic reduction *introduced*. Consequences: the
    isotropic cell is not a valid proxy and the ladder must be read from the
    decaying and CIFAR cells; and Phase 4's "self-mask is protective" now has
    both a mechanism and a scope. **The object of study is now a fixed-point
    equation in `AAᵀ` alone** — Gaussian cloud, no network, no optimizer, no
    architecture — solvable numerically to high precision.
30. **★ THE MECHANISM: the deficit is a SHAPE phenomenon.** Two measurements
    close Phase 9's contradiction. (a) **The field is nearly unbiased**:
    along data-shaped clouds its radial component crosses zero at α = 0.844
    (64 positives) and 0.830 (2048), a ~15% inward bias from the self-term —
    stable across a 32× batch change and far too small to explain a
    generator at 0.42. (b) **The generator's cloud is not data-shaped.** At
    *matched second moment*, so every difference is shape: the generator at
    0.417 has radial **+0.0006** (balanced) while a data-shaped cloud at the
    identical scale has **+0.0264** — pushed outward **43× harder**. The
    generator is at a *different* equilibrium, one belonging to its cloud's
    shape. **Why**: Algorithm 2 balances attraction against repulsion from
    the cloud's own neighbours, and repulsion is set by *packing*. The
    generator's cloud is clumped — nearest-neighbour spacing **3.47** against
    the data-shaped comparator's **6.36** at equal variance, achieved by
    concentrating energy in few directions (spectral tail **0.0034** against
    **0.142**, a factor of 42). A clumped cloud reaches the repulsion it
    needs at a small radius. Free particles do the opposite: unconstrained
    points under a repulsive interaction settle into a near-regular packing
    (nn CV **0.023** against an i.i.d. sample's 0.245) and spread across
    *more* directions than the data (tail 0.415), landing at 0.995. **A
    smooth pushforward of a low-dimensional Gaussian cannot pack regularly in
    768 dimensions** — which is why the deficit survives to the linear case
    and why nine axes are inert: none of them changes packing. It predicts
    the sign of both things that ever moved the deficit (R11 inflates spacing
    → helps, raising the tail 15×; spectral confinement reduces spread →
    hurts). **Next: 10A** measure the law (equilibrium radius as a function
    of tail fraction — three recorded points already lie on it in the right
    order: 0.0034→0.42, 0.047→0.95, 0.415→1.0); **10B** intervene on shape
    directly (repulsion/diversity penalty or a spectral-tail floor), with the
    same supersession gate — the first intervention in nine phases derived
    from a measured mechanism rather than guessed.
31. **★ THE LAW: the spectral tail sets the equilibrium radius, and R11 is an
    override.** (a) **Spectrum, not packing.** Two families: varying the
    spectrum (`S^β`) moves the radial zero across a **6× span** (1.075 at
    tail 0.505 → 0.179 at tail 0.000), while holding the covariance *fixed*
    and regularizing only the packing (nn CV 0.215 → 0.124, tail held to
    within 1.5%) moves it by **0.098** — a factor of **9**. The separation
    the mechanism pass could not make. (β = 0 is degenerate — a perfectly
    regular white cloud — excluded from the law and reported; it is why the
    family Spearman reads only +0.429, so the span is the honest statistic.)
    (b) **It is a usable law.** Interpolating equilibrium radius against tail
    predicts every 10B arm to within **0.11** — E0 0.274/0.321, E2+E3
    0.405/0.431, E3 0.342/0.335 — and, out of sample, the **free-particle**
    system measured two passes ago to within **0.015** (law 1.010, measured
    0.995). Independent check: real data's tail 0.1376 → law 0.810 against
    the directly measured data-shaped radial zero of 0.844, 3% apart.
    (c) **R11 is the sole outlier**, missing by **+0.511** — an order of
    magnitude worse than any other arm — reaching 1.029 where its shape
    warrants 0.518. That is exactly the earlier direct measurement (radial
    −0.126 at R11's operating point, only 38% pushed outward): **R11 holds
    the cloud past the field's equilibrium against a restoring force.** The
    law makes "override, not repair" quantitative. (d) **The 10B gate did not
    fire, and this is NOT the refutation branch** — the declared refutation
    was "shape moves but the second moment does not". Instead the
    interventions moved the shape (tail up to **3.95×** baseline) and moved
    the second moment by *precisely* the predicted amount, while *improving*
    quality (ED² 1.347 → 0.767). They are simply under-powered: the best arm
    reaches tail 0.0199, still **7× short** of data's 0.1376, and the law
    says that buys 0.405 — and it did. **To reach the band a generator needs
    tail ≈ 0.12.** The mechanism now *predicts* rather than explains, which
    is a first for this program.
32. **★ The tail is DESTROYED, not absent — and the chain is now complete.**
    My hypothesis that the field is *tail-blind* is **refuted, in the
    opposite direction**: the field carries tail energy 0.2893 at the
    generator's cloud against the cloud's own 0.0182 — it offers **16× more
    tail than the cloud has** — and at real data it matches the data's tail
    exactly (0.1649 against 0.1637). The teacher asks for tail every step.
    **The generator starts with a healthy tail (0.221, comparable to real
    data's 0.164) and training destroys it ~29× within the first 100 steps**
    (per-step retention ≈ 0.96), flat thereafter at ≈0.004. So the complete
    chain: least-squares regression onto a smooth map discards trailing
    directions ~4%/step → the RMS-normalized field (‖ηV‖ ≈ 0.5 against an
    output norm ~9) cannot compensate → equilibrium tail ≈ 0.004 → the
    Phase-10 law maps that to second moment 0.27–0.32 → **the generator sits
    at 0.32.** This says *why* the standing negatives are negative: capacity
    and latent dimension are inert because spectral bias is not a capacity
    limit, and fitting the teacher *better* (8B) fits the *dominant
    directions* better. **Causal confirmation**: free particles started from
    a zero-tail cloud never rebuild one (0.0000 → 0.0001 over 600 steps) and
    land at 0.559/0.710 instead of 0.995. **Honest limit**: the Phase-10 law
    predicts 0.18 for those, and the two zero-tail starts differ by 0.15
    despite equal tail, so tail alone is not sufficient outside the family
    the law was fitted on. **Next: whiten the regression metric** —
    `‖Σ^(−1/2)(f − T)‖²` with detached, shrinkage-regularized Σ. Every
    intervention so far acted on the *target*; the measurement says the
    target is fine (16× the tail) and the *metric* is the problem. Primary
    readout is the tail trace, not the score; refuted if the tail rises and
    the second moment does not follow.
33. **Phase 11: whitening refuted, and the cause localized to the
    self-referential teacher.** The derived intervention **failed its primary
    prediction**: with the metric conditioned up to **137×** in favour of the
    trailing directions, the tail still collapsed ~20× in the first 100 steps
    (0.0090–0.0109 against the baseline's 0.0095, ratios 0.94–1.15). At γ ≥
    0.9 it actively *hurt* — second moment 0.233/0.231 against 0.418, ED²
    roughly doubled. **A correction to my own reading**: the 120-step smoke
    showed γ=0.99 improving (0.362 → 0.526) and I reported that as
    encouraging; at 3 seeds it reverses. R11 remains the only arm that
    preserves tail (**0.0665 against 0.0050, 13×**) and survives a fifth
    supersession gate. **The discriminating measurement** — same
    architecture, same optimizer, same ordinary `‖·‖²` loss, differing only
    in the target — gives: drifting teacher, tail 0.216 → **0.0037**; a
    *fixed* particle cloud (tail 0.275), 0.217 → **0.0960 and still rising**;
    a fixed data cloud, 0.217 → 0.0897. **A 26× difference with everything
    else identical.** So the generator is fully capable of building and
    holding a tail, and does not under drifting because of *what the teacher
    asks for*. `T = f + ηV` anchors the target to the generator's own current
    cloud; `‖ηV‖ ≈ 0.5` against an output norm ~9 (≈6%), recomputed from a
    fresh batch each step, so it never accumulates into a persistent demand
    for **shape** — only for a small displacement. Chain link 1 (metric) is
    replaced; links 2–5 stand, and the Phase-10 law was never reached by this
    phase so is not in question. **Next: make the shape demand persist** —
    rollout teachers (apply the field K times before regressing; Phase 2 and
    `AdaptiveRolloutConfirmationProtocol.md` have the machinery but never
    evaluated it against the *tail*) or an EMA/persistent target cloud. Tail
    trace stays the primary readout. *Caveat: the discriminating measurement
    is one seed and must be replicated at 3 before it carries weight.*
34. **★ Formal audit: the cause is self-reference itself, and my last
    hypothesis was wrong.** (a) **Coherence is real but not causal.** At the
    generator's cloud the field's bulk demand is reproducible across disjoint
    batches while its tail demand is not (0.375 vs 0.066, **5.7×**). But if
    that were the cause, enlarging the demand (rollout: K committed field
    steps) or averaging it (M batches at fixed displacement) would help.
    **Neither does**: K=16 leaves the tail at 0.0035 and 16× averaging at
    0.0035 while making everything worse (ED² 1.049). Hypothesis 14 — mine
    from the previous pass — is **refuted**. (b) **What distinguishes the
    arms that work is self-reference.** Every failing arm has the form
    `T = f + Δ`, anchored to the generator's own output; rollout does not
    change that. The two working arms are the two whose target never
    references `f`: fixed particle cloud (tail **0.159**, ED² 0.466) and
    fixed data cloud (tail 0.095, second moment **1.165**, ED² 0.360)
    against the moving teacher's tail 0.0048 and ED² 0.815. (c) **A confound
    in my own Phase-11 probe, checked and survived**: it measured the fixed
    arms' tail on their *training* latents. Re-measured on a fresh probe for
    every arm, the 20–45× gap holds — the tail is in the learned map, not in
    memorized points. (d) **A claim I made that was simply wrong**: Phase 2
    carries a rollout *probe* but it was **never executed** (the artifact has
    only `F1_paper_field`); rollout was run for the first time here.
    (e) Verified from sealed artifacts: **five supersession gates, all
    `passed=false`** — R11 has never been superseded. **Next: replace the
    self-referential teacher with an external one** — have the generator
    chase an *evolving particle cloud* with a transport assignment
    (nearest-neighbour or Sinkhorn). A fixed particle cloud already beats the
    moving teacher on every metric with an *arbitrary* pairing, so the
    assignment is the missing piece. This meets the repository's existing
    `coherent_transport` / `ConditionedTransportAmortization` track on
    evidence rather than analogy. Refuted if the assignment arms behave like
    A0, which would mean the target must be *static* rather than *external*.
35. **★ Phase 12: the external target works; the assignment does not.** The
    gate does **not** fire — R11 survives a **sixth** supersession test — but
    the hypothesis behind the phase is confirmed. **A2** (external particle
    target, index pairing, no self-reference) reaches **tail 0.3243 against
    A0's 0.0041 — a factor of 79**, and 2.5× real data's own 0.132; **second
    moment 0.922, inside the band**, against 0.331; **ED² 0.2091 against
    1.2996, 6.2× better**; per-seed 0.946/0.910/0.922. All measured on a
    **fresh probe**, so the learned *map* carries the tail, not just the 256
    fitted points. It misses the gate ceiling (0.139) by 1.5× at **3.6×**
    training cost; inference is unchanged at NFE = 1. **Both amortizing
    assignments fail, for two distinct and identifiable reasons.**
    *Nearest-neighbour* (A3, A3P) collapses to second moment **0.000** on
    every seed with ED² 13.9–23.7 — a greedy match lets many samples claim
    one particle, so the target degenerates onto a few points. Note its tail
    reads *high* (0.36–0.47) while the generator is destroyed, a reminder
    that tail is a means and ED² the outcome. *Sinkhorn's barycentric
    projection* (A4, A4P) gives tail **0.0007**, a fifth of the
    self-referential baseline and 190× below A2: a barycentric projection is
    a conditional expectation, so by the law of total variance it contracts —
    **it reintroduces the very pathology the phase was built to remove**, and
    more severely. That was a design error of mine, foreseeable from this
    program's own findings. **Next: a balanced AND hard assignment** —
    Sinkhorn plus rounding, Hungarian/auction on the batch (n=256 is
    tractable), or an ε sweep connecting the soft and hard regimes. Refuted
    if it collapses like A3 (balance was not the issue) or contracts like A4
    (hardness was not the issue), in which case A2's fixed pairing does
    something neither captures and the line should be abandoned.
36. **★ Investigating Phase 12: the assignment was never the problem — the
    *correspondence* is.** (a) **The confound is resolved and the Phase-12
    attribution holds.** A2 differed from the baseline in two ways (external
    target *and* fixed latents) and only one control was missing; run now,
    self-reference with **fixed** latents gives ED² **1.0079** against fresh
    latents' 0.9908 — **fixed latents alone do nothing**. (b) **Both failures
    diagnosed directly** by measuring the *target* before training: greedy
    nearest claims **1.0 distinct particles** (target tail 0.0000, second
    moment 0.000 — a single point from step one), and the barycentric
    projection gives target tail **0.0037** against the cloud's 0.4059, a
    factor of 110. (c) **The Hungarian assignment fixes exactly that** —
    target tail 0.327–0.367, second moment 0.976–1.105, essentially perfect.
    **And it still fails on fresh latents.** Same assignment, same target
    quality: generator tail **0.3430** with fixed latents against **0.0009**
    with fresh, a factor of **380**; ED² 0.3014 against 0.8238. So Phase 12's
    stated open problem ("a balanced *and* hard assignment") was the wrong
    diagnosis — an exactly optimal assignment does not help. **What fixed
    latents supply is a stable correspondence**: the same latent maps to the
    same particle every step, so the generator learns one consistent
    function, whereas a per-step re-assignment is matched against a different
    sample each time and the generator averages the inconsistency away —
    the same averaging that destroys the tail under the self-referential
    teacher. **Next: sweep the (latent, particle) pair bank** — 256 → 1024 →
    4096 → 16384 with the correspondence assigned once and reused, scored
    against the particle cloud's own ED² **0.0752** (B2 reaches 0.1640 from
    just 256 pairs, 2.2× off). Clear success criterion, clear failure: a
    plateau well above 0.0752 means the map cannot represent the particle law
    and the line ends. Secondary arm: a *persistent* assignment with sampled
    latents, interpolating between memorization and amortization. **Do not**
    iterate further on assignment algorithms.
37. **★ Phase 13: the amortizer works, beats the particle cloud, and still
    loses to R11.** Gate **not** passed — the best bank (ED² **0.1391**)
    misses R11's ceiling (0.1193) by **1.17×**, the closest any alternative
    has come, and **R11 survives a seventh supersession test**. The
    separately declared **trend criterion passed**: banks 512 and 1024 reach
    **0.70–0.73× the particle cloud's own ED² (0.2089)** — the amortizer is
    *better than the system it amortizes* — with the second moment in band
    (0.989) and the tail restored **27×** (0.0059 → 0.1607, against real
    data's ~0.13, a quantity R11 never produces at 0.0589). Per-seed and
    stable: 0.157/0.139/0.127, from a self-referential baseline of 0.9624 —
    a factor of **6.9**. **The sweep is not monotone and the `visits/pair`
    column says why**: at a fixed 1000-step budget bank 256 gets 1000 visits
    (over-fitted: second moment **1.353**, out of band) and bank 2048 gets
    125 (under-fitted: 0.573, tail decaying back toward baseline), so the
    U-shape is the budget, not the method, and no optimal bank size is
    claimed. Jitter hurts (0.5868 vs 0.3835 on the same bank) — correspondence
    stability wants the *same* latents, not merely frequent ones. **Cost is
    14.4×** the baseline in kernel pairs, all of it the 2048-particle cloud;
    inference is unchanged at NFE = 1. **Honest position: on the program's
    own gate this is not an improvement** — R11 is a free scalar rescale and
    this is a second particle system at 14.4× training cost, 1.17× behind.
    What it is: the first *derived* structural account reproducing most of
    R11's effect, and independent confirmation that the self-reference
    diagnosis is right. **The one settling experiment**: hold visits/pair
    constant (bank 512/1024/2048/4096 at 1000/2000/4000/8000 steps). Falls
    below 0.0955 → the structural route wins on quality; plateaus near
    0.13–0.14 → 512–1024 pairs is the ceiling at this model size, R11 stands,
    and the line closes as a *mechanism* result rather than a method.
    **Higher-value work remains the consolidation**: no configuration has
    ever combined the ESS-0.9 bandwidth, R11 and the anchor.
17. **Section 6.1's anchor is the program's surviving result.** It has now
    improved every configuration it has been placed in — A5/A4 = .701 and
    A6/A4 = .658 on synthetic data, B3/B1 = .906 on CIFAR — at no measurable
    wall-clock cost, and it detects 6/6 source collisions from an independent
    audit bank. Sections 12 and 13 should be rewritten around it.

---

## 1. Executive decision

The next major research direction should be to remove the pretrained feature
encoder from the *training objective* of the original Drifting Model while
retaining:

1. a mathematically auditable signal that distinguishes source distributions;
2. usable local and multiscale image geometry;
3. one-step neural generation at inference;
4. stable minibatch estimation at realistic cost.

The proposed model separates two jobs that the original feature encoder
currently performs:

- a **source-law anchor** prevents distinct image distributions from becoming
  invisible to the objective;
- a **fixed compositional geometry kernel** makes image similarities and
  update directions statistically useful without any pretrained
  representation.

The generator remains a neural network. The removed component is the external
learned feature network that decides which real and generated images are
similar during training.

The initial architecture is:

```text
generated and target images
          |
          +--> spectral source anchor ----------------------+
          |                                                  |
          +--> fixed wavelet/convolutional kernel bank       |
                         |                                   |
                         +--> kernel-gradient drift ---------+
                                                             |
                                                             v
                                                  nonnegative total loss
                                                             |
                                                             v
                                                  neural generator update
```

The primary experiment is not ImageNet-256. It is a controlled CIFAR-10 and
structured-image study capable of answering:

> Can fixed compositional image geometry replace learned semantic geometry in
> drifting without sacrificing source-space correctness?

If the answer is negative at this scale, the program should be stopped or
redesigned before consuming large-scale compute.

---

## 2. What feature-encoder dependence means

There are two distinct problems. They must not be conflated.

### 2.1 Logical blindness

Let `φ` be the feature encoder. Feature-space drifting directly compares the
pushforward laws

\[
\phi_\# p
\qquad\text{and}\qquad
\phi_\# q.
\]

If `φ` is non-injective, then in general

\[
\phi_\#p=\phi_\#q
\quad\not\Longrightarrow\quad
p=q.
\]

This is not merely a theoretical concern. An encoder may deliberately discard
color, fine texture, phase, position, rare objects, or other distinctions that
are unimportant for its pretraining task.

The repository formalizes this boundary in
`DriftingIdentifiability/FeatureSpaceIdentifiability.lean`:

- feature-space zero drift safely gives equality of feature laws;
- source-law equality additionally requires a `MeasurableEmbedding` or another
  independently proved `MeasureDetermining` condition;
- approximate lifting requires a `FeatureStabilityCertificate`;
- a non-injective feature collision gives distinct source Dirac laws with
  equal feature laws.

This means that a stronger semantic encoder can improve generation while still
failing to supply an unconditional source-space correctness guarantee.

### 2.2 Poor raw high-dimensional geometry

A raw radial kernel in pixel space faces a different failure:

- pairwise distances concentrate;
- affinities become nearly constant or nearly zero;
- effective sample sizes collapse;
- minibatch drift becomes noisy;
- Euclidean distance does not encode local or multiscale image structure.

The feature encoder solves this practical geometry problem by mapping images
to a space in which useful neighbors are close. This is why deleting the
encoder without replacing its geometric function is unlikely to work.

### 2.3 Consequence for the architecture

A publishable encoder-independent method must address both problems:

| Required property | Mechanism |
|---|---|
| Distinct source laws remain distinguishable | Spectral characteristic anchor |
| Similar images produce useful interactions | Fixed compositional image kernel |
| Motion follows the declared kernel geometry | Kernel-gradient drift |
| No hidden learned similarity authority | Fixed features; only kernel weights adapt |
| Exact-zero claim avoids field cancellation | Separate nonnegative losses |

Replacing a pretrained encoder with a random encoder, a lossy invariant map,
or an unweighted wavelet transform is not enough.

---

## 3. Evidence from the original paper

The dependence is central to the original implementation, not an incidental
engineering choice.

The paper states that the feature extractor plays an important role on
high-dimensional data because the kernel must place semantically similar
samples close together. It uses pretrained self-supervised encoders and
computes separate losses over many scales and spatial locations.

The reported ImageNet ablations strongly track encoder quality:

| Training feature encoder | Reported FID |
|---|---:|
| SimCLR ResNet, width 256 | 11.05 |
| MoCo-v2 ResNet, width 256 | 8.41 |
| Latent MAE, width 256 | 8.46 |
| Latent MAE, width 384 | 7.26 |
| Latent MAE, width 512 | 6.49 |
| Latent MAE, width 640 | 6.30 |
| Longer-trained width-640 MAE | 4.28 |
| Classification-fine-tuned encoder | 3.36 |

For pixel-space generation the dependence is stronger:

| Training feature configuration | Reported FID |
|---|---:|
| Weaker MAE | 32.11 |
| Stronger MAE with classification fine-tuning | 9.35 |
| Additional pretrained ConvNeXt-V2 MAE | 3.70 |

The paper explicitly reports that it was unable to make ImageNet generation
work without a feature encoder, even with a latent VAE. It attributes this to
a nearly flat kernel in which affinities vanish because all samples are far
apart, and leaves the limitation to future work.

The successful system should therefore be understood as:

\[
\text{drifting}
+
\text{pretrained multiscale perceptual kernel bank},
\]

not as raw-data kernel drifting alone.

Primary source:
[Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770).

---

## 4. Repository results that constrain the design

### 4.1 Source-space Laplace identifiability

`DriftingIdentifiability/LaplaceEuclideanConverse.lean` proves that zero
population Laplace mean-shift drift identifies arbitrary probability measures
on every finite-dimensional Euclidean space:

\[
V_{p,q}^{\mathrm{Laplace}}\equiv0
\Longrightarrow
p=q.
\]

This is an ideal population theorem. It does not imply that the raw pixel
kernel has acceptable finite-batch conditioning.

### 4.2 Feature-law/source-law boundary

`DriftingIdentifiability/FeatureSpaceIdentifiability.lean` provides the exact
architecture rule:

- semantic or geometry features may improve optimization;
- they cannot be the only correctness authority unless their family is proved
  measure-determining;
- approximate source claims require a quantitative stability certificate.

### 4.3 Existing anchored plan

`numerics/AnchoredCoherentTransportResearchPlan.md` already records two
important guardrails:

1. use a fixed source-space anchor alongside optional semantic geometry;
2. do not infer identifiability from a sum of vector fields, because the
   fields may cancel.

The present plan strengthens that proposal by attempting to remove the
semantic encoder altogether. Fixed compositional kernels replace it.

### 4.4 Lessons from the transport experiments

The recent coherent-transport audit found that better coverage machinery does
not automatically produce correct local geometry. In particular:

- a valid transport plan is not automatically a better neural teacher;
- endpoint regression can destroy route information;
- global coordination and local geometry are different responsibilities;
- strong low-dimensional discrepancy scores can conceal structured support
  errors.

The first encoder-independent experiment should therefore isolate kernel
geometry. PQST, KLL, or Sinkhorn coverage repair must not be added until the
base kernel ablation succeeds.

---

## 5. Current literature and the remaining opening

### 5.1 Drifting developments

- [Kernel-Gradient Drifting Models](https://arxiv.org/abs/2605.10727) replaces
  Euclidean displacement with the gradient of the kernel. It supplies the
  most important mathematical ingredient for this plan, but does not
  demonstrate encoder-free natural-image generation.
- [One-Step Generative Modeling via Wasserstein Gradient Flows
  (W-Flow)](https://arxiv.org/abs/2605.11755) improves global coverage with
  Sinkhorn divergence, but its large image implementation still uses
  pretrained feature representations.
- [DriftXpress](https://arxiv.org/abs/2605.12183) accelerates kernel-field
  evaluation with projected RKHS features, but its image experiments still
  use a frozen pretrained image encoder.
- [Sinkhorn-Drifting](https://arxiv.org/abs/2603.12366) improves temperature
  robustness and coverage, but its principal image experiments use pretrained
  autoencoder latent spaces.
- [Drift-RAE](https://arxiv.org/abs/2606.15553) removes the original auxiliary
  MAE feature encoder, but uses a representation autoencoder based on
  pretrained DINO features and distills a pretrained flow. It relocates the
  dependence rather than eliminating pretrained representation geometry.
- [Generative Drifting is Secretly Score
  Matching](https://arxiv.org/abs/2603.09936) identifies the Gaussian
  high-frequency bottleneck and motivates heavy-tailed kernels and
  coarse-to-fine bandwidth schedules.

### 5.1b Fixed compositional representations already used generatively

A 2026-07-24 citation re-check found a directly relevant paper that the
first draft of this plan missed. It must be cited and distinguished before
any novelty language is used.

- [Generative Modeling via Kernelized Stochastic
  Interpolants](https://arxiv.org/abs/2602.20070) (Coeurdoux, Lempereur,
  Cuvelle-Magar, Eboli, Mallat, Borovykh, Vanden-Eijnden) replaces neural
  drift training with a kernel method inside the stochastic-interpolant
  framework, and explicitly uses **wavelet scattering transforms as the
  feature map** for image and physical-field generation.

Consequence for section 5.3: *"use a fixed compositional representation
instead of a pretrained encoder for generative modeling"* is **not novel**.
It is established, by the group that introduced scattering transforms.

What is not covered by that work, and therefore remains the only defensible
opening for this program:

- it is a stochastic-interpolant method, not drifting, and does not use the
  kernel-gradient drift field;
- it is training-free by design (the drift is a solved linear system), so it
  has no one-step neural generator and no minibatch drift-estimation
  problem;
- it carries no source-law anchor and makes no source-space identification
  claim — scattering features are treated as sufficient, not as an
  admittedly non-injective optimization aid.

Any claim in this program must therefore be about **drifting with a separate
measure-determining source anchor**, never about fixed compositional
features per se.

### 5.2 Fixed compositional representations

- [Invariant Scattering Convolution
  Networks](https://arxiv.org/abs/1203.1513) supplies fixed multiscale
  representations that are stable to deformations and retain high-frequency
  information.
- [Generalized Rectifier Wavelet Covariance Models for Texture
  Synthesis](https://arxiv.org/abs/2203.07902) shows that nonlinear wavelet
  statistics can synthesize rich texture without a pretrained representation.
- [Efficient Statistical Tests: A Neural Tangent Kernel
  Approach](https://proceedings.mlr.press/v139/jia21a.html) shows that
  convolutional NTK-style kernels can outperform ordinary shift-invariant
  kernels for image two-sample testing.
- [Learning with Invariances in Random Features and Kernel
  Models](https://proceedings.mlr.press/v134/mei21a.html) gives theoretical
  evidence that correctly encoded group invariance can improve statistical
  efficiency.
- [How Rotational Invariance of Common Kernels Prevents Generalization in
  High Dimensions](https://proceedings.mlr.press/v139/donhauser21a.html)
  supports avoiding a single unstructured radial kernel in high dimension.

### 5.3 Novelty assessment

**Downgraded on 2026-07-24 after the section 5.1b finding.**

The individual ingredients are established, and — contrary to the first
draft — so is the headline combination of "fixed compositional features
instead of a pretrained encoder for generation" (arXiv:2602.20070). The
narrower combination that still appears open is:

> a *source-identifying spectral anchor* carried as a separate nonnegative
> loss alongside a fixed compositional image kernel, used through
> kernel-gradient drifting with a one-step neural generator.

The distinguishing element is the anchor and the explicit
ideal/finite-approximation boundary around it, not the fixed features. This
remains a novelty *hypothesis*, not a certified claim. Repeat the citation
search against later versions, conference submissions, and code releases
before publication, and search specifically for characteristic-kernel or
spectral anchors combined with scattering geometry.

---

## 6. Proposed mathematical architecture

### 6.1 Branch A: spectral source-law anchor

For image distributions \(p,q\) on a finite-dimensional pixel or declared
latent space, define

\[
\mathcal L_{\mathrm{anchor}}(p,q)
=
\mathbb E_{\omega\sim\rho}
\left|
\mathbb E_{X\sim p} e^{i\langle\omega,X\rangle}
-
\mathbb E_{Y\sim q} e^{i\langle\omega,Y\rangle}
\right|^2.
\]

With a sufficiently rich full-support spectral measure \(\rho\), this is a
characteristic-kernel discrepancy. At the ideal population level,

\[
\mathcal L_{\mathrm{anchor}}(p,q)=0
\Longrightarrow
p=q.
\]

Use real random features in code:

\[
z_\omega(x)
=
\begin{bmatrix}
\cos\langle\omega,x\rangle\\
\sin\langle\omega,x\rangle
\end{bmatrix}.
\]

For a finite bank \(\Omega=\{\omega_\ell\}_{\ell=1}^L\),

\[
\widehat{\mathcal L}_{\mathrm{anchor}}
=
\frac1L\sum_{\ell=1}^L
\left\|
\frac1{N_p}\sum_i z_{\omega_\ell}(y_i)
-
\frac1{N_q}\sum_j z_{\omega_\ell}(x_j)
\right\|_2^2.
\]

#### Required implementation choices

- Use multiple frequency bands rather than one bandwidth.
- Include low, medium, and high frequencies.
- Parameterize frequencies as \(\omega=r u\), with \(u\) a unit direction and
  \(r\) drawn from a declared multiband radial law. Calibrate bands from
  target-only projected scales rather than the ambient pixel norm.
- Compare iid directions with structured orthogonal direction blocks; the
  latter may reduce redundant projections without changing the declared
  population target.
- Prefer a heavy-tailed spectral mixture associated with Laplace, Matérn, or
  inverse-multiquadric behavior.
- Resample a declared fraction of frequencies periodically so a fixed finite
  bank does not become the claimed measure-determining object.
- Keep an independent fixed audit bank that is never used for training or
  hyperparameter selection.
- Normalize pixel or latent coordinates once using frozen target-only
  statistics. Use a positive scale floor on every coordinate so this affine
  normalization remains invertible.
- Record the seed and exact frequency distribution in every run.

#### Honesty boundary

A finite feature bank is not exactly characteristic. The exact statement is
about the ideal expectation over \(\rho\). Empirical claims must say
“random-feature approximation” and report feature-bank sensitivity.

### 6.2 Branch B: fixed compositional image geometry

Define a fixed family \(F_{s,r}\), indexed by scale \(s\) and spatial region
\(r\). Candidate families are:

1. complex wavelet coefficients with a smooth modulus;
2. first- and second-order scattering coefficients;
3. cross-scale and cross-orientation wavelet covariances;
4. coarse global image-pyramid values;
5. fixed random convolutional features;
6. finite random-feature approximations of convolutional NNGP/NTK kernels.

A conservative positive-definite geometry kernel is

\[
K_{\mathrm{geom}}(x,y)
=
\sum_{s,r,b}
\alpha_{s,r,b}\,
\kappa_{s,b}\!\left(F_{s,r}(x),F_{s,r}(y)\right),
\]

where every coefficient is nonnegative and every base kernel is positive
definite.

Use the same location \(r\) on both images in the first implementation.
Cross-location coupling should be added only through a proved
positive-semidefinite location matrix. An arbitrary soft cross-patch matching
rule must not be described as a positive-definite kernel.

#### Smooth feature definitions

Replace nondifferentiable modulus by

\[
|u|_\varepsilon=\sqrt{u^2+\varepsilon^2}.
\]

Use a smooth Laplace-like base kernel

\[
\kappa_{\tau,\varepsilon}(u,v)
=
\exp\left(
-\frac{\sqrt{\|u-v\|_2^2+\varepsilon^2}}{\tau}
\right).
\]

The smoothing parameter must be logged and tested for sensitivity.

#### Position and invariance

Do not globally average every coefficient. The geometry family should contain:

- localized/equivariant terms;
- a coarse position-sensitive image pyramid;
- optional weak group-averaged terms for small translations, flips, or crops.

Invariance belongs only in the geometry branch. The spectral source anchor
retains source-space distinctions.

### 6.3 Kernel-gradient movement

The paper's standard normalized displacement field is

\[
V^{\mathrm{std}}_{p,q}(x)
=
\frac{\mathbb E_{Y\sim p}[k(x,Y)(Y-x)]}
     {\mathbb E_{Y\sim p}k(x,Y)}
-
\frac{\mathbb E_{Y\sim q}[k(x,Y)(Y-x)]}
     {\mathbb E_{Y\sim q}k(x,Y)}.
\]

For a structured image kernel, use

\[
V^{\mathrm{KG}}_{p,q}(x)
=
\frac{\mathbb E_{Y\sim p}\nabla_xk(x,Y)}
     {\mathbb E_{Y\sim p}k(x,Y)}
-
\frac{\mathbb E_{Y\sim q}\nabla_xk(x,Y)}
     {\mathbb E_{Y\sim q}k(x,Y)}.
\]

Equivalently,

\[
V^{\mathrm{KG}}_{p,q}(x)
=
\nabla_x\log Z_p(x)-\nabla_x\log Z_q(x).
\]

This change is load-bearing: the geometry kernel controls both interaction
strength and update direction. A texture discrepancy can therefore produce a
texture-directed gradient rather than only a raw linear interpolation toward
another image.

Implement both `standard_displacement` and `kernel_gradient` behind the same
interface. The standard field is a required ablation.

### 6.4 Adaptive kernel mixture

Let

\[
\alpha_j\ge0,
\qquad
\sum_j\alpha_j=1,
\qquad
\alpha_j\ge\alpha_{\min}.
\]

Each \(j\) represents a scale, feature family, or bandwidth. Candidate utility
statistics are:

- drift signal-to-noise ratio;
- effective affinity sample size;
- non-saturation rate;
- held-out two-sample witness power;
- gradient magnitude after normalization;
- gradient agreement with the anchor;
- improvement on an internal source-space discrepancy.

The first adaptive rule should be simple:

\[
\tilde u_j
=
\frac{\|\widehat V_j\|^2}
     {\widehat{\mathrm{Var}}(V_j)+\epsilon},
\qquad
\alpha
=
(1-\beta)\alpha
+
\beta\,\operatorname{softmax}(\tilde u/T),
\]

followed by projection onto the simplex with floor \(\alpha_{\min}\).

#### Cross-fitting requirement

Do not estimate \(\alpha\) and evaluate the resulting drift on exactly the
same examples. Use:

- controller batch: estimate utility and update the EMA weights;
- independent field batch: compute the generator target;
- independent audit batch: log diagnostic discrepancies.

This reduces the risk that the adaptive kernel simply overfits minibatch
noise.

### 6.5 Total objective

Do not rely on cancellation-prone vector-field addition. Use separately
nonnegative losses:

\[
\mathcal L_{\mathrm{total}}
=
\lambda_A\mathcal L_{\mathrm{anchor}}
+
\lambda_G\sum_j\alpha_j\mathcal L_{\mathrm{geom},j}
+
\lambda_R\mathcal L_{\mathrm{regularization}},
\]

with \(\lambda_A,\lambda_G>0\).
Require \(\lambda_R\ge0\) and a nonnegative regularization loss.

The initial geometry loss is the paper-style stop-gradient regression:

\[
\mathcal L_{\mathrm{geom},j}
=
\mathbb E_{x\sim q_\theta}
\left\|
f_\theta(\epsilon)
-
\operatorname{sg}
\left(f_\theta(\epsilon)+\eta_jV_j(x)\right)
\right\|_2^2.
\]

The anchor may be differentiated directly or converted to a separate
stop-gradient target. Direct differentiation is the safer first
implementation because its exact-zero meaning is immediate.

At the ideal population level,

\[
\mathcal L_{\mathrm{total}}=0
\Longrightarrow
\mathcal L_{\mathrm{anchor}}=0
\Longrightarrow
p=q.
\]

This is stronger and safer than claiming
\(V_{\mathrm{anchor}}+V_{\mathrm{geom}}=0\Rightarrow p=q\).

---

## 7. Claims and non-claims

### 7.1 Claims the project may target

- The training objective uses no pretrained or jointly learned feature
  encoder.
- The ideal spectral anchor is measure-determining in source space.
- The fixed geometry kernel improves finite-batch image geometry over a raw
  radial kernel.
- Kernel-gradient movement is more compatible with the structured kernel than
  raw Euclidean displacement.
- The method is robust to removal, weakening, or corruption of an external
  encoder because no such encoder supplies its training signal.

### 7.2 Claims forbidden before evidence

- “Encoder-free ImageNet is solved.”
- “The finite random-feature anchor is exactly characteristic.”
- “Wavelet features are injective” after modulus or pooling.
- “Patch matching identifies image distributions.”
- “A sum of vector fields is identifiable.”
- “Better FID proves better source distribution matching.”
- “Fixed features are automatically cheaper than a pretrained encoder.”
- “The method beats the paper” unless the protocol and compute are matched.

---

## 8. Implementation layout

Create a new isolated package so the experiment does not modify existing
flagship or coherent-transport behavior:

```text
numerics/encoder_independent_drifting/
    __init__.py
    config.py
    spectral_anchor.py
    fixed_features.py
    kernels.py
    kernel_gradient.py
    adaptive_mixture.py
    objectives.py
    models.py
    datasets.py
    metrics.py
    diagnostics.py
    collision_suite.py
    train.py
    evaluate.py
    tests/
        test_spectral_anchor.py
        test_fixed_features.py
        test_kernel_gradients.py
        test_positive_kernel_mixture.py
        test_crossfit_controller.py
        test_reproducibility.py
```

Reuse utilities only when their semantics are clear:

- official-style baseline logic from `numerics/driftlab.py` or the current
  faithful baseline;
- structured targets from `numerics/coherent_transport/targets.py`;
- source-space metrics from `numerics/coherent_transport/metrics.py`;
- run registries, hashes, frozen protocols, and source snapshots from the
  recent confirmation infrastructure.

Do not import the PQST controller, route planner, or neural transport teacher
in the first geometry experiment.

---

## 9. Phased implementation roadmap

### Phase 0 — mathematical and implementation sanity

#### P0.1 Spectral anchor

Implement:

- deterministic seeded frequency sampling;
- frequency-band mixtures;
- real sine/cosine feature moments;
- unbiased and biased empirical losses;
- analytic generator gradients;
- fixed audit bank and periodically refreshed training bank.

Unit tests:

1. loss is nonnegative;
2. loss is numerically zero when the sample arrays are identical;
3. gradients agree with finite differences;
4. shifted Gaussians and variance-changed Gaussians are detected;
5. high-frequency perturbations are detected only when the corresponding
   frequency band is present;
6. fresh-bank averaging converges toward a dense-bank reference.

#### P0.2 Fixed image features

Implement in order:

1. Laplacian image pyramid;
2. fixed oriented wavelet bank;
3. smooth modulus;
4. localized first-order scattering;
5. selected second-order or covariance terms;
6. optional fixed random convolutional features.

Unit tests:

- expected output shape by scale and location;
- no trainable parameters;
- deterministic outputs;
- nonzero gradients with respect to the input;
- translation/deformation sensitivity matches the declared design;
- localized features distinguish permuted patch layouts;
- full orthonormal-wavelet Euclidean distance equals pixel distance as a
  negative control.

#### P0.3 Kernel-gradient implementation

Implement a common field API:

```text
field(
    generated,
    positive,
    negative,
    feature_family,
    kernel,
    direction_mode,
    normalization,
) -> drift, diagnostics
```

Tests:

- automatic differentiation matches finite differences;
- Gaussian raw-space kernel-gradient reproduces the expected smoothed-score
  form up to convention and scale;
- denominators are finite and floored explicitly;
- batch permutations do not change the field;
- positive and negative inputs swapped reverse the field when expected;
- identical empirical laws give zero up to numerical tolerance.

#### P0.4 Mixture and cross-fitting

Implement fixed weights first, then the adaptive controller.

Tests:

- weights remain on the simplex;
- the declared floor is never violated;
- controller examples do not enter the field batch;
- disabling adaptation exactly reproduces fixed weights;
- duplicate kernel branches do not double effective weight silently;
- controller decisions are reproducible from the run registry.

#### Phase 0 exit gate

Proceed only if:

- all mathematical unit tests pass;
- the anchor detects every synthetic collision pair;
- at least one fixed geometry branch has healthier affinity ESS and drift SNR
  than raw pixel drifting;
- kernel-gradient and standard-displacement modes differ in the predicted
  structured directions.

---

### Phase 1 — structured-image mechanism screen

Use low-resolution image-valued targets designed to expose geometry failures:

- noisy checkerboard;
- pinwheel/radial spokes;
- rings and disconnected islands;
- local texture blocks;
- identical patch histograms with different global arrangements;
- color-swapped layouts;
- phase-scrambled images with preserved power spectra;
- rare small-object modes;
- translated and slightly deformed copies.

Required arms:

| ID | Anchor | Geometry | Direction | Adaptation |
|---|---|---|---|---|
| A0 | none | raw pixel Laplace | standard | none |
| A1 | none | raw pixel smooth Laplace | kernel gradient | none |
| A2 | spectral | none | direct anchor gradient | none |
| A3 | none | wavelet/scattering | standard | fixed |
| A4 | none | wavelet/scattering | kernel gradient | fixed |
| A5 | spectral | wavelet/scattering | kernel gradient | fixed |
| A6 | spectral | random convolutional | kernel gradient | fixed |
| A7 | spectral | geometry dictionary | kernel gradient | adaptive |
| A8 | none | pretrained paper encoder | paper protocol | fixed |

Metrics:

- energy distance and sliced Wasserstein;
- support precision and recall;
- component occupancy error;
- patch-distribution discrepancy;
- multiscale spectral error;
- topology-sensitive metrics appropriate to each target;
- anchor and geometry loss separately;
- affinity ESS, entropy, saturation, and drift SNR by branch.

Do not promote a method solely because it wins ED or SW. Inspect particle/image
geometry and collision cases.

#### Phase 1 exit gate

The program passes only if:

1. `A4` materially beats `A1` on structured geometry;
2. `A5` stays within 10% of `A4` on a pre-registered normalized geometry
   score (not an informal comparison across incomparable raw metrics);
3. `A5` passes collision cases on which `A4` fails;
4. the anchor contributes at least a declared minimum gradient share for a
   meaningful portion of training;
5. gains hold over multiple seeds and target families.

If `A4` fails, fixed wavelet geometry is not supplying the needed image prior.
Test the convolutional kernel before considering a learned encoder.

---

### Phase 2 — CIFAR-10 development

#### P2.1 Dataset and model

- CIFAR-10 at \(32\times32\);
- unconditional generation first;
- one fixed moderate-capacity generator;
- matched batch size and generated/target sample accesses;
- no pretrained network in the training graph;
- external pretrained evaluators allowed only for frozen evaluation.

Development choices should use the pre-registered internal discrepancies and
mechanism diagnostics. External encoder metrics are report-only and must not
choose bandwidths, kernel weights, stopping times, or checkpoints.

#### P2.2 Development arms

Run:

1. faithful original method with no encoder;
2. faithful original method with the paper-style available encoder;
3. raw kernel-gradient drifting;
4. anchor only;
5. fixed wavelet geometry only;
6. fixed convolutional kernel only;
7. anchor plus each geometry family;
8. fixed mixed kernel;
9. adaptive mixed kernel.

Freeze the development protocol before broad seeds. Do not choose checkpoints
or bandwidths from final test metrics.

#### P2.3 Cost matching

Report:

- training wall-clock;
- peak memory;
- generator FLOPs;
- feature/kernel FLOPs;
- pairwise interaction cost;
- number of target examples consumed;
- number of controller and audit examples;
- inference NFE and latency.

“No pretrained encoder” is not synonymous with “cheap.” Fixed wavelet and
kernel banks must be included in the training cost.

#### P2.4 Evaluation

Primary:

- FID and KID from a frozen independent evaluator;
- precision/recall or density/coverage;
- nearest-neighbor and memorization audit;
- class coverage from a frozen classifier used only after training;
- human-readable sample grids using fixed seeds.

Internal mechanism:

- anchor discrepancy on a held-out frequency bank;
- patch and wavelet discrepancies;
- gradient contribution by branch;
- kernel ESS and saturation;
- frequency progression over training;
- controller weight trajectories;
- branch-gradient cosine similarities.

#### Phase 2 exit gate

Continue to larger images only if:

- anchor plus fixed geometry substantially beats raw encoder-free drifting;
- quality is competitive enough with a pretrained-encoder baseline to justify
  scaling;
- collision robustness is retained;
- gains survive at least five frozen seeds;
- the adaptive mixture beats the best fixed mixture on more than one
  development split or target family;
- cost is reported honestly and is not prohibitive.

---

### Phase 3 — independent confirmation

Before any publication language:

1. freeze code, configurations, metrics, and checkpoints;
2. hash the protocol and source snapshots;
3. run untouched confirmation seeds;
4. generate tables directly from registered artifacts;
5. retain negative and failed arms;
6. audit for controller/evaluation overlap;
7. compare all methods under the same target-sample and compute accounting.

The confirmation report must distinguish:

- development-selected results;
- confirmatory results;
- exact population theory;
- finite random-feature approximations;
- external evaluator metrics;
- qualitative observations.

---

### Phase 4 — moderate-scale natural images

Scale in this order:

1. Tiny ImageNet or ImageNet-32;
2. FFHQ-64 or ImageNet-64;
3. class-conditional variants;
4. ImageNet-256 only after a positive moderate-scale confirmation.

At higher resolution:

- use separable or FFT wavelet computation;
- use low-rank/random-feature approximations where audited;
- subsample spatial locations with fixed unbiased estimators;
- preserve at least one coarse global and one high-frequency branch;
- profile kernel cost before increasing generator size.

Do not add Sinkhorn, PQST, or route-conditioned transport until the fixed
kernel baseline is understood. Those are optional coverage improvements, not
solutions to encoder geometry.

---

## 10. Diagnostics and hard failure indicators

### 10.1 Kernel health

Log per branch:

- mean and quantiles of pairwise distances;
- mean and quantiles of affinities;
- effective sample size;
- affinity entropy;
- percentage below a numerical-zero threshold;
- denominator minima and maxima;
- drift norm and variance;
- gradient norm;
- spectral distribution of the input gradient.

### 10.2 Anchor health

Log:

- loss by frequency band;
- gradient share relative to geometry;
- refreshed-bank versus fixed-audit-bank discrepancy;
- sensitivity to feature count;
- collision-suite detection rate;
- cosine similarity with each geometry branch.

The anchor is considered rhetorically present but practically absent if its
gradient share remains below the frozen threshold for most of training and
removing it changes no collision result.

### 10.3 Geometry health

Log:

- scale and orientation weights;
- localized versus global contribution;
- response to small translations and deformations;
- global-arrangement collision error;
- feature saturation;
- standard-displacement versus kernel-gradient update visualizations.

### 10.4 Stop conditions

Stop or redesign if:

- fixed geometry does not beat raw kernel-gradient drift on structured images;
- the anchor destroys sample quality at every nontrivial weight;
- the anchor becomes numerically invisible;
- adaptive weights collapse permanently to one branch;
- results depend on a particular evaluator or one target family;
- fixed feature cost approaches or exceeds the avoided pretrained encoder
  without a correctness or robustness benefit.

---

## 11. Risk register and planned repairs

| Risk | Why it matters | First repair |
|---|---|---|
| Spectral anchor gradients are noisy | High-dimensional frequencies oscillate | Frequency bands, antithetic frequencies, EMA, larger anchor batch |
| Raw anchor is too weak | Correctness exists only on paper | Minimum weight and gradient-share gate |
| Wavelet geometry misses semantics | Fixed local statistics may not encode objects | Add fixed convolutional NNGP/NTK branch |
| Geometry loses global arrangement | Patch marginals are insufficient | Position-sensitive pyramid and collision suite |
| Excessive invariance causes blindness | Orbit laws replace source laws | Keep invariance out of anchor; restrict group radius |
| Kernel affinities flatten | Recreates paper's failure | Per-branch whitening, median diagnostics, multiband kernels |
| Adaptive weights overfit batches | False development gains | Cross-fitting, EMA, entropy, held-out audit |
| Branch gradients cancel | Training stalls despite active losses | Separate losses and report gradient cosine; avoid field-sum theorem |
| Fixed kernel is too expensive | Encoder removal gives no practical benefit | FFT/separable wavelets, spatial subsampling, low-rank features |
| External metric leakage | Encoder re-enters through selection | Frozen dev protocol and independent confirmatory metrics |
| Finite RFF bank is called exact | Overstates theory | Explicit ideal/finite distinction in every report |

---

## 12. Formal follow-up

The first phase is empirical, but the architecture was chosen to admit a clean
formal story.

Potential Lean tasks after a successful mechanism screen:

1. define a finite family of nonnegative source and geometry losses;
2. prove zero total loss forces zero anchor loss when its weight is positive;
3. instantiate a measure-determining source discrepancy;
4. formalize that adding arbitrary geometry losses cannot destroy the
   exact-zero implication;
5. distinguish the ideal spectral expectation from a finite random-feature
   approximation;
6. formulate a `FeatureStabilityCertificate` for an injective multiscale
   linear anchor or a source MMD;
7. connect a characteristic kernel-gradient theorem to the repository's
   `IdentifiesAtZero` interface only after checking its exact normalization and
   support hypotheses.

Do not formalize the geometry branch as injective unless it actually is. Its
role is efficient optimization, not source-law authority.

---

## 13. Publication criteria

### 13.1 Minimum interesting result

A credible workshop or focused paper result would show:

- encoder-free training on CIFAR-10 or Tiny ImageNet;
- clear improvement over raw pixel drifting;
- competitive quality with a learned-encoder drifting baseline under matched
  accounting;
- source-collision robustness from the anchor;
- successful confirmatory seeds;
- transparent cost and negative ablations.

### 13.2 Strong result

A stronger paper would show:

- results across multiple natural-image datasets;
- performance close to or better than pretrained-encoder drifting;
- stable scaling from \(32^2\) to \(64^2\) or higher;
- an explicit source-space exact-zero theorem;
- an efficiency result using random features or structured transforms;
- robust performance under encoder removal because the method never uses one.

### 13.3 Defensible headline

If the evidence supports it:

> A spectrally anchored compositional kernel can replace pretrained
> representation learning in drifting at moderate image scale, while
> preserving an explicit source-distribution identification mechanism.

Do not claim that all representation dependence has disappeared if the final
system uses:

- a pretrained tokenizer whose latent geometry is load-bearing;
- a pretrained teacher;
- a pretrained network for training-time loss or controller decisions;
- an evaluation encoder for checkpoint or hyperparameter selection.

Evaluation-only encoders are acceptable when frozen and clearly disclosed.

---

## 14. Recommended immediate work order

1. **Implement the spectral anchor and its tests.**
2. **Implement raw kernel-gradient drifting and verify it against finite
   differences.**
3. **Implement a minimal Laplacian-pyramid/wavelet geometry bank.**
4. **Run the structured-image mechanism screen.**
5. **Add smooth scattering only if the minimal wavelet bank shows signal.**
6. **Add fixed convolutional kernels if wavelets miss object-level geometry.**
7. **Combine the successful geometry branch with the anchor.**
8. **Only then implement adaptive mixture weights with cross-fitting.**
9. **Freeze a CIFAR-10 development protocol.**
10. **Run independent confirmation before scaling.**

The first useful checkpoint is the end of Step 4. It can falsify the central
idea cheaply.

---

## 15. Consistency review

This plan was reviewed against the paper, the formal feature-space boundary,
and the repository's empirical audit rules.

### 15.1 Corrections incorporated during review

- The source anchor is a separate nonnegative discrepancy rather than merely
  another vector field. This avoids the cancellation loophole.
- The document distinguishes the ideal characteristic spectral expectation
  from its finite random-feature approximation.
- It does not claim that wavelet/scattering or patch features are injective.
- It explicitly notes that an orthonormal wavelet transform with an ordinary
  Euclidean radial kernel changes no distances.
- Cross-location patch coupling is restricted to a positive-semidefinite
  construction when a positive-definite kernel is claimed.
- Adaptive weights use cross-fitting so controller noise cannot define and
  evaluate the same field.
- Evaluation encoders are separated from training encoders and cannot silently
  enter model selection.
- Geometry, coverage, and route amortization are treated as separate
  mechanisms.
- Matched compute includes fixed feature and kernel cost.

### 15.2 Remaining uncertainties

- It is unknown whether fixed compositional kernels contain enough semantic
  information to compete with a large pretrained encoder.
- The best source-anchor weight and frequency schedule are empirical.
- Characteristic population theory does not guarantee acceptable minibatch
  conditioning.
- A fixed convolutional kernel may be more successful than wavelets but more
  expensive.
- The precise combination appears novel from the reviewed literature, but
  novelty must be checked again before submission.

These uncertainties are exactly what Phases 0--2 are designed to resolve.
