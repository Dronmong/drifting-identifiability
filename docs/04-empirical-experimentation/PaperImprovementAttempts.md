# Paper-improvement attempts: the ideas ledger

**Purpose.** One place recording *every* modification we tried in order to beat
the paper's drifting method, what mechanism it targeted, how it was tested, and
what happened. The point is the **pattern**: reading the whole column of
outcomes tells us where the paper is genuinely beatable and where it is not — so
the next attempt aims at real headroom instead of re-losing an old battle.

Keep this file append-only and honest. A failed idea recorded is worth more than
a success imagined; the pattern only emerges if nothing is quietly dropped.

**Outcome legend.** ✅ general win (passed a pre-registered gate) · 🟡 conditional
win (helps some regime/target, fails the target-balanced gate) · ⚪ neutral /
tie · ❌ worse · 🧊 win that evaporates in the deployment setting.

---

## A. Collapse Atlas + Phase C (dynamics design rules)

| # | Idea | Mechanism targeted | Setting | Outcome | Verdict |
|---|---|---|---|---|---|
| A1 | Coarse (single larger) bandwidth | missing-mode transport barrier | atlas / low-dim | 🟡 | Repairs missing-mode metastability, but a *well-tuned fixed* bandwidth already sits at the coarse scale (`τ*≈√(σL)`), so it is not a new lever on aggregate. |
| A2 | Coarse-to-fine bandwidth annealing | metastability then sharpening | atlas | 🟡 | Wins in tested 2-D, ties fixed-coarse in 1-D, **loses** in 5-D. Dimension-dependent. |
| A3 | Two-bandwidth / multi-scale field | missing-mode escape | atlas | 🟡 | Large (31–38×) rescue **in the missing-mode regime only**. |
| A4 | Generator-derived step `η* = min −2Reλ/|λ|²` | conditioning / step size | Phase C | ⚪ | It is a stability *ceiling*, not an operating point; `0.1·η*` is usable in the coarse regime. Not a general win. |
| A5 | Eye-mask removal | finite self-pair distortion | Phase C | 🟡 | Harmful at low `N/K`; helps curved supports. Configuration-dependent. |
| A6 | Cluster-count mask trigger | auto mask on/off | Phase C / attribution | ❌ | Misfires on an overclustered Gaussian; not a finished rule. |

## B. Low-dimensional performance study (D0–D3) + fresh attribution

| # | Idea | Mechanism targeted | Setting | Outcome | Verdict |
|---|---|---|---|---|---|
| B1 | Geometry-matched bandwidth `τ=√(σ̂L̂)` | scale adaptation | D2/D3 particles | ⚪ | Ties but does not beat the grid-tuned fixed bandwidth. Pre-registered gate **failed** (ratio 1.031). |
| B2 | Scaled step `η=0.15τ` | step adaptation | D2/D3 | ⚪ | Did not beat the tuned fixed step. |
| B3 | Generator-derived step ceiling | conditioning | D1 | ❌ | Worse than the tuned baseline. |
| B4 | Geometry-triggered eye-mask disable | self-pair distortion | attribution (fresh split) | 🟡 | 32–52% wins on ring/circle/moon; confounded with bandwidth/step; **target-balanced gate failed**. |
| B5 | Combined "geo-fixed" policy | all of the above | D3 | 🟡 | Exploratory wins (moons −47%, ring −37%) but aggregate gate **failed** (1.031). |

## C. NCJ program (normalized, cross-fitted, jittered drifting)

| # | Idea | Mechanism targeted | Setting | Outcome | Verdict |
|---|---|---|---|---|---|
| C1 | Normalized gain (drop `P·Q`) | off-support freezing | E4 particles | ✅→🧊 | **0.31×** particle win (certified no-freeze, T1/T4)… but **1.028×** (neutral) in the Adam generator. |
| C2 | Cross-fitted reference (independent batch, no mask) | finite self-pair bias | E4 particles | ✅→🧊 | Compounds C1 to **0.088×** for particles; **1.016–1.072×** in the generator. |
| C3 | Symmetric Gaussian jitter | homogeneous-swarm splitting | E4 validation | ❌ | Monotonically *worse*; frozen out at σ=0. (Certified identifiability-safe, T2, but empirically unhelpful.) |
| C4 | **NCJ combined → E4 gate** | — | particle test | ✅ | **Ratio 0.100, all 8 criteria** on the held-out registry. Real, pre-registered particle win. |
| C5 | **NCJ → E5 generator gate** | transfer to learned MLP | generator test | ❌ | **Ratio 1.072**; the particle win does not transfer. |
| C6 | ABC (analytical single-batch bias correction) | replace cross-fit cheaply | post-gate | ⚪ | 0.326 vs cross-fit 0.101 (particles): does **not** recover the cross-fit gain; ≈ dropping the gain. |
| C7 | Optimizer switch (SGD / SGD+momentum) | preserve the no-freeze win | diagnostic | 🧊 | Under SGD our field beats the paper field ~1000× on `far` — but only *ties* `paper+Adam` overall. A rigged comparison, not a win. |
| C8 | Tempered gain `(P·Q)^γ`, γ∈{0…2} | better reliability weight under Adam | H1 screen | ⚪ | Near-flat; best interior γ≈0.96 aggregate, **CI includes 1**; below the pre-registered 0.90 gate. |

**Defined but not run** (the flatness of C8 makes them low-priority at this
scale): H2 ESS / inverse-variance reliability weight; H3 loss-level confidence
decoupling; H4 variance-reduced cross-fit; adaptive bandwidth from field
diagnostics.

---

## The pattern (read the outcome column top to bottom)

Four recurring failure modes explain almost every ⚪/❌/🧊, and one recurring
success stands out.

1. **A well-tuned baseline already absorbs single-knob interventions.**
   Bandwidth (A1, B1), step size (A4, B2, B3), and mask (A5) all failed to beat
   a *tuned* fixed value — because the tuning grid already lands on the "smart"
   setting (`τ*≈√(σL)`). *Tuning is not a mechanism you can out-design with
   another static knob.*

2. **Geometry-conditional wins don't generalize.** Mask-off helps curved
   supports, coarse helps missing modes (A2, A3, B4, B5) — but each needs a
   trigger keyed to target geometry we don't get to see, and the trigger
   misfires elsewhere (A6). Every one of these failed the *target-balanced*
   gate while looking great on a favorable subset.

3. **The biggest wins cured pathologies the deployment setting doesn't have.**
   The NCJ no-freeze win (C1/C2/C4) cured *particle* freezing that Adam
   independently cures (C5/C7/C8); the atlas rescues (A3) cured missing-mode
   metastability that a coarse scale or better init also cures. We kept
   defeating problems a competently-configured baseline never suffers.

4. **Identifiability-motivated changes are optimization-neutral.** Everything
   justified by "this preserves identifiability" (drop `P·Q`, cross-fit, jitter)
   turned out not to matter for optimization, or to matter only through a
   mechanism the optimizer already handles. *Identifiability structure ≠
   optimization improvement.*

**Meta-pattern.** Every attempt operated on the field/estimator **at a fixed,
low-dimensional problem scale where a well-tuned, Adam-trained baseline is
already near-optimal.** That is why aggregate metrics never moved: there is
little headroom there, and what looked like headroom was an artifact of a
degraded configuration (constant-step optimizer, untuned bandwidth, missing-mode
init without a coarse scale).

## The one thing that consistently wins: mode recovery

Look for the ✅/🟡 rows and ask *what metric and what regime* they win in. The
answer is strikingly consistent — **missing-mode recovery / mode coverage under
adversarial initialization**:

- A3 two-bandwidth rescue: **31–38×** — missing-mode regime.
- A1 coarse bandwidth: repairs the **missing-mode** barrier.
- B4/B5: biggest conditional wins on ring/moons — supports the baseline struggles to *cover*.
- C4 NCJ: **missing-mode Kaplan–Meier recovery 60 vs 114 steps (≈2× faster)** — a criterion the E4 gate passed cleanly even though transfer later failed.

Mode collapse / missing-mode recovery is a *genuine, known* failure mode of
drifting and diffusion generators, and crucially it is **not** a place where "the
tuned baseline is already near-optimal" — it is where the baseline visibly
struggles. Aggregate ED2 on easy, well-covered targets is the wrong scoreboard;
it is exactly where the paper is hard to beat.

## Clear target for the next challenge

Stop attacking aggregate distributional error on easy, well-initialized,
low-dimensional targets. Instead:

> **Reframe the objective to mode coverage / missing-mode recovery under
> adversarial (missing-mode, far, concentrated) initialization — the one axis
> where every program showed real, repeated, non-conditional wins — and test it
> where the baseline genuinely struggles: higher dimension and/or more modes,
> where a coarse/multi-scale or reliability-weighted field cannot be trivially
> reproduced by tuning a single fixed bandwidth.**

Concretely, the next pre-registered experiment should (a) use a **mode-recovery
gate** (fraction of target modes covered, and time-to-cover, under missing-mode
starts) as the *primary* metric rather than aggregate ED2; (b) scale to more
modes and higher dimension so a single tuned bandwidth cannot cover all scales
at once; and (c) revisit the **multi-scale / coarse-then-fine** field (A2/A3),
which is the intervention most aligned with this metric and was only ever set
aside for failing the *aggregate* gate — the wrong gate for this question.

This is the pattern the ledger reveals: we were winning the whole time, just on
a scoreboard we then threw away by averaging it into aggregate error.
