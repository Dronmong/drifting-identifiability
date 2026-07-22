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

## D. Mode-recovery program (coverage scoreboard)

Reframed the objective from aggregate ED2 to mode coverage under missing-mode
init (`ModeRecoveryRoadmap.md`), targeting the axis where prior programs won.

| # | Idea | Mechanism targeted | Setting | Outcome | Verdict |
|---|---|---|---|---|---|
| D0 | Coverage metric (reach vs resolution) | change the scoreboard | `mode_recovery.py` | — | Retained infrastructure. Reach = basin mass; resolution = intra-mode precision. |
| D1 | Headroom precondition | verify a real deficit | K∈{16,32,64}, d∈{2,5,10} | ⚪ | Confounded first (selected η by reach); corrected. No headroom at K=16 d=2 (single bandwidth solves it, resolve 1.0); genuine headroom only at K≥32 or d≥5. |
| D2 | Additive two-scale field `V(τc)+αV(τf)` | reach + resolve simultaneously | hard regimes | ❌ | Resolve gains +0.06/+0.03/+0.00/−0.06/+0.00 = zero within noise. Fine scale active only within ~3σ; coarse spreads across the basin — no overlap. |
| D3 | Coarse-to-fine annealing `τ0→τ1` | contract the resolving region with the cloud | hard regimes | ❌ | +0.03 (d=2), +0.12 (d=5); best resolution ~0.34, still ~⅔ modes unresolved. |
| D4 | Capacity check (scale N, steps) | is the deficit field-fixable? | K=32 d=2 | — | More particles INERT; more steps lift reach not resolve. Barrier is dynamical, not capacity. |

**Pattern reinforced.** The mode-recovery program hit the same meta-pattern:
where the baseline works no candidate is needed; where it fails every candidate
fails too. The binding constraint is the swarm's inability to split from a
degenerate start and concentrate onto many tiny modes — a dynamical barrier no
bandwidth/schedule/particle-count change overcomes. Closed as a characterized
negative; details in [`numerics/M1_headroom.md`](../../numerics/M1_headroom.md).

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

**→ Active roadmap for this target:**
[`ModeRecoveryRoadmap.md`](../../numerics/ModeRecoveryRoadmap.md).

---

## E. Global assignment / functional-geometry probes

The closed mode-recovery program made the missing mechanism more precise: the
learned generator needs different members of a concentrated minibatch to
receive different destinations. These probes therefore change coupling or
parameter geometry instead of applying another scalar gain or bandwidth.

| # | Idea | Mechanism targeted | Setting | Outcome | Verdict |
|---|---|---|---|---|---|
| E1 | Random sliced-rank field / warm start | globally balanced particle-to-quantile assignment | K=32, d=2 missing-mode generator | ❌ | A tuned paper bandwidth matched or beat every variant; the naive high-dimensional extension is rejected. |
| E2 | **70% exact 1-D quantile fission, then 30% paper Laplace refinement** | split a homogeneous swarm globally, sharpen locally | 1-D learned generators | 🟡 | Frozen 16-target test: ED2 ratio **.9105**, target-bootstrap CI `[.8270,.9873]`, SW1 `.8949`, wins 23/32 cells, ~half wall time. **Gate failed** because the predeclared effect threshold was `.80`; oracle-paper ratio `.9774`, unequal-weight family worsened, and far starts had already failed. Retain as a modest scoped/compute win, not a general paper-beating result. |
| E3 | Output-space natural / Gauss--Newton field tracking | make the generator realize the requested particle motion | K=32, d=2 | ❌ | Large apparent win against `tau=.4` evaporated after tuning the paper arm (`tau=.8`); also expensive and did not solve mode resolution. |
| E4 | **Resolution-gated virtual-large-batch QLD** | resolve rare quantile regions without applying deterministic global transport to connected/overlap controls | 12-target 1-D development suite, 8 seeds | 🟡 | **ED2 `.7948` vs selected paper and `.8685` vs per-cell paper oracle**, CI `[.7063,.8910]`, 19/24 cells, all families favorable. Costs `1.22x` wall and `7.89x` generator evaluations; far start remains `3.31x` worse. Strongest development result, but not confirmatory until a new frozen registry passes. |
| E5 | **Resolution-gated LB-QCD frozen test** | confirm E4 without registry reuse or retuning | 16 untouched 1-D targets, 2 inits, 20 seeds | 🟡 | ED2 **`.8218`**, CI `[.7649,.8915]`, SW1 `.8679`, 30/32 wins, every family favorable, and **`.9437` vs per-cell paper oracle**. Five of six checks pass; frozen gate **FAILS narrowly** because `.8218 > .80`. Strong scoped improvement, not the predeclared 20% effect. |

| E6 | **Persistent Quantile Transport (PQT)** | put monotone mass assignment into the 1-D generator rather than reconstructing and forgetting a batchwise rank coupling | 12-target development + separate 16-target/20-seed replication | scoped win | Budget-matched 128-knot PQT obtains **`.6231` vs LB-QCD** (CI `[.4817,.8111]`) in development and **`.6427`** (CI `[.5167,.7884]`) on the separate prior-confirmatory registry; all families and both inits improve. A matched-B128 arm is also favorable. This is the largest learned-generator improvement so far, but it is a scoped nonparametric 1-D architecture result and still needs a new untouched confirmatory registry. |
| E7 | **PQT frozen confirmation** | test the fixed transport-aligned generator against paper, QLD, and LB-QCD on a new registry under equal routed target-sample budgets | 20 fresh 1-D targets, 3 inits, 20 seeds, 9 arms | PASS | All 13 frozen gates pass. Primary PQT obtains **`.4230` vs LB-QCD** (CI `[.3258,.5344]`), **`.3487` vs selected paper**, and **`.5183` vs the per-cell paper oracle**; SW1/LB-QCD is `.5602`, all seven families and all three inits improve, 57/60 cells win, and divergences are zero. This is a confirmed scoped 1-D architecture improvement, not a high-dimensional image claim. |

**New hypothesis.** Local Laplace drift is useful for refinement but is a poor
fission operator. In one dimension, exact rank coupling supplies the missing
global mass allocation without labels or target-mode knowledge. The current
candidate deliberately uses it only as a warm phase and returns to the paper
field for its certified local endpoint.

**→ Completed candidate assessment:**
[`QuantileFissionConfirmatoryResults.md`](../../numerics/QuantileFissionConfirmatoryResults.md).
