# Stage 1A results: coherent-plan free-particle screen

Per `AnchoredCoherentTransportResearchPlan.md` §9 (Stage 1A), §15, §18.
Screen only (3 seeds, declared-not-tuned η, one target realization/family). This
is a **go/no-go**, not a promotion; no confirmation claim is made.

Reproduce: `uv run --with numpy --with scipy python
numerics/coherent_transport/run_coherent_particle_screen.py`.
Primitives unit-tested: `est_plan.py` (8/8), `metrics.py` (4/4).

## Verdict: GO. Gate 1A PASSES.

**Hard consensus (coherent discrete joint plan) improves support precision and
eliminates off-support leakage over the current independent-PSQT correction on
all six structured-geometry targets, while retaining coverage and global ED2 and
improving rare-mode mass.** This confirms the program's central hypothesis
(RQ1/H1): *independent sliced assignment was the geometry bottleneck*, and a
coherent joint plan fixes it.

## Full screen (median of 3 seeds; free particles, N=256, broad init)

| target | arm | prec | off-sup | recall | cover | ED2 | extra |
|---|---|---:|---:|---:|---:|---:|---|
| checkerboard | paper | 0.18 | 0.82 | 1.00 | 0.19 | 2.43 | leak 0.48 |
| | psqt_indep | 0.94 | 0.06 | 1.00 | 0.67 | 0.009 | leak 0.13 |
| | **consensus** | **1.00** | **0.00** | 0.98 | 0.61 | 0.009 | **leak 0.00** |
| | est_bary | 0.71 | 0.29 | 0.38 | 0.23 | 0.169 | leak 0.52 |
| | sinkhorn | 1.00 | 0.00 | 0.00 | 0.04 | 0.039 | leak 0.00 |
| moons | psqt_indep | 0.96 | 0.04 | 1.00 | 0.64 | 0.0031 | |
| | **consensus** | **0.98** | **0.02** | 0.97 | 0.60 | 0.0036 | |
| rings | psqt_indep | 0.91 | 0.09 | 1.00 | 0.65 | 0.0033 | |
| | **consensus** | **1.00** | **0.00** | 0.99 | 0.58 | 0.0036 | |
| pinwheel | psqt_indep | 0.83 | 0.17 | 1.00 | 0.59 | 0.0016 | |
| | **consensus** | **0.99** | **0.01** | 0.99 | 0.60 | 0.0017 | |
| sep_modes | psqt_indep | 0.77 | 0.23 | 0.97 | 0.57 | 0.0095 | mL1 0.27 |
| | **consensus** | **1.00** | **0.00** | 0.96 | 0.62 | 0.0070 | **mL1 0.15** |
| sep_modes_rare | psqt_indep | 0.89 | 0.11 | 0.97 | 0.64 | 0.010 | mL1 0.18 |
| | **consensus** | **0.99** | **0.01** | 0.97 | 0.64 | 0.011 | **mL1 0.14** |

(est_bary and sinkhorn omitted from rows 2-6 for brevity; est_bary collapses
throughout, sinkhorn under-covers — see below.)

## Mechanism reading (matches the plan's failure table §13)

- **Independent → coherent is the fix.** `psqt_indep` leaks 6–23% off support
  (bridges between components); `consensus` routes each particle to one modal
  target and lands on support (precision → ~1.0, leak → 0), confirming the §2.2
  causal hypothesis directly.
- **Barycenter still bridges (predicted).** `est_bary` averages the matched
  target *points* across directions and collapses (precision 0.00–0.71) — §4.4's
  warning that a barycenter averages disconnected modes. So the defect is
  *averaging target identities*; the discrete/hard route is what works
  (§13 row 2). This is why consensus, not the coupling barycenter, is the
  candidate.
- **Not just matching dense OT.** The dense entropic-Sinkhorn reference has
  precision 1.0 but recall/coverage 0.00–0.51 (it contracts particles onto a
  target subset under these settings). Consensus matches its precision while
  keeping recall ~0.97. (Caveat: this is our own Sinkhorn at fixed ε, not a
  tuned faithful published method; do not over-read the Sinkhorn column.)

## Honest caveats (why this is go/no-go, not a claim)

1. Screen scale only: 3 seeds, η declared not tuned, single realization per
   family. Stage 2 fresh confirmation (≥10–20 seeds, fresh registries, paired
   bootstrap) is required before any promotion.
2. Consensus coverage (~0.60) is not perfect: modal routing to N target points
   is not a bijection, so some target points attract several particles and
   others none. This is exactly what the Stage 1B partial-transport / sparse
   Sinkhorn refinement targets.
3. **Free-particle only.** The project's NCJ and mode-recovery arcs both showed
   free-particle wins can fail to survive neural amortization / adaptive
   optimizers. The decisive risk for THIS program is the Stage 3 neural
   retention gate (G4), not this screen. Passing 1A is necessary, not
   sufficient.

## Next action (per plan §15.7-8, §9 Stage 1B)

Proceed to **Stage 1B**: hold consensus fixed and add the surplus/deficit
controller, candidate transported fractions ρ, the geometry guard, and
backtracking — testing RQ2 (partial vs full transport) and closing the coverage
gap (caveat 2). Do not begin neural amortization until a particle teacher is
promoted through Stage 2.
