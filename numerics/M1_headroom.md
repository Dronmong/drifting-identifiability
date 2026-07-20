# M1 — headroom precondition (2026-07-20)

*Roadmap: `ModeRecoveryRoadmap.md` phase M1, design invariant **L2** ("verify
headroom first"). Probe: `m1_headroom.py`. Metrics: `mode_recovery.py`.*

## Verdict

**Headroom confirmed in every tested regime.** Under missing-mode initialization
(all particles start in one mode), no single fixed Laplace bandwidth — each given
its best shot via an `η` sweep — can both **reach** the distant modes and
**resolve** them. This is a real, large coverage deficit for a multi-scale field
to attack, so the program has a genuine problem worth solving (unlike the
aggregate-ED2 setting, where the tuned baseline was already near-optimal).

## Evidence (paper field, missing-mode init, best over η)

`reach` = basin coverage (mass in the Voronoi basin, needs coarse τ) ·
`resolve` = σ-precision coverage (needs fine τ) · `spread` = cloud std / target
std (did the swarm split).

**K=16, d=2, L/σ=33**

| τ | reach | resolve | spread | reading |
|---:|---:|---:|---:|---|
| ≤0.10 (fine) | 0.06 | 0.06 | 0.02 | swarm **frozen** at the start mode — can't even move |
| 0.20 | 0.38 | 0.31 | 0.57 | partial |
| 0.40 | **1.00** | 0.31 | 0.97 | reaches every basin, resolves only 31% |
| 0.80 (coarse) | **1.00** | 0.06 | 1.02 | reaches every basin, resolves ~nothing (over-smoothed) |

Best resolution at full reach (reach ≥ 0.9): **0.31** → HEADROOM.

**K=16, d=5, L/σ=33:** best resolution at full reach **0.00** → HEADROOM.
**K=32, d=2, L/σ=50:** best resolution at full reach **0.16** → HEADROOM.

## Mechanism (why one bandwidth cannot win)

Two distinct failures bracket the bandwidth axis, exactly as the theory predicts:

- **Fine τ freezes the swarm.** The pull toward a mode at distance `L` decays
  like `e^{-L/τ}`; with `τ ≲ σ` the cloud cannot feel any distant mode, and
  because every particle starts identical it receives identical drift — a
  homogeneous blob that neither moves nor splits (spread ≈ 0.02). This is the
  fission-instability / homogeneous-swarm failure certified in
  `LaplaceFissionInstability.lean`.
- **Coarse τ reaches and splits but cannot resolve.** With `τ ~ L` the cloud
  feels all modes, spreads (spread ≈ 1.0), and its symmetry breaks so it
  populates every basin (reach → 1.0) — but the same wide kernel over-smooths,
  so particles sit spread across each basin rather than concentrating on the
  mode (resolve ≤ 0.31).

The **gap between the reach-optimal τ (~0.4–0.8) and the resolve-optimal τ (~σ)**
is the headroom. A field carrying *both* scales — coarse for reach + split, fine
for resolution once particles are in the right basin — is the natural remedy
(the M2 candidate), and it is the atlas two-bandwidth mechanism (A3) aimed at the
right scoreboard.

## Consequence for the next steps

- The precondition (L2) is satisfied → proceed to freeze fresh
  validation/test registries (M1b) drawn from this regime family (large `L/σ`,
  `K ∈ {8,16,32}`, `d ∈ {2,5,10}`, adversarial inits) and build the M2 two-scale
  field.
- The pre-registered coverage gate (M3) should target lifting **resolution at
  full reach** from ≈0.16–0.31 toward ≈1.0 while keeping reach at 1.0 — a large,
  well-defined target that no single bandwidth achieves.
- Reproduce with: `uv run --with numpy --with scipy python numerics/m1_headroom.py`.
