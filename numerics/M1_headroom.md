# M1 — headroom precondition (2026-07-20, CORRECTED)

*Roadmap: `ModeRecoveryRoadmap.md` phase M1, design invariant **L2**. Probe:
`m1_headroom.py`. Metrics: `mode_recovery.py`.*

> **Correction notice.** The first version of this file (commit 1f7ec8b) claimed
> broad headroom including at K=16, d=2. That was **confounded**: it selected the
> step size `eta` to maximize *reach* and then reported *resolution* at that
> `eta`. Selecting `eta` for the actual objective (resolution) removes the
> apparent headroom at K=16, d=2 — a single tuned bandwidth solves it. This file
> is the corrected analysis and supersedes the earlier claim.

## Verdict (corrected)

1. **No headroom in easy regimes.** At K=16, d=2 a single tuned Laplace
   bandwidth achieves reach = 1.00 and resolve = 1.00 from a missing-mode start.
   There is nothing to beat.
2. **Genuine headroom exists only in hard regimes** (K ≥ 32, or d ≥ 5): the best
   single bandwidth resolves only a small fraction of modes.
3. **But the proposed additive two-scale field does NOT close that headroom.**
   Where the baseline fails, the two-scale field fails too: it sometimes lifts
   *reach* but never *resolution*, the actual target.

## Evidence (best over τ and η, selecting for resolution)

| Regime | single (reach, resolve) | two-scale (reach, resolve) | headroom | 2-scale resolve gain |
|---|---|---|---|---:|
| K=16 d=2 L/σ=33 | (1.00, **1.00**) | (1.00, 1.00) | no | +0.00 |
| K=32 d=2 L/σ=50 | (0.41, 0.22) | (0.59, 0.28) | yes | +0.06 |
| K=64 d=2 L/σ=67 | (0.22, 0.06) | (0.23, 0.09) | yes | +0.03 |
| K=16 d=5 L/σ=33 | (0.19, 0.12) | (0.44, 0.12) | yes | +0.00 |
| K=32 d=5 L/σ=50 | (0.28, 0.12) | (0.28, 0.06) | yes | −0.06 |
| K=16 d=10 L/σ=33 | (0.25, 0.06) | (0.44, 0.06) | yes | +0.00 |

Two-scale resolution gains are zero within run-to-run noise across every hard
regime. Reach improves in some cases (e.g. K=16 d=5: 0.19 → 0.44) without a
matching resolution improvement — the cloud reaches more basins but still does
not concentrate onto the modes.

## Why the two-scale idea does not work here

The reach-vs-resolve framing was incomplete. Adding a fine scale only helps
resolution *for particles that already sit in the correct basin*. In the hard
regimes the binding constraint is upstream of that: from a single degenerate
start, the swarm cannot **split** into enough correctly-placed sub-clusters —
the homogeneous-swarm / fission bottleneck (`LaplaceFissionInstability.lean`),
made worse by capacity limits (K = 32–64 modes with N = 128 particles is only
2–4 particles per mode). A fine resolution scale does nothing for a mode the
cloud never populated, so it cannot lift resolution where reach/split already
failed. This is the same squeeze the ledger's meta-pattern describes: where the
baseline is fine (K=16 d=2), the candidate is unnecessary; where the baseline
fails (hard regimes), the candidate fails too — the bottleneck is
splitting/capacity, not bandwidth.

## Consequence

- The M2 additive two-scale field is **rejected** as the mode-recovery candidate:
  no resolution gain in any regime with genuine headroom.
- Do **not** proceed to freeze registries / build the full M3 gate for this
  candidate.
- **Capacity diagnostic (`m1_cap2` probe), K=32 d=2, single τ=0.4:**

  | N | steps | reach | resolve |
  |---:|---:|---:|---:|
  | 128 | 800 | 0.42 | 0.20 |
  | 128 | 2400 | 0.84 | 0.27 |
  | 384 | 800 | 0.41 | 0.16 |
  | 384 | 2400 | 0.83 | 0.23 |
  | 1024 | 800 | 0.42 | 0.19 |

  **More particles do not help at all** (resolve ~0.16–0.20 flat across
  N = 128/384/1024). **More steps lift reach** (0.42 → 0.84) **but not
  resolution** (plateau ~0.23–0.27). So resolution is *barriered* around ~0.25 —
  it is neither a capacity limit (N is inert) nor fixed by running longer.

- **Why additive two-scale cannot break the barrier.** The fine field
  `V(τ_fine≈3σ)` is active only within ~`3σ` of a mode; but a coarse field
  spreads particles across the whole basin (radius ~0.2–0.3 ≫ 3σ). Most
  particles sit in a "no-man's-land" — inside the basin, outside the fine
  field's reach — so nothing pulls them onto the mode. The two scales do not
  overlap in their active regions, which is exactly why the additive field lifts
  reach but not resolution.

- **Net:** the hard-regime deficit is a genuine dynamical barrier, but it is
  neither capacity-bound nor closed by the proposed candidate. The only untested
  bridge is a **coarse-to-fine anneal** (shrink τ so the active region contracts
  with the cloud), which the ledger flags as dimension-fragile (A2: 2-D win, 5-D
  loss). Absent evidence that annealing survives high dimension, the
  mode-recovery program is following the ledger's meta-pattern (candidate fails
  where the baseline fails) and should not proceed to a frozen gate without a new
  idea that demonstrably breaks the resolution barrier.
- Reproduce: `uv run --with numpy --with scipy python numerics/m1_headroom.py`.
