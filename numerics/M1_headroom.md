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
- The open question the data raises is whether the hard-regime deficit is a
  *field-design* opportunity at all, or a *capacity* limit (too few
  particles/steps to populate many modes from a degenerate start). Distinguishing
  these — hold the field fixed, scale N and steps — is the honest next diagnostic
  before proposing any new candidate. If it is capacity-bound, there is no
  field-design win to be had, matching every prior program.
- Reproduce: `uv run --with numpy --with scipy python numerics/m1_headroom.py`.
