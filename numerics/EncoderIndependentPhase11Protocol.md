# Encoder-Independent Kernel Drifting — Phase 11 protocol

## Whiten the regression metric

*Frozen pre-outcome design. Source: `EncoderIndependentTailDestruction.md`.
Results go to `EncoderIndependentPhase11Results.md`.*

---

## 1. Why this phase exists

The tail-destruction pass established the chain end to end:

- the field is **not** tail-blind — it carries 0.2893 tail energy at the
  generator's cloud against the cloud's own 0.0182, **16× more than the cloud
  has**, and at real data it matches the data's tail exactly (0.1649 vs
  0.1637);
- the generator **starts** with a healthy tail (0.221, comparable to real
  data's 0.164) and **training destroys it ~29× within the first 100 steps**;
- equilibrium tail ≈ 0.004, which the Phase-10 law maps to second moment
  0.27–0.32 — and the generator sits at 0.32.

So the target is fine and the **metric** is the problem. `‖f − T‖²` weights
each direction by its own variance, so trailing directions contribute almost
nothing to the loss and an imperfect fit drops them first.

Every intervention this program has tried — R11, the Phase-10 penalties —
acts on the *target*. This one acts on the metric, and it is the first
derived from a measured cause rather than from the symptom.

---

## 2. The intervention

Replace the geometry loss with

```
L = ‖ Σ_γ^(−1/2) ( f − T ) ‖²,   Σ_γ = γ·Σ + λI,   λ = max((1−γ)·tr(Σ)/d, φ·tr(Σ)/d)
```

with `Σ` the **detached** batch output covariance — a metric, not a second
objective — and `φ = 1e-3` a declared floor. The floor is load-bearing: the
output covariance has rank ≤ 255 in 768 dimensions, so without it the ~512
unspanned directions would be amplified without bound.

`γ = 0` reproduces the current recipe exactly up to a constant factor, which
Adam is invariant to (R24). Applied through the SVD of the centred batch, so
no `d × d` matrix is ever formed.

| arm | γ | R11 |
|---|---|---|
| W0 | 0 *(baseline = current recipe)* | off |
| W1 | 0.5 | off |
| W2 | 0.9 | off |
| W3 | 0.99 | off |
| W2R | 0.9 | **on** |
| E1 | — | **on** *(R11 alone, the incumbent)* |

CIFAR-16, target ESS 0.9, field cloud 256, Adam/2e-3, 3 fresh seeds
(`MASTER_SEED + 20000..`), 600 steps. No tuning: the γ grid and the floor are
declared here and swept in full.

---

## 3. What is measured, and in what order

**The primary readout is the tail trace, not the score.** The hypothesis is
about tail destruction, so the first question is whether the tail still
collapses in the first 100 steps. Every arm logs the output tail at
initialization and every 50 steps.

Declared predictions, in order:

1. **the tail should not collapse** — with γ > 0 the step-100 tail should be
   materially above the baseline's ~0.005;
2. the equilibrium tail should rise toward the field's own 0.16–0.29;
3. **and the second moment should follow, with no scale intervention at
   all**, by the Phase-10 law.

### 11 gate

> **If a whitened arm without R11 reaches a second-moment ratio inside
> `[0.7, 1.3]` and an ED² within 25% of the best R11 cell, the metric fix
> supersedes R11** — the program's empirical reform replaced by one derived
> from a measured cause.

Same gate R11 has now survived in Phases 6, 7, 8 and 10.

---

## 4. Declared failure branches

- **Tail rises, second moment follows, gate fires** → the mechanism is
  confirmed and R11 is superseded.
- **Tail rises, second moment does *not* follow** → this **refutes the
  Phase-10 law's causal reading**, and the question returns to the field.
  This is the most informative failure and the reason the tail trace is
  logged separately from the score.
- **Tail does not rise** → whitening did not do what it was derived to do;
  report the realized metric conditioning (`λ`, the eigenvalue span) before
  concluding anything about the mechanism.
- **Quality degrades while the tail rises** → report as a trade, not a win;
  the tail is a means, and ED² is the outcome.

## 5. What Phase 11 cannot conclude

- Nothing about ImageNet, FID, or the paper's trained model.
- "Spectral bias" remains the *reading* of the tail-destruction measurement;
  this phase tests the intervention it implies, not the reading directly.
- A positive result is scoped to CIFAR-16, raw pixel geometry, one generator
  family, target ESS 0.9, 3 fresh seeds, 600 steps.
- The anchor stays disabled; the geometry thread stays closed.
