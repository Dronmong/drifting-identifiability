# Encoder-Independent Kernel Drifting — the tail is destroyed, not absent

## The regression discards what the field supplies, and the Phase-10 law does the rest

*Deeper pass on Phase 10. Code: `diagnose_phase11.py`. Artifact:
`phase11_probe.json` (+ `.sha256`). Development seeds
(`MASTER_SEED + 19000..`), 3 seeds, 600 steps, CIFAR-16, target ESS 0.9.
Nothing feeds a gate.*

---

## 0. The finding

Phase 10 left one puzzle: the generator's spectral tail is stuck near 0.005
against real data's 0.138, and nothing raises it — not latent dimension (N4),
not capacity (8A), not an explicit tail penalty (only 4× at weight 1.0).

My hypothesis was that the field is **tail-blind** — that `V`, built from a
kernel on raw pixels, carries no energy in the data's trailing directions, so
the teacher never asks for tail. **That is refuted, and the truth is the
opposite.**

| cloud | cloud tail | **field tail** | ratio |
|---|---:|---:|---:|
| **generator** | 0.0182 | **0.2893** | **15.9×** |
| particles | 0.2627 | 0.1948 | 0.74× |
| real data | 0.1637 | 0.1649 | **1.01×** |
| white noise | 0.8945 | 0.0815 | 0.09× |

The field is generous with tail. At real data it carries **exactly** the
data's own tail fraction (0.1649 against 0.1637). At the generator's cloud it
offers **16× more tail than the cloud has**.

So the teacher asks for tail every single step, and the generator does not
keep it:

> **The generator's tail is not missing. It starts at 0.221 — comparable to
> real data's 0.164 — and training destroys it, ~29× within the first 100
> steps.**

| step | 0 | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---:|---:|---:|---:|---:|---:|---:|
| output tail (seed 0) | **0.2273** | 0.0079 | 0.0078 | 0.0068 | 0.0055 | 0.0080 | 0.0040 |
| output tail (seed 1) | 0.2210 | 0.0046 | 0.0050 | 0.0051 | 0.0051 | 0.0041 | 0.0029 |
| output tail (seed 2) | 0.1975 | 0.0073 | 0.0063 | 0.0057 | 0.0046 | 0.0049 | 0.0047 |

The collapse is essentially complete by step 100 — a per-step retention of
about **0.96** — and the trace is flat thereafter. This is not a slow drift
to an attractor; it is a fast, early destruction followed by a balance
between a field that keeps supplying tail and a regression that keeps
removing it.

---

## 1. The mechanism, end to end

Every link is now measured:

1. **The regression discards tail.** The generator loses ~4% of its tail per
   step. Least-squares fitting of a smooth map onto point targets
   preferentially reproduces high-variance directions — the trailing
   directions are exactly what an imperfect fit drops first.
2. **The field cannot compensate.** It supplies tail generously in *relative*
   terms (16× the cloud's fraction) but in *absolute* terms it is a small
   correction: the drift is RMS-normalized to ‖ηV‖ ≈ 0.5 against an output
   norm of order 9, so each step adds only a few percent of the tail energy
   it would take to close the gap.
3. **Equilibrium tail ≈ 0.004**, where destruction and supply balance.
4. **The Phase-10 law converts that into a scale.** A cloud with tail 0.004
   balances the field's radial component near second moment 0.27–0.32.
5. **The generator sits at 0.32.**

That is a complete chain from a property of least-squares regression to the
number this program has been chasing since Phase 3.

### It explains the standing negatives, and now says *why*

| observation | why |
|---|---|
| capacity is inert (36×) | spectral bias is not a capacity limit — a wider net still fits dominant directions first |
| latent dimension is inert (8→512) | same; the constraint is the fit, not the manifold |
| teacher-fitting quality does not help (2.3×) | fitting the same target *better* in the least-squares metric fits the *dominant directions* better |
| free particles are fine | no regression, so no spectral bias — they keep the tail their initialization gave them |
| R11 works but has low tail (0.0496) | it overrides the scale directly rather than restoring shape — exactly the Phase-10 outlier |

---

## 2. Causal confirmation, and an honest limit on the law

If tail sets the equilibrium scale, then manipulating tail *at
initialization* must move the equilibrium. Free particles, identical field
and budget, differing only in where they start:

| start | tail: start → end | **2nd moment** | ED² |
|---|---|---:|---:|
| isotropic noise *(the usual start)* | 0.822 → 0.405 | **0.995** | 0.068 |
| low-tail noise | 0.000 → 0.0001 | **0.559** | 0.456 |
| low-tail data | 0.000 → 0.0003 | **0.710** | 0.198 |

Two things follow.

**The causal direction holds.** Particles that begin without a tail never
build one — 0.0000 → 0.0001 over 600 steps — and they land far below the
0.995 they otherwise reach. The tail a particle system ends with is
essentially the tail it started with.

**But the law under-predicts these clouds.** Phase 10's law maps tail ≈ 0 to
a second moment of ~0.18; the measurement gives 0.559 and 0.710. And the two
zero-tail starts differ from each other by 0.15 despite having the *same*
tail. So **tail alone does not determine the equilibrium** — structure within
the retained subspace matters too. The law is directionally right and
quantitatively incomplete outside the family it was fitted on (rescaled and
spectrally reweighted data clouds). That limit is stated here rather than
discovered later.

---

## 3. The next experiment

### The intervention the mechanism implies: whiten the regression

Every intervention this program has tried acts on the *target* — R11 rescales
it, the Phase-10 penalties add terms to it. The measurement above says the
problem is not the target at all. The target already carries 16× the tail the
cloud has. **The problem is the metric the regression uses to chase it.**

`‖f − T‖²` weights every direction by its own variance, so the trailing
directions contribute almost nothing to the loss and are dropped first. The
principled fix is to make the regression direction-blind:

```
L = ‖ Σ^(−1/2) ( f − T ) ‖²
```

with `Σ` the batch's output covariance (shrinkage-regularized, and detached
so it is a metric and not a second objective). Equal weight per direction
means the tail is fitted as hard as the bulk.

**This is the first intervention derived from a measured cause rather than
from the symptom.** It predicts specifically:

- the tail should *not* collapse in the first 100 steps — that trace is the
  primary readout, not the score;
- the equilibrium tail should rise toward the field's own 0.16–0.29;
- and by the Phase-10 law, the second moment should follow **without any
  scale intervention at all**.

Proposed arms: full whitening, shrinkage-interpolated whitening
(`(1−γ)I + γΣ` for declared γ), and a top-k/tail split weighting, each
crossed against R11 off/on, with the same supersession gate R11 has now
survived four times.

**What would refute it:** the tail rises but the second moment does not
follow. That would break the Phase-10 law's causal reading and send the
question back to the field.

### Why not continue on the shape penalties

Phase 10's penalties reached tail 0.0199 at weight 1.0 by *fighting* the
regression every step. Whitening removes the force they were fighting, so it
should be far cheaper — and it needs no weight to be tuned, which the
penalties did.

---

## 4. Scope and caveats

- "Spectral bias of least-squares regression" is the *reading* that fits the
  measurements; the whitening experiment is what would test it directly.
  This pass measures the destruction, not its cause inside the optimizer.
- The field-tail measurements decompose `V` about zero and the cloud tails
  about their own mean; they are comparable as fractions but not as energies.
- R3's two zero-tail starts disagree by 0.15, so the Phase-10 law is known
  incomplete on clouds outside its fitted family (§2).
- One dataset, one geometry, one bandwidth, 3 seeds.
- Nothing here concerns ImageNet, FID, or the paper's trained model. The
  anchor stays disabled; the geometry thread stays closed.

## 5. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.diagnose_phase11 `
  --stage all --seeds 3 --steps 600
```
