# Neural pooled-rank Phase-1 smoke result

**Status:** engineering smoke passed; standard development run paused  
**Artifact:** `neural_pooled_rank_runs/20260722-153925-smoke/`  
**Registry SHA-256:**
`7b4cc89a68d73784303b0e2bc417d53b2f91f02511099c88c3c8e6be535a3dc6`

## Engineering outcome

The four-target smoke profile completed all 28 registered rows without a
numerical divergence. The following checks passed before training:

- complete Phase-0 atlas, gradient, RSR, dropout, edge-case, and frame suite;
- Torch paper-field equality with the repository NumPy implementation;
- matched positive/negative cancellation;
- full rank and tightness of all 32 registered direction banks; and
- bitwise-paired neural initialization.

Every neural arm consumed exactly 2,048 training generator-example
evaluations. All 256 realized Apache KLL projection states were serialized in
the artifact and replayed before training.

## Endpoint smoke diagnostics

These four targets and single initialization are engineering diagnostics, not
independent evidence or a model-selection result.

| Arm | Median ED2 | Median held-out SW1 | Updates |
|---|---:|---:|---:|
| Paper neural | 0.2366 | 0.3924 | 64 |
| Minibatch SW | 0.1613 | 0.2577 | 64 |
| Exact-atlas large RSR | 0.9148 | 0.6693 | 8 |
| KLL-atlas large RSR | 0.9212 | 0.6708 | 8 |
| KLL-atlas small RSR | 0.2758 | 0.3884 | 32 |
| KLL-plus-paper hybrid | 0.9243 | 0.6721 | 8 |
| Exact free-particle ceiling | 0.0219 | 0.1061 | 30 sweeps |

The large-population neural arms were worse than both ordinary baselines on
all four smoke targets. The standard run should therefore not be launched with
the current optimizer schedule.

## Diagnosis

Three comparisons sharply localize the problem.

### KLL is not the failure

Exact-atlas and KLL-atlas large RSR were nearly indistinguishable:

```text
median ED2        0.9148 exact versus 0.9212 KLL
median heldout    0.6693 exact versus 0.6708 KLL
```

The same poor endpoint under an exact target atlas rules out KLL rank error as
the primary cause.

### The projected target is not impossible

The exact free-particle ceiling reached median ED2 `0.0219` and held-out SW1
`0.1061`. Thus the fixed direction bank and exact target table contain a strong
transport signal. The remaining gap is neural amortization/optimization.

### Equal example budgets created very unequal optimizer time

The ordinary arms used 64 optimizer updates. Large RSR used only 8 because
each update evaluates 128 generated samples twice. Small-population RSR used
32 updates and recovered much of the deficit, despite having the same KLL
target atlas. This monotone relation between update count and quality is the
strongest current explanation.

The failure is not a mathematical error in RSR: full and microbatched RSR
gradients are already equal to machine precision. It is a training-dynamics
problem. Adam's parameter displacement depends strongly on the number of
optimizer steps; equal example evaluations do not make 8 Adam steps equivalent
to 64.

## Immediate repair screen

Before changing architecture or spending the standard registry, run a
development-only learning-rate screen on these already consumed smoke targets.
The candidate multiplier should be tied to the registered update-count ratio,
not selected independently per target:

```text
ordinary updates / large-RSR updates = 8 in smoke
ordinary updates / large-RSR updates = 16 in standard
```

Test base Adam learning-rate multipliers `1, 2, 4, 8, 16` for exact-atlas
large RSR, plus `1, 2, 4` for small-population KLL RSR. The exact-atlas arm is
the clean optimizer diagnostic. Only after a stable scaling rule is chosen
should KLL and the hybrid be rerun.

This is a development repair, not an amendment to a sealed confirmation. Any
selected scaling rule must be frozen in an updated standard-development
protocol, and a later confirmation must use new targets.
