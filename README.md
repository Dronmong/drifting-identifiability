# Drifting Identifiability

Lean and mathematical infrastructure for finding conditions under which the
drifting field from *Generative Modeling via Drifting* identifies a target
distribution.

## Setup

The project pins Lean and Mathlib through `lean-toolchain` and
`lake-manifest.json`.

```powershell
lake update
lake build
./scripts/Check.ps1
```

`Check.ps1` runs the trust-boundary audit, a warning-free Lean build,
conditional-module compilation, and the promoted-theorem axiom audit.
To inspect the trusted dependencies of a theorem without editing a Lean file:

```powershell
./scripts/PrintAxioms.ps1 -Declaration Fully.Qualified.theoremName
```

## Project map

- `papers/2602.04770v2.pdf` — source paper.
- `DriftingIdentifiability/Paperaxioms.lean` — reviewed paper definitions and
  axioms, frozen by a SHA-256 trust manifest.
- `DriftingIdentifiability/TrustedBoundary.lean` — exact/asymptotic targets and
  nonvacuity requirements.
- `DriftingIdentifiability/PopulationIdentifiability.lean` — promoted
  paper-native theorem for genuine finite-basis probability measures, including
  finite-probe and full-support population-energy forms.
- `DriftingIdentifiability/PopulationCandidate.lean` — preliminary
  finite-family candidate plus the accepted full-setup candidate, its
  legitimacy theorem, and its `IdentifiesAtZero` theorem.
- `DriftingIdentifiability/FiniteGrouping.lean`,
  `PaperFiniteIdentifiability.lean`, and `FiniteStability.lean` — coefficient
  algebra, frame bounds, stability, dimension obstructions, and certificates.
- `DriftingIdentifiability/EmpiricalFrameBound.lean` — actual integral-induced
  empirical interactions, including the general-`m` Gaussian Vandermonde
  construction and two-atom specializations.
- `DriftingIdentifiability/PracticalModelClasses.lean` and
  `SmoothBumpBasis.lean` — higher-dimensional/adaptive-probe infrastructure, a
  two-atom paper-Laplace theorem, and a concrete non-atomic two-bump theorem.
- `DriftingIdentifiability/Algorithm2SNIS.lean`,
  `DeletedEstimatorConsistency.lean`, and `DenominatorTail.lean` —
  fixed-anchor/sample-split finite-sample theory. These do not yet cover the
  paper's coupled reuse `x = y_neg`.
- `DriftingIdentifiability/FeatureSpaceIdentifiability.lean` and
  `CFGAffine.lean` — feature-law and signed/affine CFG guardrails.
- `DriftingIdentifiability/Extensions.lean` — opt-in root for audited Sinkhorn
  extension research; these are not paper claims.
- `DriftingIdentifiability/Conditional.lean` — opt-in conditional
  Gaussian/RKHS research. Its synthetic or externally axiomatized results do
  not support promoted claims.
- `DriftingIdentifiability/AGENTS.md` — mandatory workflow and anti-cheating
  rules.
- `DriftingIdentifiability/ResearchStatus.md` — authoritative scope and open-gap
  ledger.
- `DriftingIdentifiability/WrittenProof.md` and `LoggedFailures.md` — written
  proof and permanent failure log.
- `scripts/Check.ps1` — complete local verification entry point.
- `scripts/PrintAxioms.ps1` — theorem dependency inspection.

## Current scope

The exact restricted population result is verified: under finite-basis,
regularity, and positive interaction-frame conditions, zero normalized
population mean-shift drift identifies the represented probability measures.
The paper's stronger end-to-end practical and asymptotic questions remain open:
the current sampling theorems freeze or sample-split anchors, while the paper
reuses the random negative batch as anchors; selected proof probes need not be
training anchors; and normalized multi-temperature cancellation has not yet
been ruled out.

## Trust policy

No module may add `axiom`, `constant`, `opaque`, `sorry`, `admit`, or `sorryAx`.
`Paperaxioms.lean` is the sole exception and is checked against an exact axiom
allowlist and `.trusted/Paperaxioms.sha256`. Changes to either trusted file
require explicit review and approval.

The desired implication `V = 0 → p = q`, or any equivalent uniqueness claim,
must be proved rather than introduced as an assumption.
