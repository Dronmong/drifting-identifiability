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

`Check.ps1` runs both the trust-boundary audit and a warning-free Lean build.
To inspect the trusted dependencies of a theorem without editing a Lean file:

```powershell
./scripts/PrintAxioms.ps1 -Declaration Fully.Qualified.theoremName
```

## Project map

- `papers/2602.04770v2.pdf` — source paper.
- `DriftingIdentifiability/Paperaxioms.lean` — reviewed paper definitions and
  axioms; frozen by a SHA-256 trust manifest.
- `DriftingIdentifiability/TrustedBoundary.lean` — exact and asymptotic theorem
  targets plus nonvacuity requirements for candidate conditions.
- `DriftingIdentifiability/CounterexampleHarness.lean` — finite-model and
  counterexample definitions.
- `DriftingIdentifiability/CandidateConditions.lean` — proposed conditions and
  research-stage metadata.
- `DriftingIdentifiability/PaperFiniteIdentifiability.lean` — finite-basis route
  from Appendix C.1.
- `DriftingIdentifiability/FiniteLegitimacy.lean` — machine-checked legitimacy
  witnesses for the finite route: a distinct pair before zero drift, the lift to
  distinct densities, and satisfiability of the interaction-separation premises.
- `DriftingIdentifiability/PaperDriftIdentifiability.lean` — connects the
  *actual* density-interaction drift (Appendix C.1) to the finite algebra:
  probe-wise zero drift plus nondegeneracy identifies the basis densities, with
  a specialization to the paper's mean-shift interaction kernel.
- `DriftingIdentifiability/AGENT.md` — mandatory agent workflow and anti-cheating
  rules.
- `DriftingIdentifiability/WrittenProof.md` — proof-development template.
- `DriftingIdentifiability/LoggedFailures.md` — permanent counterexample and
  failed-approach log.
- `scripts/TrustAudit.ps1` — rejects unreviewed axioms and proof escapes.
- `scripts/PrintAxioms.ps1` — prints a theorem's kernel-visible axiom
  dependencies.

## Trust policy

No module may add `axiom`, `constant`, `opaque`, `sorry`, `admit`, or `sorryAx`.
`Paperaxioms.lean` is the sole exception and is checked against an exact axiom
allowlist and `.trusted/Paperaxioms.sha256`. Changes to either trusted file
require explicit review and approval.

The desired implication `V = 0 → p = q`, or any equivalent uniqueness claim,
must be proved rather than introduced as an assumption.
