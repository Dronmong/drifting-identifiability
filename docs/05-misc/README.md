# 05 — Misc / infrastructure

Trust boundary, axioms, failure records, build/audit scripts, and top-level
status/presentation material.

## Trust boundary & axioms

- [Paperaxioms.lean](../../DriftingIdentifiability/Paperaxioms.lean) — the **only** module allowed to declare axioms (standard external results); everything else is axiom-free
- [TrustedBoundary.lean](../../DriftingIdentifiability/TrustedBoundary.lean) — the single module permitted to import `Paperaxioms`; all new files import through it
- Aggregator: [DriftingIdentifiability.lean](../../DriftingIdentifiability.lean)

## Build & audit scripts

- `scripts/Check.ps1` — full build + trust audit + promoted-theorem axiom audit (must stay green)
- `scripts/TrustAudit.ps1`, `scripts/AxiomAudit.ps1`, `scripts/PrintAxioms.ps1`

## Failure / counterexample records

- [FailureCases.lean](../../DriftingIdentifiability/FailureCases.lean), [CounterexampleHarness.lean](../../DriftingIdentifiability/CounterexampleHarness.lean)
- [LoggedFailures.md](../../DriftingIdentifiability/LoggedFailures.md) — dead ends and falsified conjectures

## Top-level status & presentation

- [ResearchStatus.md](../../DriftingIdentifiability/ResearchStatus.md) — project-wide status (includes the honest program conclusion)
- [DynamicsRoadmap.md](../../DriftingIdentifiability/DynamicsRoadmap.md) — dynamics/empirical roadmap
- [AGENTS.md](../../DriftingIdentifiability/AGENTS.md), root [README.md](../../README.md), [audit.md](../../audit.md), [presentation.md](../../presentation.md)
