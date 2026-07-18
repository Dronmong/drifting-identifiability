# G4: foliation--cancellation endgame

## Scope

This is the live implementation note for G4 of `LaplaceRnRoadmap.md`.  The
target remains the raw Euclidean Laplace field for arbitrary probability
measures.  G1--G3 supply the radial core (all finite dimensions, including the
line) and atom alignment.  G4 must prove that every non-radial component of a
zero-drift pair either already has proportional normalizers or reduces to that
radial core.

No external identifiability theorem and no new axiom is used here.

## Notation

For a probability measure `r`, put

```text
Z_r(x)   = integral exp(-norm(x-y)/tau) dr(y),
D_r(x)   = integral exp(-norm(x-y)/tau) (y-x) dr(y),
psi_r(x) = integral tau (norm(x-y)+tau) exp(-norm(x-y)/tau) dr(y).
```

The already certified potential identity is `grad psi_r = D_r`.  Under zero
drift, define the positive normalizer ratio

```text
R(x) = Z_p(x) / Z_q(x).
```

Then `D_p = R D_q`, hence `grad psi_p = R grad psi_q`.

## P1: rigorous local reduction

**Implementation status (2026-07-18): the classical P1 bridge is closed,
axiom-free.**  `LaplaceDisplacementHessian.lean` proves that the integrated
displacement field is Fréchet differentiable at every point, including atoms;
its point-source Hessian is uniformly bounded, its integrated Hessian is
symmetric, and the companion elliptic equation holds pointwise.  Thus the
earlier Rademacher/Alexandrov boundary is obsolete.  In
`LaplaceFoliationChart.lean`, the actual ratio `Z_p/Z_q` is differentiable on
`{D_q != 0}`, the differentiated alignment and tangent-annihilation statement
are instantiated for the actual measures, the exact measure-level
cancellation equation is proved, and Mathlib's implicit-function theorem
constructs the genuine regular leaf chart with first coordinate `psi_q`.
`LaplaceFoliationFactorization.lean` restricts it to a genuine `C^1` inverse
chart, proves vertical-slice constancy by the convex mean-value theorem, and
derives actual differentiable scalar factorizations for `Z_p/Z_q` and
`psi_p`.

The P1 implementation has two layers.

### Measure layer

The following facts require no leaf geometry.

1. `R` is positive and continuous.  Positivity is the strict positivity of
   Laplace normalizers; continuity is dominated convergence, already proved in
   `LaplaceConeExtraction.lean`.
2. Zero drift gives `D_p = R D_q` pointwise by dividing the certified identity
   `Z_q D_p = Z_p D_q` by `Z_q > 0`.
3. Therefore the Frechet derivative of `psi_p` is exactly `R` times the
   Frechet derivative of `psi_q` at every point.

These statements are implemented without any regularity assumption on the
measures beyond finiteness/probability.

### Leaf chart layer

At a regular point `D_q(x) != 0`, the submersion theorem supplies a local chart
whose first coordinate is `psi_q`.  Symmetry of the two Hessians, applied to
the derivative of `grad psi_p = R grad psi_q`, shows that `dR` annihilates every
vector tangent to a `psi_q` leaf.  Thus, after shrinking the chart,

```text
psi_p = G(psi_q),       R = G'(psi_q).
```

For the Laplace companion potential the pointwise elliptic identity is

```text
psi_r - tau^2 Delta psi_r = (n+1) tau^2 Z_r.             (E)
```

At every twice differentiable point in the chart, the scalar chain rule gives

```text
Delta psi_p = G'(psi_q) Delta psi_q
              + G''(psi_q) norm(grad psi_q)^2.           (C)
```

Substituting (C) into (E), using `Z_p = G'(psi_q) Z_q`, and subtracting
`G'(psi_q)` times the `q` equation cancels `Delta psi_q` exactly:

```text
tau^2 G''(s) norm(grad psi_q)^2 = G(s) - s G'(s).         (F)
```

The Lean layer now discharges the chart/factorization facts for the actual
measures.  It also differentiates the ratio-factor germ to obtain
`dR(D_q)=h'(psi_q) norm(D_q)^2`, which combines directly with the pointwise
measure cancellation theorem.  None of these statements assumes
identifiability.

### Non-degenerate branch

Take two points on the same connected leaf at which the squared gradient norms
differ.  Equation (F) at both points has the same right-hand side.  Since
`tau != 0`, subtraction forces

```text
G''(s) = 0,       G(s) = s G'(s).
```

On an interval of such leaves, `G'` is constant and the second identity removes
the affine intercept, so `G(s) = c s`.  Consequently `Z_p = c Z_q` on that
region.  The actual two-point theorem is now fully formalized: within a
regular chart, two points on one leaf with different squared gradient norms
force the genuine defect `H=psi_p-R psi_q` to vanish at both points.  This is
the seed consumed by the gradient-flow theorem.  Interval/component gluing
still belongs to P3.

## P2: degenerate branch (the mathematical core)

If every connected leaf has constant `norm(grad psi_q)`, then `psi_q` is
transnormal on that region.  Reparametrizing its value gives a unit-speed
distance coordinate, so the regular leaves are parallel tubes around a focal
set.  G1 applies immediately only when the focal set is a point.

The planned discharge is deliberately two-stage:

1. derive the second-order far-field expansion of `psi_q`, `grad psi_q`, and
   their quotient uniformly in direction under a compact-support truncation;
2. use monotone truncation/tightness to remove the compact-support assumption,
   then compare the angular coefficient forced by transnormality with the
   support function of the focal set.

Before promotion, this argument must rule out non-point compact focal sets; a
mere citation to a transnormal/isoparametric classification is insufficient,
because transnormality alone does not imply constant principal curvatures.

### Tail audit: the original far-field plan is not general enough

The first stage is meaningful for compactly supported laws, and more generally
under a directional exponential-moment hypothesis such as

```text
integral exp(<u,y>/tau) dq(y) < infinity  for every unit u.
```

It is **not** currently a route for arbitrary probability measures.  The
leading coefficient is the directional tilt transform

```text
L_q(u) = integral exp(<u,y>/tau) dq(y),
```

which may be infinite for a heavy-tailed law.  More importantly, for such a
law the far field at `r u` can be dominated by the shrinking fraction of mass
near the moving probe rather than by a compact core multiplied by
`exp(-r/tau)`.  Truncation followed only by tightness does not give a uniform
error relative to that exponentially small core term.  Therefore the sentence
"remove compact support by monotone truncation/tightness" is not a proof and
must not be used in the arbitrary-measure theorem.

The safe P2 split is now:

1. prove tube rigidity for compact support / finite directional exponential
   moments, including the one-order-deeper expansion and its nonlinear sphere
   rigidity equation;
2. separately handle the tail-dominated regime, or replace the far-field pin
   by a local PDE/tube-averaging argument that is insensitive to tails.

### Seeded degenerate components are now closed

`LaplaceFoliationFlow.lean` gives a stronger local reduction that avoids
choosing a scalar leaf factorization.  Along every actual gradient curve
`gamma' = D_q`, the genuine defect

```text
H = psi_p - (Z_p/Z_q) psi_q
```

satisfies the homogeneous ODE

```text
H' = -(psi_q/tau^2) H.
```

The file proves the equation from the pointwise cancellation theorem, proves
two-sided Gronwall uniqueness, derives global Lipschitz and boundedness
estimates for `D_q`, constructs the needed local gradient curves by
Picard--Lindelof, and concludes that one zero seed propagates along the whole
regular orbit segment.  If the defect vanishes on a connected regular open
region, the full derivative of `Z_p/Z_q` vanishes there and the ratio is
constant.

This removes every degenerate component for which a zero seed has been
constructed.  The implicit-chart packaging and non-degenerate seed
construction are now complete.  The irreducible P2 question is therefore to
rule out, or classify, a **seedless fully degenerate component** on which `H`
never vanishes.  No seed-coverage or tube-rigidity statement has been
axiomatized.

Even in the first class, matching the leading support function only identifies
a candidate focal body.  The next coefficient must still prove that this body
is a point (equivalently, that the tilt transform is radial up to a linear
factor).  That classification remains the irreducible new theorem in G4.

## P3: gluing

Once P2 turns every degenerate component into a translated radial component,
G1--G3 give normalizer proportionality there.  P3 must then prove:

1. constants agree across overlapping regular leaf charts by continuity of
   `R`;
2. constants pass across branch boundaries;
3. `int {D_q = 0}` cannot separate two different constants (use the elliptic
   identity, decay, and the already certified atom-alignment interface);
4. the resulting global equality `Z_p = c Z_q` has `c = 1` by total mass and
   Euclidean Laplace smoothing injectivity.

**Implementation status (2026-07-18): critical interiors and the downstream
endgame landed.**
`LaplaceFoliationCancellation.lean` proves that both actual companion
potentials are constant on every preconnected open component of `{D_q=0}`.
`LaplaceFoliationFlow.lean` now uses the pointwise elliptic equation to prove
that both normalizers, and hence `Z_p/Z_q`, are constant on those components;
it also proves local constancy at every point of `interior {D_q=0}`.
`LaplaceFoliationEndgame.lean` proves the complete connectedness and mass
endgame: once the geometric work shows that `Z_p/Z_q` is locally constant,
the ratio is globally constant and smoothing injectivity plus total mass gives
`p=q`.  What remains in P3 is branch-boundary gluing at the boundary of the
critical set and between seeded and seedless regular components.  Critical-set
interiors, final connectedness, and mass normalization are no longer gaps.

## Acceptance

G4 is closed only when the final arbitrary-measure theorem has no leaf-chart,
tube-rigidity, radiality, or gluing hypothesis.  Intermediate structures are
certificates for staged development and must not be advertised as the final
converse.
