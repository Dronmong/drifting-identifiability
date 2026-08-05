"""ASFD: anchored self-feature drifting.

Specification: ``numerics/AnchoredSelfFeatureDriftingSpecification.md``.

The generator supplies its own training geometry.  A frozen copy of the
CAP-EMF-1 trunk provides feature-space neighbourhoods for a drifting field,
while an independently squared **raw-pixel** Laplace energy carries correctness
so that no collision in the learned feature map can become an equilibrium.

Three properties this package must never lose, each of which has already cost
this program a run when it was assumed rather than tested:

* freezing parameters is **not** the same as detaching inputs -- the generated
  branch needs the frozen trunk's input Jacobian or the whole mechanism is a
  no-op that still trains and still logs a falling loss;
* fields are squared **before** they are averaged, on every index, or two wrong
  fields cancel;
* the dependency manifest is an explicit list, never a directory glob.
"""

ASFD_PHASE = "asfd"
ASFD_DEVELOPMENT_UNITS = (910, 911, 912)
