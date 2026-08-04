"""CAP-EMF-1: the one-call raw-pixel Euler Mean Flow capability foundation.

Protocol: ``numerics/EncoderIndependentCAPEMF1Protocol.md``.

This package trains **no correction of any kind**.  Laplace, spectral and
Sinkhorn terms are absent by construction so that a failed foundation can never
be mistaken for a failed correction.

It also supplies the frozen feature trunk consumed by
``numerics/AnchoredSelfFeatureDriftingSpecification.md``, which is why feature
taps and their parity test exist here from the start rather than being
retrofitted after the run.
"""

CAP_PHASE = "cap-emf-1"
CAP_UNIT = 900
