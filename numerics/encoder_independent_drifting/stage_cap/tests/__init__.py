"""CAP-EMF-1 regression suite.

These modules are deliberately **absent** from
``stage_cap/artifacts.py:_DEPENDENCIES``.  They do not execute during training,
and binding test files to a frozen artifact is exactly what broke the B1 freeze
and cost the B2.5 continuation.
"""
