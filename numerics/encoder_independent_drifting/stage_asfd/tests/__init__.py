"""ASFD regression suite.

Deliberately absent from ``stage_asfd/artifacts.py:_DEPENDENCIES``: these do not
execute during training, and binding test files to a frozen artifact is exactly
what broke the B1 freeze and cost the B2.5 continuation.
"""
