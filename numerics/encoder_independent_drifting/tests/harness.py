"""Tiny PASS/FAIL harness, matching the repository's inline `_tests` style.

Test functions are named ``test_*`` and raise ``AssertionError`` on failure,
so they are also collectable by pytest if it is ever installed.  No test
depends on a package that is not already used by the numerics suite.
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable


def run_module(name: str, namespace: dict,
               log: Callable[[str], None] = print) -> int:
    """Run every ``test_*`` callable in ``namespace``; return the fail count."""
    tests = sorted((k, v) for k, v in namespace.items()
                   if k.startswith("test_") and callable(v))
    log(f"\n=== {name} ({len(tests)} tests) ===")
    failures: list[str] = []
    for key, function in tests:
        label = (function.__doc__ or key).strip().splitlines()[0]
        try:
            function()
        except Exception as error:                      # noqa: BLE001
            failures.append(key)
            log(f"  [FAIL] {label}")
            log("         " + "".join(
                traceback.format_exception_only(type(error), error)).strip())
        else:
            log(f"  [PASS] {label}")
    if failures:
        log(f"  {len(failures)} FAILED in {name}: {failures}")
    return len(failures)


def main(name: str, namespace: dict) -> None:
    failures = run_module(name, namespace)
    if failures:
        sys.exit(1)
