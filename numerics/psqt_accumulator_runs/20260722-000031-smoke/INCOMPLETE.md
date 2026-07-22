# Incomplete development artifact

This first smoke attempt completed its numerical cells but stopped before
publishing `summary.json`, `manifest.json`, or `RESULTS.md` because a NumPy
boolean was not JSON serializable.  It is not a result artifact and should be
ignored.  The serialization issue was fixed and the successful smoke artifact
is `../20260722-000930-smoke`.

