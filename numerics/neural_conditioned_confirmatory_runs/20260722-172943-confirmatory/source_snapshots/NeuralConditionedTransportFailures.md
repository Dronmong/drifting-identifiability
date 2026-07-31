# Neural conditioned-transport confirmation failures

## V1 registry: invalidated before endpoint evaluation

**Registry hash:** `102b808acb8b096e8a4b03924bbf51d783bcbcfc2969cb20aeeee87f3ec916c3`  
**Freeze hash:** `3ec5af39b9741a1c4c0cc3b4cbdfd65c78fe3470abb68589725dfbf730ae6e07`

The first execution trained the sixteen 2D targets in memory and stopped at
the frozen-atlas integrity check for the first 4D target. It wrote no result
artifact and printed no endpoint metric. The check reconstructed a
`QuantileAtlas`, which normalizes its stored directions, then demanded bitwise
equality after recomputing projected quantiles from those re-normalized rows.
The extra floating-point normalization changed projections at roundoff scale.

Repair: recompute the audit table directly from the exact direction array in
the frozen NPZ, before `QuantileAtlas` normalization. The equality check remains
bitwise; it is not weakened to a performance-dependent tolerance.

Under the preregistered contamination rule, v1 is abandoned despite the lack
of displayed outcomes. V2 uses master seed `20260802`, distinct filenames, and
a new complete source/atlas freeze. V1 files remain preserved for audit.

## V2 registry: invalidated during artifact serialization

**Registry hash:** `09f0e0c99a0bd7ce1d803bccb5ad63f6c3d511aa5c27e245ddadb955134345da`  
**Freeze hash:** `308d0db3ffbbd9bf1922be538e2e5f359606beaa12fb97a78b30782d5aee570d`

V2 passed every frozen-atlas check and trained all 640 cells. Before any
endpoint metric or gate was printed, CSV serialization failed: field names
were taken from the first baseline row, while candidate rows additionally
contained four transport diagnostics. No result artifact was completed.

Repair: form the CSV schema from the ordered union of fields across all rows.
A heterogeneous 640-row artifact fixture must pass before the next registry is
generated. V3 uses master seed `20260803`; all v2 files and the partial artifact
remain preserved and excluded.
