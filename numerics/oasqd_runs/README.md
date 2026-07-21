# OA-SQD run ledger

These directories are immutable development artifacts. Each completed run
contains rows, a machine-readable summary, source snapshots, hashes, and the
pre-run Git state. The authoritative interpretation is
[`../OASQDDevelopmentResults.md`](../OASQDDevelopmentResults.md).

| run | role |
|---|---|
| `20260721-140355-o1-screen` | superseded O1 attempt; exposed an atlas local-window setting that crossed both gaps of a small interior region |
| `20260721-140511-o1-screen` | authoritative O1 atlas validation after the sample-scale window correction |
| `20260721-140523-o2-smoke` | O2 wiring smoke test |
| `20260721-140550-o2-screen` | authoritative O2 state-aware stopping screen |
| `20260721-140706-o2-screen` | bounded one-shot pulse diagnostic |
| `20260721-140836-o3-smoke` | O3 wiring smoke test |
| `20260721-140854-o3-screen` | authoritative O3 full-versus-stratified screen |
| `20260721-141036-o4-smoke` | O4 wiring smoke test |
| `20260721-141101-o4-screen` | O4 systematic-target/adaptive-`M` attribution |
| `20260721-141342-o4-screen` | initial coarse generated-edge gate screen |
| `20260721-141724-o4-screen` | edge-threshold preselection; froze `Q=.08` for O5 |
| `20260721-142019-o5-standard` | authoritative eight-seed standard development tournament; gate failed |

The empty `20260721-141710-o4-screen` directory, if present locally, was
created by a rejected command-line arm label before any task ran and has no
artifact or evidential role.
