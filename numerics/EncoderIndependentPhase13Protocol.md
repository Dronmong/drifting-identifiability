# Encoder-Independent Kernel Drifting — Phase 13 protocol

## Does the amortizer close the gap to the particle cloud as pairs grow?

*Frozen pre-outcome design. Source: `EncoderIndependentPhase12Investigation.md`
§4. Results go to `EncoderIndependentPhase13Results.md`.*

---

## 1. Why this phase exists

The investigation established that the requirement is **correspondence
stability**, not assignment quality: an exact Hungarian assignment preserves
the target almost perfectly (target tail 0.327 against the cloud's 0.406) and
still leaves the generator at tail **0.0009** on fresh latents against
**0.3430** on fixed ones — a factor of 380 from the correspondence alone.

With a stable correspondence the generator reaches **ED² 0.1640 from just 256
pairs**, against the particle cloud's own **0.0752**. It is 2.2× off with a
tiny bank, and the bank size has never been varied.

**This is the amortization question properly posed**, and it has a clear
ending either way.

---

## 2. Design

One particle cloud of 2048 is evolved per seed under the field; the pair
banks are nested prefixes of it, so every bank draws targets of the same
quality and only the **number of pairs** changes. Negatives in the particle
field are capped at 512 (a declared approximation that keeps the cloud's cost
linear in its size; at 512 particles it is a no-op, which anchors the sweep
to the previously measured configuration).

Each bank fixes `N` latents, paired to particles by index — assigned once and
reused, which is the stable correspondence the investigation isolated.
Training draws minibatches from the bank; **evaluation is always on fresh
latents**, which is the amortization test.

| arm | target |
|---|---|
| C0 | `T = f + ηV` *(self-referential baseline)* |
| C1 | C0 + R11 *(the incumbent)* |
| C2–C5 | pair bank, `N ∈ {256, 512, 1024, 2048}` |
| C6 | bank of 2048 with **jittered** latents (`z_i + 0.1·ξ`) |

C6 is the secondary arm from the investigation: sampling near the bank rather
than exactly on it, which interpolates between memorization and amortization
and shows whether stability alone suffices without literal repetition.

CIFAR-16, target ESS 0.9, 64 positives, generator batch 256, Adam/2e-3,
3 fresh seeds (`MASTER_SEED + 25000..`), 600 steps.

---

## 3. What is measured

**The reference is the particle cloud's own ED²**, measured per seed on the
same pools. The question is whether the amortizer approaches it.

Reported per arm: ED² on fresh latents, second moment, spectral tail, the
ratio to the particle cloud's ED², and the cost ledger.

### 13 gate

> **If a pair-bank arm reaches a second-moment ratio inside `[0.7, 1.3]` and
> an ED² within 25% of the best R11 cell, the external target with a stable
> correspondence supersedes R11.**

The same gate R11 has survived six times.

### The trend criterion, declared separately

> **Success:** ED² falls monotonically with bank size and approaches the
> particle cloud's ED² (within ~25%).
> **Failure:** ED² plateaus well above the particle cloud regardless of bank
> size — the map cannot represent the particle law, and **this line ends.**

Both are reported; the gate can fail while the trend succeeds, and that
combination would mean the method works but needs more pairs than were
affordable here.

---

## 4. Declared failure branches

- **Gate fires** → R11 superseded by a structural change; report with the
  cost ledger.
- **Trend succeeds, gate fails** → report the extrapolated bank size needed,
  and the cost it would imply.
- **Trend plateaus** → the amortizer cannot reach the particle law. Say so
  plainly and **close the line**; do not propose a further variant.
- **C6 collapses while C5 works** → the effect requires literal repetition of
  the same latents, i.e. memorization, and does not amortize. That is also a
  closing result.
- **No tuning.** Bank sizes, jitter and all thresholds are declared above.

## 5. What Phase 13 cannot conclude

- Nothing about ImageNet, FID, or the paper's trained model.
- Banks beyond 2048 are untested; the particle cloud's cost grows with its
  size and 2048 is what is affordable here.
- The nested-prefix design means every bank draws from a 2048-particle
  cloud, so it isolates pair count from cloud quality — a 256-particle cloud
  evolved alone would be a worse target set, and that is deliberately not
  what is measured.
- The anchor stays disabled; the geometry thread stays closed.
