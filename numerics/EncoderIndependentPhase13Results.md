# Encoder-Independent Kernel Drifting — Phase 13 results

## The amortizer works, beats the particle cloud it fits, and still loses to R11

*Protocol: `EncoderIndependentPhase13Protocol.md`, frozen before the run.
Code: `run_phase13.py`. Artifact: `phase13.json` (+ `.sha256`), stdout in
`phase13.stdout.txt`. 39.2 minutes, 3 fresh seeds (`MASTER_SEED + 25000..`),
CIFAR-16, target ESS 0.9, 2048-particle cloud, 1000 steps.*

---

## 0. Summary

| arm | bank | visits/pair | ED² | ÷ particles | 2nd moment | tail | score | band |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C0 self-referential | — | — | 0.9624 | 5.05 | 0.380 | 0.0059 | 9.285 | out |
| **C1 self + R11** | — | — | **0.0955** | 0.44 | 0.972 | 0.0589 | **2.333** | in |
| C2 bank 256 | 256 | 1000 | 0.6184 | 3.00 | 1.353 | 0.4200 | 8.037 | out |
| **C3 bank 512** | 512 | 500 | **0.1391** | **0.73** | **0.989** | 0.1607 | 2.888 | **in** |
| **C4 bank 1024** | 1024 | 250 | **0.1449** | **0.70** | 0.771 | 0.0442 | 3.552 | **in** |
| C5 bank 2048 | 2048 | 125 | 0.3835 | 1.84 | 0.573 | 0.0248 | 5.289 | out |
| C6 bank 2048, jittered | 2048 | 125 | 0.5868 | 2.73 | 0.471 | 0.0261 | 7.057 | out |

*(particle cloud: ED² 0.2089, tail 0.4705; gate ceiling 0.1193)*

**Gate: not passed.** The best bank (0.1391) misses R11's ceiling (0.1193) by
**1.17×**. R11 survives a **seventh** supersession test — the closest any
alternative has come.

**Trend criterion: passed.** C3 and C4 reach **0.70–0.73× the particle
cloud's own ED²** — the amortizer is *better than the particle system it is
fitting* — and both land inside the second-moment band, from a
self-referential baseline of 0.9624.

---

## 1. The bank sweep is not monotone, and the reason is in the table

ED² by bank size: 0.618 → **0.139** → **0.145** → 0.384. A U, not a
monotone fall.

The `visits/pair` column explains it. At a fixed 1000-step budget, a bank of
256 gets 1000 visits per pair while a bank of 2048 gets 125. **The sweep
confounds "more pairs" with "less training per pair"**, and the U is the
product of the two:

- **bank 256** is over-fitted — 1000 visits on 256 points gives second moment
  **1.353** (over-dispersed, out of band) and tail 0.42, well above real
  data's;
- **bank 2048** is under-fitted — 125 visits leaves second moment 0.573 and
  tail 0.0248, drifting back toward the self-referential baseline's 0.0059;
- **512–1024 is where the two meet**, and that is an artifact of the budget,
  not a property of the method.

This was flagged in the design and the column was added for it; the honest
reading is that the optimum here is "the best bank *at 1000 steps*", not an
optimal bank size.

---

## 2. What is established

- **The structural alternative works.** An external target with a stable
  correspondence takes the generator from ED² 0.9624 to **0.1391**, a factor
  of **6.9**, with the second moment in band (0.989) and the tail restored
  27× (0.0059 → 0.1607). Per-seed and stable: 0.157 / 0.139 / 0.127.
- **It beats the particle system it amortizes** — 0.73× and 0.70× the cloud's
  own ED². The generator does not merely copy the particles; interpolating
  across the bank produces a *better* sample than the cloud it was fitted to.
- **It does not beat R11** — 0.1391 against 0.0955, and R11 also beats the
  particle cloud (0.44×). R11 remains the strongest thing in this program.
- **Jitter hurts.** C6 (0.5868) is worse than C5 (0.3835) on the same bank,
  so perturbing the latents away from the bank costs quality. Correspondence
  stability is not just about repetition frequency — it wants the *same*
  latents.
- **The cost is 14.4×** the self-referential baseline in kernel pairs
  (1.18e9 against 8.19e7), driven entirely by the 2048-particle cloud.
  Inference is unchanged at NFE = 1.

---

## 3. Honest position

R11 is a scalar rescaling of the teacher that costs nothing. The amortizer is
a second particle system, a fixed pair bank and 14.4× the training kernel
work, and it lands 1.17× behind. **On the program's own gate it is not an
improvement**, and it should not be presented as one.

What it *is*: the first structural account that reproduces most of R11's
effect through a mechanism that was derived rather than found — external
target, stable correspondence — and independent evidence that the
self-reference diagnosis is right. It also produces the tail (0.1607 against
real data's ~0.13) that R11 never does (0.0589), which is the quantity the
mechanism work identified as controlling.

---

## 4. The one experiment that would settle it

The sweep is confounded by visits per pair (§1). The clean version holds
visits constant and varies only the bank:

| bank | steps (to hold ~500 visits/pair) |
|---|---|
| 512 | 1000 |
| 1024 | 2000 |
| 2048 | 4000 |
| 4096 | 8000 |

If ED² falls monotonically under matched visits and crosses R11's 0.0955,
the structural route wins on quality and the remaining question is only cost.
If it plateaus near 0.13–0.14, then **512–1024 pairs is the method's ceiling
at this model size**, R11 stands, and this line should be closed and written
up as a mechanism result rather than a method.

That is a single well-posed run with a decisive outcome either way, and it is
the last thing I would spend compute on before consolidating.

**The consolidation, which needs no new mechanism**, remains outstanding and
is now the higher-value work: no configuration has ever combined the
ESS-0.9 bandwidth (worth 4.9× and unused before Phase 8), R11, and the
anchor (~3.5%, untouched since Phase 2).

---

## 5. Scope and caveats

- The bank sweep confounds pair count with visits per pair (§1); no
  conclusion about an optimal bank size is drawn.
- Banks are nested prefixes of one 2048-particle cloud, so pair count is
  isolated from cloud quality by construction — a 256-particle cloud evolved
  alone would be a worse target set and is not what was measured.
- The particle field caps negatives at 512 (declared); at 512 particles this
  is a no-op, which anchors the sweep to previously measured configurations.
- Banks beyond 2048 untested; the cloud's cost grows with its size.
- One dataset, one geometry, one bandwidth, 3 seeds, 1000 steps.
- Nothing here concerns ImageNet, FID, or the paper's trained model. The
  anchor stays disabled; the geometry thread stays closed.

## 6. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.run_phase13 `
  --seeds 3 --steps 1000
```
