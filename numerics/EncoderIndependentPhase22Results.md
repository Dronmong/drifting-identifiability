# Encoder-Independent Kernel Drifting — Phase 22 results

## The metrics rank the samples backwards

*Protocol: `EncoderIndependentPhase22Protocol.md` (frozen before the run).
Runner: `run_phase22.py`. Artifacts: `phase22.json` (+ `.sha256`),
`phase22.stdout.txt`, six sample grids. 6 arms × 4 seeds × 30 000 steps,
paired within seed, GPU, 7.31 h.*

---

## 1. Results

| arm | ESS | lv | pos | KID | FID2048 | FID512 | tail | realized | images |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_control | 0.500 | 1 | 64 | +0.16084 | 191.48 | 222.32 | **0.1611** | 0.815 | 52.2 |
| B_sharp | 0.050 | 1 | 64 | +0.14660 | 180.10 | 207.89 | 0.0382 | 0.536 | 34.3 |
| C_sharper | 0.010 | 1 | 64 | +0.25988 | 286.21 | 302.94 | 0.0015 | 0.279 | 17.8 |
| D_pos | 0.050 | 1 | 256 | +0.13634 | **170.81** | 198.85 | 0.0429 | 0.524 | 134.2 |
| E_sharper_pos | 0.010 | 1 | 256 | +0.26413 | 288.09 | 306.07 | 0.0017 | 0.237 | 60.6 |
| **F_mix** | 0.050 | 5 | 64 | **+0.13116** | 172.36 | 203.58 | 0.0467 | 0.707 | 45.2 |

*floor KID −0.00008 / FID2048 23.20 · bar KID +0.21805 / FID2048 229.07 ·
real tail 0.1962*

| contrast | KID mean | t | p | FID2048 | p |
|---|---:|---:|---:|---:|---:|
| `best_vs_baseline` | **−0.02048** | −3.34 | **0.045** | −13.30 | 0.105 |
| `mixture` | −0.01075 | −1.80 | 0.170 | −4.93 | 0.161 |
| `sharp` | −0.00973 | −2.10 | 0.127 | −8.37 | 0.084 |
| `positives_at_ess05` | −0.00869 | −0.96 | 0.408 | −8.82 | 0.138 |
| `positives_at_ess005` | +0.00654 | +1.64 | 0.200 | +6.06 | 0.220 |
| `sharper_at_pos64` | **+0.11314** | +8.58 | **0.0033** | +102.75 | 0.0010 |
| `sharper_at_pos256` | **+0.12837** | +24.91 | **0.0001** | +117.63 | 0.0001 |

---

## 2. The finding: the ranking inverts when you look

`C_sharper` is the **worst** arm by every summary statistic — KID +0.2599
(above the bar), FID2048 286.21, tail 0.0015 against real data's 0.1962.

**Its samples look the most like photographs of anything in this program.**
Soft natural tones, sky-like tops, ground-like bottoms — extremely
out-of-focus photographs. `A_control` and `F_mix`, which the metrics rank
best, produce garish coloured worm patterns.

*(`phase22_samples_C_sharper.png` versus `phase22_samples_A_control.png` /
`_F_mix.png`.)*

So on the structural readout declared in §5 of the protocol, **the metric
ordering and the visual ordering are opposed.** KID and FID at this quality
level reward high-frequency coloured texture that matches CIFAR's spectral
statistics, over plausible image composition. The spectral tail agrees with
KID and FID — A_control's 0.1611 is closest to real — and therefore also
disagrees with what the images look like.

This is the third time this program has found its metric pointing the wrong
way: ED² was saturated by a moment-matched Gaussian; FID at n = 512 was
bias-dominated; and now KID and FID both rank abstract texture above blurry
photographs.

### The R11 hypothesis — proposed here, then refuted by this run's own data

**Superseded. Retained because it was published in this document.**

I proposed that R11 (which rescales the teacher to the real batch's second
moment) must inflate a heavily-averaged, low-variance target harder, and that
inflating a blurry signal manufactures the high-contrast artifacts — flat
kernels garish, sharp kernels natural.

**That prediction fails against the second moments in `phase22.json`:**

| arm | 2nd moment | tail |
|---|---:|---:|
| A_control | 0.977 | 0.1611 |
| B_sharp | 0.991 | 0.0382 |
| C_sharper | 0.976 | 0.0015 |
| D_pos | 1.002 | 0.0429 |
| E_sharper_pos | 0.972 | 0.0017 |
| F_mix | 0.984 | 0.0467 |

All six arms land at ~1.0. R11 succeeds identically everywhere, so differential
inflation cannot be what separates the garish arms from the natural one. I
should have checked this column before writing the hypothesis down.

### What actually separates them: rank, not scale

Total variance is the same; its *distribution* is not. `tail` is the fraction
of variance beyond the top 32 directions:

- **real CIFAR 0.1962** — ~80% of variance in the top 32 PCs
- **A_control 0.1611** — closest to real
- **C_sharper 0.0015** — 99.85% inside 32 directions, i.e. a ~32-dimensional
  family of images

So `C_sharper` produces a **low-rank family of individually plausible blurry
images**, and `A_control` **spans more of the space with wrong content**. That
is a precision/recall split: high precision and low recall versus lower
precision and higher recall. KID and FID are distribution-level moment
matches that reward coverage, so they rank A and F above C — exactly the
inversion §2 observed.

This is a diagnosis, not yet a measurement: this program has never computed
precision and recall, which is why a coverage/quality trade-off could sit
undetected behind every configuration decision made so far.

---

## 3. What resolved

**The sharp cliff is real, large and unaffected by batch size.** Sharpening
to ESS 0.01 costs +0.113 KID at 64 positives (p = 0.0033) and +0.128 at 256
(p = 0.0001), all eight seed-pairs positive, with no kernel collapse
(realized ESS 0.24–0.28, no floor-pinning). Phase 21's reversal at 0.02 was
not an artifact of the underflow boundary.

**The interaction this run was built for is null.** I hypothesised the cliff
came from having too few positives to select among — at 64 positives,
averaging 29 is 45% of the batch and barely selective. Quadrupling the
attractor set changes nothing: `positives_at_ess005` is +0.0065, p = 0.20.
The controlling variable is the selectivity *fraction*, not the count.

**The best configuration beats the baseline** — `F_mix` vs `A_control`,
−0.0205 KID, p = 0.045. FID2048 agrees in direction but not significance
(−13.30, p = 0.105).

---

## 4. What did not, and two things I got wrong

**No single component contrast is significant.** `sharp` (p = 0.127),
`positives_at_ess05` (p = 0.408) and `mixture` (p = 0.170) all fail, while
their combination clears the bar. With 4 seeds this run can establish that
the joint change helps without attributing it to any part — a real limit of
the design, and a direct consequence of covering six arms instead of honouring
this program's own ≥ 8-seed rule.

**Phase 21's sharpening effect shrank by half.** 17 FID at 3 000 steps became
−8.37 FID2048 (p = 0.084) at 30 000. Short-budget screens overstate; that is
now the second time this program has measured it.

**My mixture prediction was wrong.** The protocol declared the mixture should
*hurt*, because its realized ESS rises to 0.707 — back toward the flat regime
that A_control occupies. It was instead the best arm (−0.0108 vs B, n.s.).
**Aggregate ESS is a poor summary for a mixture**: the summed kernel's
effective sample size is dominated by its widest component while the drift
still receives contributions from the sharp ones. The diagnostic I have been
reasoning with does not transfer from single bandwidths to mixtures.

**The runner's auto-verdict is wrong and should be ignored.** It printed
*"nothing resolves at 30k"* because the `an_axis_adds` test iterates only
over the component contrasts and never consults `best_vs_baseline` — which is
significant. Three contrasts resolved. The bug is in the verdict logic, not
the measurements; the numbers in §1 stand.

---

## 5. The declared outcome that fired

From §7 of the protocol: **"No grid shows structure at any setting"** — the
outcome named in advance as *the strongest likely one*.

Six configurations spanning an 18× range on the axis the mechanism identifies,
at 30 000 steps, and **not one produces a recognizable object**. The best is
blurry-photograph-like; the rest are abstract texture.

Raw-pixel kernel drifting does not generate images at this scale. The
encoder-independence result from Phases 17–18 stands, and it is a statement
about a method that does not work.

---

## 6. What to do next

1. **Test the R11 inflation hypothesis** (§2). Cheap, and it is the only
   live explanation for why the metrics invert. If R11 is manufacturing the
   artifacts that KID rewards, the program's headline reform is actively
   harming sample quality.
2. **Log-space stabilization** of the bi-softmax remains built-but-unbuilt
   work. It is now *less* urgent: the sharp regime past ESS 0.01 is
   measurably worse, so opening it is unlikely to help.
3. **Stop treating KID/FID as sufficient.** Every configuration decision in
   this program has been made on a statistic that §2 shows can be opposed to
   visual quality.

---

## 7. Scope

- CIFAR-10 32×32; only arms measured identically are comparable, none with
  published FIDs.
- 4 seeds, below this program's own ≥ 8 rule; §4 states what that cost.
- One grid per arm, seed 0 only. §2's inversion is a visual judgement over
  64 samples per arm, not a measurement — it needs a human-judgement or
  precision/recall protocol to become a result.
- Raw geometry only; licenses no claim about encoder dependence.
- Still not the paper's method: pixel-space drift with feature-space kernel
  weights.
