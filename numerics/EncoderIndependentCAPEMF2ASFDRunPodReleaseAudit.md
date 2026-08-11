# RunPod CAP-ASFD release audit

**Scope:** one on-demand RunPod Secure Cloud RTX 4090, one 750k foundation,
one exact 50k ASFD continuation, and fixed final evaluation<br>
**Status:** release configuration validated locally; no paid Pod or training
launched

## Release result

The RunPod port preserves the existing scientific implementation. It replaces
only the Windows/provider orchestration with two source-bound Linux launchers:

- `stage_cap2/runpod_bootstrap.sh` installs and verifies the pinned environment;
- `stage_cap2/runpod_pipeline.sh` exposes explicit fail-closed phase commands.

The canonical operator run card is
[`EncoderIndependentCAPEMF2ASFDRunPodProtocol.md`](EncoderIndependentCAPEMF2ASFDRunPodProtocol.md).
The CAP2 and ASFD source manifests bind the protocol and both shell scripts, so
changing any of them after admission invalidates downstream evidence.

## What was checked

- Both shell scripts pass Bash syntax checking and ShellCheck.
- All 18 Bash blocks in the run card pass Bash syntax checking and ShellCheck
  (apart from the intentional external `/workspace/runpod_operator.env` source).
- Ruff lint and formatting checks pass for CAP, CAP2 and ASFD.
- The combined CAP/CAP2/ASFD regression passes: **248/248 tests**.
- The bootstrap pins CPython 3.11.15, Torch 2.7.1+cu126, torchvision
  0.22.1+cu126, NumPy 1.26.4 and Pillow 12.2.0.
- The pipeline verifies the release commit, exact tracked source bytes, GPU
  model, Network Volume capacity and identity, provider rate, four durable-root
  attestations and the historical checkpoint SHA before admission.
- The historical admission checkpoint was locally rehashed as
  `b55b2a62bfc44e546f347cb348b8e7e63aef6686d8a97527f6d4d232a5023f49`.
- Foundation training is one model with an exact optimizer/EMA/RNG recovery:
  update 0 to a mandatory 50k pause, raw numerical readmission, then recovery to
  750k. There is no multi-arm budget split.
- ASFD cannot start without the foundation quantitative and human gate. Its
  preflight binds the same GPU/runtime/rate, measures the actual correction
  graph, and enforces a mirrored recovery-first hard wall.
- Final success requires prospectively calibrated FID/KID improvement,
  non-regression, coverage/noncollapse/memorization gates, artifact integrity
  and an uncurated paired visual review.

## Remaining provider-only checks

Local validation cannot prove that a RunPod console selection is genuine. The
operator must still:

1. create a 200 GB Secure Cloud Network Volume;
2. deploy an on-demand RTX 4090 Pod with that exact Volume at `/workspace`;
3. enter the displayed Pod hourly price and actual Volume ID;
4. upload the preserved admission checkpoint and verify its SHA;
5. run `prepare`, fresh evidence generation and the real 2,000-update admission
   benchmark;
6. proceed only if the generated preflight says `GO` and its measured cost is
   acceptable.

No local test can guarantee provider persistence, billing behavior, network
availability or throughput. Auto-pay should remain disabled, the account must
not be allowed to reach zero while the Network Volume contains the only durable
copy, and the provider-side spending balance remains a human authorization.

## Scientific boundary

This is a deliberately concentrated one-seed development screen. The exact
hybrid is supported by nearby MeanFlow, self-feature and Drifting literature,
but not already validated as a complete published recipe. A successful run is
proof-of-concept evidence for a one-call model without an external/separately
trained encoder. It is not a causal ablation, replication, held-out claim,
representation-independence theorem or ImageNet comparison. The dynamically
capped ASFD correction is a stabilization heuristic and is not claimed to be
the gradient of one fixed identifiable objective.
