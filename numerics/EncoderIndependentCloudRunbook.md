# Cloud runbook — renting the GPU and running CAP-EMF-1

**Date:** 2026-08-04
**Target:** ~34 h on an RTX 4090, ~$17, inside a 48 h / $25 envelope

---

## 0. How the connection actually works

**I cannot reach a rented machine on my own.** My tools run shell commands *on
your Windows machine*. So the path is:

```
me  ──►  your Windows box  ──ssh──►  the rented GPU
```

Once you have `ssh cap-gpu` working from your machine, I can run
`ssh cap-gpu "<command>"` and read the output. Everything below is arranged so
that one alias is the only thing you have to set up by hand.

---

## 1. Which provider

**RunPod**, RTX 4090, **Secure Cloud** if the price gap over Community is under
about 40%.

| | why |
|---|---|
| RTX 4090 | our estimates are calibrated to it; a 5090 only pays off in BF16, which §3.1 of the protocol rules out |
| Secure Cloud | a 34 h run on a Community host that disappears costs restart time; our recovery makes that survivable, not free |
| persistent volume | checkpoints and snapshots must outlive a stopped pod |

Vast.ai is cheaper and fine if you prefer it — pick a host with a high
reliability score and an NVMe disk. The instructions below are provider-neutral
apart from §2.

**Do not** pick an A100 or H100. Our effective batch is 64, which cannot
saturate them; you would pay 3–6× for maybe 1.4×.

---

## 2. Renting it

1. Create the account, add credit (**$25 is enough; ~$17 expected**).
2. **Storage:** container disk **20 GB**, persistent volume **50 GB**.
   Budget: PyTorch ~8 GB, checkpoints 4.5 GB, post-hoc EMA snapshots 4.5 GB,
   recovery file ~0.5 GB, CIFAR-10 0.2 GB.
3. **Template:** any "RunPod PyTorch 2.x" image. We install exact versions
   ourselves, so the template only needs CUDA and Python 3.10+.
4. **Deploy**, then open the pod's *Connect* panel and copy the SSH command.

### 2.1 SSH from Windows

Windows 10/11 ships OpenSSH. In PowerShell:

```powershell
# once, if you have no key yet
ssh-keygen -t ed25519 -C "cap-emf-1"
# paste this into RunPod -> Settings -> SSH Public Keys
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

Then create the alias so every later command is short. Edit
`C:\Users\Dronm\.ssh\config` (create it if absent):

```
Host cap-gpu
    HostName <ip-from-runpod>
    User root
    Port <port-from-runpod>
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    StrictHostKeyChecking accept-new
```

Verify:

```powershell
ssh cap-gpu "nvidia-smi --query-gpu=name,memory.total --format=csv"
```

**Tell me when that prints a 4090 and I can drive the rest.**

---

## 3. Getting the code across

The GitHub remote is **222 commits behind**, so cloning would fetch code from
before any of this work. Two options; the first needs no decision from you.

### 3.1 Ship a snapshot (recommended)

`git archive` exports only *tracked* files, so none of the 11 GB of prior-stage
checkpoints comes along — the archive is **38 MB**.

```powershell
cd C:\Users\Dronm\Documents\drifting_identifiability
git archive --format=tar.gz -o $env:TEMP\cap_code.tgz HEAD -- `
  numerics/encoder_independent_drifting `
  numerics/EncoderIndependentCAPEMF1Protocol.md `
  numerics/AnchoredSelfFeatureDriftingSpecification.md
scp $env:TEMP\cap_code.tgz cap-gpu:/workspace/
ssh cap-gpu "mkdir -p /workspace/drifting && tar -xzf /workspace/cap_code.tgz -C /workspace/drifting && ls /workspace/drifting/numerics"
```

### 3.2 Or push first

`git push origin main` (222 commits), then clone on the pod. Cleaner
provenance, but it publishes the whole research history — your call, and not
needed for the run.

---

## 4. Environment and data

```bash
ssh cap-gpu bash -lc '
  pip install --quiet torch==2.7.1 torchvision==0.22.1 numpy scipy pillow
  python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
'
```

CIFAR-10 must be downloaded once — the loaders run with `download=False` on
purpose, so a missing dataset fails loudly rather than silently fetching
during a timed run:

```bash
ssh cap-gpu bash -lc '
  python -c "
from torchvision.datasets import CIFAR10
for train in (True, False):
    d = CIFAR10(root=\"/root/.cache/cifar\", train=train, download=True)
    print(\"train\" if train else \"test\", len(d.data))
"
'
```

Expect `train 50000` and `test 10000`.

---

## 5. Tests, then the benchmark

### 5.1 Tests

```bash
ssh cap-gpu bash -lc '
  cd /workspace/drifting && export PYTHONPATH=/workspace/drifting
  python -m numerics.encoder_independent_drifting.stage_cap.tests.test_cap
  python -m numerics.encoder_independent_drifting.stage_asfd.tests.test_asfd
'
```

Expect **43** and **47** passes.

### 5.2 The preflight is regenerated on the box, not copied

This matters. The committed `cap_preflight.json` records source hashes and a
throughput probe **from the laptop**. Two reasons it cannot be reused:

- it hashes source files, and line endings can differ across a transfer;
- its throughput number is an RTX 4050 measurement, and the whole point of the
  benchmark is to replace an extrapolation with a measurement.

Generate the box's own, to a separate path so both records survive:

```bash
ssh cap-gpu bash -lc '
  cd /workspace/drifting
  export PYTHONPATH=/workspace/drifting CUBLAS_WORKSPACE_CONFIG=:4096:8
  python -m numerics.encoder_independent_drifting.stage_cap.preflight \
    --device cuda --probe-micro-batch 64 \
    --out numerics/encoder_independent_drifting/stage_cap/cap_preflight_gpu.json
'
```

`--probe-micro-batch 64` asks for the whole batch in one microbatch. At 3.25 GiB
for a microbatch of 16 on a 6 GiB card, 64 should need ~13 GiB and fit a 24 GB
4090 comfortably — which removes three quarters of the kernel launches. If it
does not fit, the probe steps down on its own and reports which shape it used.

**This is the $1 measurement.** Read `seconds_per_update` from the throughput
check and:

```
projected hours = seconds_per_update x 750000 / 3600
```

| measured s/update | foundation | verdict |
|---|---:|---|
| ≤ 0.165 | ≤ 34 h | on plan |
| 0.165–0.21 | 34–44 h | fits, thin margin |
| > 0.21 | > 44 h | budget-stop will bite; expect a checkpoint short of 750 k |

---

## 6. The run

**`tmux` is not optional.** A 34-hour job must survive an SSH disconnection.

```bash
ssh cap-gpu bash -lc '
  cd /workspace/drifting
  tmux new-session -d -s cap "
    export PYTHONPATH=/workspace/drifting CUBLAS_WORKSPACE_CONFIG=:4096:8
    python -m numerics.encoder_independent_drifting.stage_cap.run_unit \
      --device cuda \
      --preflight numerics/encoder_independent_drifting/stage_cap/cap_preflight_gpu.json \
      --data-root /root/.cache/cifar \
      --micro-batch 64 \
      --i-have-authorized-the-budget-run 2>&1 | tee /workspace/cap_run.log
  "
  sleep 5 && tail -5 /workspace/cap_run.log
'
```

Notes on the flags:

- `--micro-batch 64` is the only shape the protocol lets the benchmark change.
  It holds the effective batch at 64 by setting accumulation to 1, and the
  artifact records the realized split. Drop it if the probe showed 64 did not
  fit.
- `--i-have-authorized-the-budget-run` is the deliberate opt-in. Without it the
  runner refuses.
- Determinism is on by default and needs `CUBLAS_WORKSPACE_CONFIG`. Add
  `--nondeterministic` only if the benchmark shows it costs real money; the
  choice is recorded either way.

### 6.1 Monitoring

I can run these:

```bash
ssh cap-gpu "tail -20 /workspace/cap_run.log"
ssh cap-gpu "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv"
```

Health lines print at every health interval:

```
step 50000: moment 0.812 var 0.798 rank 0.94 HH 0.41
```

**Watch `HH`.** It is the discriminating threshold (H4 ≥ 0.50) and the leading
indicator for whether the ASFD feature gate can pass at all.

### 6.2 If it dies

Recovery is exact — optimizer, model, EMA and all four RNG streams are written
atomically every 1 000 updates. **Re-run the identical command.** It continues
from the last recovery point and `test_restart_reproduces_an_uninterrupted_run`
asserts a resumed run reaches the same loss as an uninterrupted one.

The one manual step: if the pod was destroyed rather than stopped, restore
`/workspace` from the persistent volume first.

---

## 7. Evaluation and retrieval

The sealed split opens **once**, after the final checkpoint is frozen:

```bash
ssh cap-gpu bash -lc '
  cd /workspace/drifting
  export PYTHONPATH=/workspace/drifting
  python -m numerics.encoder_independent_drifting.stage_cap.evaluation \
    --device cuda --data-root /root/.cache/cifar --acknowledge-sealed
'
```

Then pull everything back — the artifacts and grids are small; the checkpoints
are 151 MB each:

```powershell
$D = "C:\Users\Dronm\Documents\drifting_identifiability\numerics\encoder_independent_drifting\stage_cap"
scp cap-gpu:/workspace/drifting/numerics/encoder_independent_drifting/stage_cap/cap_*.json* $D\
scp -r cap-gpu:/workspace/drifting/numerics/encoder_independent_drifting/stage_cap/grids $D\
scp cap-gpu:/workspace/cap_run.log $D\
# the final EMA checkpoint: the frozen trunk ASFD forks from
scp cap-gpu:/workspace/drifting/numerics/encoder_independent_drifting/stage_cap/checkpoints/cap_emf1_step750000_ema.pt $D\checkpoints\
```

**Retrieve before you terminate the pod.** Also worth pulling the post-hoc EMA
snapshots (~4.5 GB) if you want the option of synthesising a different EMA
horizon later; they are secondary and the run is valid without them.

---

## 8. Cost control

- RunPod bills while the pod **runs**; stopping it keeps the volume at a much
  lower rate. Stop it the moment the run finishes.
- Set a spending alert at $20.
- The benchmark (§5.2) costs minutes. Do not skip it to save them — every
  number in §6 depends on the extrapolation it replaces.

---

## 9. Order of operations

1. rent, set up `ssh cap-gpu`, verify `nvidia-smi` — **you**
2. ship the snapshot — either of us
3. environment, CIFAR-10, tests — me
4. **preflight benchmark, read s/update, project cost** — me, **you decide**
5. launch under `tmux` — me, once you approve
6. monitor; the coherence gate is the real decision point — both
7. sealed evaluation, retrieve, stop the pod — me

Step 4 is the gate. Nothing after it should start until the measured number is
on the table.
