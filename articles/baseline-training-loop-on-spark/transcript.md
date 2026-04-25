# Transcript — A2: baseline-training-loop-on-spark

Provenance for the throughput-envelope sweep article. Cleaned source material from the 2026-04-25 session that produced this piece.

## Working setup at session start

Carried forward from the article #16 (`nemo-framework-on-spark`) handoff:
- `nvcr.io/nvidia/nemo:26.04.00` already on disk (70.1 GB)
- `nvcr.io/nvidia/pytorch:25.11-py3` already on disk (29.8 GB) — not used in A2
- `articles/nemo-framework-on-spark/evidence/nemo_train.py` — 215-line single-config run that A2's harness is built on top of
- pgvector and NIM Embed running (sibling services, not used by A2)

## What I built

`evidence/sweep.py` — same model shape, same training loop, parameterized over three axes:

```python
for precision in ("bf16", "fp8"):
    for seq in (1024, 2048):
        for batch in (2, 4, 8, 16):
            ...
```

30 steps per config, 5 warmup excluded. Wraps `run_one()` in a try/except for OOM
(none hit). FP8 path uses `te.fp8_autocast` with `DelayedScaling(format=HYBRID)`.

## Initial bug

First run failed immediately:

```
AttributeError: module 'transformer_engine.pytorch' has no attribute '__version__'
```

Fix: `__version__` lives on the parent package. Imported `transformer_engine`
(in addition to `transformer_engine.pytorch as te`) and read the version from
the parent module. One-line patch.

## Sweep run output (clean tail)

```
sweep on NVIDIA GB10  torch=2.11.0a0+eb65b36914.nv26.02  te=2.14.0+71bbefbf
[ 1/16] batch= 2 seq=1024 prec=bf16  → tok/s=   11044  step= 185.4ms  peak= 5.26GiB  loss  11.03→  7.64  (7s)
[ 2/16] batch= 4 seq=1024 prec=bf16  → tok/s=   12641  step= 324.0ms  peak= 7.94GiB  loss  11.02→  7.50  (10s)
[ 3/16] batch= 8 seq=1024 prec=bf16  → tok/s=   12819  step= 639.0ms  peak=13.83GiB  loss  11.03→  8.22  (20s)
[ 4/16] batch=16 seq=1024 prec=bf16  → tok/s=   13626  step=1202.4ms  peak=25.63GiB  loss  11.03→  8.55  (39s)
[ 5/16] batch= 2 seq=2048 prec=bf16  → tok/s=   12422  step= 329.7ms  peak= 7.95GiB  loss  11.04→  7.92  (11s)
[ 6/16] batch= 4 seq=2048 prec=bf16  → tok/s=   12729  step= 643.5ms  peak=13.85GiB  loss  11.03→  8.47  (20s)
[ 7/16] batch= 8 seq=2048 prec=bf16  → tok/s=   13036  step=1256.8ms  peak=25.65GiB  loss  11.03→  9.52  (39s)
[ 8/16] batch=16 seq=2048 prec=bf16  → tok/s=   13036  step=2513.7ms  peak=49.24GiB  loss  11.03→  9.77  (79s)
[ 9/16] batch= 2 seq=1024 prec=fp8  → tok/s=   11944  step= 171.5ms  peak= 5.60GiB  loss  11.03→  6.46  (7s)
[10/16] batch= 4 seq=1024 prec=fp8  → tok/s=   13462  step= 304.3ms  peak= 8.03GiB  loss  11.03→  7.08  (10s)
[11/16] batch= 8 seq=1024 prec=fp8  → tok/s=   13777  step= 594.6ms  peak=13.49GiB  loss  11.04→  7.72  (19s)
[12/16] batch=16 seq=1024 prec=fp8  → tok/s=   14266  step=1148.5ms  peak=24.33GiB  loss  11.03→  8.41  (36s)
[13/16] batch= 2 seq=2048 prec=fp8  → tok/s=   12873  step= 318.2ms  peak= 8.05GiB  loss  11.04→  6.88  (10s)
[14/16] batch= 4 seq=2048 prec=fp8  → tok/s=   13202  step= 620.5ms  peak=13.52GiB  loss  11.03→  7.56  (20s)
[15/16] batch= 8 seq=2048 prec=fp8  → tok/s=   13641  step=1201.1ms  peak=24.34GiB  loss  11.03→  8.31  (37s)
[16/16] batch=16 seq=2048 prec=fp8  → tok/s=   13370  step=2450.9ms  peak=46.06GiB  loss  11.03→  8.85  (76s)

sweep done in 7.4 min

peak throughput  :    14266 tok/s @ batch=16 seq=1024 prec=fp8
best mem/throughput: batch=2 seq=1024 prec=fp8 (5.60 GiB / 11944 tok/s)
```

## nvidia-smi summary across the 7.4-min sweep

225 samples at 2-second intervals.

```
gpu_util  : mean=86.8%  peak=96%
power     : mean=55.8 W  peak=77.1 W
temp      : mean=65.8 °C peak=77 °C
```

## Cross-check vs A1 setpoint

| metric | A1 (100 steps) | A2 sweep (30 steps) | drift |
|---|---:|---:|---:|
| tokens/sec | 12,820 | 12,641 | −1.4 % |
| step ms    | 319.5  | 324.0  | +1.4 % |
| peak GPU GiB | 7.94 | 7.94 | 0.0 % |

Within 1.4% — short-run sweep methodology is reproducible against the long-run baseline.

## Findings I'm carrying forward

1. **FP8 is free throughput on the GB10.** Wins every shape, 2.6–8.1%. Memory is also a touch lower at large configs.
2. **bf16 plateaus past batch=8 at seq=2048.** Doubling batch from 8 → 16 buys ~0% throughput at long sequence, costs 2× memory and 2× wall.
3. **Memory-per-token is ~1.5 KiB on this 354M.** Predict OOMs ahead of time; never tried > 50 GiB and never hit one.
4. **Sustained 87% GPU util at 56W mean.** Whole-night training costs about as much electricity as a desk lamp.
5. **Per-iteration budget for the A4 agent: 4.3M tokens at peak config in 5 minutes.** 100 iters = 430M tokens overnight.
