# Server Resource Guide

Updated: 2026-04-19

## Purpose

This document records the current server hardware topology and provides practical usage guidance for running multiple experiment families in parallel without stepping on each other.

It is written as a reusable operations note for this repository and other projects on the same machine.

## Hardware Summary

- CPU: `2 x Intel Xeon Platinum 8473C`
- Physical cores: `104`
- Logical CPUs: `208`
- NUMA nodes: `2`
- GPUs: `4 x NVIDIA GeForce RTX 4090`
- Per-GPU memory: `49140 MiB`
- System memory: `251 GiB`

## NUMA and GPU Topology

This machine is dual-socket, and the 4 GPUs are split cleanly across the 2 NUMA nodes.

### CPU / NUMA layout

- `NUMA0`: CPUs `0-51,104-155`
- `NUMA1`: CPUs `52-103,156-207`

### GPU / NUMA affinity

- `GPU0` -> `NUMA0`
- `GPU1` -> `NUMA0`
- `GPU2` -> `NUMA1`
- `GPU3` -> `NUMA1`

### Cross-GPU relation

- `GPU0 <-> GPU1`: same NUMA side
- `GPU2 <-> GPU3`: same NUMA side
- `GPU0/1 <-> GPU2/3`: cross-socket path (`SYS`)

## Why This Matters

If one experiment family is spread across all GPUs while another family also spreads across all GPUs, they compete in three places at once:

- GPU compute and memory
- CPU threads
- cross-NUMA memory traffic

The cleanest pattern on this machine is:

- keep one experiment family on `GPU0/1 + NUMA0`
- keep another experiment family on `GPU2/3 + NUMA1`

This is much more stable than full-machine round-robin scheduling.

## Recommended Allocation Patterns

### Pattern A: DLCR + MBA isolation

Use this when `DLCR` and `MBA` are both active.

- `DLCR`: `GPU0, GPU1` + `NUMA0`
- `MBA`: `GPU2, GPU3` + `NUMA1`

Recommended interpretation:

- `GPU0`: primary DLCR run
- `GPU1`: second DLCR run or reserved buffer GPU
- `GPU2/3`: MBA-only zone

### Pattern B: single-project full control

Use this when only one project is active.

- Prefer `GPU0/1` together or `GPU2/3` together
- Avoid mixing `GPU0` with `GPU2`, or `GPU1` with `GPU3`, unless there is a strong reason

### Pattern C: light probe jobs

For very small probe or smoke jobs:

- use one GPU only
- keep them on the same NUMA side as the main project
- do not let probe jobs float across all 4 GPUs

## Recommended Launch Rules

### 1. Reserve GPUs explicitly

Do not rely on implicit default CUDA behavior.

Preferred options:

- `CUDA_VISIBLE_DEVICES=0`
- or explicit `--device cuda:0`

When a project owns a GPU zone, it should only launch inside that zone.

### 2. Bind to NUMA when running heavy jobs

For jobs on `GPU0/1`, prefer:

```bash
numactl --cpunodebind=0 --membind=0 <command>
```

For jobs on `GPU2/3`, prefer:

```bash
numactl --cpunodebind=1 --membind=1 <command>
```

This is especially helpful for:

- heavy dataloading
- large CPU preprocessing
- many concurrent subprocesses
- high-memory runs

### 3. Cap BLAS / OpenMP threads

Unbounded thread pools are a common source of fake parallelism and poor throughput.

Safe defaults for many-worker experiment managers:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

If a job is launched alone and needs more CPU help, raise these carefully to `2-4`.

### 4. Be conservative with workers-per-GPU

Heavy host backbones such as `PatchTST` and `TimesNet` should not be launched with aggressive workers-per-GPU settings by default.

Recommended starting points:

- heavy jobs: `1` worker per GPU
- medium jobs: `1-2` workers per GPU
- only increase after checking:
  - GPU utilization
  - memory pressure
  - load average

## Practical Advice For This Repository

### DLCR runs

Current DLCR-style runs benefit from stable GPU ownership more than extreme parallel fanout.

Recommended:

- keep current run on one GPU if it is already halfway done
- for the next full-scale run, use `GPU0/1` in parallel
- increase dataloader workers from `0` to `4-8` when safe
- increase batch size only after checking GPU memory headroom

### MBA runs

MBA sweeps should not use full-machine round-robin by default.

Recommended:

- only use `GPU2/3`
- keep `WORKERS_PER_GPU = 1` initially
- use thread caps of `1`
- avoid launching duplicate managers into the same output root

## Anti-Patterns To Avoid

### 1. All-GPU round-robin from multiple projects

Bad:

- Project A uses `0,1,2,3`
- Project B also uses `0,1,2,3`

This causes unstable latency and hard-to-debug interference.

### 2. High workers-per-GPU on heavy models

Bad:

- `WORKERS_PER_GPU = 10` for `PatchTST` / `TimesNet`

This tends to explode CPU load and memory pressure before GPU utilization actually improves.

### 3. Duplicate experiment managers

Bad:

- launching the same sweep manager twice into the same output root

This creates self-overlap, duplicate work, and misleading resource readings.

### 4. Ignoring NUMA locality

Bad:

- GPU-heavy jobs on `GPU0/1` while CPU threads and memory traffic spread heavily across NUMA1

This is wasted bandwidth and can reduce effective throughput.

## Quick Inspection Commands

### Hardware topology

```bash
lscpu
nvidia-smi topo -m
```

### GPU status

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv
```

### CPU and memory

```bash
uptime
free -h
ps -eo pid,user,psr,pcpu,pmem,etime,cmd --sort=-pcpu | head -n 20
```

### Check running experiment jobs

```bash
ps -ef | rg 'run_mba_pilot.py|run_resnet1d_local_closed_form_fixedsplit.py|parallel_sweep_manager.py'
```

## Suggested Team Operating Rules

When multiple experiment families share this machine:

1. Assign a GPU zone before launching.
2. Assign a NUMA side that matches that GPU zone.
3. Cap thread pools explicitly.
4. Prefer fewer, cleaner workers over many speculative workers.
5. Before starting a new sweep, confirm there is no earlier duplicate manager still alive.

## Current House Default

Unless there is a specific reason not to:

- `DLCR`: `GPU0/1 + NUMA0`
- `MBA`: `GPU2/3 + NUMA1`
- `WORKERS_PER_GPU`: start from `1`
- BLAS/OpenMP thread caps: start from `1`

This is the default operating mode that should be reused by other projects on this machine.
