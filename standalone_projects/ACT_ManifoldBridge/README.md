# ACT_ManifoldBridge

`ACT_ManifoldBridge` is the active standalone workspace for the original
ACT/MBA augmentation line and its current ACT-V2 follow-up stack.

The stable publishable method body remains small:

```text
x -> z(x) -> z_aug -> x_aug
```

That core path means:

- embed each trial as one static SPD / Log-Euclidean state;
- build a direction bank in latent space;
- generate manifold candidates with the curriculum sampler;
- realize candidates back to raw time series through `bridge_single()`;
- train the host on original and augmented samples.

## Current Mainline

There are two active pipelines:

- `--pipeline act`
  - the original ACT/MBA external augmentation path;
  - `--pipeline mba` is kept as a legacy alias for this.
- `--pipeline mba_feedback`
  - the current ACT-V2 utilization stack;
  - keeps the original CE stream intact and adds weighted augmentation loss.

Supported direction engines:

- `lraes`
- `zpia`
- `adaptive`

`adaptive` is the current ACT-V2 experimental path. It supports:

- dual-engine augmentation pools (`lraes` + `zpia`);
- adaptive routing during training;
- optional on-the-fly augmentation;
- focal weighting, tau scheduling, consistency regularization;
- orthogonal subspace fusion via `--direction-bank-source orthogonal_fusion`.

Current `orthogonal_fusion` uses RC-4 geometry governance:

- the shared safety radius is restored from the original manifold safe-step
  budget (`R = eta_safe * d_min`);
- the orthogonal risk branch restores the original LRAES risk-step **scale
  budget** after projection, rather than claiming information-preserving
  recovery;
- this replaces the earlier cap-only behavior, whose normalize target was not
  fully realized after projection;
- structure is handled first and only the remaining radius budget is allocated
  to the risk branch;
- the old global clip on the full fused `Delta_z` is no longer the default
  governance rule.

The older failed object-layer and realization-layer branches have been removed
from the active surface of this folder so the project reflects the current
research direction more clearly.

## Entry Point

Run the original ACT/MBA mainline:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset natops --pipeline act --algo lraes --model resnet1d
```

Run the feedback path with a fixed engine:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset natops --pipeline mba_feedback --algo zpia --model resnet1d
```

Run the ACT-V2 adaptive stack:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset natops --pipeline mba_feedback --algo adaptive --model resnet1d \
  --direction-bank-source orthogonal_fusion --onthefly-aug \
  --aug-weight-mode focal --tau-max 2.0 --tau-min 0.1 --tau-warmup-ratio 0.3
```

## Core Options

- `--pipeline {act,mba,mba_feedback}`
- `--algo {pia,lraes,zpia,adaptive}`
- `--model {minirocket,resnet1d,patchtst,timesnet}`
- `--k-dir`
- `--pia-gamma`
- `--multiplier`
- `--theory-diagnostics`
- `--disable-safe-step`

ACT-V2 options:

- `--onthefly-aug`
- `--steps-per-epoch`
- `--aug-weight-mode {sigmoid,focal}`
- `--tau-max`
- `--tau-min`
- `--tau-warmup-ratio`
- `--consistency-regularization`
- `--lambda-consistency`
- `--consistency-mode {mse,kl}`
- `--direction-bank-source {lraes,zpia_telm2,orthogonal_fusion}`
- `--osf-alpha`
- `--osf-beta`
- `--osf-kappa`
- `--router-temperature`
- `--router-min-prob`
- `--router-smoothing`

## Current Layout

- [run_act_pilot.py](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/run_act_pilot.py): canonical experiment entrypoint.
- [core/](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/core): latent geometry, bridge realization, and host backbones.
- [utils/](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/utils): datasets, loaders, evaluators, and ACT-V2 trainers.
- [scripts/](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/scripts): sweep, aggregation, paper-table, and visualization helpers.
- [docs/](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/docs): architecture and maintenance notes.
- [results/](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/results): tracked evidence tables and current experiment outputs.

See [docs/PROJECT_STRUCTURE.md](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/docs/PROJECT_STRUCTURE.md)
for the cleaned structure map.

## Supported Hosts

- `resnet1d`
- `patchtst`
- `timesnet`
- `minirocket` for the original `act` path only

## Environment Rule

Use the `pia` conda environment for all commands in this project:

```bash
conda run -n pia python ...
```

This avoids the missing-dependency noise present in the base environment.
