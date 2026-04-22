# ACT_ManifoldBridge

`ACT_ManifoldBridge` has been cleaned back to the original ACT mainline.

The repository now centers on a single geometric augmentation chain:

`x -> z(x) -> z_aug -> x_aug`

In practice, this means:

- each trial is embedded as a static SPD / Log-Euclidean point
- local directions come from `PIA` or `LRAES`
- augmented latent points are generated with the original curriculum sampler
- `bridge_single()` realizes them back into raw time-series samples
- the host is trained on `orig + aug` in the standard supervised setting

The older heavy feedback / ACL exploration path has been removed from the main project entrypoints so the codebase stays focused on the original ACT method body.

## Main Entry

Run the original ACT pipeline:

```bash
python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset natops --pipeline act --algo lraes --model resnet1d
```

`--pipeline mba` is still accepted as a legacy alias for the same original ACT path.

## Project Structure

- [run_act_pilot.py](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/run_act_pilot.py): single ACT protocol entrypoint
- [core/pia.py](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/core/pia.py): direction-bank construction
- [core/curriculum.py](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/core/curriculum.py): original latent candidate generation
- [core/bridge.py](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/core/bridge.py): whitening-coloring bridge
- [utils/evaluators.py](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/utils/evaluators.py): host training and evaluation

## Supported Hosts

- `resnet1d`
- `patchtst`
- `timesnet`
- `minirocket`

## Core Options

- `--algo {pia,lraes}`
- `--model {minirocket,resnet1d,patchtst,timesnet}`
- `--k-dir`
- `--pia-gamma`
- `--multiplier`
- `--theory-diagnostics`
- `--disable-safe-step`

## Quick Start

Run a small smoke:

```bash
python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset basicmotions --pipeline act --algo lraes --model resnet1d \
  --seeds 1 --epochs 30 --device cuda
```

Run a full sweep over AEON fixed-split datasets:

```bash
python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --all-datasets --pipeline act --algo lraes --model resnet1d \
  --out-root standalone_projects/ACT_ManifoldBridge/results/act_core
```
