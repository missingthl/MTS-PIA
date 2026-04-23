# ACT_ManifoldBridge

`ACT_ManifoldBridge` is the standalone implementation of the original
ManifoldBridge augmentation line. The publishable method body is intentionally
kept small:

`x -> z(x) -> z_aug -> x_aug`

The current mainline is the original ACT/MBA-style external augmentation:

- embed each trial as one static SPD / Log-Euclidean state;
- build a PIA or LRAES direction bank;
- generate latent candidates with the curriculum sampler;
- realize candidates back to raw time series through `bridge_single()`;
- train the black-box host on `original + augmented` samples.

`wavelet_mba` is kept as an opt-in experimental path for the next round of
object-layer tests. It applies the same ManifoldBridge idea only to the
low-frequency wavelet skeleton `cA`, freezes all detail coefficients `cD`, and
reconstructs each candidate with one final iDWT.

## Entry Point

Run the original ACT/MBA mainline:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset natops --pipeline act --algo lraes --model resnet1d
```

`--pipeline mba` is accepted as a legacy alias for `act`.

Run the Wavelet-cA experimental path:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset basicmotions --pipeline wavelet_mba --algo lraes --model resnet1d \
  --wavelet-name db4 --wavelet-level auto --wavelet-mode symmetric
```

All experiments in this folder should be launched from the `pia` conda
environment.

## Project Layout

- [run_act_pilot.py](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/run_act_pilot.py): single runnable protocol entrypoint.
- [core/](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/core): geometric operators, bridge realization, and host backbones.
- [utils/](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/utils): dataset loading and host evaluators.
- [scripts/](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/scripts): paper/table/visualization helpers plus legacy sweep scripts.
- [docs/](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/docs): architecture and maintenance notes.
- [results/](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/results): tracked evidence tables and historical experiment outputs.

See [docs/PROJECT_STRUCTURE.md](/home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/docs/PROJECT_STRUCTURE.md)
for the cleaned project map.

## Supported Hosts

- `resnet1d`
- `patchtst`
- `timesnet`
- `minirocket` for the original ACT/MBA path only

## Core Options

- `--pipeline {act,mba,wavelet_mba}`
- `--algo {pia,lraes}`
- `--model {minirocket,resnet1d,patchtst,timesnet}`
- `--k-dir`
- `--pia-gamma`
- `--multiplier`
- `--theory-diagnostics`
- `--disable-safe-step`

Wavelet-only options:

- `--wavelet-name`
- `--wavelet-level`
- `--wavelet-mode`
- `--wavelet-step-tier-ratios`
- `--feedback-margin-temperature`
- `--aug-loss-weight`

## Quick Smoke

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset basicmotions --pipeline act --algo lraes --model resnet1d \
  --seeds 1 --epochs 15 --device cuda
```

Wavelet smoke:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset basicmotions --pipeline wavelet_mba --algo lraes --model resnet1d \
  --seeds 1 --epochs 15 --device cuda \
  --out-root standalone_projects/ACT_ManifoldBridge/results/wavelet_mba_v1/basicmotions
```
