# Project Structure

`ACT_ManifoldBridge` is now organized around one stable core and one active
experimental stack.

## Cleaned Layout

```text
ACT_ManifoldBridge/
  run_act_pilot.py          # canonical ACT / ACT-V2 experiment entrypoint
  core/                     # manifold operators, bridge, direction banks, host backbones
  utils/                    # datasets, loaders, evaluators, ACT-V2 trainers
  scripts/                  # grand sweep, aggregation, paper assets, visualization
  docs/                     # architecture and maintenance notes
  results/                  # tracked evidence tables and current result bundles
  requirements.txt
  environment.yml
```

## Stable Method Body

The stable ACT/MBA method body is still:

```text
raw trial -> SPD / Log-Euclidean state -> latent candidate -> raw candidate
```

Relevant files:

- `core/pia.py`: PIA, LRAES, and zPIA direction-bank construction.
- `core/curriculum.py`: latent candidate generation and Safe-Step control.
- `core/bridge.py`: whitening-coloring realization.
- `run_act_pilot.py`: baseline-vs-ACT orchestration.

## Active Experimental Stack

The active follow-up stack is ACT-V2:

```text
act state -> dual engines (lraes / zpia) -> adaptive utilization / OSF fusion -> host
```

Relevant files:

- `run_act_pilot.py`: `mba_feedback` + `adaptive` orchestration.
- `utils/evaluators.py`: weighted aug-CE, adaptive routing, tau scheduling, focal weighting, consistency regularization.
- `scripts/run_v2_grand_sweep.sh`: current large-scale launcher.
- `scripts/aggregate_v2_grand_sweep.py`: sweep aggregation.

## Removed From Active Surface

Older failed side branches are no longer part of the active working surface:

- wavelet object-layer branch
- MBA white-edit realization branch
- early adaptive validation folders that were superseded by ACT-V2 RC runs

Git history still preserves them, but the current folder now reflects the work
that is still being advanced.

## Results Convention

Active result roots should now look like:

```text
results/mba_zpia_vnext/
results/v2_grand_sweep_rc3_osf_safeclip/
results/v2_taxonomy_analysis/
results/paper_matrix_v2_final/
```

Historical folders are kept only when they still serve audit or paper-evidence
purposes.

## Environment Rule

Always run this project inside the `pia` conda environment:

```bash
conda run -n pia python ...
```
