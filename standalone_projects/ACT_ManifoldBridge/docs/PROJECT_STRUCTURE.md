# Project Structure

This folder is the active home of `ACT_ManifoldBridge`. The cleaned structure is:

```text
ACT_ManifoldBridge/
  run_act_pilot.py          # canonical experiment entrypoint
  core/                     # method operators and host backbone definitions
  utils/                    # datasets and evaluators
  scripts/                  # analysis, table, visualization, and legacy runners
  docs/                     # architecture and maintenance notes
  results/                  # tracked evidence tables and historical outputs
  requirements.txt
  environment.yml
```

## Publishable Method Body

The publishable core is the original ACT/MBA-style ManifoldBridge chain:

```text
raw trial -> SPD/Log-Euclidean state -> latent candidate -> raw candidate
```

Relevant files:

- `core/pia.py`: PIA and LRAES direction-bank construction.
- `core/curriculum.py`: original latent candidate generation.
- `core/bridge.py`: whitening-coloring realization.
- `core/whitened_edit.py`: VNext white-space line edit and recoloring helpers.
- `run_act_pilot.py`: baseline-vs-ACT experiment orchestration.

## Experimental Path

`--pipeline mba_white_edit` is the MBA-VNext realization experiment. It keeps
the original raw SPD object and LRAES target-state generator, inserts a rank-1
line edit in whitened coordinates, and then trains via the existing weighted
aug-CE host path. It is opt-in and does not replace the publishable ACT/MBA
mainline.

`core/wavelet_mba.py` and `--pipeline wavelet_mba` are retained as opt-in
object-layer experiments. `ca_only` is the V1 low-frequency skeleton path.
`dual_a_dm` is the V2 dual-object path with `cA_2` plus `cD_2` and frozen
`cD_1`. Neither should be treated as replacing the original ACT/MBA method body
until a separate result matrix supports that change.

## Legacy / Historical Material

Historical sweep managers and one-off evidence scripts live under
`scripts/legacy/`. They are kept for reproducibility, but should not be read as
the main project API.

`results/` contains both publishable tables and old sweep folders. New
experiments should write to a clearly named subfolder such as:

```text
results/wavelet_mba_v1/<dataset-or-matrix-name>/
results/wavelet_dual_object_v2/<dataset-or-matrix-name>/
results/mba_white_edit_vnext/<dataset-or-matrix-name>/
```

## Environment Rule

Use the `pia` conda environment for all commands in this project:

```bash
conda run -n pia python ...
```

This avoids missing-dependency noise from the base Python environment.
