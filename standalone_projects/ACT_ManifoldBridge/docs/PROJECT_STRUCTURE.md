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
- `run_act_pilot.py`: baseline-vs-ACT experiment orchestration.

## Experimental Path

`core/wavelet_mba.py` and `--pipeline wavelet_mba` are retained as an opt-in
object-layer experiment. They should not be treated as replacing the original
ACT/MBA method body until a separate result matrix supports that change.

## Legacy / Historical Material

Historical sweep managers and one-off evidence scripts live under
`scripts/legacy/`. They are kept for reproducibility, but should not be read as
the main project API.

`results/` contains both publishable tables and old sweep folders. New
experiments should write to a clearly named subfolder such as:

```text
results/wavelet_mba_v1/<dataset-or-matrix-name>/
```

## Environment Rule

Use the `pia` conda environment for all commands in this project:

```bash
conda run -n pia python ...
```

This avoids missing-dependency noise from the base Python environment.
