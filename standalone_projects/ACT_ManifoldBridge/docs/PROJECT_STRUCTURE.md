# Project Structure

`ACT_ManifoldBridge` is now trimmed to a release-oriented surface. The folder
keeps only the runnable method code, the release matrix scripts, and compact
summary results.

## Layout

```text
ACT_ManifoldBridge/
  run_act_pilot.py          # canonical single-experiment entrypoint
  core/                     # manifold bridge, direction banks, host backbones
  utils/                    # dataset and evaluator utilities
  scripts/                  # release matrix runner and summarizer
  docs/                     # project notes
  results/release_summary/  # compact release-facing result tables
  requirements.txt
  environment.yml
```

## Release Method Surface

The maintained comparison surface is:

- `mba_core_lraes`: original Manifold Bridge Augmentation baseline.
- `mba_core_rc4_fused_concat`: RC-4 structure-risk fusion reference.
- `mba_core_zpia_top1_pool`: current default and strongest release method.

The shared method body is:

```text
raw trial -> covariance SPD -> Log-Euclidean state z
          -> candidate state z_cand
          -> bridge realization back to raw trial
          -> core concat CE training
```

## Key Files

- `core/bridge.py`: SPD/log-Euclidean bridge and raw-domain realization.
- `core/pia.py`: LRAES and zPIA/TELM2 direction-bank construction.
- `core/curriculum.py`: candidate generation and safe-step utilities.
- `run_act_pilot.py`: experiment orchestration.
- `scripts/run_mba_vs_rc4_matrix.py`: queue runner for the release arms.
- `scripts/summarize_mba_vs_rc4_matrix.py`: summary generation.

## Results Policy

Large raw experiment logs are not part of the release folder. Keep only compact
tables needed to understand the maintained claims:

```text
results/release_summary/main_comparison.csv
results/release_summary/internal_ablation_reference.csv
```

## Environment

Use:

```bash
conda run -n pia python ...
```
