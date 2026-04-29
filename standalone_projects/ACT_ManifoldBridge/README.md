# ACT ManifoldBridge

Minimal release implementation for covariance-state Manifold Bridge
Augmentation on multivariate time-series classification.

The published project is intentionally scoped to three method families:

- `mba_core_lraes`: original MBA baseline using LRAES directions in
  Log-Euclidean covariance space.
- `mba_core_rc4_fused_concat`: RC-4 orthogonal structure-risk fusion reference.
- `mba_core_zpia_top1_pool`: current SOTA in this project, using the top-response
  zPIA/TELM2 template direction with MBA core concat training.

Historical branches such as wavelet objects, adaptive routers, progressive
feedback, spectral OSF, and large paper-asset sweeps were removed from the
release tree so the folder stays readable.

## Quick Start

Run the current SOTA (`zpia_top1_pool`) on one dataset:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  --dataset natops \
  --pipeline act \
  --algo zpia_top1_pool \
  --model resnet1d \
  --seeds 1,2,3 \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-3 \
  --patience 10 \
  --val-ratio 0.2 \
  --k-dir 10 \
  --pia-gamma 0.1 \
  --multiplier 10 \
  --out-root standalone_projects/ACT_ManifoldBridge/results/local_run
```

Run the release comparison matrix:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/run_mba_vs_rc4_matrix.py \
  --out-root standalone_projects/ACT_ManifoldBridge/results/local_matrix \
  --actual-arms mba_core_lraes,mba_core_rc4_fused_concat,mba_core_zpia_top1_pool \
  --gpus 0 \
  --seeds 1,2,3
```

Then summarize:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/summarize_mba_vs_rc4_matrix.py \
  --root standalone_projects/ACT_ManifoldBridge/results/local_matrix
```

## Release Results

The lightweight release table is kept at:

```text
results/release_summary/main_comparison.csv
```

Current 20-dataset summary:

```text
baseline_ce                  mean F1 0.6874
mba_core_lraes               mean F1 0.7182
rc4_osf                      mean F1 0.7100
mba_core_rc4_fused_concat    mean F1 0.7164
mba_core_zpia_top1_pool      mean F1 0.7314
```

## Layout

- `run_act_pilot.py`: canonical single-run entrypoint.
- `core/`: Log-Euclidean bridge, zPIA/TELM2 direction banks, LRAES/MBA helpers,
  and host model definitions.
- `utils/`: dataset loading and model evaluation utilities.
- `scripts/run_mba_vs_rc4_matrix.py`: queue runner for release comparison arms.
- `scripts/summarize_mba_vs_rc4_matrix.py`: summary table generator.
- `results/release_summary/`: compact result tables only; large experiment logs
  are intentionally not part of the release tree.

## Environment

Use the `pia` conda environment:

```bash
conda run -n pia python ...
```
