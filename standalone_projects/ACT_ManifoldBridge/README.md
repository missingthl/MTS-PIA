# ACT ManifoldBridge

Research implementation for covariance-state Manifold Bridge Augmentation,
CSTA/PIA, and external time-series augmentation baselines on multivariate
time-series classification.

The original release comparison was scoped to three internal method families:

- `mba_core_lraes`: original MBA baseline using LRAES directions in
  Log-Euclidean covariance space.
- `mba_core_rc4_fused_concat`: RC-4 orthogonal structure-risk fusion reference.
- `mba_core_zpia_top1_pool`: current SOTA in this project, using the top-response
  zPIA/TELM2 template direction with MBA core concat training.

The current research tree also contains external baselines, CSTA sampling arms,
protocol summaries, and multi-backbone adapters.  The main maps are:

- `docs/PROJECT_STRUCTURE.md`: project layout and result hygiene.
- `docs/EXTERNAL_BASELINES.md`: where every external augmentation arm lives.
- `docs/PIA_OPERATOR.md`: CSTA/PIA operator contract.

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

List external baselines and their code locations:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/list_external_baselines.py
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
- `utils/`: dataset loading, model evaluation utilities, backbone dispatch, and
  external augmentation implementations.
- Current project-native backbones live in `core/`:
  `resnet1d.py`, `patchtst.py`, `timesnet.py`, and `mptsnet.py`.
- `utils/external_baselines.py`: raw-domain, DTW, guided-warping, JobDA,
  TimeVAE-style, SMOTE, and covariance-state baseline implementations.
- `utils/external_baseline_manifest.py`: searchable method catalog.
- `scripts/run_mba_vs_rc4_matrix.py`: queue runner for release comparison arms.
- `scripts/run_external_baselines_phase1.py`: historical-name runner for
  Phase 1/2/3 external baseline and CSTA sampling matrices.
- `scripts/list_external_baselines.py`: prints the baseline catalog.
- `scripts/summarize_mba_vs_rc4_matrix.py`: summary table generator.
- `results/release_summary/`: compact result tables only; large experiment logs
  are intentionally not part of the release tree.

## Environment

Use the `pia` conda environment:

```bash
conda run -n pia python ...
```
