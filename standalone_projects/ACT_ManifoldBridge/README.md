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
E1 table builders, protocol summaries, and multi-backbone adapters.  The main
workflow map is:

- `docs/WORKFLOW.md`: canonical operational workflow and readiness checks.
- `docs/DIRECTORY_GUIDE.md`: shortest map for where code lives.
- `docs/PROJECT_STRUCTURE.md`: project layout and result hygiene.
- `docs/EXTERNAL_BASELINES.md`: where every external augmentation arm lives.
- `docs/PIA_OPERATOR.md`: CSTA/PIA operator contract.

## Quick Start

Before formal matrix work, run the read-only workflow readiness check:

```bash
python standalone_projects/ACT_ManifoldBridge/scripts/check_workflow_readiness.py
```

It verifies canonical entrypoints, locked roots, Final20 roots, and E1 coverage
artifacts.

Run canonical CoSTA-U5 (`zpia_top1_pool` with `topk_uniform_top5`) on one
dataset:

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
  --eta-safe 0.75 \
  --multiplier 10 \
  --template-selection topk_uniform_top5 \
  --out-root standalone_projects/ACT_ManifoldBridge/results/local_run
```

Run the E1/external-baseline matrix through the unified runner:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/run_external_baselines_phase1.py \
  --datasets natops \
  --seeds 1 \
  --arms no_aug,raw_mixup,csta_topk_uniform_top5 \
  --epochs 1 \
  --batch-size 64 \
  --patience 1 \
  --multiplier 1 \
  --out-root /tmp/e1_runner_smoke \
  --device cuda \
  --fail-fast
```

Build auditable E1 atoms and tables from existing run outputs:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/build_e1_main_artifacts.py
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
- `external/`: vendored third-party repositories only, not the external
  comparison matrix.  Current contents are DiffusionTS and TimeVQVAE code trees.
- Current project-native backbones live in `core/`:
  `resnet1d.py`, `patchtst.py`, `timesnet.py`, `mptsnet.py`, and
  `moderntcn.py`; see `docs/BACKBONES.md`.
- `utils/external_baseline_methods/`: actual external baseline implementations:
  raw-domain, DTW, guided-warping, JobDA, TimeVAE-style, SMOTE, and
  covariance-state controls.
- `utils/external_baseline_manifest.py`: searchable method catalog.
- `scripts/run_external_baselines_phase1.py`: historical-name runner for
  Phase 1/2/3 external baseline and CSTA sampling matrices.
- `scripts/list_external_baselines.py`: prints the baseline catalog.
- `archive/release_legacy/scripts/`: pre-U5 MBA/RC4 release-era scripts kept
  for provenance only.
- `results/release_summary/`: compact result tables only; large experiment logs
  are intentionally not part of the release tree.

## Environment

Use the `pia` conda environment:

```bash
conda run -n pia python ...
```
