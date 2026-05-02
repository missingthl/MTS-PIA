# External Baseline Index

This page is the map for the external augmentation and training-strategy arms in
`ACT_ManifoldBridge`.  The implementation is intentionally centralized so
external comparisons do not leak into `run_act_pilot.py`.

## Quick Lookup

Print the current catalog:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/list_external_baselines.py
```

Filter by phase or family:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/list_external_baselines.py --phase phase2
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/list_external_baselines.py --family dtw_guided_warp
```

The source of truth for the catalog is:

```text
utils/external_baseline_manifest.py
```

The runner registry is:

```text
scripts/run_external_baselines_phase1.py
```

The historical runner name still says `phase1`, but it now dispatches Phase 1,
Phase 2, Phase 3, and CSTA sampling arms.

## Code Map

```text
utils/external_baselines.py
  Offline augmentation implementations:
  raw transforms, Mixup generation, DBA/wDBA, SPAWNER-style, RGW/DGW,
  JobDA-cleanroom, TimeVAE-style, SMOTE, random/PCA covariance-state controls.

utils/backbone_trainers.py
  Multi-backbone dispatch for hard-label, soft-label, JobDA, and manifold-mixup
  training paths.

utils/evaluators.py
  Actual model training/evaluation loops.

scripts/run_external_baselines_phase1.py
  Matrix runner and method metadata registry.

scripts/list_external_baselines.py
  Human-readable catalog printer.
```

## Phase 1

Initial baseline matrix:

| Arm | Family | Implementation |
| --- | --- | --- |
| `no_aug` | control | runner branch |
| `raw_aug_jitter` | raw time | `tsaug.AddNoise` adapter |
| `raw_aug_scaling` | raw time | native amplitude scaling |
| `raw_aug_timewarp` | raw time | `tsaug.TimeWarp` adapter |
| `raw_mixup` | vicinal | native Mixup generator + soft-label trainer |
| `dba_sameclass` | DTW barycenter | `tslearn` DBA adapter |
| `raw_smote_flatten_balanced` | flattened raw | `imbalanced-learn` SMOTE adapter |
| `random_cov_state` | covariance state | native random tangent direction |
| `pca_cov_state` | covariance state | train-only PCA tangent direction |
| `csta_top1_current` | CSTA/PIA | `run_act_pilot.py` via `_run_csta_method` |
| `csta_group_template_top` | CSTA/PIA | neighborhood-consensus extension |

Locked reference root:

```text
results/csta_external_baselines_phase1/resnet1d_s123/
```

Do not use this locked root for smoke or debug runs.
The runner now blocks writes to locked Phase 1/2 roots unless
`--allow-locked-root-overwrite` is explicitly supplied for intentional reference
regeneration.

## Phase 2

Stronger offline augmentation baselines:

| Arm | Family | Implementation |
| --- | --- | --- |
| `raw_aug_magnitude_warping` | raw time | native per-channel spline magnitude curve |
| `raw_aug_window_warping` | raw time | native local window speed warp |
| `raw_aug_window_slicing` | raw time | native crop-and-resample |
| `wdba_sameclass` | DTW barycenter | weighted `tslearn` DBA adapter |
| `spawner_sameclass_style` | DTW pattern mix | SPAWNER-style clean-room adapter |
| `jobda_cleanroom` | supervised augmentation | JobDA-style TSW + joint labels |
| `rgw_sameclass` | DTW guided warp | RGW clean-room adapter |
| `dgw_sameclass` | DTW guided warp | DGW clean-room adapter |

Locked reference root:

```text
results/csta_external_baselines_phase2/resnet1d_s123/
```

Implementation caveats:

- `jobda_cleanroom` is based on the JobDA paper description; no confirmed
  official author code has been found.
- `spawner_sameclass_style`, `rgw_sameclass`, and `dgw_sameclass` are
  clean-room/adapted implementations, not vendored external repositories.

## Phase 3

Training-strategy or generative case-study baselines:

| Arm | Family | Implementation |
| --- | --- | --- |
| `manifold_mixup` | hidden-state training | native ResNet1D hidden-state mixup |
| `timevae_classwise_optional` | generative model | PyTorch TimeVAE-style classwise adapter |

Implementation caveat:

- `timevae_classwise_optional` is currently a PyTorch translation-style adapter
  for the project environment.  It is not yet a full parity port of the official
  Keras/TensorFlow TimeVAE pipeline.

## CSTA Sampling Arms

These are not external baselines; they are CSTA/PIA policy variants routed
through the external runner for fair comparison and compact summaries.

| Arm | Meaning |
| --- | --- |
| `csta_topk_softmax_tau_0.05` | PIA top-K softmax activation, tau 0.05 |
| `csta_topk_softmax_tau_0.10` | PIA top-K softmax activation, tau 0.10 |
| `csta_topk_softmax_tau_0.20` | PIA top-K softmax activation, tau 0.20 |
| `csta_topk_uniform_top5` | PIA uniform sampling over top-5 templates |
| `csta_fv_filter_top5` | Pre-bridge feasibility-filtered high-response top-5 selector |
| `csta_fv_score_top5` | Pre-bridge fidelity-variety scored top-5 selector |
| `csta_random_feasible_selector` | Feasible-only random control for the selector layer |

## Running A Small Smoke

Use a throwaway output root under `/tmp` or a clearly named smoke root:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/run_external_baselines_phase1.py \
  --out-root /tmp/csta_external_smoke \
  --datasets natops \
  --arms rgw_sameclass,dgw_sameclass,timevae_classwise_optional \
  --seeds 1 \
  --epochs 1 \
  --batch-size 32 \
  --patience 1 \
  --val-ratio 0.2 \
  --multiplier 1 \
  --device cpu \
  --fail-fast
```

Never target locked roots with smoke runs.
