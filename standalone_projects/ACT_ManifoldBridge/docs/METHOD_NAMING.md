# Method Naming Convention

**Status**: NORMATIVE — All new code and documentation should follow these names.
**Date**: 2026-05-05

---

## 1. Canonical Method Name

The locked Final20 method is:

```text
csta_topk_uniform_top5
```

**Decoded**:
- `csta` = Covariance-State Time-Series Augmentation (framework)
- `topk` = top-K template selection
- `uniform` = uniform sampling within top-K
- `top5` = K=5

---

## 2. Naming Taxonomy

### Framework Level
| Name | Meaning |
|------|---------|
| `CSTA` | Covariance-State Time-Series Augmentation (the full framework) |
| `PIA` | Projected Interface Algorithm (the internal operator) |
| `TELM2` | The dictionary estimator used by PIA |

### Method Level — `csta_{selection_policy}`

| Name | Selection | Dictionary | Status |
|------|-----------|------------|--------|
| `csta_topk_uniform_top5` | Uniform over top-5 | TELM2 | **CANONICAL** |
| `csta_top1_current` | Top-1 response | TELM2 | Ablation |
| `csta_group_template_top` | Group-consensus top | TELM2 | Ablation |
| `csta_fv_filter_top5` | FV feasibility filter | TELM2 | Archived (negative) |
| `csta_fv_score_top5` | FV scored selection | TELM2 | Archived (negative) |
| `csta_random_feasible_selector` | Feasible-only random | TELM2 | Control |

### Direction Source Level

| Name | Source | Status |
|------|--------|--------|
| `zpia` / `telm2` | TELM2 dictionary | CANONICAL |
| `lraes` | LRAES Fisher-PIA eigensolver | Legacy |
| `pca` | PCA components | Control baseline |
| `random` | Random Gaussian | Control baseline |
| `random_orthogonal` | Random orthogonal basis | Control baseline |

### External Baseline Level

| Name | Family | Status |
|------|--------|--------|
| `no_aug` | Control | Baseline |
| `random_cov_state` | Covariance-state | Baseline |
| `pca_cov_state` | Covariance-state | Baseline |
| `raw_aug_jitter` | Raw-time | Baseline |
| `raw_aug_scaling` | Raw-time | Baseline |
| `raw_aug_timewarp` | Raw-time | Baseline |
| `raw_aug_magnitude_warping` | Raw-time | Baseline |
| `raw_aug_window_warping` | Raw-time | Baseline |
| `raw_aug_window_slicing` | Raw-time | Baseline |
| `dba_sameclass` | DTW barycenter | Baseline |
| `wdba_sameclass` | DTW barycenter | Baseline |
| `spawner_sameclass_style` | DTW pattern mix | Baseline |
| `jobda_cleanroom` | Supervised aug | Baseline |
| `rgw_sameclass` | Guided warping | Baseline |
| `dgw_sameclass` | Guided warping | Baseline |
| `raw_mixup` | Vicinal | Baseline |
| `manifold_mixup` | Vicinal (hidden state) | Case study |
| `raw_smote_flatten_balanced` | Interpolation | Baseline |
| `timevae_classwise_optional` | Generative | Case study |

---

## 3. Legacy Naming — DO NOT USE in New Code

| Legacy Name | Replaced By | Reason |
|-------------|-------------|--------|
| `zpia_top1_pool` | `csta_top1_current` | Old naming scheme |
| `zpia_multidir_pool` | `csta_topk_uniform_top5` | Old naming scheme |
| `mba_core_lraes` | `csta_lraes` (if revived) | Old MBA prefix |
| `mba_core_zpia_top1_pool` | `csta_top1_current` | Old MBA prefix |
| `mba_core_rc4_fused_concat` | `rc4_osf` | Old MBA prefix |

These legacy names appear in `run_act_pilot.py --algo` and shell scripts.
They resolve to the same code paths as the canonical names.

---

## 4. Code-to-Name Mapping

| CLI `--algo` | Canonical Method Name | Description |
|-------------|----------------------|-------------|
| `zpia` | `csta_*` | TELM2 dictionary + any selection policy |
| `lraes` | `csta_lraes_*` | LRAES dictionary |
| `pca` | `pca_cov_state` | PCA baseline |
| `random` | `random_cov_state` | Random baseline |

| CLI `--template-selection` | Policy Suffix | Description |
|---------------------------|---------------|-------------|
| `top_response` | `top1` | Top-1 response |
| `topk_uniform_top5` | `topk_uniform_top5` | Uniform over top-5 |
| `topk_softmax_tau_X` | `topk_softmax_tau_X` | Softmax sampling with temperature X |

---

## 5. Paper Writing Rules

1. **Always use canonical names** in tables and figures: `csta_topk_uniform_top5`, not `zpia_top1_pool`.
2. **Use consistent family labels**: `Covariance-State` for CSTA/random_cov/pca_cov; `Raw-Time` for jitter/warp; `DTW` for dba/wdba/spawner.
3. **Do NOT invent new names** in the paper that don't match the code or result CSVs.
