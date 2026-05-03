# Pilot7 Mechanism Conclusion (FROZEN)

**Status**: FROZEN — Do not reopen for further tuning.
**Date**: 2026-05-03

---

## 1. Phase Definition

This document covers the **Pilot7 method selection + mechanism validation** phase.

- **Pilot7 datasets**: atrialfibrillation, ering, handmovementdirection, handwriting, japanesevowels, natops, racketsports
- **Role**: Development matrix for method selection and mechanism analysis only.
- **NOT**: Final evaluation. Cannot be used as paper main table.

---

## 2. Main Comparison (Pilot7)

| Method | Mean F1 | Role |
| :--- | :--- | :--- |
| `wdba_sameclass` | 0.66792 | Strongest external baseline |
| **`csta_topk_uniform_top5`** | **0.66524** | **Main method (selected)** |
| `dba_sameclass` | 0.66331 | External baseline |
| `csta_top1_current` | 0.65047 | Ablation |
| `random_cov_state` | 0.64813 | Control |
| `no_aug` | 0.62124 | Baseline |

CSTA-U5 vs wDBA: mean delta = -0.00268, 12W/9L seed-level, bootstrap CI crosses zero.
**Conclusion: competitive, not significantly outperforming. Final20 required.**

---

## 3. Selector Ablation Conclusion

- FV Selector v1: underperforms Uniform-Top5 by -0.0192 mean F1.
- Root cause: feasible_rate=1.0, no filtering occurred; fv_score_top5 over-concentrates templates (entropy drop), hurting diversity.
- `fv_filter_top5` ≡ `random_feasible_selector` under current config (all feasible).
- **Decision: FV Selector v1 archived as negative result. Uniform-Top5 confirmed as main method.**

---

## 4. Local Tangent Audit Conclusion

Key result:
- Top1 tangent alignment (0.3036) > UniformTop5 (0.2645), but Top1 F1 < UniformTop5 F1.
- This falsifies "higher alignment → better augmentation".
- Correct interpretation: CSTA-U5 works through **diversity in high-response tangent-relevant neighborhood**, not through maximizing tangent alignment.

**Paper role**: Post-hoc mechanism analysis only. NOT part of main method.

---

## 5. Fixed Config for Final20

```
method:     csta_topk_uniform_top5
gamma:      0.1
eta_safe:   0.75
multiplier: 10
k_dir:      10
backbone:   resnet1d
seeds:      1,2,3
```

**Baselines to include in Final20**:
- no_aug
- wdba_sameclass
- dba_sameclass
- raw_aug_jitter (best raw aug)
- random_cov_state
- pca_cov_state

---

## 6. What Happens After Final20

Three possible outcomes:

| Outcome | Paper Claim |
| :--- | :--- |
| CSTA > wDBA, stable CI | "outperforms strong DTW synthesis baseline" |
| CSTA ≈ wDBA | "competitive with wDBA; better auditability / different dependency mechanism" |
| CSTA < wDBA | "competitive non-DTW covariance-state augmentation; emphasize controls, mechanism, cost" |

---

## 7. What NOT to Do

- Do NOT reopen Pilot7 for further tuning after Final20 is launched.
- Do NOT change gamma/eta/k_dir based on Final20 results (test set contamination).
- Do NOT claim "CSTA significantly outperforms wDBA" until Final20 CI confirms it.
- Do NOT integrate Local Tangent Estimator into main method pipeline.
