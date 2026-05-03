# CSTA-PIA Mechanism Evidence v1
**Status**: FROZEN — Pilot7 mechanism validation completed. Do NOT modify for further tuning.
**Date**: 2026-05-03
**Scope**: Pilot7 (7 datasets: atrialfibrillation, ering, handmovementdirection, handwriting, japanesevowels, natops, racketsports)
**Config**: resnet1d, seeds 1/2/3, gamma=0.1, eta_safe=0.75, multiplier=10, k_dir=10

---

## 1. Main Comparison Table (Pilot7)

Source: `main_comparison_pilot7.csv`

| Method | Mean F1 | Note |
| :--- | :--- | :--- |
| `wdba_sameclass` | 0.66792 | Strongest external baseline |
| `dba_sameclass` | 0.66331 | |
| **`csta_topk_uniform_top5`** | **0.66524** | **Main method** |
| `csta_top1_current` | 0.65047 | |
| `random_cov_state` | 0.64813 | Ablation control |
| `pca_cov_state` | 0.64703 | Ablation control |
| `no_aug` | 0.62124 | |

**Current status**: CSTA-U5 is 0.00268 below wDBA (mean). Seed-level win/loss = 12W/9L. Bootstrap CI crosses zero. Conclusion: **competitive, not yet significantly outperforming wDBA on Pilot7.**

---

## 2. Selector Ablation (Pilot7, 21/21 success)

Source: `selector_ablation_summary.csv`

| Selector | Mean F1 |
| :--- | :--- |
| `csta_topk_uniform_top5` | 0.66524 |
| `csta_fv_score_top5` | 0.64605 |
| `csta_fv_filter_top5` | 0.64052 |
| `csta_random_feasible_selector` | 0.64052 |

**Conclusion**: FV Selector v1 underperforms Uniform-Top5. `fv_filter` and `random_feasible` are equivalent (feasible_rate=1.0, pre_filter_reject_count=0). **FV Selector v1 is archived as negative result. Main method remains Uniform-Top5.**

---

## 3. Local Tangent Audit (Pilot7)

Source: `local_tangent_overall_summary.csv`, `local_tangent_alignment_vs_performance.csv`

Key findings:

| Method | Mean F1 | Tangent Alignment |
| :--- | :--- | :--- |
| `csta_topk_uniform_top5` | 0.66524 | 0.264496 |
| `csta_top1_current` | 0.65047 | 0.303621 |
| `random_cov_state` | ~0.648 | (lower) |
| `pca_cov_state` | ~0.647 | (higher than random) |

**Key insight**:
- Top1 has **higher** tangent alignment than UniformTop5 but **lower** F1.
- This means: tangent alignment is a **plausibility indicator**, not a **sufficient utility predictor**.
- CSTA-U5 works through: high-response candidate neighborhood + diversity preservation, NOT by selecting the most tangent-aligned direction.

---

## 4. Paper Wording Guardrails

### SAFE to write:
```
PIA-selected template directions are more locally tangent-aligned than random covariance perturbations.

The highest-alignment direction (Top1) is not the most useful augmentation direction: Top1 achieves higher tangent alignment but lower F1 than UniformTop5.

CSTA benefits from sampling diverse high-response PIA templates rather than greedily selecting a single locally tangent direction.
```

### DO NOT write:
```
Local tangent audit proves PIA recovers the true manifold.
UniformTop5 works because it has the highest tangent alignment.
CSTA significantly outperforms wDBA.
```

---

## 5. What is Fixed for Final20

Config locked:
- method: `csta_topk_uniform_top5`
- gamma: `0.1`
- eta_safe: `0.75`
- multiplier: `10`
- k_dir: `10`
- backbone: `resnet1d`
- seeds: `1,2,3`

Do NOT change any of the above during Final20.

---

## 6. Files in This Directory

| File | Content |
| :--- | :--- |
| `main_comparison_pilot7.csv` | Pilot7 overall rank, all methods |
| `selector_ablation_summary.csv` | 4 selector arms, Pilot7 |
| `selector_win_tie_loss.csv` | Dataset-level W/T/L for selectors |
| `selector_bootstrap_ci.csv` | Bootstrap CI for selector comparison |
| `fv_selector_diagnostics.csv` | FV selector mechanism diagnostics |
| `csta_vs_external_after_selector.csv` | CSTA vs external refs after selector ablation |
| `local_tangent_overall_summary.csv` | Pilot7 aggregated tangent alignment |
| `local_tangent_dataset_summary.csv` | Per-dataset tangent alignment |
| `local_tangent_alignment_vs_performance.csv` | Alignment vs F1 correlation |
