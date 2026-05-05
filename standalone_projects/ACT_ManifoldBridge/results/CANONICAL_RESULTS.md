# Canonical Results Reference

**Status**: OFFICIAL — This is the single source of truth for paper results.
**Date**: 2026-05-05
**Updated**: from finalized experiment outputs

---

## 1. Primary Method Configuration (LOCKED)

```text
method:              csta_topk_uniform_top5
gamma:               0.1
eta_safe:            0.75
multiplier:          10
k_dir:               10
backbone:            resnet1d
dictionary:          TELM2 (zpia)
activation_policy:   uniform_topk (top-5)
bridge:              whitening-coloring covariance realization
safe_generator:      local_margin_safe_step
```

---

## 2. Main Table — Final20 Core Comparison

**Source**: `final20_minimal_baseline_v1/resnet1d_s123/`

| Method | Mean F1 (20DS) | Gain vs No Aug |
|--------|---------------|----------------|
| wDBA | 0.7300 | +0.0426 |
| **CSTA-U5** | **0.7279** | **+0.0405** |
| DBA | 0.7265 | +0.0391 |
| Random Cov-State | 0.7229 | +0.0355 |
| PCA Cov-State | 0.7221 | +0.0347 |
| Raw Jitter | 0.7123 | +0.0249 |
| No Aug | 0.6874 | — |

---

## 3. CSTA vs wDBA — Formal Alignment

**Source**: `final20_main_comparison_v1/resnet1d_s123/final20_main_comparison_report.md`

| Metric | CSTA-U5 | wDBA | Delta |
|--------|---------|------|-------|
| Mean Aug F1 | 0.7279 | 0.7300 | −0.0021 |
| Dataset W/T/L | 10 / 1 / 9 | — | — |
| Seed W/T/L | 30 / 3 / 27 | — | — |
| Bootstrap CI (95%) | — | — | [−0.0183, +0.0135] crosses zero |

**Conclusion**: CSTA-U5 and wDBA are **statistically indistinguishable** on Final20.

---

## 4. Computational Efficiency

Selected datasets from `final20_main_comparison_v1`:

| Dataset | CSTA (s) | wDBA (s) | Speedup |
|---------|----------|----------|---------|
| motorimagery | 90 | 23318 | **259×** |
| ethanolconcentration | 26 | 2613 | **100×** |
| selfregulationscp2 | 17 | 508 | **30×** |
| heartbeat | 18 | 344 | **19×** |

---

## 5. Ablation — Direction Source (Pilot7)

**Source**: `csta_neurips_ablation_v1/`

Direction bank variants (all with uniform_topk selection, eta_safe=0.75):

| Direction Source | Mean F1 (Pilot7) |
|-----------------|-------------------|
| TELM2 (zpia) | 0.6652 (best) |
| PCA | — |
| Random Orthogonal | — |
| Random | — |

---

## 6. Ablation — Selection Policy (Pilot7)

**Source**: `csta_sampling_v1/`

| Policy | Mean F1 (Pilot7) |
|--------|-------------------|
| uniform_top5 | 0.6630 |
| top1 | 0.6505 |

Conclusion: uniform-topK > top1. Diversity in high-response neighborhood matters.

---

## 7. Mechanism Evidence

**Source**: `local_tangent_audit_v1/` + `csta_mechanism_evidence_v1/`

| Direction Source | Mean Tangent Alignment | Mean F1 (Pilot7) |
|-----------------|----------------------|-------------------|
| PIA selected (U5) | 0.2137 | 0.6652 |
| PIA selected (Top1) | 0.2297 | 0.6505 |
| PCA Cov-State | 0.2290 | 0.6470 |
| Random Cov-State | 0.1403 | 0.6481 |

Key insight: Higher alignment ≠ higher utility. UniformTop5 trades peak alignment for diversity.

---

## 8. Selector Ablation

**Source**: `csta_selector_ablation_v1/`

FV Selector v1 underperforms UniformTop5 by −0.0192 mean F1 (Pilot7).
Archived as negative result. Not part of main method.

---

## 9. Cross-Backbone Robustness (Pilot7)

**Source**: `backbone_robustness_moderntcn_v1/` (ModernTCN), `pilot_patchtst_v1/` (PatchTST)

CSTA-U5 maintains gains across ModernTCN and PatchTST backbones.
Full Final20 cross-backbone table pending.

---

## 10. Mixup Addendum (Final20)

**Source**: `final20_addendum_mixup_v1/`

---

## 11. Key Files for Paper

| Table / Figure | Source |
|---------------|--------|
| Main comparison table | `final20_minimal_baseline_v1/` `overall_summary_external.csv` |
| Per-dataset breakdown | `final20_minimal_baseline_v1/` `dataset_summary_external.csv` |
| CSTA vs wDBA alignment | `final20_main_comparison_v1/` `final20_csta_vs_wdba_per_dataset.csv` |
| Ablation (direction source) | `csta_neurips_ablation_v1/` |
| Ablation (selection policy) | `csta_sampling_v1/` |
| Mechanism (tangent alignment) | `local_tangent_audit_v1/` `local_tangent_audit_report.md` |
| Efficiency (cost pareto) | `csta_protocol_v1/` `figure_cost_pareto.csv` |
| Dataset characteristics | `csta_protocol_v1/` `protocol_dataset_metrics.csv` |
| Negative transfer analysis | `csta_protocol_v1/` `protocol_negative_transfer.csv` |

---

## 12. Config Drift Warning

| Directory | eta_safe | Status |
|-----------|----------|--------|
| `full_scale_resnet1d_v1/` | **0.5** | ❌ NOT canonical — do not use in main table |
| `final20_minimal_baseline_v1/` | **0.75** | ✅ CANONICAL |
| `final20_main_comparison_v1/` | **0.75** | ✅ CANONICAL |
| `csta_pia_final20/` | **0.75** | ✅ CANONICAL |

All paper tables MUST use eta_safe=0.75 results.
