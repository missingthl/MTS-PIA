# CSTA/AO-PIA Results Index

This document defines the official boundaries of experimental evidence for the CSTA (Active Manifold Bridge) framework.

## 1. Official Benchmarks (Phase 1 & Phase 2)
- **Phase 1 (Baselines)**: `results/csta_external_baselines_phase1/resnet1d_s123/`
  - Targets 7 datasets, 3 seeds, 11 methods.
  - Baseline metrics for all comparisons.
- **Phase 2 (Architecture Validation)**: `results/csta_selector_ablation_v1/resnet1d_s123/`
  - Validates E2 (Closed-form residual head) vs E0/E1.
- **Final20 (Top Performance)**: `results/run_final20_resnet1d_s123/`

## 2. Methodology Pilots (AO-PIA)
- **Pilot7 (AO Validation)**: `results/csta_ao_pia_pilot7_v2/resnet1d_s123/`
  - Current AO-Fisher and AO-Contrastive vs CSTA-U5.
  - **Official Evidence**: per_seed_external.csv.

## 3. Mechanism Evidence
- **Local Tangent Audit**: `results/local_tangent_audit_v1/resnet1d_s123/`
  - `local_tangent_overall_summary.csv`: Aggregated using **Dataset-Seed Equal-Weighted Mean**.
- **Evidence Pack**: `results/csta_mechanism_evidence_v1/resnet1d_s123/`
  - Unified table combining alignment and performance.

## 4. Engineering & Safety
- **Safe-Step Audit**: Included in per-seed CSVs as `safe_clip_rate`.
- **AO-Audit**: `is_padded_direction` and `ao_padded_template_usage_rate` are recorded in AO pilot summaries.
