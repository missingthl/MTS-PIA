# Internal Baseline Map

This page maps ACT/CSTA internal baselines to their code paths, runners, and
current naming status.

The internal baselines are **project-native method variants**.  They are not
implemented under `external/` and they are not external augmentation baselines.

## Two Naming Eras

The project currently has two internal naming layers:

```text
release-era names:
  mba_core_lraes
  rc4_osf
  mba_core_rc4_fused_concat
  mba_core_zpia_top1_pool
  mba_core_zpia_multidir_pool
  mba_core_rc4_multiz_fused_concat

current CSTA/PIA names:
  csta_top1_current
  csta_topk_uniform_top5
  csta_template_random_within_bank
  csta_fv_filter_top5
  csta_fv_score_top5
  csta_random_feasible_selector
```

Release-era names remain in historical result tables and in
`scripts/run_mba_vs_rc4_matrix.py`.  Current papers and new result tables should
prefer the `csta_*` names where possible.

## Main Internal Baselines

| Internal arm | Code path | Runner | Status |
| --- | --- | --- | --- |
| `baseline_ce` | `core/csta/training.py`, `utils/backbone_trainers.py` | implicit no-augmentation branch in ACT runs | Baseline classifier without augmentation. |
| `mba_core_lraes` | `core/pia.py`, `core/csta/direction_banks.py`, `core/csta/act_builder.py` | `scripts/run_mba_vs_rc4_matrix.py` | Legacy MBA/LRAES internal baseline. |
| `rc4_osf` | `core/csta/rc4_osf_builders.py`, `core/csta/pipelines.py` | `scripts/run_mba_vs_rc4_matrix.py` | Legacy RC4 online structure feedback reference. |
| `mba_core_rc4_fused_concat` | `core/csta/rc4_osf_builders.py`, `core/csta/pipelines.py` | `scripts/run_mba_vs_rc4_matrix.py` | Legacy RC4 fused concat baseline. |
| `mba_core_zpia_top1_pool` | `core/pia.py`, `core/csta/template_pool_builder.py`, `core/csta/template_slots.py` | `scripts/run_mba_vs_rc4_matrix.py` | Release-era name for the older top1 PIA/CSTA candidate. |
| `mba_core_zpia_multidir_pool` | `core/pia.py`, `core/csta/template_pool_builder.py`, `core/csta/template_slots.py` | `scripts/run_mba_vs_rc4_matrix.py` | Release-era multi-direction PIA pool variant. |
| `mba_core_rc4_multiz_fused_concat` | `core/csta/rc4_osf_builders.py`, `core/csta/pipelines.py` | `scripts/run_mba_vs_rc4_matrix.py` | Legacy RC4 + multi-z fused variant. |
| `csta_top1_current` | `core/csta/template_policies.py`, `core/csta/template_slots.py` | `scripts/run_external_baselines_phase1.py` | Current top1 CSTA/PIA ablation, routed through external runner for fair summaries. |
| `csta_topk_uniform_top5` | `core/csta/template_policies.py`, `core/csta/template_slots.py` | `scripts/run_external_baselines_phase1.py`, `scripts/run_csta_pia_final20.sh` | Current canonical CSTA/PIA policy. |
| `csta_template_random_within_bank` | `core/csta/template_policies.py`, `core/csta/template_slots.py` | `scripts/run_external_baselines_phase1.py` | Direction-specificity control: random TELM2 template from the PIA bank. |
| `csta_fv_filter_top5` | `core/csta/template_candidate_scoring.py`, `core/csta/template_slots.py` | `scripts/run_external_baselines_phase1.py` | Archived selector ablation. |
| `csta_fv_score_top5` | `core/csta/template_candidate_scoring.py`, `core/csta/template_slots.py` | `scripts/run_external_baselines_phase1.py` | Archived selector ablation. |
| `csta_random_feasible_selector` | `core/csta/template_candidate_scoring.py`, `core/csta/template_slots.py` | `scripts/run_external_baselines_phase1.py` | Feasible-only selector control. |

## Internal Baseline Flow

Historical release matrix:

```text
scripts/run_mba_vs_rc4_matrix.py
  arm spec
    ↓
run_act_pilot.py
  public ACT/CSTA CLI
    ↓
core/csta/cli.py
    ↓
core/csta/experiment.py
    ↓
core/csta/pipelines.py
    ↓
core/csta/act_builder.py / template_pool_builder.py / rc4_osf_builders.py
```

Current CSTA sampling and selector matrices:

```text
scripts/run_external_baselines_phase1.py
  fair matrix runner
    ↓
_run_csta_method(...)
    ↓
run_act_pilot.py
    ↓
core/csta/*
```

The external runner is used for current CSTA arms only to produce compact,
fairly comparable tables beside external baselines.  Those `csta_*` arms remain
internal methods.

## Result Files

Historical release summaries:

```text
results/release_summary/main_comparison.csv
results/release_summary/internal_ablation_reference.csv
```

Current CSTA/PIA final and mechanism summaries:

```text
results/csta_pia_final20/
results/csta_selector_ablation_v1/
results/csta_direction_specificity_stress_v1/   # when generated
results/local_tangent_audit_v1/
results/csta_mechanism_evidence_v1/
```

## Naming Guidance

Use `docs/METHOD_NAMING.md` for canonical paper names.  In short:

```text
Use `csta_topk_uniform_top5` for the current canonical CSTA/PIA method.
Use release-era `mba_core_*` names only when discussing historical release tables.
Use `random_cov_state` and `pca_cov_state` as covariance-state controls, not as CSTA methods.
```
