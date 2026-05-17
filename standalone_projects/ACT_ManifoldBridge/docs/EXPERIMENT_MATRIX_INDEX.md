# Experiment Matrix Index

This page explains how to audit existing ACT/CSTA experiment roots without
manually opening every result directory.

Generated artifacts:

```text
results/experiment_matrix_index_v1/experiment_matrix_index.csv
results/experiment_matrix_index_v1/experiment_matrix_tier_counts.csv
results/experiment_matrix_index_v1/experiment_matrix_index_report.md
```

Regenerate from existing CSVs:

```bash
python scripts/build_experiment_matrix_index.py
```

The script is read-only with respect to experiments: it scans
`per_seed_external.csv` files, classifies each result root, and writes a
governance index.  It does not launch experiments and does not modify locked
Phase1/Phase2 references.

## Tiers

| Tier | Meaning |
| --- | --- |
| `canonical_final20_component` | Locked Final20 method components, such as CSTA-U5 or wDBA. |
| `canonical_final20_controls` | Locked Final20 controls/baselines. |
| `locked_reference_phase1` | Historical locked Phase1 Pilot7 reference. |
| `locked_reference_phase2` | Historical locked Phase2 Pilot7 reference. |
| `backbone_robustness` | Backbone-host robustness roots and per-dataset backbone outputs. |
| `mechanism_or_pilot` | Development or mechanism-exploration matrices. |
| `legacy_noncanonical_eta0.5` | Historical drift-audit roots that should not be used as canonical paper numbers. |
| `smoke_or_probe` | Local validation/probe outputs only. |
| `recovery_or_rebuilt` | Recovered/rebuilt fragments.  Useful, but cite source policy. |
| `external_baseline_matrix` | External baseline matrices not in locked Phase1/2 roots. |

## Current High-Level Counts

As of the latest local index:

```text
canonical Final20 components: 2 roots / 120 rows
canonical Final20 controls:   1 root  / 240 rows
locked Phase1/2 refs:         336 rows total
mechanism or pilot roots:     44 roots / 2335 rows
backbone robustness roots:    103 roots / 722 rows
```

The backbone count is intentionally high because PatchTST, TimesNet,
MiniRocket, and ModernTCN include per-dataset and recovery roots.  For
paper-facing backbone evidence, use:

```text
docs/BACKBONE_U5_MATRIX.md
```

## Smoke Check Used During This Audit

The governance smoke ran to:

```text
/tmp/csta_governance_smoke_u5
```

Methods:

```text
csta_topk_uniform_top5
csta_template_random_within_bank
random_cov_state
```

All three succeeded.  Command JSON confirmed:

```text
csta_topk_uniform_top5            -> --template-selection topk_uniform_top5
csta_template_random_within_bank  -> --template-selection random
random_cov_state                  -> full covariance-state random control path
```

This verifies the three direction-control notions remain separated:

```text
UniformTop5 high-response PIA sampling
random sampling inside TELM2/PIA bank
full covariance-state random direction
```

## Known Schema Caveat

Candidate audit CSVs are intentionally sparse across method families.  Some
columns are `NaN` for template methods or controls.  Do not use a blanket
"all numeric cells finite" check across the entire candidate table.  Instead,
check required fields by method family, such as:

```text
template_id/template_rank/template_response_abs for PIA template methods
gamma_requested/gamma_used/safe_radius_ratio for CSTA generated candidates
transport_error_logeuc for bridge-realized candidates
```
