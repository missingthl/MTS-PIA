# Backbone U5 Matrix

This document is the paper-facing index for CSTA-U5 backbone robustness evidence.
It exists because the underlying results are spread across canonical roots,
rebuilt roots, per-dataset recovery folders, and pilot-only probes.

Generated governance artifacts:

```text
results/backbone_u5_matrix_v1/backbone_u5_sources.csv
results/backbone_u5_matrix_v1/backbone_u5_per_seed.csv
results/backbone_u5_matrix_v1/backbone_u5_summary.csv
results/backbone_u5_matrix_v1/backbone_u5_matrix_report.md
```

Regenerate with:

```bash
python scripts/build_backbone_u5_matrix.py
```

## Current Matrix

| Backbone | Scope | Evidence Tier | Datasets | Pairs | CSTA-U5 | No Aug | Delta | W/T/L | Use |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ResNet1D | Final20 | canonical | 20 | 60 | 0.7279 | 0.6874 | +0.0405 | 39/7/14 | Main canonical host. |
| ModernTCN | Final20 | rebuilt_final20 | 20 | 60 | 0.7297 | 0.6601 | +0.0696 | 46/2/12 | Strong robustness evidence with rebuilt-root caveat. |
| MiniRocket | Final20 | best_available_recovery | 19 | 57 | 0.7402 | 0.7312 | +0.0089 | 25/14/18 | Model-agnostic robustness evidence; cite recovery caveat. |
| PatchTST | Final20 | best_available_recovery | 20 | 59 | 0.6987 | 0.6744 | +0.0243 | 39/5/15 | Transformer robustness evidence; cite recovery/dedup caveat. |
| TimesNet | Final20 | best_available_recovery | 19 | 55 | 0.6750 | 0.5844 | +0.0906 | 41/1/13 | Transformer robustness evidence; cite recovery/dedup caveat. |
| ModernTCN | Pilot7 | pilot_only_u5 | 7 | 21 | 0.6693 | 0.6442 | +0.0250 | 15/1/5 | Pilot/probe only. |
| MPTSNet | Pilot7 | pilot_only_u5 | 7 | 21 | 0.5809 | 0.5246 | +0.0563 | 16/0/5 | Pilot/probe only. |

## Guardrails

- Use the ResNet1D Final20 canonical row for the main method table.
- Use backbone rows as robustness evidence, not as external augmentation baselines.
- Do not promote MPTSNet to Final20 robustness unless a paired Final20 matrix is added.
- Keep `best_available_recovery` language for MiniRocket, PatchTST, and TimesNet unless a clean consolidated root is created.
- Do not cite `full_scale_resnet1d_v1` as canonical; it is retained only for drift auditing because it used `eta_safe=0.5`.

## Why This Exists

Earlier summaries under `grand_robustness_summary_*` were useful but not
sufficiently explicit about source policy.  In particular, one summary script
read the historical ResNet1D `full_scale_resnet1d_v1` root, while the locked
canonical ResNet1D U5 result lives under:

```text
results/csta_pia_final20/resnet1d_s123/per_seed_external.csv
```

This matrix keeps the source policy visible so future audits do not accidentally
mix canonical, recovered, and pilot-only evidence.
