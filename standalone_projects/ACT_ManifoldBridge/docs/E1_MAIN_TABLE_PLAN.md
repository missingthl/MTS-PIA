# E1 Main Table Plan

This document freezes the current paper-facing E1 main-table design.  E1 is the
external comparison table for CoSTA/CSTA-U5.  It should compare against
recognizable literature families without mixing in internal mechanism controls
or exploratory CSTA variants.

## Included Methods

Use these engineering method names when building the table:

```python
E1_METHODS = [
    "no_aug",
    "raw_aug_jitter",
    "raw_aug_timewarp",
    "raw_mixup",
    "timegan_classwise",
    "diffusionts_classwise",
    "dba_sameclass",
    "wdba_sameclass",
    "rgw_sameclass",
    "dgw_sameclass",
    "csta_topk_uniform_top5",
]
```

Paper display names may differ from engineering names:

| Engineering name | Paper display name |
| --- | --- |
| `no_aug` | No-Aug |
| `raw_aug_jitter` | Jitter |
| `raw_aug_timewarp` | TimeWarp |
| `raw_mixup` | Mixup |
| `timegan_classwise` | TimeGAN-style |
| `diffusionts_classwise` | Diffusion-TS |
| `dba_sameclass` | DBA |
| `wdba_sameclass` | wDBA |
| `rgw_sameclass` | RGW |
| `dgw_sameclass` | DGW |
| `csta_topk_uniform_top5` | CoSTA-U5 |

Important naming note: the codebase uses `timegan_classwise`, not
`timegan_style`.  The paper can display it as `TimeGAN-style` because the
current implementation is a compact PyTorch TimeGAN-style adapter, not a
line-by-line official TensorFlow author-code reproduction.

## Excluded From E1 Main Table

```python
E1_EXCLUDED = [
    "timevae_classwise_optional",
    "raw_aug_magnitude_warping",
    "raw_aug_window_slicing",
    "raw_aug_window_warping",
    "manifold_mixup",
    "raw_smote_flatten_balanced",
    "spawner_sameclass_style",
    "jobda_cleanroom",
    "random_cov_state",
    "pca_cov_state",
]
```

Reasons:

- `timevae_classwise_optional`: excluded from E1 by design; keep for generative
  cost-utility stress or appendix.
- `raw_aug_magnitude_warping`, `raw_aug_window_slicing`,
  `raw_aug_window_warping`: useful transform baselines, but E1 keeps only a
  compact representative transform set.
- `manifold_mixup`: hidden-state vicinal training, not an offline augmentation
  baseline for the main external table.
- `raw_smote_flatten_balanced`: flattened feature-space oversampling; useful
  appendix/rebalancing baseline but not structure-aware.
- `spawner_sameclass_style`, `jobda_cleanroom`: clean-room/style adapters with
  stronger reproduction caveats; keep optional or appendix.
- `random_cov_state`, `pca_cov_state`: internal mechanism controls, not external
  paper methods.

## Main Table Columns

E1 main table is an accuracy-efficiency table. It should not carry provenance
columns; provenance belongs in `e1_method_registry.csv` and the appendix
provenance table.

```text
Method
Family
Avg_F1
Delta_vs_NoAug
WTL_vs_NoAug
Aug_Time
Total_Time
Rel_Cost
```

Definitions:

- `Avg_F1`: dataset-balanced mean macro-F1 over the frozen E1 benchmark.
- `Delta_vs_NoAug`: dataset-level paired improvement over No-Aug.
- `WTL_vs_NoAug`: dataset-level win/tie/loss after seed averaging.
- `Aug_Time`: dataset-balanced mean of `method_cost_sec`.
- `Total_Time`: dataset-balanced mean of `total_sec`.
- `Rel_Cost`: mean over datasets of `Aug_Time(method,d) / Aug_Time(CoSTA-U5,d)`, with CoSTA-U5 as `1.0x`.
- Full coverage is mandatory. Methods without complete E1 coverage cannot enter
  `e1_main_table.csv`; they remain visible in coverage/audit outputs.

These columns are intentionally excluded from E1 main table:

```text
Source_Level
Protocol
Cost_Type
```

They are stored in:

- `results/e1_main/e1_method_registry.csv`
- `results/e1_main/e1_method_provenance_table.csv`
- the appendix provenance table

## Source-Level Mapping

| Method | Source_Level |
| --- | --- |
| No-Aug | reference |
| Jitter / TimeWarp | standard transform baseline supported by the PLOS ONE 2021 time-series augmentation survey |
| Mixup | ICLR 2018 |
| TimeGAN-style | NeurIPS 2019 method family; project-native PyTorch style adapter |
| Diffusion-TS | ICLR 2024 |
| DBA | Pattern Recognition 2011 / tslearn implementation |
| wDBA | weighted DBA-family implementation using tslearn weighted DBA |
| RGW / DGW | guided warping methods, ICPR 2020 / survey-listed family |
| CoSTA-U5 | proposed |

## Family / Protocol Draft

| Method | Family | Protocol | Cost_Type |
| --- | --- | --- | --- |
| No-Aug | Reference | train host classifier without augmentation | none |
| Jitter | Temporal heuristic | offline raw-domain transform | cheap transform |
| TimeWarp | Temporal heuristic | offline raw-domain temporal warping | cheap transform |
| Mixup | Vicinal heuristic | soft-label vicinal training | trainer-side |
| TimeGAN-style | Deep generative | classwise generator fitting + sampling | generator training |
| Diffusion-TS | Deep generative | classwise diffusion generator fitting + sampling | heavy generator training |
| DBA | Alignment / analytical | same-class DTW barycenter synthesis | DTW barycenter |
| wDBA | Alignment / analytical | weighted same-class DTW barycenter synthesis | DTW barycenter |
| RGW | Alignment / analytical | same-class random guided warping | DTW guided warping |
| DGW | Alignment / analytical | discriminative guided warping | DTW guided warping |
| CoSTA-U5 | Proposed | covariance-state top-5 uniform PIA proposal + safe bridge | covariance-state bridge |

## Hard Rules

- Do not use arXiv-only methods in E1.
- Do not include internal controls in E1.
- Do not include hidden-state baselines in E1.
- Do not include TimeVAE in E1.
- Do not treat clean-room/style adapters as official reproductions.
- Do not call `random_cov_state` or `pca_cov_state` external baselines.

## Current Engineering Cross-Check

Known registry names:

```text
timegan_classwise
diffusionts_classwise
csta_topk_uniform_top5
```

Do not introduce a second engineering method name such as `timegan_style`.
Use `timegan_classwise` in scripts and result filters, and map it to
`TimeGAN-style` only at table-render time.
