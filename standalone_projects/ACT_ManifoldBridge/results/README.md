# Results Directory

**Updated**: 2026-05-05

## Quick Navigation

| You want to... | Go to |
|---------------|------|
| See the **paper main table** | [`CANONICAL_RESULTS.md`](CANONICAL_RESULTS.md) |
| Find which results are **stale/archived** | [`ARCHIVED.md`](ARCHIVED.md) |
| Get the **CSTA vs wDBA alignment report** | [`final20_main_comparison_v1/resnet1d_s123/final20_main_comparison_report.md`](final20_main_comparison_v1/resnet1d_s123/final20_main_comparison_report.md) |
| Get **per-dataset breakdown** | `final20_minimal_baseline_v1/resnet1d_s123/dataset_summary_external.csv` |
| Get **mechanism evidence** (tangent audit) | [`local_tangent_audit_v1/resnet1d_s123/local_tangent_audit_report.md`](local_tangent_audit_v1/resnet1d_s123/local_tangent_audit_report.md) |
| Get **paper figure CSVs** (rank/regime/cost) | [`csta_protocol_v1/`](csta_protocol_v1/) |

## Result Directory Layout

```text
results/
├── CANONICAL_RESULTS.md          ← Authoritative paper data reference
├── ARCHIVED.md                   ← Stale/superseded directories
│
├── final20_main_comparison_v1/   ← CSTA vs wDBA formal alignment (CANONICAL)
├── final20_minimal_baseline_v1/  ← 7-method main table (CANONICAL)
├── final20_addendum_mixup_v1/    ← Mixup addendum (CANONICAL)
├── csta_pia_final20/             ← CSTA-U5 only Final20 (CANONICAL)
├── wdba_final20/                 ← wDBA only Final20 (CANONICAL)
│
├── csta_protocol_v1/             ← Paper figure CSVs (rank, regime, cost)
├── local_tangent_audit_v1/       ← Post-hoc tangent alignment audit
├── csta_mechanism_evidence_v1/   ← Mechanism evidence pack
│
├── csta_step3_diagnostic_sweep_etafix/  ← Pilot7 eta-fix sweep
├── csta_neurips_ablation_v1/            ← 8-condition ablation
├── csta_sampling_v1/                    ← Uniform-top5 vs top1
├── csta_selector_ablation_v1/           ← FV Selector (negative result)
│
├── backbone_robustness_moderntcn_v1/    ← ModernTCN cross-backbone
├── backbone_robustness_mptsnet_v1/      ← MPTSNet cross-backbone
│
├── release_summary/              ← Compact release CSV
│
└── (archived directories)        ← See ARCHIVED.md
```

## Config Lock

The canonical Final20 configuration is **locked**:

```text
method:     csta_topk_uniform_top5
gamma:      0.1
eta_safe:   0.75
multiplier: 10
k_dir:      10
backbone:   resnet1d
```

**Warning**: `full_scale_resnet1d_v1/` uses eta_safe=0.5. Do NOT use in main table.

