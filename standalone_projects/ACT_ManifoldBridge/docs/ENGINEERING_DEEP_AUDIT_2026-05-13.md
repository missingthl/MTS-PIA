# Engineering Deep Audit 2026-05-13

This audit was triggered after a backbone-robustness result was missed during
manual reading.  The goal is to reduce future agent and human audit mistakes by
making ownership, result roots, and evidence tiers explicit.

## Current State

The public ACT/CSTA entrypoint is now thin:

```text
run_act_pilot.py -> core.csta.cli.main()
```

Most implementation complexity has moved into:

```text
core/csta/pipelines.py
core/csta/result_rows.py
utils/evaluators.py
utils/external_runner_registry.py
scripts/run_external_baselines_phase1.py
```

The refactor succeeded in moving the old God-file shape out of
`run_act_pilot.py`, but the project is still hard to audit because results and
report scripts are spread across many roots.

## Source-Control Hygiene

The worktree is not clean.  There are tracked modifications, tracked deletions,
and many untracked exploration modules/scripts.  This audit did not revert or
clean any user work.

Important tracked-delete signals observed:

```text
RESULTS_INDEX.md
scripts/analyze_ao_pilot7_corrected.py
utils/timevqvae_wrapper.py
```

Important untracked exploration modules observed:

```text
core/csta/ag_pia.py
core/csta/cs_flow.py
core/csta/latent_residual_flow.py
core/csta/task_guided_latent_residual_flow.py
core/csta/lc_latent_residual_flow.py
core/csta/spg_pia.py
core/csta/spg_cfm.py
```

These are mechanism-exploration lines, not replacements for canonical U5 unless
explicitly promoted by a locked result document.

## Canonical Method Boundary

The frozen main method remains:

```text
csta_topk_uniform_top5
gamma = 0.1
eta_safe = 0.75
multiplier = 10
k_dir = 10
backbone = resnet1d
```

The best-supported paper claim remains:

```text
CSTA-U5 is competitive with wDBA on Final20 and provides strong cost/auditability advantages.
```

Do not claim clear superiority over wDBA unless a later locked result updates
the formal Final20 comparison.

## Backbone Evidence Finding

Backbone robustness had already been run more deeply than the old docs made
obvious.  The current governance entry is:

```text
docs/BACKBONE_U5_MATRIX.md
results/backbone_u5_matrix_v1/
```

Key finding:

```text
ResNet1D, ModernTCN, MiniRocket, PatchTST, and TimesNet all have Final20-style
U5-vs-no_aug evidence, though some non-ResNet rows are rebuilt/recovery-tier.
MPTSNet is currently Pilot7/probe-tier only.
```

## Result Governance Risks

1. `grand_robustness_summary_final` is useful but not sufficient as the only
   paper evidence source.

2. Its generator script used `full_scale_resnet1d_v1` for ResNet1D, while
   `CANONICAL_RESULTS.md` marks that root as non-canonical due to
   `eta_safe=0.5`.

3. PatchTST and TimesNet are stored across per-dataset and recovery folders.
   Different deduplication policies can change pair counts slightly.

4. MiniRocket robustness combines core/batch/recovery roots and must keep that
   caveat unless a clean consolidated root is produced.

## Code Governance Risks

The most important remaining maintainability risks are:

```text
1. core/csta/pipelines.py is still a high-complexity orchestration file.
2. utils/evaluators.py is still a very large shared training/evaluation module.
3. external_runner_registry.py is a single runtime source but also carries a very large passthrough-field list.
4. Many mechanism exploration modules are untracked, which makes reproduction and review fragile.
5. Report scripts have overlapping responsibilities and inconsistent source policies.
```

## Completed Governance Actions

Added:

```text
scripts/build_backbone_u5_matrix.py
docs/BACKBONE_U5_MATRIX.md
results/backbone_u5_matrix_v1/
```

The new script reads existing CSVs only.  It does not run experiments and does
not modify locked Phase1/Phase2 roots.

## Recommended Next Steps

1. Decide which exploration modules are intended to be tracked and which should
   remain local scratch.

2. Treat `docs/BACKBONE_U5_MATRIX.md` as the entrypoint for backbone evidence.

3. Update paper-facing summaries to cite the canonical ResNet1D root:

```text
results/csta_pia_final20/resnet1d_s123/per_seed_external.csv
```

4. Do not run further method exploration until the current dirty worktree is
   grouped into reviewable commits or intentionally archived.

5. If more engineering cleanup is needed, prioritize:

```text
core/csta/pipelines.py -> pipeline output dataclass and method-family dispatch modules
utils/evaluators.py -> backbone-specific evaluator files
utils/external_runner_registry.py -> result schema / method catalog / governance split
```
