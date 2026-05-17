# ACT Engineering Scope

更新时间：2026-05-13

本轮工程治理只看：

```text
standalone_projects/ACT_ManifoldBridge/
```

不要用相邻目录解释 ACT 当前状态，也不要从相邻目录推断 ACT 的入口、
结果口径或模块职责。

## Scope

In scope:

```text
run_act_pilot.py
core/
utils/
scripts/
docs/
results/
release_artifacts/
```

Out of scope for this ACT audit:

```text
../ACT_Bridge_Export/
../RouteB_GeometricAugmentation/
root-level scripts/
root-level results/
root-level docs/
```

这些目录可能有历史或并行价值，但本轮不解释、不搬迁、不清理、不作为
ACT 工程成熟度判断依据。

## Current ACT Mental Model

ACT/CSTA 当前按下面的工程边界阅读：

```text
run_act_pilot.py
  public single-run CLI shim

core/csta/
  CSTA internal pipeline, state extraction, candidate construction,
  materialization, result rows, training dispatch, and mechanism probes

core/
  downstream backbone implementations and core bridge/PIA primitives

utils/external_baseline_methods/
  project-native external augmentation baseline implementations

utils/external_runner_registry.py
  external-matrix method registry, CSTA arm metadata, locked-root guard

scripts/run_external_baselines_phase1.py
  fair matrix runner for external baselines and CSTA internal arms

scripts/build_*report.py
  experiment-specific report builders

results/
  ACT experiment outputs, including canonical, pilot, mechanism, smoke,
  recovery, and locked-reference roots
```

## Governance Entrypoints

Use these before reading raw result directories:

```text
docs/DIRECTORY_GUIDE.md
docs/PROJECT_STRUCTURE.md
results/CANONICAL_RESULTS.md
docs/EXPERIMENT_MATRIX_INDEX.md
docs/BACKBONE_U5_MATRIX.md
docs/ENGINEERING_DEEP_AUDIT_2026-05-13.md
```

## Cleanup Principle

The goal is maintainability, not cosmetic directory motion.

Do:

```text
1. keep ACT public CLIs stable;
2. keep canonical U5 unchanged;
3. run smoke tests under /tmp;
4. keep locked Phase1/Phase2 roots untouched;
5. add indexes and ownership docs before moving files;
6. split modules only when it lowers review burden.
```

Do not:

```text
1. move or delete existing result roots during audit;
2. mix root-level results into ACT paper tables;
3. cite smoke/recovery roots without tier labels;
4. treat mechanism probes as canonical methods unless promoted by result docs;
5. add new generation mechanisms while engineering ownership is unclear.
```

## Immediate Engineering Priorities

1. Keep `run_act_pilot.py` thin and avoid adding logic back into it.
2. Keep `scripts/run_external_baselines_phase1.py` as orchestration only:
   method-specific command construction belongs in `utils/csta_method_commands.py`,
   and method-family execution should stay in small helpers.
3. Keep `core/csta/pipeline_registry.py` as a static dispatch table.  Prefer
   explicit routing over dynamic plugins until the paper-facing method set is
   fully frozen.
4. Use `scripts/audit_csta_schema.py --fail-on-warning` after result-field or
   passthrough changes.
5. Make `core/csta/pipelines.py` easier to audit by separating method-family
   orchestration from common output assembly.
6. Split `utils/evaluators.py` only after smoke parity is easy to check.
7. Keep `utils/external_runner_registry.py` as the running method source of
   truth, but consider later splitting schema/governance constants out of it.
8. Use generated indexes for result navigation instead of opening random
   `results/*v1/*recovery/*smoke` folders manually.

## Smoke Baseline For Refactors

Before and after non-trivial ACT refactors, run:

```bash
python scripts/run_external_baselines_phase1.py \
  --datasets natops \
  --seeds 1 \
  --arms csta_topk_uniform_top5,csta_template_random_within_bank,random_cov_state \
  --backbone resnet1d \
  --epochs 1 \
  --batch-size 64 \
  --lr 1e-3 \
  --patience 1 \
  --multiplier 1 \
  --pia-gamma 0.1 \
  --eta-safe 0.75 \
  --k-dir 10 \
  --device cpu \
  --out-root /tmp/csta_governance_smoke_u5 \
  --fail-fast
```

This smoke confirms the three key direction-control paths remain separated:

```text
csta_topk_uniform_top5
  high-response UniformTop5 PIA sampling

csta_template_random_within_bank
  random direction sampled inside TELM2/PIA support bank

random_cov_state
  full covariance-state random direction control
```
