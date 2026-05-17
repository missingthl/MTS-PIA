# ACT/CSTA Engineering Refactor Audit

## Status

This audit records the compatibility refactor that started splitting
`run_act_pilot.py` into smaller CSTA modules without changing public CLI,
result schemas, bridge logic, safe-step logic, or canonical CSTA/PIA settings.

Existing result directories are treated as read-only.  In particular, the
locked external references must not be reused as smoke/probe output roots:

```text
results/csta_external_baselines_phase1/resnet1d_s123/
results/csta_external_baselines_phase2/resnet1d_s123/
```

## Extracted Modules

The historical single-run entrypoint now imports these implementation helpers:

```text
core/csta/state.py
  TrialRecord and canonical covariance-state extraction.

core/csta/template_slots.py
  PIA candidate slot construction and audit-row materialization.

core/csta/template_policies.py
  Isolated template-id policy logic for top1, topK, group, and bank-random
  controls.  Pre-bridge FV scoring remains close to slot construction because
  it depends on candidate feasibility fields.

core/csta/template_candidate_scoring.py
  Pre-bridge candidate physics and FV selector scoring helpers.  This keeps
  safe-step scalar diagnostics reviewable without opening full slot-row output.

core/csta/materialize.py
  z-space candidate realization through the whitening-coloring bridge.

core/csta/diagnostics.py
  Host-alignment and template-usage diagnostics.

core/csta/direction_banks.py
  Direction-bank dispatch for LRAES, zPIA, PCA, random-orth, and AO variants.

core/csta/training.py
  Backbone training dispatch wrappers.

core/csta/augment_builders.py
  Compatibility re-export shim for historical imports.

core/csta/act_builder.py
  Base ACT/LRAES/zPIA realized augmentation and feedback margin scoring.

core/csta/template_pool_builder.py
  zPIA/PIA top1/topK/template-pool augmentation builder.

core/csta/rc4_osf_builders.py
  Legacy RC4/OSF fused, spectral OSF, rank-1 OSF, and multi-z fused builders.

core/csta/pipelines.py
  Pipeline orchestration for ACT/CSTA, zPIA template-pool, RC4 fused, multi-z
  fused, and feedback variants.

core/csta/experiment.py
  Dataset/seed loop, split preparation, pipeline dispatch, result-row helper
  calls, candidate-audit helper calls, and optional viz sample writing.

core/csta/result_rows.py
  Success/failure row construction, direction-bank metadata merge, CSTA
  diagnostic merge, and candidate-audit summary write/merge.

core/csta/cli.py
  Argument parser, argument validation, dataset selection, and final CSV
  writing for the public ACT runner.

utils/external_runner_registry.py
  External-runner constants, phase arm groups, method metadata, CSTA diagnostic
  passthrough fields, CSTA policy mapping, and locked-root protection.  This
  keeps the external matrix runner focused on orchestration rather than method
  registration.

utils/external_aug_dispatch.py
  Method-to-external-augmenter dispatch.  External algorithm implementations
  remain in `utils/external_baselines.py`; this layer keeps construction
  branching out of the public matrix runner.

utils/external_baseline_methods/
  Method-specific external augmentation implementations.  The historical
  `utils/external_baselines.py` file is now a compatibility facade.

references/external_baselines/
  Downloaded PDF references and source index for the external baseline papers.
```

`run_act_pilot.py` remains the canonical CLI entrypoint but is now intentionally
thin: it sets thread environment defaults and delegates to `core.csta.cli`.
This compatibility layer is deliberate so external runners and existing shell
launchers do not need new command lines.

## Current Remaining Debt

The remaining high-risk public I/O concerns live mostly in `core/csta/cli.py`
and the public output side effects coordinated by `core/csta/experiment.py`:

```text
experiment loop and CSV writing
argparse
candidate-audit writeout location policy
result schema compatibility through result_rows.py
```

Result-row/schema construction is now centralized in `core/csta/result_rows.py`.
External-runner method registration is now centralized in
`utils/external_runner_registry.py`, external augmentation construction is
dispatched through `utils/external_aug_dispatch.py`, and concrete external
baseline implementations are split under `utils/external_baseline_methods/`.
The next safe extraction step is not more line-count reduction; it is isolating
runner summary writing or splitting legacy RC4 builders further, each guarded by
smoke/identity checks.

## Compatibility Checks

After this refactor, the required checks are:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
  standalone_projects/ACT_ManifoldBridge/core/csta/*.py \
  standalone_projects/ACT_ManifoldBridge/core/pia.py \
  standalone_projects/ACT_ManifoldBridge/core/pia_audit.py \
  standalone_projects/ACT_ManifoldBridge/core/bridge.py \
  standalone_projects/ACT_ManifoldBridge/utils/external_aug_dispatch.py \
  standalone_projects/ACT_ManifoldBridge/utils/external_runner_registry.py \
  standalone_projects/ACT_ManifoldBridge/scripts/run_external_baselines_phase1.py
```

Smoke should use a local or `/tmp` output root, not a locked result root:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/run_external_baselines_phase1.py \
  --out-root /tmp/csta_refactor_smoke \
  --datasets natops \
  --arms csta_template_random_within_bank,csta_topk_uniform_top5,random_cov_state \
  --seeds 1 \
  --epochs 1 \
  --batch-size 64 \
  --lr 1e-3 \
  --patience 1 \
  --val-ratio 0.2 \
  --multiplier 1 \
  --k-dir 10 \
  --pia-gamma 0.1 \
  --eta-safe 0.75 \
  --device cpu
```

Expected smoke properties:

```text
CSTA rows status=success
candidate_audit_available=True
candidate_physics_ok=True
template_response_* diagnostics finite
gamma_used_ratio_mean finite
locked Phase1/Phase2 row counts unchanged
```

## Guardrails

This is an engineering refactor only.  It must not change:

```text
csta_topk_uniform_top5 numerical logic
csta_top1_current numerical logic
FV selector numerical logic
whitening-coloring bridge numerical logic
safe-step numerical logic
public CLI argument names
external runner method registry semantics
locked result files
```
