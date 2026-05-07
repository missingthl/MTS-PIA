# Project Structure

`ACT_ManifoldBridge` is the standalone research module for CSTA/PIA and its
external augmentation baselines.  The current project is no longer only the
early MBA/RC4/zPIA comparison surface; it now contains a fuller CSTA experiment
stack:

```text
CSTA/PIA core method
external augmentation baselines
multi-backbone training adapters
protocol summaries and audit outputs
```

For a shorter "where do I look first?" map, see:

```text
docs/DIRECTORY_GUIDE.md
```

## Layout

```text
ACT_ManifoldBridge/
  run_act_pilot.py
    Canonical single-experiment entrypoint for CSTA/PIA.

  core/
    Method internals and host backbones.

  utils/
    Dataset loading, evaluator/trainer utilities, backbone dispatch, and
    external baseline adapters.

  scripts/
    Matrix runners, protocol builders, diagnostic reports, and shell launchers.

  docs/
    Method positioning, experiment matrix notes, engineering audits, and
    release protocol docs.

  references/
    Small reference papers or citation-support files.  Do not vendor large
    external repositories here.  External-baseline PDFs and their source index
    live under `references/external_baselines/`.

  results/
    Lightweight summaries and selected protocol artifacts.  Large run outputs,
    smoke/debug roots, and `_csta_runs` should remain local-only.

  logs/
    Local execution logs.  Not required for release claims.
```

Important naming caveat:

```text
external/
  Vendored third-party repositories only.  It is not the external comparison
  matrix and currently contains only DiffusionTS / TimeVQVAE code trees.

utils/external_baseline_methods/
  Actual project-native external baseline method implementations.
```

## Core Method Surface

CSTA is the full covariance-state augmentation framework:

```text
raw multivariate time series
→ Log-Euclidean covariance state
→ PIA operator
→ safe vicinal state displacement
→ whitening-coloring bridge
→ backbone training
```

PIA is the core operator inside CSTA:

```text
train-only covariance-state cloud
→ TELM2 tangent dictionary
→ anchor-conditioned activation policy
→ local-margin safe-step generation
```

Primary PIA activation policies:

```text
top1
softmax-topK
uniform-topK
```

`group_top` is implemented through the same policy interface but is interpreted
as a neighborhood-consensus extension.

## Key Code Files

```text
core/bridge.py
  Log-Euclidean target realization back to raw time-series space.

core/csta/
  Refactored CSTA internals used by the historical `run_act_pilot.py`
  entrypoint.  This package is the preferred place for new CSTA implementation
  logic.

core/csta/state.py
  Trial records and canonical Log-Euclidean covariance-state extraction.
  Local tangent audits and CSTA training runs should share this builder.

core/csta/template_slots.py
  Converts template policy outputs into concrete candidate slots and audit rows.

core/csta/template_policies.py
  Reviewable PIA template-id policy logic: top1, topK uniform/softmax,
  group policies, bank-random controls, and policy neighbor preparation.

core/csta/template_candidate_scoring.py
  Pre-bridge candidate physics and FV selector scoring helpers used by
  template slot construction.

core/csta/materialize.py
  z-space candidate materialization through the whitening-coloring bridge and
  bridge/candidate metric aggregation.

core/csta/augment_builders.py
  Compatibility re-export shim for historical imports.  New implementation
  logic should live in the mechanism-specific builder modules below.

core/csta/act_builder.py
  Base ACT/LRAES/zPIA realized augmentation and feedback margin scoring.

core/csta/template_pool_builder.py
  zPIA/PIA top1/topK/template-pool augmentation builder.

core/csta/rc4_osf_builders.py
  Legacy RC4/OSF fused, spectral OSF, rank-1 OSF, and multi-z fused builders.

core/csta/pipelines.py
  High-level training pipeline orchestration for ACT/CSTA, zPIA template-pool,
  RC4 fused, multi-z fused, and feedback variants.

core/csta/experiment.py
  Dataset/seed loop, train/val/test split preparation, pipeline dispatch,
  result-row helper calls, candidate-audit helper calls, and optional viz sample
  writing for the public ACT runner.

core/csta/result_rows.py
  Centralized success/failure row construction, direction-bank metadata merge,
  CSTA diagnostic merge, and candidate-audit summary write/merge.

core/csta/cli.py
  Argument parser, argument validation, dataset selection, and final CSV writing
  for the public ACT runner.

core/csta/diagnostics.py
  Host-alignment probes, template usage summaries, and response diagnostics.

core/csta/direction_banks.py
  Dispatch for LRAES/zPIA/PCA/random-orth/AO direction-bank builders.

core/csta/training.py
  Thin backbone-training dispatch wrappers used by `run_act_pilot.py`.

run_act_pilot.py
  Thin public entrypoint that preserves environment-thread defaults and calls
  `core.csta.cli.main()`.  Keep command-line compatibility here.

core/pia.py
  LRAES/zPIA/TELM2 direction-bank construction.

core/pia_operator.py
  Lightweight PIA operator metadata and abstraction layer.

core/pia_audit.py
  Candidate-level audit helpers.

core/resnet1d.py
core/patchtst.py
core/timesnet.py
core/mptsnet.py
core/moderntcn.py
  Backbone implementations.

Backbone files are currently kept at the top level of `core/` for compatibility
with existing imports.  If more backbones are added, move them under a dedicated
`core/backbones/` package and keep thin compatibility shims for:

```text
core.resnet1d
core.patchtst
core.timesnet
core.mptsnet
core.moderntcn
```

utils/evaluators.py
  Training and evaluation loops.

utils/backbone_trainers.py
  Unified hard/soft/jobDA/manifold-mixup backbone dispatch.

utils/external_baselines.py
  Compatibility facade that re-exports external augmentation methods from
  `utils/external_baseline_methods/`.

utils/external_baseline_methods/
  Method-specific external baseline implementations: raw transforms, Mixup,
  DBA/wDBA, SPAWNER-style, RGW/DGW, JobDA-cleanroom, TimeVAE-style generators,
  SMOTE, and naive covariance-state controls.

utils/external_aug_dispatch.py
  Method-to-augmenter dispatch used by the external matrix runner.  This keeps
  runner orchestration separate from external augmentation construction.

utils/external_runner_registry.py
  External-runner method registry, phase arm groups, CSTA passthrough fields,
  method metadata, CSTA policy mapping, and locked-result-root guard.

utils/external_baseline_manifest.py
  Searchable catalog that maps every external/CSTA sampling arm to its
  implementation file, runner name, method family, source space, and caveats.

scripts/run_external_baselines_phase1.py
  Current general external-baseline runner despite the historical name.

scripts/list_external_baselines.py
  Prints the external baseline catalog as a table or CSV.

scripts/build_csta_protocol_summary.py
  Unified protocol summary builder.

scripts/build_step3_diagnostic_report.py
  Step3 gamma/eta diagnostic reporting.

docs/EXTERNAL_BASELINES.md
  Human-readable index for all external baseline arms and their code locations.

docs/INTERNAL_BASELINES.md
  Human-readable index for internal CSTA/MBA/RC4 baselines, release-era arm
  names, current CSTA arm names, runners, and implementation locations.

docs/BACKBONES.md
  Human-readable index for downstream host backbones, model files, dispatch
  layers, and hard-label versus soft-label support boundaries.

docs/EXTERNAL_BASELINE_PAPERS.md
  Tracked source index for baseline papers downloaded locally under ignored
  `references/external_baselines/`.

docs/DIRECTORY_GUIDE.md
  Short directory map that explains the CSTA flow, external-baseline flow, and
  why `external/` is a vendor-code directory rather than the method matrix.
```

## Result Policy

Commit-safe result roots:

```text
results/release_summary/
results/csta_protocol_v1/
selected compact final summaries only
```

Local-only result roots:

```text
results/*_smoke/
results/*_debug/
results/_logs/
results/**/_csta_runs/
logs/
__pycache__/
```

Locked references should never be used as smoke/probe `--out-root` targets:

```text
results/csta_external_baselines_phase1/resnet1d_s123/
results/csta_external_baselines_phase2/resnet1d_s123/
```

If a locked root is accidentally overwritten, restore it before committing.

## Environment

Use the project environment:

```bash
conda run -n pia python ...
```

Optional baseline dependencies should be lazy-loaded inside their specific arms.
An unrelated missing or broken dependency must not prevent importing the whole
external baseline module.
