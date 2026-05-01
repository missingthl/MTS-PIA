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

## Layout

```text
ACT_ManifoldBridge/
  run_act_pilot.py
    Canonical single-experiment entrypoint for CSTA/PIA.

  core/
    Method internals and host backbones.

  utils/
    Dataset loading, evaluator/trainer utilities, backbone dispatch, and
    external baseline generators.

  scripts/
    Matrix runners, protocol builders, diagnostic reports, and shell launchers.

  docs/
    Method positioning, experiment matrix notes, engineering audits, and
    release protocol docs.

  references/
    Small reference papers or citation-support files.  Do not vendor large
    external repositories here.

  results/
    Lightweight summaries and selected protocol artifacts.  Large run outputs,
    smoke/debug roots, and `_csta_runs` should remain local-only.

  logs/
    Local execution logs.  Not required for release claims.
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
  Backbone implementations.

Backbone files are currently kept at the top level of `core/` for compatibility
with existing imports.  If more backbones are added, move them under a dedicated
`core/backbones/` package and keep thin compatibility shims for:

```text
core.resnet1d
core.patchtst
core.timesnet
core.mptsnet
```

utils/evaluators.py
  Training and evaluation loops.

utils/backbone_trainers.py
  Unified hard/soft/jobDA/manifold-mixup backbone dispatch.

utils/external_baselines.py
  External augmentation baselines: raw transforms, Mixup, DBA/wDBA,
  SPAWNER-style, RGW/DGW, JobDA-cleanroom, TimeVAE-style, cov-state baselines.

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
