# ACT ManifoldBridge Engineering Audit

Date: 2026-05-01

This audit records the current engineering state after adding JobDA-cleanroom,
RGW/DGW guided warping, and the first TimeVAE-style Phase 3 adapter.

## Executive Summary

The code surface is moving in the right direction: CSTA/PIA is now separated
from external baselines, multi-backbone dispatch has a dedicated utility layer,
and the PIA operator documentation gives the method a stable conceptual shape.

The main engineering risk is not the algorithm code itself.  The main risk is
result-directory hygiene: one locked Phase 1 result directory has been
overwritten by a small WDBA probe.  That directory must not be committed in its
current modified state.

## Current Directory Roles

```text
ACT_ManifoldBridge/
  run_act_pilot.py
    Canonical single-run CSTA/PIA experiment entrypoint.

  core/
    Method internals: bridge, PIA dictionary/operator/audit, ResNet1D,
    PatchTST, TimesNet, and manifold raw helpers.

  utils/
    Dataset loading, training/evaluation adapters, backbone dispatch, and
    external baseline generators.

  scripts/
    Matrix runners, protocol builders, step3 diagnostics, and lightweight
    shell launchers.

  docs/
    Method positioning, experiment matrix, structure notes, and this audit.

  references/
    Small paper/reference artifacts only.  Large vendored repos should stay out.

  results/
    Mixed state: contains compact summaries, useful protocol outputs, and
    multiple local/smoke/debug roots.  This needs stricter release policy.

  logs/
    Local execution logs.  Not release-critical.
```

## Compile Check

The current Python source files under `ACT_ManifoldBridge` compile with:

```bash
conda run -n pia python -m py_compile $(find standalone_projects/ACT_ManifoldBridge \
  -path '*/__pycache__' -prune -o \
  -path '*/results/*' -prune -o \
  -type f -name '*.py' -print)
```

No syntax-level failures were observed in this audit.

## Current Method Surface

### CSTA/PIA

Maintained method stack:

```text
raw MTS
→ Log-Euclidean covariance state
→ PIA dictionary/operator
→ safe vicinal displacement
→ whitening-coloring bridge
→ backbone training
```

PIA policies currently represented in code:

```text
top1
softmax-topK
uniform-topK
group_top  # neighborhood-consensus extension
```

The current strongest development candidate remains:

```text
PIA Uniform-Top5, eta-fix Step3 candidate g0.1_e0.75
```

but the docs correctly avoid claiming final external SOTA before final20 and
statistical confirmation.

### External Baselines

Current external baseline families:

```text
Raw-domain:
  jitter, scaling, timewarp, magnitude warping, window warping, window slicing

Vicinal/interpolation:
  raw_mixup, SMOTE

DTW/pattern:
  DBA, wDBA, SPAWNER-style, RGW, DGW

Supervised augmentation:
  JobDA-cleanroom

Generative Phase 3:
  timevae_classwise_optional

Hidden-state Phase 3:
  manifold_mixup
```

Discoverability layer added:

```text
utils/external_baseline_manifest.py
  method catalog and source-of-truth lookup table

scripts/list_external_baselines.py
  CLI table/CSV listing for phases and families

docs/EXTERNAL_BASELINES.md
  human-readable index of all external arms, implementation locations, and
  caveats
```

The manifest is synchronized with
`scripts/run_external_baselines_phase1.py::METHOD_INFO` at this audit point:

```text
missing_in_manifest = []
extra_in_manifest = []
```

## Important Findings

### 1. Locked Phase1 Result Directory Was Overwritten

The tracked directory:

```text
results/csta_external_baselines_phase1/resnet1d_s123/
```

currently shows modified files.  `run_config.json` changed from the locked
pilot7 Phase 1 matrix to:

```text
arms = wdba_sameclass
datasets = cricket
epochs = 1
multiplier = 1
```

and `per_seed_external.csv` shrank from the full Phase 1 matrix to one probe row.

Action:

```text
Do not commit these modified Phase1 result files.
Restore the locked refs from git or regenerate them into a new result root.
All smoke/probe runs must use /tmp or a clearly named non-locked root.
```

### 2. TimeVAE Is Not Yet an Official-Equivalent Translation

The official TimeVAE repository is Keras/TensorFlow-based and includes a
TimeVAE decoder with interpretable level/trend/seasonality components.

Current project implementation:

```text
timevae_classwise_optional = compact PyTorch classwise dense VAE adapter
```

This is useful as a runnable Phase 3 generative baseline, but it should not be
called an official TimeVAE reproduction yet.

Policy:

```text
If we port the official Keras architecture to PyTorch with equivalent model
components, we can report it as a PyTorch translation of official TimeVAE.
Until then, report it as TimeVAE-style classwise VAE adapter.
```

### 3. Optional Heavy Dependencies Should Stay Lazy

`external_baselines.py` now lazy-loads `scipy` for magnitude warping and
`scikit-learn` for PCA cov-state.  This is good: unrelated baselines should not
fail because optional dependencies have ABI or installation issues.

Keep this policy:

```text
No top-level import for optional baseline-only dependencies.
Fail only when that specific arm is invoked.
```

### 4. RGW/DGW Dimensional Flow Looks Consistent

The native adapter receives and returns:

```text
X: [N, C, T]
```

Internally DTW operates on:

```text
[T, C]
```

and generated samples are resampled back to:

```text
[C, T]
```

Smoke confirmed both `rgw_sameclass` and `dgw_sameclass` write finite summaries.

### 5. Result Policy Needs a Hard Boundary

Recommended result tiers:

```text
Commit-safe:
  results/release_summary/
  results/csta_protocol_v1/
  selected lightweight final summaries only

Local-only:
  results/*_smoke/
  results/*_debug/
  results/_logs/
  results/**/_csta_runs/
  logs/
  __pycache__/

Locked refs:
  results/csta_external_baselines_phase1/resnet1d_s123/
  results/csta_external_baselines_phase2/resnet1d_s123/
  should never be used as out-root for smoke/probe runs.
```

## Recommended Immediate Cleanup

1. Restore the accidentally modified locked Phase1 result files before any
   GitHub upload.
2. Keep code changes for external baselines, backbone trainers, and evaluators.
3. Keep `docs/EXPERIMENT_MATRIX_AND_ISSUES.md` and this audit document.
4. Keep `references/papers/jobda_aaai2021.pdf` if the repository accepts small
   paper artifacts; otherwise replace it with a citation-only note.
5. Remove or ignore `__pycache__/` and local smoke/debug outputs.

## Recommended Structural Refactor, Later

No urgent large refactor is needed before experiments.  After final20:

```text
scripts/
  run_external_baselines_phase1.py
    should be renamed or wrapped as run_external_baselines.py because it now
    handles Phase 1, Phase 2, and Phase 3 arms.

utils/external_baselines.py
    is large.  Split later into:
      external_raw.py
      external_dtw.py
      external_generative.py
      external_covstate.py

results/
    should keep only compact release/protocol summaries in git.
```

## Current Maturity Assessment

```text
Core CSTA/PIA implementation:        mature enough for final20
External baseline runner:            usable, but file name is outdated
Phase 2 DTW baselines:               strong and expanding
Phase 3 generative baselines:         runnable but not yet official-equivalent
Multi-backbone dispatch:              smoke-level ready
Result hygiene:                       needs immediate cleanup before commit
Paper protocol layer:                 good foundation
```

## Backbone Structure Note

The backbone tree should be read as:

```text
core/resnet1d.py
core/patchtst.py
core/timesnet.py
core/mptsnet.py
utils/evaluators.py
utils/backbone_trainers.py
```

`core/mptsnet.py` is now present so MPTSNet has the same project-native status
as ResNet1D, PatchTST, and TimesNet.  It is a dependency-light PyTorch adapter
following the official MPTSNet design intent: FFT-selected periodic scales,
periodic local CNN modeling, and transformer global dependency modeling.

If the backbone list grows further, the next structural cleanup should introduce:

```text
core/backbones/
```

and keep top-level compatibility shims for the current imports.

Overall:

```text
The engineering is research-mature but not yet release-clean.
The next cleanup should focus on result protection and naming clarity, not on
rewriting the CSTA/PIA algorithm.
```
