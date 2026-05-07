# Directory Guide

This guide is the quick mental model for the ACT/CSTA repository.  It exists
because several directory names are historical and can otherwise be misleading.

## Top-Level Meaning

```text
run_act_pilot.py
  Public single-run CLI for CSTA/PIA experiments.

core/
  Project-native method and model internals.

core/csta/
  CSTA pipeline internals: state extraction, template selection, materializing
  z-space candidates, training dispatch, result rows, and experiment loop.

utils/
  Data loading, evaluator/trainer utilities, external-baseline adapters, and
  method registries.

utils/external_baseline_methods/
  The actual project-native external baseline implementations.

external/
  Vendored third-party repositories only.  This is not the external comparison
  matrix.  Currently it contains DiffusionTS and TimeVQVAE code trees.

scripts/
  Matrix runners, summary builders, and diagnostic command-line tools.

results/
  Experiment outputs.  Locked reference roots and smoke/debug roots coexist
  here, so check docs before writing.

references/
  Local paper PDFs and citation-support material.  Ignored by git.

docs/
  Human-readable method, protocol, baseline, and engineering structure notes.
```

## External Baseline Flow

When you see an external arm such as `wdba_sameclass` or `rgw_sameclass`, follow
this path:

```text
scripts/run_external_baselines_phase1.py
  matrix orchestration
    ↓
utils/external_runner_registry.py
  metadata, phase arms, locked-root guard
    ↓
utils/external_aug_dispatch.py
  method name -> builder function
    ↓
utils/external_baseline_methods/<method>.py
  actual augmentation implementation
```

If an arm needs a vendored third-party repository, the method implementation
uses a wrapper under `utils/`, and the raw third-party tree lives under:

```text
external/<ThirdPartyRepoName>/
```

## Internal Baseline Flow

Internal baselines are project-native CSTA/MBA/RC4 variants.  They do not live
under `external/`.

Historical release-era arms such as `mba_core_lraes`,
`mba_core_rc4_fused_concat`, and `mba_core_zpia_top1_pool` are defined by:

```text
scripts/run_mba_vs_rc4_matrix.py
  release-era internal arm specs
    ↓
run_act_pilot.py
    ↓
core/csta/*
```

Current CSTA/PIA arms such as `csta_topk_uniform_top5` are usually routed
through the external matrix runner for comparison tables, but they are still
internal methods:

```text
scripts/run_external_baselines_phase1.py
  fair matrix runner for external + CSTA arms
    ↓
run_act_pilot.py
    ↓
core/csta/*
```

See:

```text
docs/INTERNAL_BASELINES.md
```

## Backbone Flow

Backbones are downstream classifiers used to evaluate an augmentation method.
They are not external augmentation baselines and they are not CSTA variants.

Model files live under:

```text
core/resnet1d.py
core/patchtst.py
core/timesnet.py
core/mptsnet.py
core/moderntcn.py
```

Training dispatch is split by experiment owner:

```text
run_act_pilot.py --model <backbone>
  ↓
core/csta/training.py
  ↓
utils/evaluators.py

scripts/run_external_baselines_phase1.py --backbone <backbone>
  ↓
utils/backbone_trainers.py
  ↓
utils/evaluators.py
```

See:

```text
docs/BACKBONES.md
```

## CSTA Internal Flow

When you see a CSTA arm such as `csta_topk_uniform_top5`, follow this path:

```text
scripts/run_external_baselines_phase1.py
  fair matrix orchestration
    ↓
run_act_pilot.py
  public CSTA CLI
    ↓
core/csta/cli.py
  argument handling and final CSV
    ↓
core/csta/experiment.py
  dataset/seed loop and pipeline dispatch
    ↓
core/csta/pipelines.py
  CSTA/PIA pipeline orchestration
    ↓
core/csta/template_slots.py + template_policies.py
  template candidate policy and slot construction
    ↓
core/csta/materialize.py + core/bridge.py
  whitening-coloring bridge realization
```

## Result Root Rule

Never use these locked roots for smoke or probe runs:

```text
results/csta_external_baselines_phase1/resnet1d_s123/
results/csta_external_baselines_phase2/resnet1d_s123/
```

Use `/tmp/...` or clearly named local smoke roots instead.
