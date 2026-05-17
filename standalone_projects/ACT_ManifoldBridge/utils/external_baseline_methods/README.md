# External Baseline Method Implementations

This package is the implementation home for ACT/CSTA external comparison arms.
It is organized so the paper-facing baseline matrix can be understood from the
directory tree without reading runner code.

The repository-level `external/` directory is only for vendored third-party
repositories.  Do not use `external/` as the baseline matrix map.

## Paper-Facing Tree

```text
external_baseline_methods/
  temporal_vicinal/
    Group 1: fast temporal transforms, vicinal training, and feature-space
    oversampling baselines.

  deep_generative/
    Group 2: generator-fitting baselines such as VAE, GAN, VQ-VAE, and
    diffusion-style time-series generators.

  alignment_structure/
    Group 3: analytical or alignment-based structure-preserving baselines such
    as DBA, weighted DBA, guided warping, SPAWNER-style, and JobDA-style arms.

  internal_controls/
    Mechanism controls for CSTA ablations.  These are not external paper
    methods and should not be reported as external baselines.
```

The group directories are lightweight facades.  Runtime implementations remain
in one-method or one-family modules at this package root to preserve stable
imports and avoid duplicating algorithm code.

## Group 1: Temporal / Vicinal Heuristic Augmentations

| Arm | Implementation | Paper-table role |
| --- | --- | --- |
| `raw_aug_jitter` | native/tsaug-style raw transform | E1 classical transform baseline |
| `raw_aug_scaling` | native raw transform | E1 classical transform baseline |
| `raw_aug_timewarp` | tsaug adapter when available | E1 classical transform baseline |
| `raw_aug_magnitude_warping` | native raw transform | E1 classical transform baseline |
| `raw_aug_window_slicing` | native raw transform | E1 classical transform baseline |
| `raw_aug_window_warping` | native raw transform | E1 classical transform baseline |
| `raw_mixup` | native soft-label Mixup training | E1b / secondary vicinal baseline |
| `manifold_mixup` | native hidden-state Mixup training | E1b / secondary vicinal baseline |
| `raw_smote_flatten_balanced` | imbalanced-learn SMOTE adapter | E1b / appendix rebalancing baseline |

These methods are useful because they are fast, simple, and widely recognized.
They do not explicitly control multivariate covariance-state dependency
structure, which is where CoSTA/CSTA enters.

## Group 2: Deep Generative Augmentations

| Arm | Implementation status | Paper-table role |
| --- | --- | --- |
| `timevae_classwise_optional` | PyTorch clean-room / translation-style adapter | E1c generative cost-utility stress test |
| `timegan_classwise` | compact PyTorch TimeGAN-style adapter | E1c GAN representative |
| `timevqvae_classwise` | adapter around vendored TimeVQVAE code | E1c heavy generator baseline |
| `diffusionts_classwise` | adapter around vendored Diffusion-TS code | E1c diffusion representative |

Generative baselines fit an additional model before producing augmented samples.
Report them with compute/protocol caveats where appropriate.  `timevae` and
`timegan` are not official line-by-line reproductions; they are project-native
adapters following the corresponding method families.

## Group 3: Analytical / Alignment-Based Structure-Preserving Augmentation

| Arm | Implementation | Paper-table role |
| --- | --- | --- |
| `dba_sameclass` | tslearn DTW barycenter averaging | E1a classical structural baseline |
| `wdba_sameclass` | weighted DBA-family tslearn barycenter | E1a weighted DBA-family baseline |
| `rgw_sameclass` | clean-room random guided warping | E1a guided warping baseline |
| `dgw_sameclass` | clean-room discriminative guided warping | E1a guided warping baseline |
| `spawner_sameclass_style` | clean-room SPAWNER-style adapter | Optional E1a / appendix with caveat |
| `jobda_cleanroom` | clean-room JobDA-style joint-label adapter | Appendix or optional E1a with caveat |

This group is the closest external family to CoSTA/CSTA in spirit: it tries to
preserve or exploit time-series structure rather than blindly perturbing raw
signals.  Use precise wording for clean-room or style adapters; do not claim
official reproduction unless an official implementation is actually wired in.

## Internal Controls, Not External Baselines

| Arm | Role |
| --- | --- |
| `random_cov_state` | covariance-state random direction control |
| `pca_cov_state` | covariance-state PCA direction control |
| `no_aug` | backbone / no-augmentation reference |

These controls are important for mechanism analysis, but they are not methods
from external augmentation papers.  Keep them out of the external main table.

## Runtime Module Map

```text
base.py
  Shared `ExternalAugResult` dataclass and small array helpers.

raw_jitter.py / raw_scaling.py / raw_timewarp.py
raw_magnitude_warping.py / raw_window_warping.py / raw_window_slicing.py
  Group 1 transform implementations.

raw_mixup.py / manifold_mixup.py / smote.py
  Group 1 vicinal or feature-space baselines.

timevae.py / timegan.py / timevqvae.py / diffusionts.py
  Group 2 deep generative adapters.

dba.py / wdba.py / rgw.py / dgw.py / spawner.py / jobda.py
  Group 3 alignment and structure-preserving adapters.

random_cov_state.py / pca_cov_state.py
  Internal CSTA controls.
```

## Related Files

```text
utils/external_aug_dispatch.py
  Dispatches a method name to one function in this package.

utils/external_baseline_groups.py
  Paper-facing group definitions and method-to-group mapping.

utils/external_runner_registry.py
  Runtime method registry, phase arm groups, CSTA passthrough fields, and
  locked-root protection.

utils/external_baseline_manifest.py
  Searchable display catalog used by `scripts/list_external_baselines.py`.

scripts/run_external_baselines_phase1.py
  Matrix runner.  It should orchestrate experiments, not implement algorithms.
```

## CLI Lookup

List all methods with paper-facing groups:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/list_external_baselines.py
```

Filter one paper group:

```bash
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/list_external_baselines.py \
  --paper-group deep_generative
```

## Maintenance Rules

- Add new external paper methods to one of the three paper-facing groups.
- Add CSTA mechanism controls to `internal_controls/`, not to the external
  baseline groups.
- Keep group directories as facades and documentation anchors; put actual
  implementation code in one-method or one-family modules.
- If a method is a clean-room or style adapter, say that explicitly in docs and
  table captions.
- If a method uses vendored official code, keep the third-party tree under
  `external/` and keep only the adapter here.
