# CSTA/PIA Experiment Matrix and Issue Log

Last updated: 2026-05-01 17:37 CST.

This note records the current experimental design, known issues, and next execution order for the ACT ManifoldBridge / CSTA-PIA project.  It is intended to keep the experiment protocol stable before further tuning or paper-table expansion.

For a direct map from each external baseline name to its implementation file,
runner arm, and caveats, see:

```text
docs/EXTERNAL_BASELINES.md
utils/external_baseline_manifest.py
```

## Current Method Positioning

- `CSTA` is the full covariance-state augmentation framework.
- `PIA` is the CSTA-internal data-adaptive tangent dictionary operator.
- `TELM2` is the current PIA dictionary estimator.
- The current strongest primary PIA activation policy is `uniform-topK`, specifically `csta_topk_uniform_top5`.
- `group_top` is implemented through the same policy interface, but conceptually remains a neighborhood-consensus extension rather than a primary single-anchor activation policy.

## Locked Pilot7 Datasets

Pilot7 is used as the development matrix:

```text
atrialfibrillation
ering
handmovementdirection
handwriting
japanesevowels
natops
racketsports
```

All locked pilot7 comparisons use:

```text
backbone = ResNet1D
seeds = 1,2,3
epochs = 30
batch_size = 64
lr = 1e-3
patience = 10
val_ratio = 0.2
metric = macro-F1
```

## External Baseline Matrix

### Phase 1

Phase 1 covers initial external baselines:

```text
no_aug
raw_aug_jitter
raw_aug_scaling
raw_aug_timewarp
raw_mixup
dba_sameclass
raw_smote_flatten_balanced
random_cov_state
pca_cov_state
csta_top1_current
csta_group_template_top
```

Phase 1 locked root:

```text
results/csta_external_baselines_phase1/resnet1d_s123/
```

### Phase 2

Phase 2 adds stronger offline time-series augmentations:

```text
raw_aug_magnitude_warping
raw_aug_window_warping
raw_aug_window_slicing
wdba_sameclass
spawner_sameclass_style
jobda_cleanroom
rgw_sameclass
dgw_sameclass
```

Phase 2 locked root:

```text
results/csta_external_baselines_phase2/resnet1d_s123/
```

### Current External Height

Current pilot7 external leaderboard from the eta-fix Step3 report:

```text
wdba_sameclass                 0.667922
dba_sameclass                  0.663309
raw_aug_jitter                 0.661657
raw_aug_timewarp               0.660861
spawner_sameclass_style        0.660813
raw_aug_window_slicing         0.659249
```

Interpretation:

- `wdba_sameclass` remains the current strongest locked external pilot7 baseline.
- CSTA/PIA should be compared primarily against `wdba_sameclass`, `dba_sameclass`, `best_rawaug`, `random_cov_state`, and `pca_cov_state`.
- SMOTE is not budget-matched and must remain labeled as a class-balancing interpolation baseline.

## Additional External Baselines Under Integration

### JobDA-cleanroom

`jobda_cleanroom` is implemented as a clean-room adapter based on the JobDA
paper description.  No confirmed official author code has been found, so this
baseline must be reported as:

```text
based on Ma et al., AAAI 2021; clean-room implementation; official code not confirmed
```

The implementation uses Time-Series Warping (TSW) transforms and JobDA-style
joint labels.  At inference, class probabilities are recovered by summing over
the transform axis.  It is not budget-matched to `multiplier=10`; its effective
augmentation ratio is `num_transforms - 1`.

### RGW/DGW guided warping

`rgw_sameclass` and `dgw_sameclass` are implemented as native `[N,C,T]`
clean-room adapters inspired by the Uchida laboratory guided-warping codebase
(`uchidalab/time_series_augmentation`, Apache-2.0):

```text
rgw_sameclass:
  random same-class prototype
  DTW path prototype -> anchor
  warp anchor along the sample-side path
  interpolate back to original T

dgw_sameclass:
  sample same-class positives and different-class negatives
  select the positive prototype maximizing distance-to-negatives minus distance-to-positives
  DTW-guided warp as above
  optional variable window slicing based on warp amount
```

The current project implementation does not vendor the external Keras/TensorFlow
code.  It provides a reproducible adapter with explicit metadata:

```text
guided_warp_mode
guided_warp_slope_constraint
guided_warp_use_window
guided_warp_use_variable_slice
guided_warp_dtw_value_mean
guided_warp_amount_mean
guided_warp_score_mean
guided_warp_cleanroom_adapter
```

These two arms are pure offline augmentation baselines and should be compared
with DBA/wDBA/SPAWNER-style, not with backbone-only methods.

### TimeVAE classwise optional

`timevae_classwise_optional` is now wired as a Phase 3 generative case-study
arm.  The official TimeVAE repository (`abudesai/timeVAE`, MIT) is
Keras/TensorFlow-based and expects data in `[N,T,D]`.  The current `pia`
environment does not include TensorFlow/Keras, so the project runner uses a
compact PyTorch classwise VAE adapter with native `[N,C,T]` tensors.

This must be reported carefully:

```text
TimeVAE-style classwise generator, clean-room PyTorch adapter.
Official Keras TimeVAE pipeline was inspected but not vendored or executed.
```

The arm records:

```text
target_aug_ratio
actual_aug_ratio
class_fit_success_rate
generation_fail_count
timevae_skipped_classes
timevae_latent_dim
timevae_hidden_dim
timevae_epochs
timevae_beta
timevae_final_loss_mean
timevae_cleanroom_adapter
timevae_official_keras_pipeline
```

Because this is a generative-model baseline, it remains Phase 3 case study
material rather than a Phase 1/2 offline augmentation main-table method.

## CSTA/PIA Sampling and Step3 Sweep

### Sampling Policy Results

Sampling V1 established that `uniform-top5` is stronger than `top1`:

```text
csta_top1_current              about 0.650467
csta_topk_uniform_top5         about 0.663028
```

This supports the PIA narrative shift:

```text
single top-response direction
→ high-response template-neighborhood sampling
```

### Eta-Fix Step3 Sweep

Eta-fix Step3 root:

```text
results/csta_step3_diagnostic_sweep_etafix/resnet1d_s123/
```

Sweep matrix:

```text
policy = csta_topk_uniform_top5
gamma ∈ {0.05, 0.10, 0.20}
eta_safe ∈ {0.25, 0.50, 0.75}
datasets = pilot7
seeds = 1,2,3
```

Completion status:

```text
9 / 9 gamma-eta combinations completed
21 / 21 dataset-seed rows per combination
no failed rows
no NaN / Inf rows
```

Best eta-fix Step3 combo:

```text
g0.1_e0.75
mean F1 over pilot7 = 0.665242
```

Comparison of best Step3 combo:

```text
vs no_aug:             +0.044005 mean delta, 18/21 wins
vs csta_top1_current:  +0.014775 mean delta, 14/21 wins
vs dba_sameclass:      +0.001933 mean delta, 11/21 wins
vs wdba_sameclass:     -0.002680 mean delta, 12/21 wins
```

Interpretation:

- Eta-fix Step3 does not support the earlier stale claim that `g0.2_e0.5` clearly beats wDBA.
- The current fixed candidate is `PIA Uniform-Top5, gamma=0.1, eta_safe=0.75`.
- This candidate matches or slightly trails wDBA on mean F1, while winning 12/21 dataset-seed pairs against wDBA.
- The gap is small and statistically weak; paper wording should be "competitive with wDBA" unless final20 changes the conclusion.

## Candidate Audit Layer

Candidate-level audit is now part of the intended CSTA/PIA evidence layer:

```text
candidate_audit.csv.gz
→ per_seed_external.csv
→ protocol summaries and figures
```

Minimum candidate audit fields include:

```text
candidate_uid
dataset
seed
method
tid
class_id
candidate_order
activation_policy
template_id
template_rank
template_sign
template_response_abs
gamma_requested
gamma_used
safe_radius_ratio
safe_clip_flag
manifold_margin
transport_error_logeuc
bridge_success
candidate_status
```

Known candidate-audit issue to watch:

- `step3_safe_audit.csv` currently reports `gamma_used_mean_mean` as `0.0` in the formal report, while per-seed passthrough fields include nonzero audit-level `gamma_used_mean_audit`.  This suggests a summary-column selection issue rather than a generator issue.
- Before paper use, `build_step3_diagnostic_report.py` should prefer the audit-level fields when present:

```text
gamma_requested_mean_audit
gamma_used_mean_audit
safe_clip_rate_audit
safe_radius_ratio_mean_audit
transport_error_logeuc_mean_audit
```

## Final20 Design

Final20 should only use a fixed configuration selected from pilot7 development results.  Current fixed candidate:

```text
method = csta_topk_uniform_top5
gamma = 0.1
eta_safe = 0.75
multiplier = 10
k_dir = 10
backbone = resnet1d
```

Proposed final20 root:

```text
results/csta_pia_final20/resnet1d_s123/
```

Final20 datasets:

```text
articularywordrecognition
atrialfibrillation
basicmotions
cricket
epilepsy
ering
ethanolconcentration
fingermovements
handmovementdirection
handwriting
har
heartbeat
japanesevowels
libras
motorimagery
natops
pendigits
racketsports
selfregulationscp2
uwavegesturelibrary
```

Important caution:

- Do not use pilot7 test results to keep selecting hyperparameters after this point.
- Treat pilot7 as the development matrix and final20 as the fixed-config evaluation.
- If final20 is used as the main paper table, no further hyperparameter selection should be performed on final20 test results.

## WDBA Final20 Status

There is an active/ongoing or recently launched wDBA final20 process under:

```text
results/wdba_final20/resnet1d_s123/
```

Known issue:

- wDBA is computationally heavy.
- Some shards may run for a very long time.
- Before launching additional heavy jobs, check GPU/process state to avoid oversubscription.

Suggested command:

```bash
ps -ef | rg 'wdba_final20|run_external_baselines_phase1' | rg -v rg
nvidia-smi
```

## Multi-Backbone Plan

Currently supported backbone dispatch includes:

```text
resnet1d
minirocket
patchtst
timesnet
```

Policy:

- Hard-label baselines may use the multi-backbone dispatch.
- `raw_mixup` remains soft-label and is supported only for `resnet1d`.
- `manifold_mixup` remains supported only for `resnet1d`.
- Non-ResNet soft-label attempts must fail fast.

## MPTSNet Candidate Backbone

Repository:

```text
https://github.com/MUYang99/MPTSNet
```

Source status:

- README identifies the work as AAAI 2025.
- README states MIT License, but the cloned repository did not expose a separate `LICENSE` file.
- The model code is concentrated in `model/MPTSNet.py` and `model/layers.py`.

Integration assessment:

- MPTSNet can likely be integrated as `core/mptsnet.py`.
- Its forward interface can remain compatible with our `[B, C, T] → logits` convention.
- The original model internally permutes `[B, C, T]` to `[B, T, C]`.

Required adaptations:

```text
1. Replace timm.models.layers.trunc_normal_ with torch.nn.init.trunc_normal_.
2. Remove direct imports from repository-level utils.py.
3. Implement local FFT period detection and per-batch amplitude weighting.
4. Avoid using test accuracy for checkpoint selection; use our existing val/test evaluator.
5. Add mptsnet to backbone dispatch only after smoke tests pass.
```

Performance caution:

- Original MPTSNet computes FFT amplitude via CPU numpy inside `forward`, which can slow training.
- A torch-native FFT implementation is preferable before full experiments.

Suggested first smoke:

```text
dataset = atrialfibrillation
seed = 1
epochs = 1
arm = no_aug
backbone = mptsnet
```

Do not launch full MPTSNet matrices before this smoke passes.

## Next External Baseline Candidates

This section tracks external methods that may be added after the current CSTA/PIA and wDBA final20 state is stable.  These are not part of the current locked Phase 1/2 matrix.

### Priority Order

First batch:

```text
1. JobDA-cleanroom
2. RGW/DGW adapter
```

Reason:

- Both are direct supervised/time-series augmentation baselines.
- They match the CSTA/PIA comparison protocol better than representation-only or heavy generative methods.
- JobDA covers AAAI supervised augmentation pressure.
- RGW/DGW covers DTW-guided shape-aware augmentation pressure.

Second batch:

```text
3. TimeVAE-classwise
4. Diffusion-TS subset
```

Reason:

- These are generative baselines and should be treated as quality-cost stress tests.
- They should not be mixed into the main offline-augmentation table until their training and generation budgets are explicitly recorded.

Not recommended as a main augmentation baseline:

```text
InfoTS
```

Reason:

- InfoTS is primarily contrastive representation learning with information-aware augmentations.
- It is better treated as an idea source for a future `pia_info_aware_selector` or as an appendix representation baseline, not as a direct supervised augmentation generator.

### Candidate Status Table

| Method | Public implementation status | Engineering recommendation |
| --- | --- | --- |
| `JobDA` | No confirmed official implementation found yet.  The AAAI 2021 paper exists, but current public search did not reveal an author-maintained repository.  A local copy of the paper is stored at `references/papers/jobda_aaai2021.pdf`. | Implemented as `jobda_cleanroom` clean-room v1; document as "based on Ma et al., AAAI 2021; no confirmed official code found." |
| `RGW / DGW guided warping` | `uchidalab/time_series_augmentation` is public, Apache-2.0, and the README links the repository to time-warping/guided-warping time-series augmentation work. | Worth adding after current final20 stabilization.  Prefer extracting the core RGW/DGW logic into a clean adapter rather than vendoring the full Keras/TensorFlow project. |
| `TimeVAE` | `abudesai/timeVAE` is public, MIT, and described as a Keras/TensorFlow implementation for synthetic time-series generation with TimeVAE plus dense/convolutional VAE baselines. | Use as an external classwise generator script, not a `core` model.  Record generation time, fit success rate, target/actual augmentation ratio, and failure counts. |
| `Diffusion-TS` | `Y-debug-sys/Diffusion-TS` is public, MIT, and marked as the ICLR 2024 official implementation. | Use only as a selected subset stress test.  Compare F1 and generation cost.  Do not run first-batch full matrix. |
| `InfoTS` | `chengw07/InfoTS` is public and identifies itself as AAAI 2023 code, but the repository mainly exposes `InfoTS.zip` plus README. | Do not use as a main supervised augmentation baseline.  Optionally inspect later for information-aware selection ideas. |
| `MPTSNet` | `MUYang99/MPTSNet` is public, README identifies AAAI 2025, and states MIT License. | Use as a multi-backbone robustness test, not an augmentation competitor.  Integrate as `core/mptsnet.py` only after smoke tests. |

### Implementation Notes

`JobDA-cleanroom`:

- Scope is small and explicit.
- Do not claim official reproduction.
- Current implementation is registered as `jobda_cleanroom` in the external baseline runner.
- It follows the JobDA paper structure:

```text
sample augmentation = TSW clean-room transform
label augmentation = joint label (original class, transform id)
train output space = C × M
test inference = sum probabilities across transform ids for each original class
```

- Default transform set:

```text
M = 4
transform_subseqs = 0,2,4,8
0 = identity/original time series
2,4,8 = TSW subsequence counts
```

- Current support:

```text
backbone = resnet1d only
label_mode = joint_hard
budget_matched = False
actual_aug_ratio = M - 1 = 3
```

- Add/keep metadata:

```text
source_status = no_confirmed_official_code
implementation_style = cleanroom
paper_basis = JobDA_AAAI2021
jobda_num_transforms
jobda_transform_subseqs
jobda_official_code_confirmed = 0
```

- Smoke status:

```text
dataset = natops
seed = 1
epochs = 1
status = success
```

- Important comparison note:

`jobda_cleanroom` is not budget-matched to CSTA multiplier 10.  The paper uses four transformations including identity, so the effective added-sample ratio is 3.  Do not compare it as if it generated 10N samples.

`RGW/DGW adapter`:

- Keep it train-split-only.
- Reuse the existing external baseline runner.
- Input/output should stay `[N, C, T]`.
- Record:

```text
dtw_elapsed_sec
guided_warping_mode = random | discriminative
teacher_or_discriminative_rule
fallback_count
actual_aug_ratio
```

`TimeVAE-classwise`:

- Keep outside `core`.
- It may require a separate TensorFlow/Keras environment.
- Treat each class model as a generator artifact.
- Record:

```text
target_aug_ratio
actual_aug_ratio
class_fit_success_rate
generation_fail_count
method_elapsed_sec
```

`Diffusion-TS subset`:

- Use only a small case-study subset first, for example:

```text
natops
handwriting
atrialfibrillation
```

- Compare against:

```text
no_aug
best_rawaug
wdba_sameclass
csta_pia_fixed
```

`InfoTS`:

- Do not include in the main augmentation matrix.
- Possible future use:

```text
pia_info_aware_selector
```

where template candidates are selected by fidelity/variety signals derived from candidate audit statistics.

### Source Links

- RGW/DGW implementation source: <https://github.com/uchidalab/time_series_augmentation>
- TimeVAE source: <https://github.com/abudesai/timeVAE>
- Diffusion-TS source: <https://github.com/Y-debug-sys/Diffusion-TS>
- InfoTS source: <https://github.com/chengw07/InfoTS>
- MPTSNet source: <https://github.com/MUYang99/MPTSNet>

## Known Engineering Issues

### 1. Stale Step3 Results

Old root:

```text
results/csta_step3_diagnostic_sweep/resnet1d_s123/
```

Issue:

- Generated before eta-safe propagation was fixed.
- Must be retained only as stale audit evidence.
- Must not be used for final eta/gamma conclusions.

Use instead:

```text
results/csta_step3_diagnostic_sweep_etafix/resnet1d_s123/
```

### 2. Safe Audit Summary Column Ambiguity

Issue:

- Formal safe audit may read legacy summary columns where `gamma_used_mean` appears as zero.
- Candidate audit passthrough contains more reliable audit-level fields.

Required fix:

- Update Step3/protocol summary scripts to prefer `*_audit` fields when available.

### 3. Phase1 Locked Result Mutability

Current worktree shows modified files under:

```text
results/csta_external_baselines_phase1/resnet1d_s123/
```

Risk:

- Locked references should not drift silently.
- If these changes are intentional, document why.
- If not intentional, do not commit them until reviewed.

### 4. Large Result Hygiene

Do not commit:

```text
_csta_runs/
_job_logs/
full raw candidate artifacts beyond lightweight csv.gz audit summaries
large per-job logs
```

Commit only:

```text
code
docs
lightweight protocol summaries
lightweight audit reports
selected per_seed_external.csv when needed for model readability
```

## Immediate Next Actions

1. Fix Step3 report field preference so safe audit uses audit-level candidate diagnostics.
2. Review whether modified Phase1 locked files are intentional.
3. If final20 is launched, use fixed config:

```text
csta_topk_uniform_top5
gamma = 0.1
eta_safe = 0.75
```

4. Keep wDBA final20 status visible before launching additional long jobs.
5. Implement MPTSNet only after current CSTA/PIA run state is stable.

## Paper-Wording Guardrails

Allowed current wording:

```text
CSTA-PIA with Uniform-Top5 and tuned safe radius is competitive with strong DTW-based augmentation on pilot7.
```

Avoid current wording:

```text
CSTA-PIA clearly outperforms wDBA.
```

Reason:

- Best eta-fix Step3 mean F1 is `0.665242`.
- Locked wDBA mean F1 is `0.667922`.
- The mean delta vs wDBA is `-0.002680`, with 12/21 wins but a broad bootstrap CI.
