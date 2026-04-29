# PIA Operator in CSTA

## Positioning

CSTA is the complete covariance-state augmentation framework:

1. map a raw multivariate time series into a Log-Euclidean covariance state;
2. apply PIA in that covariance-state tangent space;
3. realize the target state back into raw time-series space with the whitening-coloring bridge;
4. train the downstream ResNet1D on the original plus realized augmentations.

PIA is the CSTA-internal operator.  It is a data-adaptive tangent dictionary operator for covariance-state vicinal augmentation.  In implementation terms, PIA owns the template dictionary, the anchor-conditioned activation policy, and the local safe-step state displacement.  It is not a classifier, a feedback controller, or a raw waveform generator.

## Four-Step Operator Contract

### 1. Covariance-State Input

Each train split sample is represented as a centered Log-Euclidean covariance-state vector `z`.  PIA operates on this train-only state cloud and never uses validation or test samples to estimate templates.

### 2. Template Dictionary Estimation

The default dictionary estimator is `TELM2`.  The existing `build_zpia_direction_bank(...)` path estimates a train-only dictionary, centers and normalizes the template directions, and records dictionary-level metadata such as `bank_source`, `z_dim`, row-norm summaries, and TELM2 reconstruction diagnostics.

### 3. Anchor-Conditioned Activation

Primary PIA activation policies are single-anchor distributions `q(d | z_i)` over high-response dictionary directions:

- `top1`: choose the highest absolute-response template.
- `softmax-topK`: sample from the top-K response set with a softmax temperature.
- `uniform-topK`: sample uniformly from the top-K response set.

`group_top` is implemented through the same policy interface for engineering convenience, but it is conceptually a neighborhood-consensus extension: it activates templates from a group or neighborhood state `q(d | z_G)`, not from a purely single-anchor `q(d | z_i)`.

## Safe Vicinal Generation

PIA uses the existing local-margin safe-step rule.  A requested displacement scale is clipped by a local label-preserving radius estimate, then applied in covariance-state tangent space.  This keeps PIA as a conservative vicinal augmentation operator rather than an unrestricted generator.

The standard operator metadata is:

- `operator_name=PIA`
- `dictionary_estimator=TELM2`
- `activation_policy=top1|softmax_topk|uniform_topk|group_top`
- `activation_scope=single_anchor|neighborhood_consensus`
- `safe_generator=local_margin_safe_step`
- `bridge_realizer=whitening_coloring`

## Diagnostics

Every CSTA/PIA result row should preserve diagnostics from the run-level result CSV when available:

- dictionary-level: `direction_bank_source`, `zpia_z_dim`, `telm2_recon_mean`, row-norm summaries;
- activation-level: `template_usage_entropy`, `selected_template_entropy`, `top_template_concentration`;
- safe-generation-level: `gamma_requested_mean`, `gamma_used_mean`, `safe_clip_rate`, `safe_radius_ratio_mean`;
- bridge-level: `transport_error_logeuc_mean`, `aug_valid_rate` when available.

Historical locked results may miss some diagnostics.  Protocol summaries should keep these values as `NaN` and set availability flags rather than re-running locked experiments.

## Current Empirical Status

On the pilot7 sampling audit, `uniform-top5` is currently the strongest primary PIA policy:

- `csta_top1_current`: mean F1 around `0.650467`;
- `csta_topk_uniform_top5`: mean F1 around `0.663028`;
- locked `dba_sameclass`: mean F1 around `0.663309`;
- locked `wdba_sameclass`: mean F1 around `0.667922`.

This motivates describing PIA as high-response template-neighborhood sampling rather than a single top-response template trick.  Final main-method status still depends on the later gamma and safe-radius tuning stage.

## Non-Goals for This Layer

PIA does not currently include feedback, OSF, LRAES, VIB, routers, or learned gates.  Those remain historical baselines or future extensions.  This operator layer is intentionally a lightweight abstraction over the existing zPIA/CSTA numerical path.
