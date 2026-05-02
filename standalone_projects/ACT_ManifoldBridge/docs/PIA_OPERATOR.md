# PIA Operator in CSTA

## Positioning

CSTA is the complete covariance-state augmentation framework:

1. map a raw multivariate time series into a Log-Euclidean covariance state;
2. apply PIA in that covariance-state tangent space;
3. realize the target state back into raw time-series space with the whitening-coloring bridge;
4. train the downstream ResNet1D on the original plus realized augmentations.

PIA is the CSTA-internal proposal operator. It is a data-adaptive tangent
dictionary operator for covariance-state vicinal augmentation. In the current
engineering split, PIA owns the train-only template dictionary and high-response
candidate proposal. Candidate selection, safe generation, and bridge
realization are separate layers around PIA. It is not a classifier, a feedback
controller, or a raw waveform generator.

## Operator Contract

### 1. Covariance-State Input

Each train split sample is represented as a centered Log-Euclidean covariance-state vector `z`.  PIA operates on this train-only state cloud and never uses validation or test samples to estimate templates.

### 2. Template Dictionary Estimation

The default dictionary estimator is `TELM2`.  The existing `build_zpia_direction_bank(...)` path estimates a train-only dictionary, centers and normalizes the template directions, and records dictionary-level metadata such as `bank_source`, `z_dim`, row-norm summaries, and TELM2 reconstruction diagnostics.

### 3. Anchor-Conditioned Candidate Proposal

Primary PIA proposal policies are single-anchor distributions `q(d | z_i)`
over high-response dictionary directions:

- `top1`: choose the highest absolute-response template.
- `softmax-topK`: sample from the top-K response set with a softmax temperature.
- `uniform-topK`: sample uniformly from the top-K response set.

`group_top` is implemented through the same policy interface for engineering
convenience, but it is conceptually a neighborhood-consensus extension: it
activates templates from a group or neighborhood state `q(d | z_G)`, not from a
purely single-anchor `q(d | z_i)`.

## Pre-Bridge FV Selection

The FV selector is intentionally separate from PIA proposal. Version 1 uses
only pre-bridge fields:

- feasibility: `gamma_used > eps`, `safe_radius_ratio <= 1`,
  `manifold_margin > eps`, `direction_norm > eps`;
- relevance: normalized template response;
- safety balance: preference for a moderate safe-radius ratio around `rho=0.75`;
- variety: displacement magnitude plus template diversity bonus.

`csta_fv_filter_top5`, `csta_fv_score_top5`, and
`csta_random_feasible_selector` are selector arms. They do not use
`bridge_success` or `transport_error_logeuc` in the pre-bridge score.

## Safe Vicinal Generation

CSTA uses the existing local-margin safe-step rule. A requested displacement
scale is clipped by a local inter-class margin proxy, then applied in
covariance-state tangent space. This keeps the generated target state
conservative, but it is not a strict label-preservation guarantee.

## Whitening-Coloring Bridge and Post-Bridge Audit

The bridge is a whitening-coloring covariance realization map. It realizes the
target covariance in raw time-series space and records covariance-target
fidelity diagnostics. `transport_error_logeuc` and `bridge_success` are
post-bridge audit/gate fields, not inputs to the FV selector v1 score.

The legacy field `metric_preservation_error` is kept for CSV compatibility and
should be interpreted as a bridge deformation diagnostic.

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
- selector-level: `feasible_rate`, `selector_accept_rate`,
  `fidelity_score_mean`, `variety_score_mean`, reject counts;
- bridge-level: `transport_error_logeuc_mean`, `aug_valid_rate` when available.

Historical locked results may miss some diagnostics.  Protocol summaries should keep these values as `NaN` and set availability flags rather than re-running locked experiments.

## Current Empirical Status

On the pilot7 sampling audit, `uniform-top5` is currently the strongest primary PIA policy:

- `csta_top1_current`: mean F1 around `0.650467`;
- `csta_topk_uniform_top5`: mean F1 around `0.663028`;
- locked `dba_sameclass`: mean F1 around `0.663309`;
- locked `wdba_sameclass`: mean F1 around `0.667922`.

This motivates describing PIA as high-response template-neighborhood proposal
plus an explicit selector layer, rather than a single top-response template
trick. Final main-method status still depends on restored locked references,
standardized Step3 summaries, and final20 validation.

## Non-Goals for This Layer

PIA does not currently include feedback, OSF, LRAES, VIB, routers, learned
gates, strict label guarantees, exact Bures-Wasserstein OT, or full temporal
motif generation. Those remain historical baselines, limitations, or future
extensions. This operator layer is intentionally a lightweight abstraction over
the existing zPIA/CSTA numerical path.
