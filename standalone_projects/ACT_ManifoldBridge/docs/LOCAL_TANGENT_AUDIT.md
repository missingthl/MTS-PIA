# Local Tangent Audit

## Purpose

The local tangent audit is a post-hoc geometric diagnostic for CSTA/PIA. It does not change the augmentation generator, the bridge, the training loop, or any locked performance result.

The audit asks whether PIA-selected template directions behave like train-only, class-conditional tangent directions in the Log-Euclidean covariance-state cloud.

## Diagnostic Definition

For each train anchor `z_i`, the audit estimates a same-class local tangent basis `U_i`:

1. find same-class nearest neighbors in the train split only;
2. run local PCA on those neighbor states;
3. keep either an explicit tangent dimension or the smallest dimension explaining a configured variance threshold.

For a direction `d`, the tangent alignment is:

```text
alignment(i, d) = ||U_i^T d||^2 / (||d||^2 + eps)
normal_leakage = 1 - alignment
```

The audit compares three direction sources:

```text
pia_selected: selected PIA template directions
random_cov: fixed-seed random covariance-state unit directions
pca_cov: train-only global PCA covariance-state directions
```

For CSTA/PIA methods, the runner prefers real `candidate_audit.csv.gz`
artifacts when available. If a real candidate audit is unavailable, it falls
back to deterministic policy replay and marks the output with:

```text
audit_source = actual_candidate_audit | policy_replay
actual_candidate_audit_available = true | false
```

This distinction matters: `actual_candidate_audit` is the strongest mechanism
evidence because it audits the templates that were actually materialized by a
CSTA run, while `policy_replay` is a faithful but reconstructed diagnostic.

## Scope

The first audit version covers:

```text
csta_topk_uniform_top5
csta_top1_current
random_cov_state
pca_cov_state
```

It writes only diagnostic files under:

```text
results/local_tangent_audit_v1/
```

It must not write to Phase 1 or Phase 2 locked external-baseline roots.

## Top-K Neighborhood Diagnostics

UniformTop5 is not expected to maximize tangent alignment for every anchor.
The audit therefore also records high-response neighborhood statistics:

```text
top1_alignment
top5_alignment_mean
top5_alignment_std
selected_alignment_rank_within_top5
selected_alignment_minus_top5_mean
top5_response_mean
top5_response_std
```

These fields help separate two claims:

```text
1. PIA proposals are more tangent-aligned than random covariance directions.
2. UniformTop5 trades the single highest-alignment direction for high-response
   neighborhood diversity.
```

## Limitations

This audit does not claim to recover the true continuous data manifold. Local PCA is only a finite-sample proxy for class-conditional covariance-state geometry.

The audit also does not provide a strict label-preservation guarantee. It is intended to support or falsify the interpretation that PIA templates align with local same-class covariance-state tangent structure.

Recommended paper wording:

```text
We use local tangent alignment as a diagnostic to support the interpretation
that PIA templates behave as local class-conditional covariance-state tangent
directions.
```
