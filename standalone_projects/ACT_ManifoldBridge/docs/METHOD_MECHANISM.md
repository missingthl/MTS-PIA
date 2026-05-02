# CSTA/PIA Mechanism

## Problem Setting

CSTA is a train-split-only data augmentation framework for multivariate time
series classification. It augments the training set, then trains a downstream
backbone with the same evaluation protocol used by external baselines.

The method targets covariance-dependency augmentation. It does not claim to
model arbitrary temporal motifs or to replace raw-time, DTW, or generative
baselines in every data regime.

## CSTA Overview

The current framework is decomposed into five layers:

1. PIA proposal: estimate a train-only tangent dictionary and propose
   high-response covariance-state candidates.
2. Pre-bridge FV selection: choose candidates using relevance, local safety,
   and variety before raw-space realization.
3. Safe generation: apply the local inter-class margin proxy to produce a
   conservative target covariance state.
4. Whitening-coloring bridge: realize the target covariance in raw time-series
   space.
5. Post-bridge fidelity gate and audit: record bridge success and
   covariance-target fidelity diagnostics.

This split keeps candidate selection independent from bridge diagnostics in
the first selector version.

## PIA Candidate Proposal

PIA is the data-adaptive tangent dictionary proposal operator inside CSTA. It
learns a TELM2/zPIA dictionary from train-only Log-Euclidean covariance states.
For each anchor state `z_i`, PIA proposes high-response template directions.

Primary single-anchor proposal policies are:

- `top1`: choose the highest-response template.
- `softmax-topK`: sample within the top-K response set by softmax.
- `uniform-topK`: sample uniformly within the top-K response set.

`group_top` reuses the policy interface, but conceptually it is a
neighborhood-consensus extension using `q(d | z_G)` rather than a purely
single-anchor `q(d | z_i)`.

## Pre-Bridge FV Selection

FV selection is a separate pre-bridge layer. Version 1 does not use
`bridge_success` or `transport_error_logeuc` in its main score.

The pre-bridge feasibility filter is:

```text
gamma_used > eps
safe_radius_ratio <= 1
manifold_margin > eps
direction_norm > eps
```

The scored selector uses:

```text
score = 1.0 * relevance + 1.0 * safe_balance + 0.5 * variety
```

with:

```text
relevance    = normalized(template_response_abs)
safe_balance = normalized(-abs(safe_radius_ratio - 0.75))
variety      = normalized(z_displacement_norm) + template_diversity_bonus
```

The selector arms are:

- `csta_fv_filter_top5`: feasibility-filtered high-response sampling.
- `csta_fv_score_top5`: fidelity-variety scored top-5 candidate sampling.
- `csta_random_feasible_selector`: feasible-only random control.

The random feasible control is important: it tests whether satisfying the safe
constraint alone is enough, or whether candidate relevance and variety still
matter.

## Local-Margin Safe Generation

Safe generation clips the requested displacement by a local inter-class margin
proxy:

```text
gamma_used <= eta_safe * d_min / ||direction||
```

This is a conservative radius controller, not a formal label-preservation
guarantee. The audit fields `safe_radius_ratio`, `safe_clip_rate`, and
`gamma_zero_rate` are used to verify whether augmentation is too weak, too
aggressive, or frequently clipped.

## Whitening-Coloring Bridge

The bridge is a deterministic whitening-coloring covariance realization map.
Given an original sample and a target covariance, it whitens the sample with
the original covariance and colors it with the target covariance.

The bridge is transport-inspired, but the implementation does not claim exact
Gaussian optimal transport or Bures-Wasserstein optimality. The legacy field
`metric_preservation_error` remains in CSVs for compatibility and should be
read as a bridge deformation diagnostic.

`transport_error_logeuc` is a covariance-target fidelity diagnostic. In FV
selector v1 it is recorded after bridge realization and is not used in the
pre-bridge score.

## Candidate Audit

Each generated candidate should expose:

- identity: dataset, seed, method, anchor, class, candidate order;
- proposal: template id, rank, sign, response;
- selection: feasible rate, selector accept rate, FV scores, reject counts;
- safe generation: gamma requested/used, safe radius ratio, margin;
- bridge audit: bridge success, transport error, deformation diagnostics.

Per-candidate audit rows aggregate into per-run summary fields used by protocol
tables and mechanism plots.

## Limitations

CSTA does not claim strict label preservation. It uses a local margin proxy.

CSTA does not claim exact Bures-Wasserstein optimal transport. It uses a
whitening-coloring covariance realization bridge.

CSTA targets dependency-structure augmentation. It does not fully generate new
temporal motifs, phase patterns, or event timing dynamics.

The current FV selector is hand-designed and intentionally lightweight. Learned
selectors, feedback weighting, and post-bridge risk-aware scoring are future
extensions rather than part of the current main method.
