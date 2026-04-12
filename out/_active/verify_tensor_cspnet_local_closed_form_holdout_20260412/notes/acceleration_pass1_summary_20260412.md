# Tensor-CSPNet Acceleration Pass 1 Summary

Date: `2026-04-12`

## Scope

This note records the first implementation-level acceleration pass for the Stage-2 Tensor-CSPNet host line.

The goal of this pass is:

- keep the method definition unchanged
- accelerate the host implementation before changing training budget
- verify equivalence on single-subject `E0` runs

## Implemented changes

1. `modeig_forward` was rewritten from nested Python loops to batched `torch.linalg.eigh`
2. `BatchDiag` was vectorized with `torch.diag_embed`
3. `BiMap` / `Graph_BiMap` channel loops were replaced by batched `einsum`
4. structured mixed precision was introduced:
   - SPD path remains `float64`
   - `Temporal_Block`, `Classifier`, residual heads, `beta`, and prototypes use `float32`
5. subject-level cache was added for `BCIC holdout Tensor-CSPNet`

## First equivalence result

Reference baseline (old implementation):

- `subject 5`
  - `acc = 0.6076`
  - `loss = 1.090845`
  - `wallclock = 6188.9s`

Accelerated implementation:

- [subject 5 per_subject.csv](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/e0/seed1_accel_equiv_sub5_seed1/per_subject.csv)
  - `acc = 0.6006944444444444`
  - `loss = 1.0977438036352396`
  - `wallclock = 486.26946663856506s`

## Readout

- absolute accuracy change: about `-0.0069`
- loss change: about `+0.0069`
- wallclock speedup: about `12.7x`

Current interpretation:

- the first acceleration pass appears to preserve result quality closely enough for Stage-2 engineering use
- the implementation-level speed gain is very large
- the next checks are:
  - `subject 1` equivalence
  - `batch 29 / 58 / 87` validation on accelerated `E0`

