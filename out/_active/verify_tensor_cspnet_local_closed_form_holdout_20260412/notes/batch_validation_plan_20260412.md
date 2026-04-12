# Batch Validation Plan

Date: 2026-04-12

## Group A: Equivalence validation
- arm: `e0`
- subject: `1`
- seed: `1`
- batch size: `29`
- epochs: `100`
- target baseline:
  - subject 1 acc: `0.8368`
  - source: prior full baseline run / partial snapshot

## Group B: Batch-size validation
- arm: `e0`
- subjects: `1`, `5`
- seed: `1`
- epochs: `100`
- batch sizes: `29`, `58`, `87`
- objective:
  - compare accuracy/loss/wallclock under accelerated implementation
  - quantify further speedup from larger batch sizes after implementation-level acceleration
