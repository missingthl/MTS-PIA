# External Baseline Validity Report

Phase 1/2 performance references were treated as locked; no training performance was rerun.

- Global best external method: `wdba_sameclass`
- Global best external mean F1: `0.667922`
- Phase 3 reference rows: `3`
- Phase 3 refs have no all-NaN reference columns: `True`

## wDBA Audit

The historical `fallback_count` mixed class-size replacement with any true DBA/tau/barycenter fallback. This audit separates class-size replacement exactly from the train split and reports the remaining count as true fallback.

- Total with-replacement count: `2280`
- Total true fallback count estimate: `0`
- Detail entropy/distance available: `False`

## SPAWNER-style Audit

- Total same-class scarcity count: `0`
- Total DTW-path true fallback estimate: `0`
- Detail path-level audit available: `False`
