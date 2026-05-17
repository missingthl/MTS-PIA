# Analytical / Alignment-Based Structure-Preserving Generation

Paper-facing group for baselines that preserve structure through barycenters,
DTW alignment, guided warping, or supervised time-series warping.

Included arms:

- `dba_sameclass`
- `wdba_sameclass`
- `spawner_sameclass_style`
- `rgw_sameclass`
- `dgw_sameclass`
- `jobda_cleanroom`

These are the strongest currently implemented external structural baselines
against CoSTA/CSTA-U5.

Status notes:

- `dba_sameclass` is the direct DBA-family baseline through tslearn DTW
  barycenter averaging.
- `wdba_sameclass` is a weighted DBA-family implementation.  Do not describe it
  as a separate official algorithm unless a specific weighted-DBA citation is
  used.
- `rgw_sameclass` and `dgw_sameclass` are clean-room guided-warping adapters
  following the guided warping / discriminative guided warping protocol.
- `spawner_sameclass_style` and `jobda_cleanroom` are style / clean-room
  adapters.  Use explicit caveats in the paper table or move them to appendix
  if the main table should contain only strict reproductions.
