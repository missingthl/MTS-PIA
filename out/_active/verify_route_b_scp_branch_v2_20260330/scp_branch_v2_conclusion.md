# SCP-Branch v2 Conclusion

两条线共用同一套 dense backbone、prototype-memory 与 v1b tight anchors/local shaping 口径。
v2 只测 single replay round 的训练闭环吸收，不做 online / multi-round / test-time update。

## selfregulationscp1

- `same_backbone_no_shaping`: 0.6348 +/- 0.0000
- `v1b_local_shaping`: 0.6418 +/- 0.0000
- `v2_single_replay`: 0.6330 +/- 0.0000
- `delta_v2_vs_v1b`: -0.0088 +/- 0.0000
- `stitch_boundary_jump_ratio_mean`: 1.9652 +/- 0.0000
- `replay_continuity_distortion_ratio`: 1.0001 +/- 0.0000
