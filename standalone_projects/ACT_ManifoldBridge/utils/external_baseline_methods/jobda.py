from __future__ import annotations

from typing import List, Tuple

import numpy as np

from utils.external_baseline_methods.base import ExternalAugResult, finite_stack, resample_ct


def _downsample_avg_stride2_ct(segment_ct: np.ndarray) -> np.ndarray:
    segment_ct = np.asarray(segment_ct, dtype=np.float32)
    c, length = segment_ct.shape
    if length <= 1:
        return segment_ct.astype(np.float32, copy=True)
    out_len = int(np.ceil(length / 2.0))
    out = np.empty((c, out_len), dtype=np.float32)
    for i in range(out_len):
        start = 2 * i
        stop = min(start + 2, length)
        out[:, i] = np.mean(segment_ct[:, start:stop], axis=1)
    return out


def _upsample_insert_avg_ct(segment_ct: np.ndarray) -> np.ndarray:
    segment_ct = np.asarray(segment_ct, dtype=np.float32)
    c, length = segment_ct.shape
    if length <= 1:
        return segment_ct.astype(np.float32, copy=True)
    out = np.empty((c, 2 * length - 1), dtype=np.float32)
    out[:, 0::2] = segment_ct
    out[:, 1::2] = 0.5 * (segment_ct[:, :-1] + segment_ct[:, 1:])
    return out


def time_series_warping_cleanroom(x_ct: np.ndarray, n_subseq: int) -> np.ndarray:
    """Clean-room Time-Series Warping (TSW) transform for JobDA."""
    x_ct = np.asarray(x_ct, dtype=np.float32)
    c, t = x_ct.shape
    n_subseq = max(1, min(int(n_subseq), max(1, t)))
    segments = np.array_split(x_ct, n_subseq, axis=1)
    warped: List[np.ndarray] = []
    for seg_idx, segment in enumerate(segments):
        if segment.shape[1] == 0:
            continue
        if seg_idx % 2 == 0:
            warped.append(_downsample_avg_stride2_ct(segment))
        else:
            warped.append(_upsample_insert_avg_ct(segment))
    if not warped:
        return x_ct.astype(np.float32, copy=True)
    stitched = np.concatenate(warped, axis=1)
    return resample_ct(stitched, t).reshape(c, t).astype(np.float32)


def jobda_cleanroom_augmented_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    transform_subseqs: Tuple[int, ...] = (0, 2, 4, 8),
) -> ExternalAugResult:
    """Build the JobDA joint-label training set."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    transforms = tuple(int(x) for x in transform_subseqs)
    if not transforms or transforms[0] != 0:
        transforms = (0,) + tuple(x for x in transforms if int(x) != 0)
    X_out: List[np.ndarray] = []
    y_joint: List[int] = []
    warning_count = 0
    n_transforms = len(transforms)
    for i, x in enumerate(X_train):
        cls = int(y_train[i])
        for transform_id, n_subseq in enumerate(transforms):
            if int(n_subseq) <= 0:
                x_aug = x.astype(np.float32, copy=True)
            else:
                if int(n_subseq) > x.shape[1]:
                    warning_count += 1
                x_aug = time_series_warping_cleanroom(x, int(n_subseq))
            X_out.append(x_aug)
            y_joint.append(cls * n_transforms + int(transform_id))
    return ExternalAugResult(
        X_aug=finite_stack(X_out),
        y_aug=np.asarray(y_joint, dtype=np.int64),
        source_space="raw_time",
        label_mode="joint_hard",
        uses_external_library=False,
        library_name="",
        budget_matched=False,
        selection_rule="jobda_cleanroom_tsw_joint_label",
        warning_count=int(warning_count),
        fallback_count=0,
        meta={
            "jobda_num_transforms": float(n_transforms),
            "jobda_transform_subseqs": ",".join(str(x) for x in transforms),
            "jobda_cleanroom": 1.0,
            "jobda_official_code_confirmed": 0.0,
            "actual_aug_ratio": float(max(0, n_transforms - 1)),
        },
    )
