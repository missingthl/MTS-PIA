#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
datasets/seed1_preprocess.py

封装 SEED1 的数据/特征读取，保持与 SEED_DEMO.ipynb 一致的切分方式：
- DE 特征：从 pickle 读取 train / test，StandardScaler 标准化
- 频域特征：从 ./data/SEED/features 或 ./features 读取 YYM_H
- 固定的帧级切分参数：train_len/test_len 及每个电影的帧数列表，用于 movie-level voting
"""

import os
from typing import List, Tuple

import numpy as np
import scipy.io

from .seed_preprocessed import extract_seed_feature

# ---- 切分参数（与原 SEED_DEMO.ipynb 保持一致） ----
TRAIN_LEN = 2010
TEST_LEN = 1384
TRAIN_INDEX = [235, 233, 206, 238, 185, 195, 237, 216, 265]  # 9 部电影帧数之和 = 2010
TEST_INDEX = [237, 235, 233, 235, 238, 206]                  # 6 部电影帧数之和 = 1384

# 频域文件的两个可能路径（内容相同）
FREQ_DIR_CANDIDATES = ["./data/SEED/features", "./features"]


def load_seed1_de() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回标准化后的 DE 特征与标签：(trainx, trainy, testx, testy)。
    """
    return extract_seed_feature()


def _find_freq_dir() -> str:
    """优先使用 data/SEED/features，没有则回退到 ./features。"""
    for d in FREQ_DIR_CANDIDATES:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError("找不到 SEED1 频域特征目录，检查 ./data/SEED/features 或 ./features")


def load_seed1_freq_features() -> np.ndarray:
    """
    读取频域特征 YYM_H，返回 shape = (N_subjects, F_freq, T_total) 的数组。
    排序规则与 ipynb 相同：按文件名中第四段的数字排序。
    """
    freq_dir = _find_freq_dir()
    files = sorted(
        [f for f in os.listdir(freq_dir) if f.endswith(".mat")],
        key=lambda x: int(x.split("_")[3].split("f")[0]),
    )

    feature_set = []
    for f in files:
        path = os.path.join(freq_dir, f)
        mat = scipy.io.loadmat(path)
        if "YYM_H" not in mat:
            raise KeyError(f"{path} 缺少键 'YYM_H'")
        cur_feature = np.array(mat["YYM_H"])
        feature_set.append(cur_feature)

    return np.array(feature_set)


def seed1_num_segments(feature_set: np.ndarray) -> int:
    """
    返回按照 ipynb 切分时的段数（通常是 42 = 14 被试 × 3 session）。
    """
    return feature_set.shape[0]

