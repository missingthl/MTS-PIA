#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dreamer_preprocess.py — DREAMER 数据集预处理

功能：
- 从 ./data/dreamer/DREAMER.mat 读取 EEG，做基线扣除
- 生成帧级特征 (T, 14) 和二分类标签（高/低 valence）
- 提供一个简单的 train/test 划分，返回适配器可用的 FoldData
"""

import os
from typing import Tuple, Dict, List

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

DEFAULT_MAT_PATH = "./data/dreamer/DREAMER.mat"

# 18 段视频的帧长度（每段 128Hz 采样，原脚本给出的长度）
VIDEO_LENGTHS = [
    25472, 16768, 44544, 21248, 17408, 24320, 24576, 50432, 18560,
    8576, 12288, 23168, 47104, 21760, 39424, 24960, 32768, 23808,
]


def _baseline_correct_all(raw) -> Tuple[np.ndarray, np.ndarray]:
    """
    按 participant × video 遍历，做基线扣除，返回：
        X: (N_total_frames, 14)  — 每帧 14 通道
        y: (N_total_frames,)    — 0/1，低/高 valence

    依据论文步骤：
      - 对每个参与者，先把 baseline 按 1s 切片，求均值/方差（等价于整体 mean/std）
      - 对应的 stimuli 做 (stim - baseline_mean)/baseline_std
    """
    participants = 23
    videos = 18

    # Step1: 生成视频级标签（valence < 4 → 0，否则 1）
    video_valence = np.zeros((participants, videos), dtype=np.int64)
    for p in range(participants):
        vals = raw["DREAMER"][0, 0]["Data"][0, p]["ScoreValence"][0, 0][:, 0].astype(float)
        video_valence[p] = (vals >= 4.0).astype(np.int64)

    # Step2: baseline 扣除 + 按视频长度展开到帧级标签
    # 基线长度写死为 61*128，对应原始脚本
    all_frames = []
    all_labels = []
    for p in range(participants):
        for v in range(videos):
            baseline = raw["DREAMER"][0, 0]["Data"][0, p]["EEG"][0, 0]["baseline"][0, 0][v, 0]
            stimuli = raw["DREAMER"][0, 0]["Data"][0, p]["EEG"][0, 0]["stimuli"][0, 0][v, 0]

            # baseline mean/std（按 1s=128 点切片，等价整体）
            base_mean = baseline.mean(axis=0)
            base_std = baseline.std(axis=0)
            base_std[base_std == 0] = 1.0  # 避免除零

            # stimuli: baseline 去均值/方差
            stim = (stimuli - base_mean) / base_std

            label_video = video_valence[p, v]
            labels = np.full((stim.shape[0],), label_video, dtype=np.int64)

            all_frames.append(stim)
            all_labels.append(labels)

    X = np.concatenate(all_frames, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def load_dreamer_valence(
    mat_path: str = DEFAULT_MAT_PATH,
    test_size: float = 0.1,
    random_state: int = 42,
):
    """
    加载 DREAMER.mat，得到标准化后的 (X_train, y_train, X_test, y_test)。
    默认做 9:1 的随机划分。
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"DREAMER.mat not found at {mat_path}")

    raw = scipy.io.loadmat(mat_path)
    X, y = _baseline_correct_all(raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train.astype(np.float32), y_train, X_test.astype(np.float32), y_test


# ----------------------------------------------------------------------
# 频域映射（ELM 风格）与 fold 构造
# ----------------------------------------------------------------------

N_HIDDEN_FREQ = 500
FREQ_RANDOM_SEED = 123


def _build_freq_mapping(input_dim: int = 14,
                        n_hidden: int = N_HIDDEN_FREQ,
                        seed: int = FREQ_RANDOM_SEED):
    rng = np.random.RandomState(seed)
    W = rng.normal(loc=0.0, scale=1.0, size=(n_hidden, input_dim))
    b = rng.normal(loc=0.0, scale=1.0, size=(n_hidden, 1))
    return W, b


def dreamer_offline_extract_freq_features(
    mat_path: str = DEFAULT_MAT_PATH,
    n_hidden: int = N_HIDDEN_FREQ,
    seed: int = FREQ_RANDOM_SEED,
):
    """
    用固定 W,b 将 baseline 扣除后的 EEG (T,14) 映射到 sigmoid 隐层，返回：
        H_all: (N_total_frames, n_hidden)
        y_all: (N_total_frames,)
        trial_lengths: list[int] 对应 23*18 段的帧数
    """
    raw = scipy.io.loadmat(mat_path)
    X, y = _baseline_correct_all(raw)

    W, b = _build_freq_mapping(input_dim=X.shape[1], n_hidden=n_hidden, seed=seed)
    H_lin = W @ X.T + b
    H = 1.0 / (1.0 + np.exp(-H_lin))
    H = H.T.astype(np.float32)

    # trial_lengths 直接用 VIDEO_LENGTHS * participants
    trial_lengths = VIDEO_LENGTHS * 23
    return H, y, trial_lengths


def dreamer_split_train_test_freq(
    mat_path: str = DEFAULT_MAT_PATH,
    test_size: float = 0.1,
    random_state: int = 42,
    n_hidden: int = N_HIDDEN_FREQ,
    seed: int = FREQ_RANDOM_SEED,
):
    """
    构造频域 train/test，保持与空间划分一致（随机 9:1）。
    返回 X_train_freq, y_train, X_test_freq, y_test, trial_lengths
    """
    H, y, trial_lengths = dreamer_offline_extract_freq_features(
        mat_path=mat_path, n_hidden=n_hidden, seed=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(
        H, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train.astype(np.float32), y_train, X_test.astype(np.float32), y_test, trial_lengths


def dreamer_get_trial_lengths() -> List[int]:
    """
    返回视频级 trial 长度列表（23*18 段），便于 trial-level voting。
    """
    return VIDEO_LENGTHS * 23
