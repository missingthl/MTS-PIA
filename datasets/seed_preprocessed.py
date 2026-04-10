#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
seed_preprocessed.py — SEED1 特征级数据加载与标准化

功能：
1. 从 ./seed_python 目录读取 train / test 特征（作者提供的 SEED1 预处理数据）
2. 封装 read_data_sets()，返回 DataSet 风格的数据结构
3. 提供 extract_seed_feature()，直接返回标准化后的 (trainx, trainy, testx, testy)

注意：
- 这里不建网络、不训练，只做数据准备。
"""

import os
import pickle
import numpy as np
from six.moves import urllib   # 保留一致性，虽然当前 maybe_download 不真的下载

# 数据根目录优先使用 data/SEED/SEED baseline/seed_python，其次 data/SEED/seed_python，再回落到 ./seed_python
SEED1_DATA_DIR_CANDIDATES = [
    os.path.join(".", "data", "SEED", "SEED baseline", "seed_python"),
    os.path.join(".", "data", "SEED", "seed_python"),
    "./seed_python",
]


def _resolve_seed1_data_dir() -> str:
    candidates = [os.path.abspath(d) for d in SEED1_DATA_DIR_CANDIDATES]
    for d in candidates:
        train_path = os.path.join(d, "train")
        test_path = os.path.join(d, "test")
        if os.path.isfile(train_path) and os.path.isfile(test_path):
            print(f"[seed1] using seed_python dir: {d}")
            return d
    msg = [
        "SEED1 seed_python not found.",
        "Expected train/test files under one of the following directories:",
    ]
    msg.extend([f"  - {d}" for d in candidates])
    raise FileNotFoundError("\n".join(msg))


DATA_DIRECTORY = _resolve_seed1_data_dir()
SOURCE_URL = "http://35.183.27.27/"


def maybe_download(filename: str) -> str:
    """返回数据文件的路径（如果不存在，原代码是打算下载，这里只返回路径）"""
    filepath = os.path.join(DATA_DIRECTORY, filename)
    # 原来下载部分被注释掉了，这里也保持不动
    return filepath


def dense_to_one_hot(labels_dense: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """把 [N,] 的标签转成 [N, num_classes] 的 one-hot"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_data(filename: str, one_hot: bool = False):
    """从 pickle 文件中读取 data / label，按需 one-hot。"""
    with open(filename, "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
        data = d["data"]
        labels = d["label"]

        if one_hot:
            labels = dense_to_one_hot(labels)
        return data, labels


class DataSet(object):
    """和 SEED_DEMO.py 里的 DataSet 保持一致，用于 batch 访问。"""

    def __init__(self, data, labels):
        assert data.shape[0] == labels.shape[0]
        self._num_examples = data.shape[0]
        self._data = data
        self._lables = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._lables

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._lables = self._lables[perm]

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._data[start:end], self._lables[start:end]


def read_data_sets(one_hot: bool = False):
    """
    读取 ./seed_python 目录下的 train / test pickle，
    返回带 .train / .test 属性的简单对象（和原 SEED_DEMO 保持一致）。
    """
    class DataSets(object):
        pass

    train_filename = maybe_download("train")
    test_filename = maybe_download("test")

    train_data, train_labels = load_data(train_filename, one_hot)
    test_data, test_labels = load_data(test_filename, one_hot)

    data_sets = DataSets()
    data_sets.train = DataSet(train_data, train_labels)
    data_sets.test = DataSet(test_data, test_labels)
    return data_sets


def extract_seed_feature():
    """
    只做：读取 SEED1 的 train/test，标准化后返回 numpy 数组。
    不建网络，不训练，不画图。
    """
    SAMPLE_SIZE = 400  # 原来的行保留，方便你以后对照；现在函数里其实没用到

    data = read_data_sets(one_hot=False)

    train_x = data.train.data
    train_label = data.train.labels

    test_x = data.test.data
    test_labels = data.test.labels

    n_samples = data.train.num_examples  # 如有需要可一起返回

    trainx = np.asarray(train_x)
    trainy = np.asarray(train_label)
    testx = np.asarray(test_x)
    testy = np.asarray(test_labels)

    from sklearn import preprocessing
    std = preprocessing.StandardScaler()
    trainx = std.fit_transform(trainx)
    testx = std.transform(testx)

    return trainx, trainy, testx, testy
