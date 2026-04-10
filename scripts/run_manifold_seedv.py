#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_manifold_seedv.py

用途：在 SEED-V 上跑“流形单流”基线
  DE 序列 -> SPD 协方差(分频带) -> RA -> log 向量化 -> 线性 SVM

fold 划分与官方一致：
  set_1: trial 0~4
  set_2: trial 5~9
  set_3: trial 10~14
  fold1: train=set_1+set_2, test=set_3
  fold2: train=set_1+set_3, test=set_2
  fold3: train=set_2+set_3, test=set_1

运行：
  python scripts/run_manifold_seedv.py
"""

import argparse
import os
from typing import Dict, List, Tuple, Optional
import numpy as np

# 确保可以找到 telm_aec 包
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
TELM_SRC = os.path.join(PROJECT_ROOT, "archive", "legacy_code", "block", "src")
if TELM_SRC not in sys.path:
    sys.path.append(TELM_SRC)

from telm_aec.manifold import LinearManifoldClassifier
from datasets.seedv_preprocess import iter_seedv_trials


def load_seedv_trials() -> List[Tuple[np.ndarray, int, int]]:
    """
    返回列表，每个元素 (X_trial, label, trial_idx)。
    X_trial: (T, 310)
    label:   int
    trial_idx: 0~14
    """
    out = []
    for _tid, Xt, y, t_idx in iter_seedv_trials(source="de"):
        out.append((Xt, int(y), int(t_idx)))
    return out


def build_folds(trials: List[Tuple[np.ndarray, int, int]]) -> Dict[str, Dict[str, List]]:
    """
    根据 trial_idx 分组，构造 fold1/2/3 的 train/test 列表。
    返回:
      {
        "fold1": {"train": [X...], "train_y": [...], "test": [...], "test_y": [...]},
        ...
      }
    """
    set1, set2, set3 = [], [], []
    for Xt, y, t_idx in trials:
        if 0 <= t_idx <= 4:
            set1.append((Xt, y))
        elif 5 <= t_idx <= 9:
            set2.append((Xt, y))
        elif 10 <= t_idx <= 14:
            set3.append((Xt, y))
        else:
            raise ValueError(f"Unexpected trial_idx {t_idx}")

    def split(pair_list):
        Xs = [p[0] for p in pair_list]
        ys = np.asarray([p[1] for p in pair_list], dtype=int)
        return Xs, ys

    folds = {
        "fold1": {  # train set1+set2, test set3
            "train": split(set1 + set2)[0],
            "train_y": split(set1 + set2)[1],
            "test": split(set3)[0],
            "test_y": split(set3)[1],
        },
        "fold2": {  # train set1+set3, test set2
            "train": split(set1 + set3)[0],
            "train_y": split(set1 + set3)[1],
            "test": split(set2)[0],
            "test_y": split(set2)[1],
        },
        "fold3": {  # train set2+set3, test set1
            "train": split(set2 + set3)[0],
            "train_y": split(set2 + set3)[1],
            "test": split(set1)[0],
            "test_y": split(set1)[1],
        },
    }
    return folds


def run_fold(
    name: str,
    X_tr: List[np.ndarray],
    y_tr: np.ndarray,
    X_te: List[np.ndarray],
    y_te: np.ndarray,
    *,
    mode: str,
    pca_dim: Optional[int],
    eps: float,
    classifier: str,
    C: float,
    C_grid: Optional[List[float]],
    standardize: bool,
    debug_save_feats: Optional[str] = None,
):
    clf = LinearManifoldClassifier(
        mode=mode,
        classifier=classifier,
        C=C,
        C_list=C_grid,
        eps=eps,
        pca_dim=pca_dim,
        standardize=standardize,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = float(np.mean(y_pred == y_te))
    print(f"[{name}] manifold ({mode}) trial-acc = {acc:.4f}")

    if debug_save_feats:
        feats_tr = clf.featurize(X_tr)
        feats_te = clf.featurize(X_te)
        np.savez_compressed(
            debug_save_feats.replace(".npz", f"_{name}.npz"),
            feats_tr=feats_tr,
            y_tr=y_tr,
            feats_te=feats_te,
            y_te=y_te,
        )
        print(f"[{name}] debug feats saved to {debug_save_feats.replace('.npz', f'_{name}.npz')}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="band", choices=["band", "pca", "flat"])
    ap.add_argument("--pca-dim", type=int, default=None)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--classifier", type=str, default="svm", choices=["svm", "logreg"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--C-grid", type=str, default=None,
                    help="逗号分隔的 C 候选，例如 0.1,1,10；留空则只用 --C")
    ap.add_argument("--no-standardize", action="store_true",
                    help="关闭 log 特征标准化（默认开启）")
    ap.add_argument("--debug-save-feats", type=str, default=None,
                    help="如指定，保存每个 fold 的 train/test 特征到 npz（用于可视化/调试）")
    args = ap.parse_args()

    trials = load_seedv_trials()
    folds = build_folds(trials)

    accs = []
    C_grid = None
    if args.C_grid:
        C_grid = [float(v) for v in args.C_grid.split(",") if v.strip()]

    for name, fd in folds.items():
        acc = run_fold(
            name,
            fd["train"],
            fd["train_y"],
            fd["test"],
            fd["test_y"],
            mode=args.mode,
            pca_dim=args.pca_dim,
            eps=args.eps,
            classifier=args.classifier,
            C=args.C,
            C_grid=C_grid,
            standardize=not args.no_standardize,
            debug_save_feats=args.debug_save_feats,
        )
        accs.append(acc)
    accs = np.asarray(accs)
    print("\n====== Manifold-stream (SEED-V) summary ======")
    print("Mean acc: {0:.4f} ± {1:.4f}".format(accs.mean(), accs.std()))


if __name__ == "__main__":
    main()
