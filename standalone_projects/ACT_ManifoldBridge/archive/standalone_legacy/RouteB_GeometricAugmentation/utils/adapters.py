#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
datasets/adapters.py

统一的数据集适配层（Dataset Adapter）：

- 把“怎么读某个数据集、怎么划分 fold、多少类别”等细节封装起来
- 对外提供统一接口，供核心实验框架调用

当前实现：
- SeedVAdapter: 支持 SEED-V 的空间流 / 流形流
- Seed1Adapter / DreamerAdapter: 先占位，后续你补充具体实现即可
"""

from typing import Dict, Optional

import numpy as np

from .types import FoldData, TrialFoldData


# ======================================================================
# SeedVAdapter: 当前重点支持的数据集
# ======================================================================

class SeedVAdapter:
    """
    SEED-V 数据集适配器。

    - 空间流（spatial stream）：使用 DE 特征，给 CNN / DCNet 使用
    - 流形流（manifold stream）：使用 trial-level 序列
    """

    name: str = "seedv"
    num_classes: int = 5

    def __init__(self):
        # 如果将来需要，可以在这里加载共享参数（比如 freq_params.npz）
        pass

    # -------------------- 空间流接口 --------------------

    def get_spatial_folds_for_cnn(
        self,
        split_by: str = "trial",
        subject_split: str = "loso",
        subject_k: int = 5,
        subject_seed: int = 0,
        de_lds: bool = False,
        de_lds_level: str = "session",
        de_lds_q: float = 1e-3,
        de_lds_r: float = 1.0,
        de_lds_method: str = "fixed",
        de_lds_em_iters: int = 10,
        de_lds_em_tol: float = 1e-4,
    ) -> Dict[str, FoldData]:
        """
        返回给 CNN 使用的 3 个 fold。

        split_by:
          - "trial": trials 0~4 / 5~9 / 10~14
          - "session": sessions 0 / 1 / 2
          - "subject": subject-based split (LOSO / K-fold)

            folds["fold1"] = FoldData(X_tr, y_tr, X_te, y_te)
            ...

        其中 X_* 为 2D：(N, 310)
             y_* 为 1D：(N,)
        """
        from sklearn import preprocessing
        from .seedv_preprocess import split_fold_with_trial_id, split_subject_folds_with_trial_id

        split_by = (split_by or "trial").lower()
        if split_by == "subject":
            raw_folds = split_subject_folds_with_trial_id(
                source="de",
                mode=subject_split,
                n_splits=subject_k,
                seed=subject_seed,
                de_lds=de_lds,
                de_lds_level=de_lds_level,
                de_lds_q=de_lds_q,
                de_lds_r=de_lds_r,
                de_lds_method=de_lds_method,
                de_lds_em_iters=de_lds_em_iters,
                de_lds_em_tol=de_lds_em_tol,
            )
            folds: Dict[str, FoldData] = {}
            for name, (X_tr, y_tr, tid_tr, X_te, y_te, tid_te) in raw_folds.items():
                scaler = preprocessing.StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
                folds[name] = FoldData(
                    X_train=X_tr,
                    y_train=y_tr,
                    X_test=X_te,
                    y_test=y_te,
                    trial_id_train=tid_tr,
                    trial_id_test=tid_te,
                )
            return folds

        X1, y1, tid1, X2, y2, tid2, X3, y3, tid3 = split_fold_with_trial_id(
            split_by=split_by,
            de_lds=de_lds,
            de_lds_level=de_lds_level,
            de_lds_q=de_lds_q,
            de_lds_r=de_lds_r,
            de_lds_method=de_lds_method,
            de_lds_em_iters=de_lds_em_iters,
            de_lds_em_tol=de_lds_em_tol,
        )

        # 构造 3-fold
        X_fold1 = np.concatenate((X1, X2), axis=0)
        X_fold2 = np.concatenate((X1, X3), axis=0)
        X_fold3 = np.concatenate((X2, X3), axis=0)
        y_fold1 = np.concatenate((y1, y2), axis=0)
        y_fold2 = np.concatenate((y1, y3), axis=0)
        y_fold3 = np.concatenate((y2, y3), axis=0)
        tid_fold1 = np.concatenate((tid1, tid2), axis=0)
        tid_fold2 = np.concatenate((tid1, tid3), axis=0)
        tid_fold3 = np.concatenate((tid2, tid3), axis=0)

        std_fold1 = preprocessing.StandardScaler()
        std_fold2 = preprocessing.StandardScaler()
        std_fold3 = preprocessing.StandardScaler()

        X_fold1 = std_fold1.fit_transform(X_fold1)
        X_fold2 = std_fold2.fit_transform(X_fold2)
        X_fold3 = std_fold3.fit_transform(X_fold3)

        x_test1 = std_fold1.transform(X3)
        x_test2 = std_fold2.transform(X2)
        x_test3 = std_fold3.transform(X1)

        folds: Dict[str, FoldData] = {
            "fold1": FoldData(
                X_train=X_fold1,
                y_train=y_fold1,
                X_test=x_test1,
                y_test=y3,
                trial_id_train=tid_fold1,
                trial_id_test=tid3,
            ),
            "fold2": FoldData(
                X_train=X_fold2,
                y_train=y_fold2,
                X_test=x_test2,
                y_test=y2,
                trial_id_train=tid_fold2,
                trial_id_test=tid2,
            ),
            "fold3": FoldData(
                X_train=X_fold3,
                y_train=y_fold3,
                X_test=x_test3,
                y_test=y1,
                trial_id_train=tid_fold3,
                trial_id_test=tid1,
            ),
        }
        return folds

    # -------------------- 频域流接口 --------------------

    def get_freq_folds_for_pia(
        self,
        source: str = "de",
        split_by: str = "trial",
        subject_split: str = "loso",
        subject_k: int = 5,
        subject_seed: int = 0,
    ) -> Dict[str, FoldData]:
        """
        返回给频域 ELM / PIA 使用的 3 个 fold（已废弃，仅保留接口）：

            folds["fold1"] = FoldData(X_tr, y_tr, X_te, y_te)

        X_* 形状大致为 (N, F_freq)，例如 (N, 500)
        y_* 形状为 (N,)
        """
        source = (source or "de").lower()
        if source in {"de", "eeg"}:
            # 论文对齐：频域流与空间流共享同一套预处理输入（DE）
            return self.get_spatial_folds_for_cnn(
                split_by=split_by,
                subject_split=subject_split,
                subject_k=subject_k,
                subject_seed=subject_seed,
            )

        raise ValueError("SeedV freq stream is deprecated; use spatial/manifold streams instead.")

    @property
    def trial_lengths_for_fusion(self):
        """
        返回 SEED-V 测试集的 trial 长度列表，用于 trial-level majority voting。
        """
        from .seedv_preprocess import get_seedv_trial_lengths_for_folds

        return get_seedv_trial_lengths_for_folds()

    def get_manifold_trial_folds(
        self,
        source: str = "de",
        split_by: str = "trial",
        subject_split: str = "loso",
        subject_k: int = 5,
        subject_seed: int = 0,
        raw_repr: str = "signal",
        raw_cache: str = "auto",
        raw_channel_policy: str = "strict",
        raw_shrinkage: Optional[float] = None,
        raw_bands: Optional[list] = None,
        de_lds: bool = False,
        de_lds_level: str = "session",
        de_lds_q: float = 1e-3,
        de_lds_r: float = 1.0,
        de_lds_method: str = "fixed",
        de_lds_em_iters: int = 10,
        de_lds_em_tol: float = 1e-4,
    ) -> Dict[str, TrialFoldData]:
        """
        返回给流形流使用的 trial-level 3-fold。

        split_by:
          - "trial": trials 0~4 / 5~9 / 10~14
          - "session": sessions 0 / 1 / 2
          - "subject": subject-based split (LOSO / K-fold)
        source:
          - "de": DE 特征序列
          - "raw": raw EEG (T, C)
        """
        from .seedv_preprocess import iter_seedv_trials_meta, split_subject_trial_folds

        set1, set2, set3 = [], [], []
        source = (source or "de").lower()
        if source not in {"de", "raw"}:
            raise ValueError("SeedV manifold stream supports source: de | raw.")
        split_by = (split_by or "trial").lower()
        if split_by == "subject":
            raw_folds = split_subject_trial_folds(
                source=source,
                mode=subject_split,
                n_splits=subject_k,
                seed=subject_seed,
                raw_repr=raw_repr,
                raw_cache=raw_cache,
                raw_channel_policy=raw_channel_policy,
                raw_shrinkage=raw_shrinkage,
                raw_bands=raw_bands,
                de_lds=de_lds,
                de_lds_level=de_lds_level,
                de_lds_q=de_lds_q,
                de_lds_r=de_lds_r,
                de_lds_method=de_lds_method,
                de_lds_em_iters=de_lds_em_iters,
                de_lds_em_tol=de_lds_em_tol,
            )
            folds: Dict[str, TrialFoldData] = {}
            for name, (tr_tr, y_tr, tid_tr, tr_te, y_te, tid_te) in raw_folds.items():
                folds[name] = TrialFoldData(
                    trials_train=tr_tr,
                    y_trial_train=y_tr,
                    trials_test=tr_te,
                    y_trial_test=y_te,
                    trial_id_train=tid_tr,
                    trial_id_test=tid_te,
                )
            return folds

        for trial_id, Xt, y, t_idx, session_idx, _ in iter_seedv_trials_meta(
            source=source,
            raw_repr=raw_repr,
            raw_cache=raw_cache,
            raw_channel_policy=raw_channel_policy,
            raw_shrinkage=raw_shrinkage,
            raw_bands=raw_bands,
            de_lds=de_lds,
            de_lds_level=de_lds_level,
            de_lds_q=de_lds_q,
            de_lds_r=de_lds_r,
            de_lds_method=de_lds_method,
            de_lds_em_iters=de_lds_em_iters,
            de_lds_em_tol=de_lds_em_tol,
        ):
            if split_by == "trial":
                if 0 <= t_idx <= 4:
                    set1.append((trial_id, Xt, y))
                elif 5 <= t_idx <= 9:
                    set2.append((trial_id, Xt, y))
                elif 10 <= t_idx <= 14:
                    set3.append((trial_id, Xt, y))
                else:
                    raise ValueError(f"Unexpected trial_idx {t_idx}")
            elif split_by == "session":
                if session_idx == 0:
                    set1.append((trial_id, Xt, y))
                elif session_idx == 1:
                    set2.append((trial_id, Xt, y))
                elif session_idx == 2:
                    set3.append((trial_id, Xt, y))
                else:
                    raise ValueError(f"Unexpected session_idx {session_idx}")
            else:
                raise ValueError(f"Unknown split_by: {split_by}")

        def _pack(pairs):
            trials = [p[1] for p in pairs]
            ys = np.asarray([p[2] for p in pairs], dtype=int)
            tids = np.asarray([p[0] for p in pairs])
            return trials, ys, tids

        folds = {
            "fold1": TrialFoldData(  # train set1+set2, test set3
                trials_train=_pack(set1 + set2)[0],
                y_trial_train=_pack(set1 + set2)[1],
                trial_id_train=_pack(set1 + set2)[2],
                trials_test=_pack(set3)[0],
                y_trial_test=_pack(set3)[1],
                trial_id_test=_pack(set3)[2],
            ),
            "fold2": TrialFoldData(  # train set1+set3, test set2
                trials_train=_pack(set1 + set3)[0],
                y_trial_train=_pack(set1 + set3)[1],
                trial_id_train=_pack(set1 + set3)[2],
                trials_test=_pack(set2)[0],
                y_trial_test=_pack(set2)[1],
                trial_id_test=_pack(set2)[2],
            ),
            "fold3": TrialFoldData(  # train set2+set3, test set1
                trials_train=_pack(set2 + set3)[0],
                y_trial_train=_pack(set2 + set3)[1],
                trial_id_train=_pack(set2 + set3)[2],
                trials_test=_pack(set1)[0],
                y_trial_test=_pack(set1)[1],
                trial_id_test=_pack(set1)[2],
            ),
        }
        return folds


# ======================================================================
# 预留适配器：Seed1 / DREAMER
# ======================================================================

class Seed1Adapter:
    """
    SEED1 数据集适配器占位。

    未来可以在这里封装：
    - extract_seed_feature() 提供的 DE train/test
    - ./features/ 目录下的 YYM_H 频域特征
    - 电影级 voting 的 index 信息
    """
    name: str = "seed1"
    num_classes: int = 3
    last_align_index_path: Optional[str] = None
    last_de_mode: Optional[str] = None
    last_de_root: Optional[str] = None
    last_de_var: Optional[str] = None
    last_freeze_align: Optional[bool] = None

    def get_spatial_folds_for_cnn(
        self,
        *,
        seed_de_root: str | None = None,
        seed_de_var: str | None = None,
        seed_de_mode: str = "official",
        seed_freeze_align: bool = True,
        seed_manifest: str | None = None,
        seed_de_window: str | None = None,
    ) -> Dict[str, FoldData]:
        """
        使用现有的 pickle 预处理（./data/SEED/seed_python 或 ./seed_python）：
        - 只提供 1 个 fold：train / test
        - 特征已做标准化，形状 (N, 310)
        """
        mode = (seed_de_mode or "official").lower()
        self.last_de_mode = mode
        self.last_de_root = seed_de_root
        self.last_de_var = seed_de_var
        self.last_freeze_align = seed_freeze_align
        if mode == "official":
            from .seed_official_de import load_seed_official_de, write_seed_train_test_index

            if not seed_de_root:
                raise ValueError("seed_de_root is required for seed_de_mode=official")
            if not seed_de_var:
                raise ValueError("seed_de_var is required for seed_de_mode=official")
            manifest_path = seed_manifest or "logs/seed_raw_trial_manifest_full.json"
            trainx, trainy, testx, testy, trial_index, skipped = load_seed_official_de(
                seed_de_root=seed_de_root,
                seed_de_var=seed_de_var,
                manifest_path=manifest_path,
                freeze_align=seed_freeze_align,
                seed_de_window=seed_de_window,
            )
            out_path = write_seed_train_test_index(trial_index)
            self.last_align_index_path = out_path
            align_key = "subject/session/trial" if seed_freeze_align else "manifest_order"
            print(f"[seed1][align] freeze={'on' if seed_freeze_align else 'off'} key={align_key}")
            print(f"[seed1][align] wrote {out_path}")
            if skipped:
                print(f"[seed1][de] skipped_files={skipped}")
            
            # Reconstruct per-sample trial_id arrays for aggregation (Official Mode Only)
            train_meta = [m for m in trial_index if m['split'] == 'train']
            test_meta = [m for m in trial_index if m['split'] == 'test']
            
            def expand_ids(meta_list):
                ids = []
                for m in meta_list:
                    # m['trial_id'] e.g. "1_s1_t0"
                    # We repeat it n_windows times
                    n = m['n_windows']
                    t_id = m['trial_id']
                    ids.extend([t_id] * n)
                return np.array(ids)
            
            tid_train = expand_ids(train_meta)
            tid_test = expand_ids(test_meta)
            
            # Verify lengths matches data
            if len(tid_train) != len(trainx):
                 print(f"[seed1][warn] tid_train len {len(tid_train)} != x {len(trainx)}")
            if len(tid_test) != len(testx):
                 print(f"[seed1][warn] tid_test len {len(tid_test)} != x {len(testx)}")

        elif mode == "author":
            from .seed_preprocessed import extract_seed_feature

            print("[seed1][de] mode=author (seed_python)")
            trainx, trainy, testx, testy = extract_seed_feature()
            self.last_align_index_path = None
            tid_train, tid_test = None, None # Not supported for author mode yet
            print(
                f"[seed1][de] samples_train={len(trainy)} samples_test={len(testy)} "
                f"labels_unique={sorted(set(np.concatenate([trainy, testy]).tolist()))}"
            )
        else:
            raise ValueError(f"Unknown seed_de_mode: {seed_de_mode}")
        folds = {
            "fold1": FoldData(
                X_train=trainx,
                y_train=trainy,
                X_test=testx,
                y_test=testy,
                trial_id_train=tid_train,
                trial_id_test=tid_test
            )
        }
        return folds

    def get_manifold_trial_folds(
        self,
        source: str = "de",
        split_by: str = "trial",
        # Arguments not used but kept for signature compatibility
        subject_split: str = "loso",
        subject_k: int = 5,
        subject_seed: int = 0,
        raw_repr: str = "signal",
        raw_cache: str = "auto",
        raw_channel_policy: str = "strict",
        raw_shrinkage: any = None,
        raw_bands: any = None,
        de_lds: bool = False,
        de_lds_level: str = "session",
        de_lds_q: float = 1e-3,
        de_lds_r: float = 1.0,
        de_lds_method: str = "fixed",
        de_lds_em_iters: int = 10,
        de_lds_em_tol: float = 1e-4,
    ) -> Dict[str, TrialFoldData]:
        """
        Support SEED1 manifold folds (Trial-level).
        Reuses logic from load_seed_official_de but structures as TrialFoldData.
        """
        if self.last_de_mode == "official" and self.last_de_root:
             # Reuse last config if available, else default
             seed_de_root = self.last_de_root
             seed_de_var = self.last_de_var if self.last_de_var else "de_LDS1"
             seed_manifest = "logs/seed_raw_trial_manifest_full.json"
        else:
             # Fallback defaults if get_spatial wasn't called first
             # This might require arguments to be passed explicitly if not using defaults
             seed_de_root = "data/SEED/SEED_EEG/ExtractedFeatures_1s"
             seed_de_var = "de_LDS1"
             seed_manifest = "logs/seed_raw_trial_manifest_full.json"
        
        from .seed_official_de import load_seed_official_de
        # Import local helper to reconstruct trials
        def reconstruct(X, y, trial_indices):
            trials = []
            y_trials = []
            ids = []
            cursor = 0
            for meta in trial_indices:
                n = meta['n_windows']
                if n == 0: continue
                segment = X[cursor : cursor + n]
                label = int(meta['label'])
                tid = meta['trial_id']
                trials.append(segment)
                y_trials.append(label)
                ids.append(tid)
                cursor += n
            return trials, np.asarray(y_trials), np.asarray(ids)

        trainx, trainy, testx, testy, trial_index, _ = load_seed_official_de(
            seed_de_root=seed_de_root,
            seed_de_var=seed_de_var,
            manifest_path=seed_manifest,
            freeze_align=True
        )
        train_meta = [m for m in trial_index if m['split'] == 'train']
        test_meta = [m for m in trial_index if m['split'] == 'test']
        
        tr_trials, tr_y, tr_id = reconstruct(trainx, trainy, train_meta)
        te_trials, te_y, te_id = reconstruct(testx, testy, test_meta)
        
        return {
            "fold1": TrialFoldData(
                trials_train=tr_trials,
                y_trial_train=tr_y,
                trials_test=te_trials,
                y_trial_test=te_y,
                trial_id_train=tr_id,
                trial_id_test=te_id,
            )
        }


class DreamerAdapter:
    """
    DREAMER 数据集适配器占位。

    未来可以在这里封装：
    - baseline-subtracted EEG 的预处理
    - 对应的 fold 划分
    - 频域映射（legacy，已废弃）
    """
    name: str = "dreamer"
    num_classes: int = 2   # 默认二分类：高/低 valence

    def get_spatial_folds_for_cnn(self) -> Dict[str, FoldData]:
        """
        基于 DREAMER.mat，返回一个简单的 train/test fold。
        预处理逻辑在 datasets/dreamer_preprocess.py。
        """
        from .dreamer_preprocess import load_dreamer_valence

        X_train, y_train, X_test, y_test = load_dreamer_valence()
        folds = {
            "fold1": FoldData(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        }
        return folds

    def get_freq_folds_for_pia(self, source: str = "eeg") -> Dict[str, FoldData]:
        """
        返回给频域流使用的 fold。

        - source="eeg": 与空间流共享 baseline-corrected + StandardScaler 后的 EEG (14-d)
        - source="yym": 使用随机映射生成的 YYM-like 特征（legacy，对齐旧实现）
        """
        source = (source or "eeg").lower()
        if source in {"de", "eeg"}:
            return self.get_spatial_folds_for_cnn()

        if source in {"yym", "yym_like", "legacy_yym"}:
            from .dreamer_preprocess import dreamer_split_train_test_freq

            X_train, y_train, X_test, y_test, _ = dreamer_split_train_test_freq()
            return {
                "fold1": FoldData(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                )
            }

        raise ValueError(f"Unknown Dreamer freq source: {source}")

    @property
    def trial_lengths_for_fusion(self):
        """
        返回视频级 trial 长度列表（23*18 段），可用于 trial-level voting。
        """
        from .dreamer_preprocess import dreamer_get_trial_lengths
        return {"fold1": dreamer_get_trial_lengths()}


# ======================================================================
# 工厂函数：根据名字获取 adapter
# ======================================================================

def get_adapter(dataset: str):
    """
    根据字符串名字返回对应的适配器实例。

    目前支持：
        - "seedv"   → SeedVAdapter
    预留：
        - "seed1"   → Seed1Adapter
        - "dreamer" → DreamerAdapter
    """
    dataset = dataset.lower()
    # Convenience alias: "seed" -> SEED1 (3-class)
    if dataset == "seed":
        dataset = "seed1"
    if dataset == "seedv":
        return SeedVAdapter()
    elif dataset == "seed1":
        return Seed1Adapter()
    elif dataset == "dreamer":
        return DreamerAdapter()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
