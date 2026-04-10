import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets.adapters import get_adapter
from models.dual_stream_net import DualStreamNet
from models.prototype_mdm import logm_spd, expm_sym
from models.spdnet import DeepSPDClassifier
from runners.spatial_dcnet_torch import DCNetTorch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ordered_unique(ids: List[str]) -> List[str]:
    seen = set()
    out = []
    for tid in ids:
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def _build_spatial_trial_map(
    X: np.ndarray,
    y: np.ndarray,
    trial_ids: np.ndarray,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, int]]:
    windows_by_id: Dict[str, List[np.ndarray]] = {}
    labels_by_id: Dict[str, int] = {}
    for idx, tid in enumerate(trial_ids):
        tid = str(tid)
        windows_by_id.setdefault(tid, []).append(X[idx])
        if tid not in labels_by_id:
            labels_by_id[tid] = int(y[idx])
        else:
            if labels_by_id[tid] != int(y[idx]):
                raise RuntimeError(f"Spatial label mismatch within trial {tid}: {labels_by_id[tid]} vs {int(y[idx])}")
    ordered_ids = _ordered_unique([str(t) for t in trial_ids])
    windows_by_id_np = {k: np.stack(v, axis=0) for k, v in windows_by_id.items()}
    return ordered_ids, windows_by_id_np, labels_by_id


def _align_trials(
    spatial_ids: List[str],
    spatial_windows: Dict[str, np.ndarray],
    spatial_labels: Dict[str, int],
    manifold_trials: List[np.ndarray],
    manifold_labels: np.ndarray,
    manifold_ids: np.ndarray,
    split_name: str,
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], List[int], dict]:
    manifold_ids_list = [str(t) for t in manifold_ids]
    spatial_set = set(spatial_ids)
    manifold_set = set(manifold_ids_list)
    intersection = spatial_set.intersection(manifold_set)
    only_spatial = sorted(list(spatial_set - manifold_set))[:5]
    only_manifold = sorted(list(manifold_set - spatial_set))[:5]

    audit = {
        "split": split_name,
        "n_spatial": len(spatial_set),
        "n_manifold": len(manifold_set),
        "n_intersection": len(intersection),
        "only_spatial_sample": only_spatial,
        "only_manifold_sample": only_manifold,
    }

    if len(spatial_set) != len(manifold_set) or len(intersection) != len(spatial_set):
        raise RuntimeError(
            f"[Alignment:{split_name}] ID set mismatch: "
            f"n_spatial={len(spatial_set)} n_manifold={len(manifold_set)} n_intersection={len(intersection)}"
        )

    order_match = spatial_ids == manifold_ids_list
    audit["order_match"] = bool(order_match)

    aligned_spatial_windows = []
    aligned_manifold_trials = []
    aligned_labels = []
    length_mismatch = []

    for idx, tid in enumerate(manifold_ids_list):
        if tid not in spatial_windows:
            raise RuntimeError(f"[Alignment:{split_name}] Missing spatial trial id {tid}")
        aligned_spatial_windows.append(spatial_windows[tid])
        aligned_manifold_trials.append(manifold_trials[idx])
        label = int(manifold_labels[idx])
        aligned_labels.append(label)
        if spatial_labels.get(tid) != label:
            raise RuntimeError(
                f"[Alignment:{split_name}] Label mismatch for {tid}: spatial={spatial_labels.get(tid)} manifold={label}"
            )
        if spatial_windows[tid].shape[0] != manifold_trials[idx].shape[0]:
            length_mismatch.append(tid)

    audit["reordered"] = bool(not order_match)
    audit["length_mismatch_count"] = len(length_mismatch)
    audit["length_mismatch_sample"] = length_mismatch[:5]

    return manifold_ids_list, aligned_spatial_windows, aligned_manifold_trials, aligned_labels, audit


def _build_manifold_windows(
    trial: np.ndarray,
    window_len: int = 24,
    stride: int = 12,
) -> np.ndarray:
    T_total = trial.shape[0]
    if T_total < window_len:
        raise RuntimeError(f"Trial length {T_total} < window_len {window_len}")
    t_reshaped = trial.reshape(T_total, 5, 62).transpose(0, 2, 1)  # (T, 62, 5)
    windows = []
    for start in range(0, T_total - window_len + 1, stride):
        end = start + window_len
        windows.append(t_reshaped[start:end, :, :])  # (win, 62, 5)
    if not windows:
        raise RuntimeError("No manifold windows generated (check window_len/stride).")
    return np.stack(windows, axis=0)  # (n_win, win, 62, 5)


def _prepare_manifold_input(
    xb: torch.Tensor,
    band_norm_mode: str = "per_band_global_z",
) -> torch.Tensor:
    xb_perm = xb.permute(0, 3, 2, 1).contiguous()  # (B, 5, 62, Win)
    if band_norm_mode == "per_band_global_z":
        mean = xb_perm.mean(dim=(2, 3), keepdim=True)
        std = xb_perm.std(dim=(2, 3), keepdim=True) + 1e-6
        xb_perm = (xb_perm - mean) / std
    model_in = xb_perm.permute(0, 2, 1, 3).reshape(xb.size(0), 62, -1)  # (B, 62, 5*Win)
    return model_in


def _compute_corr_matrix(model_in: torch.Tensor, spd_eps: float) -> torch.Tensor:
    B, C, T = model_in.shape
    x_c = model_in - model_in.mean(dim=2, keepdim=True)
    x_std = x_c.std(dim=2, keepdim=True) + 1e-6
    x_z = x_c / x_std
    mat = torch.matmul(x_z, x_z.transpose(1, 2)) / (T - 1)
    mat = 0.5 * (mat + mat.transpose(1, 2))
    mat = mat + torch.eye(C, device=model_in.device).double() * spd_eps
    return mat


def _compute_global_log_ref(
    manifold_windows_list: List[np.ndarray],
    seed: int,
    device: torch.device,
    spd_eps: float,
    band_norm_mode: str,
) -> torch.Tensor:
    n_total = len(manifold_windows_list)
    n_sub = int(0.8 * n_total)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)[:n_sub]
    sum_log = None
    count = 0
    with torch.no_grad():
        for idx in indices:
            windows = torch.tensor(manifold_windows_list[idx], dtype=torch.float64, device=device)
            model_in = _prepare_manifold_input(windows, band_norm_mode=band_norm_mode)
            mats = _compute_corr_matrix(model_in, spd_eps)
            log_mats = logm_spd(mats)
            if sum_log is None:
                sum_log = log_mats.sum(dim=0)
            else:
                sum_log = sum_log + log_mats.sum(dim=0)
            count += mats.shape[0]
    if sum_log is None or count == 0:
        raise RuntimeError("Failed to compute global log-ref (no samples).")
    mean_log = sum_log / float(count)
    return mean_log


class JointTrialDataset(Dataset):
    def __init__(
        self,
        trial_ids: List[str],
        spatial_windows: List[np.ndarray],
        manifold_trials: List[np.ndarray],
        labels: List[int],
        window_len: int = 24,
        stride: int = 12,
    ):
        self.trial_ids = trial_ids
        self.spatial_windows = spatial_windows
        self.manifold_trials = manifold_trials
        self.labels = labels
        self.window_len = window_len
        self.stride = stride
        self.manifold_windows = [
            _build_manifold_windows(trial, window_len=window_len, stride=stride)
            for trial in manifold_trials
        ]

    def __len__(self) -> int:
        return len(self.trial_ids)

    def __getitem__(self, idx: int):
        return (
            self.trial_ids[idx],
            self.spatial_windows[idx],
            self.manifold_windows[idx],
            int(self.labels[idx]),
        )


def _collate_batch(batch):
    trial_ids = [b[0] for b in batch]
    spatial_windows = [b[1] for b in batch]
    manifold_windows = [b[2] for b in batch]
    labels = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return trial_ids, spatial_windows, manifold_windows, labels


@dataclass
class JointFusionConfig:
    seeds: List[int]
    dataset: str = "seed1"
    out_root: str = "promoted_results/phase14/step2/seed1"
    spatial_ckpt_fmt: str = "experiments/checkpoints/seedv_spatial_torch_seed{}_refactor.pt"
    manifold_ckpt_fmt: str = "experiments/checkpoints/phase13e/step4/seed1/seed{}/global_centered_corr_tsm/manifold/report_last.pt"
    bands_mode: str = "all5_timecat"
    band_norm_mode: str = "per_band_global_z"
    matrix_mode: str = "corr"
    global_centering: bool = True
    spd_eps: float = 1e-3
    epochs: int = 30
    batch_size: int = 8
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.0


class JointFusionRunner:
    def __init__(self, cfg: JointFusionConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_spatial_model(self, seed: int) -> DCNetTorch:
        ckpt_path = self.cfg.spatial_ckpt_fmt.format(seed)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Spatial checkpoint not found: {ckpt_path}")
        model = DCNetTorch(310, 3).to(self.device)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        model.train()
        return model

    def _load_manifold_model(self, seed: int) -> DeepSPDClassifier:
        ckpt_path = self.cfg.manifold_ckpt_fmt.format(seed)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Manifold checkpoint not found: {ckpt_path}")
        model = DeepSPDClassifier(
            n_channels=62,
            deep_layers=2,
            n_classes=3,
            output_dim=32,
            cov_eps=self.cfg.spd_eps,
            hidden_dim=96,
        ).to(self.device).double()
        state = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(state)
        model.train()
        return model

    def _compute_spatial_features_batch(
        self, model: DCNetTorch, spatial_windows: List[np.ndarray]
    ) -> torch.Tensor:
        counts = [w.shape[0] for w in spatial_windows]
        concat = np.concatenate(spatial_windows, axis=0)
        xb = torch.tensor(concat, dtype=torch.float32, device=self.device)
        logits, feats = model(xb, return_features=True)
        feats = feats.float()
        split_feats = torch.split(feats, counts, dim=0)
        trial_feats = torch.stack([f.mean(dim=0) for f in split_feats], dim=0)
        return trial_feats

    def _embedding_from_cov(self, model: DeepSPDClassifier, cov: torch.Tensor) -> torch.Tensor:
        for layer in model.manifold_layers:
            cov = layer(cov)
        log_cov = model.log_eig(cov)
        vec = model.vectorize(log_cov)
        return vec

    def _compute_spatial_logits_batch(
        self, model: DCNetTorch, spatial_windows: List[np.ndarray]
    ) -> np.ndarray:
        counts = [w.shape[0] for w in spatial_windows]
        concat = np.concatenate(spatial_windows, axis=0)
        xb = torch.tensor(concat, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = model(xb)
        split_logits = torch.split(logits, counts, dim=0)
        trial_logits = torch.stack([l.mean(dim=0) for l in split_logits], dim=0)
        return trial_logits.cpu().numpy()

    def _compute_manifold_features_batch(
        self,
        model: DeepSPDClassifier,
        manifold_windows: List[np.ndarray],
        log_ref_global: torch.Tensor,
    ) -> torch.Tensor:
        trial_feats = []
        for windows in manifold_windows:
            xb = torch.tensor(windows, dtype=torch.float64, device=self.device)
            model_in = _prepare_manifold_input(xb, band_norm_mode=self.cfg.band_norm_mode)
            mats = _compute_corr_matrix(model_in, self.cfg.spd_eps)
            if self.cfg.global_centering:
                log_C = logm_spd(mats)
                log_centered = log_C - log_ref_global
                mats = expm_sym(log_centered)
            vec = self._embedding_from_cov(model, mats)
            trial_feats.append(vec.mean(dim=0))
        feats = torch.stack(trial_feats, dim=0)
        return feats.float()

    def _compute_manifold_logits_batch(
        self,
        model: DeepSPDClassifier,
        manifold_windows: List[np.ndarray],
        log_ref_global: torch.Tensor,
    ) -> np.ndarray:
        trial_logits = []
        with torch.no_grad():
            for windows in manifold_windows:
                xb = torch.tensor(windows, dtype=torch.float64, device=self.device)
                model_in = _prepare_manifold_input(xb, band_norm_mode=self.cfg.band_norm_mode)
                mats = _compute_corr_matrix(model_in, self.cfg.spd_eps)
                if self.cfg.global_centering:
                    log_C = logm_spd(mats)
                    log_centered = log_C - log_ref_global
                    mats = expm_sym(log_centered)
                logits = model.forward_from_cov(mats)
                trial_logits.append(logits.mean(dim=0))
        return torch.stack(trial_logits, dim=0).cpu().numpy()

    def run_seed(self, seed: int) -> dict:
        set_seed(seed)
        adapter = get_adapter(self.cfg.dataset)

        spatial_folds = adapter.get_spatial_folds_for_cnn(
            seed_de_root="data/SEED/SEED_EEG/ExtractedFeatures_1s",
            seed_de_var="de_LDS1",
            seed_de_mode="official",
            seed_freeze_align=True,
        )
        manifold_folds = adapter.get_manifold_trial_folds()

        s_fold = spatial_folds["fold1"]
        m_fold = manifold_folds["fold1"]

        s_train_ids, s_train_windows, s_train_labels = _build_spatial_trial_map(
            s_fold.X_train, s_fold.y_train.ravel(), s_fold.trial_id_train
        )
        s_test_ids, s_test_windows, s_test_labels = _build_spatial_trial_map(
            s_fold.X_test, s_fold.y_test.ravel(), s_fold.trial_id_test
        )

        train_ids, train_spatial, train_manifold, train_labels, audit_train = _align_trials(
            s_train_ids,
            s_train_windows,
            s_train_labels,
            m_fold.trials_train,
            m_fold.y_trial_train,
            m_fold.trial_id_train,
            "train",
        )
        test_ids, test_spatial, test_manifold, test_labels, audit_test = _align_trials(
            s_test_ids,
            s_test_windows,
            s_test_labels,
            m_fold.trials_test,
            m_fold.y_trial_test,
            m_fold.trial_id_test,
            "test",
        )

        out_dir = os.path.join(self.cfg.out_root, f"seed{seed}")
        ensure_dir(out_dir)
        audit = {
            "seed": seed,
            "train": audit_train,
            "test": audit_test,
        }
        write_json(os.path.join(out_dir, "alignment_audit.json"), audit)

        align_pass = True
        if (
            audit_train["n_spatial"] != audit_train["n_manifold"]
            or audit_test["n_spatial"] != audit_test["n_manifold"]
        ):
            align_pass = False

        # Prepare datasets
        train_dataset = JointTrialDataset(
            train_ids, train_spatial, train_manifold, train_labels, window_len=24, stride=12
        )
        test_dataset = JointTrialDataset(
            test_ids, test_spatial, test_manifold, test_labels, window_len=24, stride=12
        )

        # Compute global log ref (using train manifold windows)
        log_ref_global = _compute_global_log_ref(
            train_dataset.manifold_windows,
            seed=seed,
            device=self.device,
            spd_eps=self.cfg.spd_eps,
            band_norm_mode=self.cfg.band_norm_mode,
        )
        if self.cfg.global_centering:
            ref_dir = os.path.join(out_dir, "global_centering")
            ensure_dir(ref_dir)
            np.save(os.path.join(ref_dir, "log_ref_global.npy"), log_ref_global.cpu().numpy())

        # Baseline models (frozen)
        spatial_base = self._load_spatial_model(seed)
        manifold_base = self._load_manifold_model(seed)
        spatial_base.eval()
        manifold_base.eval()

        # Trainable models
        spatial_model = self._load_spatial_model(seed)
        manifold_model = self._load_manifold_model(seed)

        # Determine feature dims
        with torch.no_grad():
            spatial_mode = spatial_model.training
            manifold_mode = manifold_model.training
            spatial_model.eval()
            manifold_model.eval()
            dummy = torch.zeros(2, 310, device=self.device)
            _, s_feat = spatial_model(dummy, return_features=True)
            spatial_feat_dim = int(s_feat.shape[1])
            dummy_m = torch.zeros(2, 62, 24 * 5, device=self.device, dtype=torch.float64)
            m_feat = manifold_model.get_embedding(dummy_m)
            manifold_feat_dim = int(m_feat.shape[1])
            if spatial_mode:
                spatial_model.train()
            if manifold_mode:
                manifold_model.train()

        joint_model = DualStreamNet(
            spatial_model, manifold_model, spatial_feat_dim, manifold_feat_dim, num_classes=3
        ).to(self.device)

        # Optimizer
        params = [
            {"params": spatial_model.parameters(), "lr": self.cfg.backbone_lr},
            {"params": manifold_model.parameters(), "lr": self.cfg.backbone_lr},
            {"params": joint_model.fusion_head.parameters(), "lr": self.cfg.head_lr},
        ]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.head_lr,
            weight_decay=self.cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # Training
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=_collate_batch,
        )

        instability_events: List[dict] = []

        for epoch in range(1, self.cfg.epochs + 1):
            joint_model.train()
            for batch_idx, (trial_ids, s_windows, m_windows, labels) in enumerate(train_loader, start=1):
                optimizer.zero_grad()
                feats_s = self._compute_spatial_features_batch(spatial_model, s_windows)
                feats_m = self._compute_manifold_features_batch(manifold_model, m_windows, log_ref_global)
                logits = joint_model.forward_fusion(feats_s, feats_m)
                loss = criterion(logits, labels.to(self.device))
                if not torch.isfinite(loss):
                    if len(instability_events) < 5:
                        instability_events.append(
                            {
                                "epoch": epoch,
                                "batch": batch_idx,
                                "issue": "loss_nonfinite",
                                "trial_ids": trial_ids[:5],
                                "loss": float(loss.detach().cpu().item()),
                            }
                        )
                    continue
                loss.backward()
                # Gradient check
                total_norm = 0.0
                for p in joint_model.parameters():
                    if p.grad is None:
                        continue
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                if not np.isfinite(total_norm):
                    if len(instability_events) < 5:
                        instability_events.append(
                            {
                                "epoch": epoch,
                                "batch": batch_idx,
                                "issue": "grad_nonfinite",
                                "trial_ids": trial_ids[:5],
                                "grad_norm": float(total_norm),
                            }
                        )
                optimizer.step()

        # Evaluation
        spatial_logits = self._compute_spatial_logits_batch(spatial_base, test_dataset.spatial_windows)
        manifold_logits = self._compute_manifold_logits_batch(manifold_base, test_dataset.manifold_windows, log_ref_global)

        # Fusion logits
        joint_model.eval()
        fusion_logits_list = []
        with torch.no_grad():
            for trial_ids, s_windows, m_windows, labels in DataLoader(
                test_dataset, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=_collate_batch
            ):
                feats_s = self._compute_spatial_features_batch(spatial_model, s_windows)
                feats_m = self._compute_manifold_features_batch(manifold_model, m_windows, log_ref_global)
                logits = joint_model.forward_fusion(feats_s, feats_m)
                fusion_logits_list.append(logits.cpu().numpy())
        fusion_logits = np.concatenate(fusion_logits_list, axis=0)

        y_true = np.array(test_labels, dtype=int)
        pred_s = spatial_logits.argmax(axis=1)
        pred_m = manifold_logits.argmax(axis=1)
        pred_f = fusion_logits.argmax(axis=1)

        s_ok = pred_s == y_true
        m_ok = pred_m == y_true
        f_ok = pred_f == y_true

        both = int((s_ok & m_ok).sum())
        spatial_only = int((s_ok & ~m_ok).sum())
        manifold_only = int((~s_ok & m_ok).sum())
        neither = int((~s_ok & ~m_ok).sum())

        rescued = int((f_ok & ~s_ok).sum())
        lost = int((~f_ok & s_ok).sum())
        net_gain = int(rescued - lost)
        complementary_count = int(manifold_only)

        acc_spatial = float(s_ok.mean())
        acc_manifold = float(m_ok.mean())
        acc_fusion = float(f_ok.mean())

        # Save predictions
        def _save_pred(path, logits, preds):
            df = {
                "trial_id": test_ids,
                "true_label": y_true.tolist(),
                "pred_label": preds.tolist(),
                "logit0": logits[:, 0].tolist(),
                "logit1": logits[:, 1].tolist(),
                "logit2": logits[:, 2].tolist(),
            }
            import pandas as pd

            pd.DataFrame(df).to_csv(path, index=False)

        _save_pred(os.path.join(out_dir, "spatial_trial_pred.csv"), spatial_logits, pred_s)
        _save_pred(os.path.join(out_dir, "manifold_trial_pred.csv"), manifold_logits, pred_m)
        _save_pred(os.path.join(out_dir, "fusion_trial_pred.csv"), fusion_logits, pred_f)

        # Console summary
        print(
            f"[Seed {seed}] Alignment: PASS train(n_spatial={audit_train['n_spatial']} "
            f"n_manifold={audit_train['n_manifold']} n_intersection={audit_train['n_intersection']}) "
            f"test(n_spatial={audit_test['n_spatial']} n_manifold={audit_test['n_manifold']} "
            f"n_intersection={audit_test['n_intersection']}) | "
            f"AccS={acc_spatial:.4f} AccM={acc_manifold:.4f} AccF={acc_fusion:.4f} | "
            f"rescued={rescued} lost={lost} net={net_gain} complementary={complementary_count}"
        )

        return {
            "seed": seed,
            "acc_spatial": acc_spatial,
            "acc_manifold": acc_manifold,
            "acc_fusion": acc_fusion,
            "rescued": rescued,
            "lost": lost,
            "net_gain": net_gain,
            "complementary_count": complementary_count,
            "both_correct": both,
            "spatial_only": spatial_only,
            "manifold_only": manifold_only,
            "neither": neither,
            "alignment_pass": align_pass,
            "alignment_train_n_spatial": audit_train["n_spatial"],
            "alignment_train_n_manifold": audit_train["n_manifold"],
            "alignment_train_n_intersection": audit_train["n_intersection"],
            "alignment_test_n_spatial": audit_test["n_spatial"],
            "alignment_test_n_manifold": audit_test["n_manifold"],
            "alignment_test_n_intersection": audit_test["n_intersection"],
            "spatial_feat_dim": spatial_feat_dim,
            "manifold_feat_dim": manifold_feat_dim,
            "backbone_lr": self.cfg.backbone_lr,
            "head_lr": self.cfg.head_lr,
            "weight_decay": self.cfg.weight_decay,
            "epochs": self.cfg.epochs,
            "batch_size": self.cfg.batch_size,
            "bands_mode": self.cfg.bands_mode,
            "band_norm_mode": self.cfg.band_norm_mode,
            "matrix_mode": self.cfg.matrix_mode,
            "global_centering": self.cfg.global_centering,
            "instability_events": instability_events,
        }

    def run_all(self) -> Dict[str, Any]:
        results = []
        for seed in self.cfg.seeds:
            res = self.run_seed(seed)
            results.append(res)

        import pandas as pd

        out_dir = self.cfg.out_root
        ensure_dir(out_dir)
        df = pd.DataFrame(results)
        summary_path = os.path.join(out_dir, "summary.csv")
        df.to_csv(summary_path, index=False)

        # Build report
        report_path = os.path.join(out_dir, "EXPERIMENT_REPORT.md")
        self._write_report(df, report_path)
        return {"summary_csv": summary_path, "report_md": report_path}

    def _write_report(self, df, path: str) -> None:
        lines = []
        lines.append("# Phase 14 Step 2: End-to-End Feature Fusion (Joint Training)\n")
        lines.append("## Config\n")
        cfg = self.cfg
        lines.append(f"- seeds: {cfg.seeds}")
        lines.append(f"- dataset: {cfg.dataset}")
        lines.append(f"- epochs: {cfg.epochs}")
        lines.append(f"- batch_size: {cfg.batch_size}")
        lines.append(f"- backbone_lr: {cfg.backbone_lr}")
        lines.append(f"- head_lr: {cfg.head_lr}")
        lines.append(f"- weight_decay: {cfg.weight_decay}")
        lines.append(f"- bands_mode: {cfg.bands_mode}")
        lines.append(f"- band_norm_mode: {cfg.band_norm_mode}")
        lines.append(f"- matrix_mode: {cfg.matrix_mode}")
        lines.append(f"- global_centering: {cfg.global_centering}")
        lines.append("")

        lines.append("## Alignment Audit Summary\n")
        lines.append("| Seed | Train (S/M/∩) | Test (S/M/∩) |")
        lines.append("| --- | --- | --- |")
        for _, r in df.iterrows():
            lines.append(
                f"| {int(r['seed'])} | "
                f"{int(r['alignment_train_n_spatial'])}/{int(r['alignment_train_n_manifold'])}/"
                f"{int(r['alignment_train_n_intersection'])} | "
                f"{int(r['alignment_test_n_spatial'])}/{int(r['alignment_test_n_manifold'])}/"
                f"{int(r['alignment_test_n_intersection'])} |"
            )
        lines.append("")

        lines.append("## Per-Seed Metrics\n")
        lines.append("| Seed | AccS | AccM | AccF | Rescued | Lost | Net | Complementary | Both | S-Only | M-Only | Neither |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, r in df.iterrows():
            lines.append(
                f"| {int(r['seed'])} | {r['acc_spatial']:.4f} | {r['acc_manifold']:.4f} | {r['acc_fusion']:.4f} | "
                f"{int(r['rescued'])} | {int(r['lost'])} | {int(r['net_gain'])} | {int(r['complementary_count'])} | "
                f"{int(r['both_correct'])} | {int(r['spatial_only'])} | {int(r['manifold_only'])} | {int(r['neither'])} |"
            )
        lines.append("")

        lines.append("## Feature Dimensions\n")
        lines.append("| Seed | Spatial Dim | Manifold Dim |")
        lines.append("| --- | --- | --- |")
        for _, r in df.iterrows():
            lines.append(
                f"| {int(r['seed'])} | {int(r['spatial_feat_dim'])} | {int(r['manifold_feat_dim'])} |"
            )
        lines.append("")

        # Step13F comparison
        step13f_path = "promoted_results/phase13f/step1/seed1/summary.csv"
        if os.path.exists(step13f_path):
            try:
                import pandas as pd

                step13f = pd.read_csv(step13f_path)
                lines.append("## Comparison to Phase 13F Step1 (Late Fusion)\n")
                lines.append("| Seed | Step13F AccF | Step14 AccF | Δ(AccF) |")
                lines.append("| --- | --- | --- | --- |")
                for _, r in df.iterrows():
                    seed = int(r["seed"])
                    match = step13f[step13f["Seed"] == seed]
                    if len(match) > 0:
                        acc_f_13f = float(match.iloc[0]["Fusion Acc"])
                        acc_f_14 = float(r["acc_fusion"])
                        lines.append(f"| {seed} | {acc_f_13f:.4f} | {acc_f_14:.4f} | {acc_f_14 - acc_f_13f:.4f} |")
                lines.append("")
            except Exception:
                lines.append("## Comparison to Phase 13F Step1 (Late Fusion)\n")
                lines.append("Failed to load Step13F summary.csv for comparison.\n")

        # Instability events
        lines.append("## Training Stability\n")
        any_instability = False
        for _, r in df.iterrows():
            events = r.get("instability_events", [])
            if isinstance(events, str):
                try:
                    events = json.loads(events)
                except Exception:
                    events = []
            if events:
                any_instability = True
                lines.append(f"- Seed {int(r['seed'])}:")
                for e in events[:5]:
                    lines.append(f"  - {e}")
        if not any_instability:
            lines.append("- No loss/gradient anomalies detected.")
        lines.append("")

        lines.append("## Risk Monitoring\n")
        for _, r in df.iterrows():
            seed = int(r["seed"])
            acc_s = float(r["acc_spatial"])
            acc_f = float(r["acc_fusion"])
            rescued = int(r["rescued"])
            if acc_f <= acc_s and rescued <= 5:
                lines.append(
                    f"- Seed {seed}: fusion_acc ({acc_f:.4f}) <= spatial_acc ({acc_s:.4f}) "
                    f"and rescued={rescued}. Check alignment audit and manifold feature output."
                )
        lines.append("")

        lines.append("## Conclusion\n")
        lines.append("Joint fusion results above summarize end-to-end feature fusion with strict trial alignment.")

        ensure_dir(os.path.dirname(path))
        with open(path, "w") as f:
            f.write("\n".join(lines))
