from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple
import gc
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from datasets.types import FoldData
from PIA.snn import SNNClassifier


def _log_stats(tag: str, X: np.ndarray) -> None:
    X = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(X)
    finite_ratio = float(np.mean(finite)) if X.size > 0 else 0.0
    if finite_ratio < 1.0:
        print(f"{tag} finite ratio: {finite_ratio:.6f}")
    x_min = float(np.nanmin(X))
    x_max = float(np.nanmax(X))
    x_mean = float(np.nanmean(X))
    x_std = float(np.nanstd(X))
    print(f"{tag} stats: min={x_min:.6f} max={x_max:.6f} mean={x_mean:.6f} std={x_std:.6f}")


def _format_array_stats(X: np.ndarray) -> str:
    X = np.asarray(X, dtype=np.float64)
    return (
        f"shape={X.shape} dtype={X.dtype} "
        f"min={float(np.nanmin(X)):.6f} max={float(np.nanmax(X)):.6f} "
        f"mean={float(np.nanmean(X)):.6f} std={float(np.nanstd(X)):.6f}"
    )


def _format_tensor_stats(x: torch.Tensor) -> str:
    x = x.detach()
    return (
        f"shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={float(x.min().item()):.6f} max={float(x.max().item()):.6f} "
        f"mean={float(x.mean().item()):.6f} std={float(x.std().item()):.6f}"
    )


def _format_label_stats(y: np.ndarray, limit: int = 20) -> str:
    y_arr = np.asarray(y)
    uniq = np.unique(y_arr)
    uniq_head = uniq[:limit]
    return (
        f"shape={y_arr.shape} dtype={y_arr.dtype} "
        f"unique_head={uniq_head.tolist()}"
    )


@lru_cache(maxsize=1)
def _seedv_topo_index(grid_size: Tuple[int, int] = (9, 9)):
    """
    Build a fixed, reproducible 62-channel -> 2D grid mapping for SEED-V.
    Uses data/SEED-V/channel_62_pos.locs (theta, radius) and maps to a 9x9 grid.
    Collisions are averaged later during mapping.
    """
    try:
        from datasets.seedv_preprocess import SEEDV_CHANNELS_PATH
    except Exception as exc:
        raise RuntimeError("Failed to locate SEED-V channel list for topo mapping.") from exc
    if not os.path.isfile(SEEDV_CHANNELS_PATH):
        raise FileNotFoundError(f"SEED-V channel list not found: {SEEDV_CHANNELS_PATH}")

    thetas = []
    radii = []
    with open(SEEDV_CHANNELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            thetas.append(float(parts[1]))
            radii.append(float(parts[2]))
    if len(thetas) != 62:
        raise ValueError(f"Unexpected channel count in {SEEDV_CHANNELS_PATH}: {len(thetas)}")

    theta = np.deg2rad(np.asarray(thetas, dtype=np.float64))
    radius = np.asarray(radii, dtype=np.float64)
    # Polar -> Cartesian (Fpz near top center)
    x = radius * np.sin(theta)
    y = radius * np.cos(theta)

    H, W = grid_size
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_norm = (x - x_min) / (x_max - x_min + 1e-12)
    y_norm = (y - y_min) / (y_max - y_min + 1e-12)
    cols = np.rint(x_norm * (W - 1)).astype(int)
    rows = np.rint((1.0 - y_norm) * (H - 1)).astype(int)
    cols = np.clip(cols, 0, W - 1)
    rows = np.clip(rows, 0, H - 1)

    counts = np.zeros((H, W), dtype=np.float32)
    for r, c in zip(rows, cols):
        counts[r, c] += 1.0
    counts[counts == 0] = 1.0

    return rows, cols, counts


def _de_flat_to_topo(
    X_flat: np.ndarray,
    *,
    n_channels: int = 62,
    n_bands: int = 5,
    grid_size: Tuple[int, int] = (9, 9),
) -> np.ndarray:
    """
    Map DE features (band-major, C=62, B=5) to a fixed 9x9 topo grid.
    Uses channel_62_pos.locs mapping; collisions are averaged.
    """
    X_flat = np.asarray(X_flat, dtype=np.float32)
    if X_flat.ndim != 2:
        raise ValueError(f"Expected flat DE features [N,D], got {X_flat.shape}")
    D = X_flat.shape[1]
    expected = n_channels * n_bands
    if D != expected:
        raise ValueError(f"Topo input expects D={expected} (C={n_channels}, B={n_bands}), got {D}")
    X = X_flat.reshape(X_flat.shape[0], n_bands, n_channels)
    H, W = grid_size
    rows, cols, counts = _seedv_topo_index(grid_size)
    out = np.zeros((X.shape[0], n_bands, H, W), dtype=np.float32)
    for ch in range(n_channels):
        r = rows[ch]
        c = cols[ch]
        out[:, :, r, c] += X[:, :, ch]
    out = out / counts[None, None, :, :]
    return out


class DCNetTorch(nn.Module):
    """
    PyTorch implementation of DCNet (Deep Convolutional Network) for EEG spatial analysis.
    Based on the original TensorFlow implementation.
    Features a deconvolutional top-down pathway followed by a convolutional bottom-up pathway.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        deconv_first_filters: Optional[int] = None,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        init: str = "default",
        classifier_type: str = "conv",
    ):
        super().__init__()
        # Reference Architecture:
        # Input: (B, input_dim, 1, 1)
        # Deconv Path (Upsampling):
        # 1. (input_dim -> input_dim) k=2 s=2 -> 2x2
        # 2. (input_dim -> 120)       k=2 s=2 -> 4x4
        # 3. (120 -> 50)              k=2 s=2 -> 8x8
        # 4. (50 -> 18)               k=2 s=2 -> 16x16
        # 5. (18 -> 3)                k=2 s=2 -> 32x32
        
        # Conv Path (Downsampling):
        # 1. (3 -> 3)   k=2 s=2 -> 16x16
        # 2. (3 -> 18)  k=2 s=2 -> 8x8
        # 3. (18 -> 120) k=2 s=2 -> 4x4
        # 4. (120 -> 200) k=2 s=2 -> 2x2
        # 5. (200 -> 500) k=2 s=2 -> 1x1
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=2, stride=2),
            nn.SELU(inplace=True),
            nn.ConvTranspose2d(input_dim, 120, kernel_size=2, stride=2),
            nn.SELU(inplace=True),
            nn.ConvTranspose2d(120, 50, kernel_size=2, stride=2),
            nn.SELU(inplace=True),
            nn.ConvTranspose2d(50, 18, kernel_size=2, stride=2),
            nn.SELU(inplace=True),
            nn.ConvTranspose2d(18, 3, kernel_size=2, stride=2), # Output 32x32
            nn.SELU(inplace=True),
        )

        def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.SELU(inplace=True),
                nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum),
            )

        self.conv_blocks = nn.Sequential(
            _conv_block(3, 3),    # -> 16x16
            _conv_block(3, 18),   # -> 8x8
            _conv_block(18, 120), # -> 4x4
            _conv_block(120, 200),# -> 2x2
            _conv_block(200, 500),# -> 1x1
        )
        self.dropout = nn.AlphaDropout(p=0.5)
        self.classifier_type = classifier_type.lower()
        
        # Classifier operates on 1x1 spatial map (500 filters)
        if self.classifier_type == "conv":
            self.classifier = nn.Conv2d(500, num_classes, kernel_size=1, stride=1)
        elif self.classifier_type == "linear":
            self.classifier = nn.Linear(500, num_classes)
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")
            
        if init and init != "default":
            self._init_weights(init)

    def _init_weights(self, init: str) -> None:
        mode = init.lower()
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                if mode == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif mode == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif mode == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="selu")
                else:
                    raise ValueError(f"Unknown init mode: {init}")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # x input: (B, C, 1, 1) or (B, C)
        if x.ndim == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        
        x = self.deconv(x)
        x = self.conv_blocks(x)
        features = x
        x = self.dropout(x)
        
        if self.classifier_type == "conv":
            x = self.classifier(x)
            # x is (B, num_classes, 1, 1) -> (B, num_classes)
            logits = x.flatten(1)
        else:
            x = x.flatten(1)
            logits = self.classifier(x)
            
        if return_features:
            return logits, features.flatten(1)
        return logits


def _set_bn_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


@dataclass
class SpatialDCNetRunnerTorch:
    """
    Runner for training and evaluating DCNetTorch model on spatial stream data.
    Handles data loading, model initialization, training loop, validation, and checkpointing.
    """
    num_classes: int
    epochs: int = 40
    batch_size: int = 2048
    eval_batch_size: Optional[int] = None
    num_workers: int = 4
    spatial_head: str = "softmax"  # softmax | snn
    spatial_input: str = "flat"  # flat | topo
    C_head: float = 4.0
    snn_nodes: int = 3
    snn_activation: str = "sigmoid"
    checkpoint_root: str = "experiments/checkpoints"
    checkpoint_prefix: str = "seedv_spatial_torch"
    learning_rate: float = 1e-4
    adam_eps: float = 1e-8
    bn_eps: float = 1e-5
    bn_momentum: float = 0.1
    init: str = "default"
    deconv_first_filters: Optional[int] = None
    classifier_type: str = "conv"
    clipnorm: float = 1.0
    check_inputs: bool = True
    log_input_stats: bool = True
    freeze_bn: bool = False
    terminate_on_nan: bool = True
    device: Optional[str] = None
    seed_de_debug: bool = False

    def _get_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_eval_batch_size(self) -> int:
        if self.eval_batch_size is None or self.eval_batch_size <= 0:
            return int(self.batch_size)
        return int(self.eval_batch_size)

    def _batched_logits(
        self,
        model: nn.Module,
        X: torch.Tensor,
        device: torch.device,
        batch_size: int,
    ) -> np.ndarray:
        logits_list = []
        n = X.shape[0]
        for start in range(0, n, batch_size):
            xb = X[start : start + batch_size].to(device, non_blocking=True)
            logits = model(xb)
            logits_list.append(logits.detach().cpu())
            del xb, logits
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return torch.cat(logits_list, dim=0).numpy()

    def _batched_features(
        self,
        model: nn.Module,
        X: torch.Tensor,
        device: torch.device,
        batch_size: int,
    ) -> np.ndarray:
        feats = []
        n = X.shape[0]
        for start in range(0, n, batch_size):
            xb = X[start : start + batch_size].to(device, non_blocking=True)
            _, feat = model(xb, return_features=True)
            feats.append(feat.detach().cpu())
            del xb, feat
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return torch.cat(feats, dim=0).numpy()

    def fit_predict(self, fold: FoldData, fold_name: str) -> Dict[str, np.ndarray]:
        X_tr_flat = np.asarray(fold.X_train, dtype=np.float32)
        y_tr_raw = np.asarray(fold.y_train)
        X_te_flat = np.asarray(fold.X_test, dtype=np.float32)
        y_te_raw = np.asarray(fold.y_test)
        y_tr = np.asarray(fold.y_train).astype(int).ravel()
        y_te = np.asarray(fold.y_test).astype(int).ravel()
        tid_te = fold.trial_id_test if hasattr(fold, 'trial_id_test') and fold.trial_id_test is not None else None
        tid_tr = fold.trial_id_train if hasattr(fold, 'trial_id_train') and fold.trial_id_train is not None else None

        if X_tr_flat.ndim != 2 or X_te_flat.ndim != 2:
            raise ValueError(
                f"[{fold_name}] expected flat features [N,D]; got train {X_tr_flat.shape} "
                f"test {X_te_flat.shape}."
            )

        if self.log_input_stats:
            _log_stats(f"[{fold_name}][input][train]", X_tr_flat)
            _log_stats(f"[{fold_name}][input][test]", X_te_flat)
        if self.check_inputs:
            if not np.isfinite(X_tr_flat).all():
                raise ValueError(f"[{fold_name}] NaN/Inf detected in X_train")
            if not np.isfinite(X_te_flat).all():
                raise ValueError(f"[{fold_name}] NaN/Inf detected in X_test")
        if y_tr.min(initial=0) < 0 or y_tr.max(initial=0) >= self.num_classes:
            raise ValueError(f"[{fold_name}] y_train out of range [0,{self.num_classes-1}]")
        if y_te.min(initial=0) < 0 or y_te.max(initial=0) >= self.num_classes:
            raise ValueError(f"[{fold_name}] y_test out of range [0,{self.num_classes-1}]")

        debug_labels = bool(self.seed_de_debug and fold_name == "fold1")
        if debug_labels:
            print(f"[{fold_name}][debug][label_raw][train] {_format_label_stats(y_tr_raw)}")
            print(f"[{fold_name}][debug][label_raw][test] {_format_label_stats(y_te_raw)}")
            print(f"[{fold_name}][debug][label_idx][train] {_format_label_stats(y_tr)}")
            print(f"[{fold_name}][debug][label_idx][test] {_format_label_stats(y_te)}")
            if y_tr_raw.ndim > 1:
                print(
                    f"[{fold_name}][debug][label_raw] ndim>1 detected; "
                    f"train_shape={y_tr_raw.shape} test_shape={y_te_raw.shape}"
                )
                if y_tr_raw.ndim == 2 and y_tr_raw.shape[1] == self.num_classes:
                    row_sum = np.sum(y_tr_raw, axis=1)
                    row_sum_min = float(np.min(row_sum))
                    row_sum_max = float(np.max(row_sum))
                    is_binary = np.all((y_tr_raw == 0) | (y_tr_raw == 1))
                    print(
                        f"[{fold_name}][debug][label_raw] one_hot_candidate={bool(is_binary)} "
                        f"row_sum_min={row_sum_min:.3f} row_sum_max={row_sum_max:.3f}"
                    )
                    if is_binary:
                        y_tr_argmax = np.argmax(y_tr_raw, axis=1)
                        y_te_argmax = np.argmax(y_te_raw, axis=1)
                        print(
                            f"[{fold_name}][debug][label_argmax][train] "
                            f"{_format_label_stats(y_tr_argmax)}"
                        )
                        print(
                            f"[{fold_name}][debug][label_argmax][test] "
                            f"{_format_label_stats(y_te_argmax)}"
                        )
            if y_tr.dtype != np.int64 or y_tr.ndim != 1:
                raise AssertionError(
                    f"[{fold_name}] y_train must be int64 1D after processing; got {y_tr.dtype} {y_tr.shape}"
                )
            if y_te.dtype != np.int64 or y_te.ndim != 1:
                raise AssertionError(
                    f"[{fold_name}] y_test must be int64 1D after processing; got {y_te.dtype} {y_te.shape}"
                )
            if y_tr.min(initial=0) < 0 or y_tr.max(initial=0) >= self.num_classes:
                raise AssertionError(f"[{fold_name}] y_train out of range [0,{self.num_classes-1}]")
            if y_te.min(initial=0) < 0 or y_te.max(initial=0) >= self.num_classes:
                raise AssertionError(f"[{fold_name}] y_test out of range [0,{self.num_classes-1}]")

        topo_grid = None
        topo_debug = None
        if self.spatial_input.lower() == "topo":
            if self.classifier_type.lower() == "linear":
                raise ValueError(
                    f"[{fold_name}] topo input is incompatible with linear classifier; use conv."
                )
            if (
                self.deconv_first_filters is not None
                and int(self.deconv_first_filters) == 310
                and X_tr_flat.shape[1] != 310
            ):
                raise ValueError(
                    f"[{fold_name}] deconv_first=310 expects D=310 in topo mode, got {X_tr_flat.shape[1]}."
                )
            X_tr_topo = _de_flat_to_topo(X_tr_flat)
        # Remove topo mapping, pass flat features to DCNet.
        # The DCNetTorch model expects input in [B, C, H, W] format.
        # For flat features, we treat C as the feature dimension and H=W=1.
        input_dim = X_tr_flat.shape[1]
        if (
            self.deconv_first_filters is not None
            and int(self.deconv_first_filters) == 310
            and input_dim != 310
        ):
            raise ValueError(
                f"[{fold_name}] deconv_first=310 expects input_dim=310, got {input_dim}."
            )

        # Use 1x1 spatial input for DCNet refactored architecture
        X_tr = X_tr_flat.reshape(X_tr_flat.shape[0], input_dim, 1, 1)
        X_te = X_te_flat.reshape(X_te_flat.shape[0], input_dim, 1, 1)

        # Dataset / Loader
        train_dataset = TensorDataset(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(y_tr).long(),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_te).float(),
            torch.from_numpy(y_te).long(),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device != "cpu" else False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device != "cpu" else False,
        )

        os.makedirs(self.checkpoint_root, exist_ok=True)
        checkpoint_path = os.path.join(
            self.checkpoint_root,
            f"{self.checkpoint_prefix}_{fold_name}.pt",
        )
        print(f"[{fold_name}] checkpoint path: {checkpoint_path}")

        device = self._get_device()
        eval_batch_size = self._get_eval_batch_size()
        model = DCNetTorch(
            input_dim=input_dim,
            num_classes=self.num_classes,
            deconv_first_filters=None, # Automatically uses input_dim
            bn_eps=self.bn_eps,
            bn_momentum=self.bn_momentum,
            init=self.init,
            classifier_type="conv",
        ).to(device).float()
        if self.freeze_bn:
            _set_bn_eval(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, eps=self.adam_eps)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_model_state = None

        debug_printed = False
        eval_debug_printed = False
        for epoch in range(self.epochs):
            model.train()
            if self.freeze_bn:
                _set_bn_eval(model)
            
            total_loss = 0.0
            correct = 0
            total = 0
            nan_stop = False
            
            for xb, yb in train_loader:
                if not debug_printed and fold_name == "fold1" and total == 0: # Check total == 0 for first batch
                    print(f"[{fold_name}][debug][raw] {_format_array_stats(X_tr_flat)}")
                    print(f"[{fold_name}][debug][input] {_format_tensor_stats(xb)}")
                    print(
                        f"[{fold_name}][debug][label_batch] shape={tuple(yb.shape)} "
                        f"dtype={yb.dtype} unique_head={torch.unique(yb)[:20].cpu().tolist()}"
                    )
                    if self.spatial_input.lower() == "topo":
                        pass # Refactor removed topo support
                    else:
                        print(
                            f"[{fold_name}][debug][layout] input=[B,C,H,W]=[batch,{input_dim},1,1]; "
                            "channels=flat features"
                        )
                    debug_printed = True
                
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                
                loss = criterion(logits, yb)
                if self.terminate_on_nan and not torch.isfinite(loss):
                    print(f"[{fold_name}] NaN detected at epoch {epoch + 1} (train loss)")
                    nan_stop = True
                    break
                loss.backward()
                if self.clipnorm and self.clipnorm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.clipnorm)
                optimizer.step()
                
                total_loss += float(loss.item()) * xb.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == yb).sum().item())
                total += xb.size(0)
                
            if nan_stop or total == 0:
                break
            train_loss = total_loss / total
            train_acc = correct / total

            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    logits = model(xb)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == yb).sum().item()
                    val_total += xb.size(0)
            
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            
            if debug_labels and not eval_debug_printed:
                 print(f"[{fold_name}][debug][eval] val_acc={val_acc:.4f}")
                 eval_debug_printed = True

            print(
                f"Epoch {epoch + 1}/{self.epochs} "
                f"- loss: {train_loss:.4f} - accuracy: {train_acc:.4f} "
                f"- val_accuracy: {val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = {
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "num_classes": self.num_classes,
                }
                torch.save(best_model_state, checkpoint_path)

        try:
            best_model_state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(best_model_state["model_state"])
            model_source = "checkpoint"
        except Exception as exc:
            print(f"[{fold_name}] Warning: failed to load checkpoint, using last model: {exc}")
            model_source = "last_model"
        print(f"[{fold_name}] model source: {model_source}")

        model.eval()
        with torch.no_grad():
            X_tr_t = torch.from_numpy(X_tr_flat).float()
            X_te_t = torch.from_numpy(X_te_flat).float()
            if self.spatial_head.lower() == "softmax":
                logits_te = self._batched_logits(model, X_te_t, device, eval_batch_size)
                logits_tr = self._batched_logits(model, X_tr_t, device, eval_batch_size)
                proba_test = torch.softmax(torch.from_numpy(logits_te), dim=1).numpy()
                proba_train = torch.softmax(torch.from_numpy(logits_tr), dim=1).numpy()
            elif self.spatial_head.lower() == "snn":
                H_tr = self._batched_features(model, X_tr, device, eval_batch_size)
                H_te = self._batched_features(model, X_te, device, eval_batch_size)
                snn = SNNClassifier(
                    n_nodes=self.snn_nodes,
                    C=self.C_head,
                    activation="sigmoid" if self.snn_activation == "sigmoid" else "sine",
                ).fit(H_tr, y_tr)
                proba_test = snn.predict_proba(H_te)
                proba_train = snn.predict_proba(H_tr)
            else:
                raise ValueError(f"Unknown spatial_head: {self.spatial_head}")
        
        # Phase 9.0: Export Trial-Level Predictions
        seed = "unknown"
        # Try to parse seed from checkpoint_prefix or args (not stored in runner directly)
        # But we can look at fold_name if passed, or just use a generic path if out_prefix not available.
        # Actually we need the Seed.
        # We can pass seed via construction or assume it is handled by caller script setting an env var or file arg?
        # Better: caller script sets `out_prefix` attribute on runner if possible.
        # Check `run_phase9_export.py` plan: we will likely instantiate runner with known seed.
        # But `SpatialDCNetRunnerTorch` doesn't have `out_prefix`.
        # I will rely on `fold_name` or add `out_prefix` support.
        # Or just export if `tid_te` is present.
        
        if tid_te is not None:
             import pandas as pd
             # Using generic name pattern, assuming seed can be inferred or we use a standard name that the script renames.
             # Actually safer to let the script handle renaming if we output to a known location?
             # Or just parse seed from somewhere.
             # Let's use `experiments/phase9_fusion/preds/spatial_debug.csv` and let script move it?
             # No, parallel runs might conflict.
             # I will generate filename based on `fold_name`.
             # If fold_name contains "seedX", use it.
             
             import re
             m = re.search(r"seed(\d+)", fold_name)
             if m:
                 s_val = m.group(1)
                 export_path = f"experiments/phase9_fusion/preds/spatial_trial_preds_seed{s_val}.csv"
                 os.makedirs(os.path.dirname(export_path), exist_ok=True)
                 
                 # Create DF for Test
                 df_te = pd.DataFrame({
                     "trial_id": tid_te,
                     "true_label": y_te,
                     "prob_0": proba_test[:, 0],
                     "prob_1": proba_test[:, 1],
                     "prob_2": proba_test[:, 2] if proba_test.shape[1] > 2 else 0.0,
                 })
                 
                 # Create DF for Train (if available)
                 df_tr = pd.DataFrame()
                 if tid_tr is not None and proba_train is not None:
                     df_tr = pd.DataFrame({
                         "trial_id": tid_tr,
                         "true_label": y_tr,
                         "prob_0": proba_train[:, 0],
                         "prob_1": proba_train[:, 1],
                         "prob_2": proba_train[:, 2] if proba_train.shape[1] > 2 else 0.0,
                         "split": "train"
                     })
                     
                 # Concat
                 df_all = pd.concat([df_te, df_tr], ignore_index=True)
                 
                 # Aggregate
                 # Mode: Mean Prob
                 agg = df_all.groupby(["trial_id", "split"]).agg({
                     "true_label": "first", 
                     "prob_0": "mean",
                     "prob_1": "mean",
                     "prob_2": "mean",
                     "trial_id": "count" # Count windows (will result in 'n_windows' after rename)
                 })
                 # Reset index to make trial_id and split columns
                 agg = agg.rename(columns={"trial_id": "n_windows"}).reset_index()
                 
                 agg["pred_label"] = np.argmax(agg[["prob_0", "prob_1", "prob_2"]].values, axis=1)
                 agg["mean_prob_max"] = np.max(agg[["prob_0", "prob_1", "prob_2"]].values, axis=1)
                 
                 # Phase 13A: Entropy & Margin
                 probs = agg[["prob_0", "prob_1", "prob_2"]].values
                 p_safe = np.clip(probs, 1e-9, 1.0)
                 agg["entropy"] = -np.sum(p_safe * np.log(p_safe), axis=1)
                 p_sorted = np.sort(probs, axis=1)
                 agg["margin"] = p_sorted[:, -1] - p_sorted[:, -2]
                 
                 agg["seed"] = s_val

                 
                 # Save
                 # Reorder columns
                 cols = ["seed", "true_label", "n_windows", "prob_0", "prob_1", "prob_2", "pred_label", "entropy", "margin", "trial_id", "split"]
                 agg.to_csv(export_path, columns=cols, index=False)
                 print(f"[{fold_name}] Exported spatial preds to {export_path}")

        if tid_tr is not None:
             import pandas as pd
             import re
             m = re.search(r"seed(\d+)", fold_name)
             if m:
                 s_val = m.group(1)
                 export_path_tr = f"experiments/phase9_fusion/preds/spatial_train_preds_seed{s_val}.csv"
                 os.makedirs(os.path.dirname(export_path_tr), exist_ok=True)
                 
                 # Helper to export
                 # proba_train is (N, K)
                 # y_tr is (N,) or (N, K)?
                 # y_tr was forced to ravel int earlier.
                 
                 df = pd.DataFrame({
                     "trial_id": tid_tr,
                     "true_label": y_tr, # sample-level labels
                     "prob_0": proba_train[:, 0],
                     "prob_1": proba_train[:, 1],
                     "prob_2": proba_train[:, 2] if proba_train.shape[1] > 2 else 0.0
                 })
                 
                 agg = df.groupby("trial_id").agg({
                     "true_label": "first",
                     "prob_0": "mean",
                     "prob_1": "mean",
                     "prob_2": "mean",
                     "trial_id": "count"
                 }).rename(columns={"trial_id": "n_windows"})
                 
                 agg["pred_label"] = np.argmax(agg[["prob_0", "prob_1", "prob_2"]].values, axis=1)
                 agg["mean_prob_max"] = np.max(agg[["prob_0", "prob_1", "prob_2"]].values, axis=1)
                 
                 # Phase 13A: Entropy & Margin
                 probs = agg[["prob_0", "prob_1", "prob_2"]].values
                 p_safe = np.clip(probs, 1e-9, 1.0)
                 agg["entropy"] = -np.sum(p_safe * np.log(p_safe), axis=1)
                 p_sorted = np.sort(probs, axis=1)
                 agg["margin"] = p_sorted[:, -1] - p_sorted[:, -2]
                 
                 agg["seed"] = s_val
                 
                 cols = ["seed", "true_label", "n_windows", "prob_0", "prob_1", "prob_2", "pred_label", "entropy", "margin"]
                 agg.to_csv(export_path_tr, columns=cols, index=True)
                 print(f"[{fold_name}] Exported spatial TRAIN preds to {export_path_tr}")


        result = {
            "sample_proba_test": np.asarray(proba_test, dtype=np.float64),
            "sample_proba_train": np.asarray(proba_train, dtype=np.float64),
            "y_test": y_te,
            "y_train": y_tr,
            "best_val_acc": float(best_acc),
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return result
