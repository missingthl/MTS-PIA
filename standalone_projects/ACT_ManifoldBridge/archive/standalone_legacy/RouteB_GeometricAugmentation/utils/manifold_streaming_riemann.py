from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class RiemannianUtils:
    """Utilities for SPD manifold processing."""

    @staticmethod
    def cal_riemannian_mean(cov_matrices: torch.Tensor, max_iter: int = 50, tol: float = 1e-6):
        """
        Compute Fréchet mean (Riemannian mean) for SPD matrices.
        cov_matrices: [..., 62, 62]
        returns: [62, 62]
        """
        assert cov_matrices.shape[-1] == cov_matrices.shape[-2] == 62
        # mild symmetrization to counter numeric drift
        cov_matrices = 0.5 * (cov_matrices + cov_matrices.transpose(-1, -2))
        mean = cov_matrices.mean(dim=0)
        for _ in range(max_iter):
            eigvals, eigvecs = torch.linalg.eigh(mean)
            eigvals = torch.clamp(eigvals, min=1e-6)
            A_half = eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.transpose(-1, -2)
            A_nhalf = eigvecs @ torch.diag(1.0 / eigvals.sqrt()) @ eigvecs.transpose(-1, -2)

            centered = A_nhalf @ cov_matrices @ A_nhalf
            centered = 0.5 * (centered + centered.transpose(-1, -2))
            eig_c, vec_c = torch.linalg.eigh(centered)
            eig_c = torch.clamp(eig_c, min=1e-6)
            log_centered = vec_c @ torch.diag_embed(torch.log(eig_c)) @ vec_c.transpose(-1, -2)

            update = log_centered.mean(dim=0)
            norm = torch.linalg.norm(update)
            mean = A_half @ torch.linalg.matrix_exp(update) @ A_half
            if norm < tol:
                break
        return mean

    @staticmethod
    def tangent_space_mapping(
        cov_matrices: torch.Tensor,
        reference_mean: torch.Tensor,
        *,
        debug_dtype: bool = False,
    ):
        """
        Log-Euclidean mapping to tangent space at reference_mean.
        cov_matrices: [..., 62, 62]
        reference_mean: [62, 62]
        returns: same leading shape, mapped SPD -> symmetric log.
        """
        assert cov_matrices.shape[-1] == cov_matrices.shape[-2] == 62
        assert reference_mean.shape == (62, 62)
        cov64 = cov_matrices.to(dtype=torch.float64)
        ref64 = reference_mean.to(device=cov64.device, dtype=torch.float64)
        if debug_dtype:
            print(
                f"[riemann][tsm] cov dtype={cov_matrices.dtype} ref dtype={reference_mean.dtype} "
                f"cov64 dtype={cov64.dtype} ref64 dtype={ref64.dtype}",
                flush=True,
            )
        eigvals, eigvecs = torch.linalg.eigh(ref64)
        eigvals = torch.clamp(eigvals, min=1e-6)
        A_nhalf = eigvecs @ torch.diag(1.0 / eigvals.sqrt()) @ eigvecs.transpose(-1, -2)

        centered = A_nhalf @ cov64 @ A_nhalf
        eig_c, vec_c = torch.linalg.eigh(centered)
        eig_c = torch.clamp(eig_c, min=1e-6)
        log_centered = vec_c @ torch.diag_embed(torch.log(eig_c)) @ vec_c.transpose(-1, -2)
        out = log_centered.to(dtype=torch.float32)
        if debug_dtype:
            print(f"[riemann][tsm] out dtype={out.dtype}", flush=True)
        return out


class ManifoldStreamingDataset(Dataset):
    """
    Lazy-loading Dataset: loads SPD sequences from disk and applies TSM per band.
    Expected file data shape: [T, 62, 62, 5] or [T, 5, 62, 62]; output shape: [T, 5, 62, 62].
    """

    def __init__(
        self,
        manifest_path: str,
        reference_mean: torch.Tensor,
        transform: Optional = None,
        *,
        tsm_smooth_mode: str = "none",
        tsm_ema_alpha: float = 0.05,
        tsm_kalman_qr: float = 1e-4,
    ):
        self.manifest_path = Path(manifest_path)
        data = json.loads(self.manifest_path.read_text())
        self.file_paths, self.labels = self._parse_manifest(data)
        if len(self.file_paths) != len(self.labels):
            raise ValueError("manifest length mismatch")
        # reference_mean: [5, 62, 62]
        assert reference_mean.shape == (5, 62, 62)
        self.reference_mean = reference_mean
        self.tsm_smooth_mode = self._normalize_smooth_mode(tsm_smooth_mode)
        self.tsm_ema_alpha = float(tsm_ema_alpha)
        self.tsm_kalman_qr = float(tsm_kalman_qr)
        if self.tsm_smooth_mode == "ema":
            if not 0.0 < self.tsm_ema_alpha <= 1.0:
                raise ValueError(f"tsm_ema_alpha must be in (0,1], got {tsm_ema_alpha}")
        if self.tsm_smooth_mode == "kalman":
            if self.tsm_kalman_qr <= 0.0:
                raise ValueError(f"tsm_kalman_qr must be > 0, got {tsm_kalman_qr}")
        # cache A_nhalf per band to avoid repeated eig
        self.A_nhalf = []
        ref64 = reference_mean.to(dtype=torch.float64)
        for b in range(5):
            eigvals, eigvecs = torch.linalg.eigh(ref64[b])
            eigvals = torch.clamp(eigvals, min=1e-6)
            A_nhalf = eigvecs @ torch.diag(1.0 / eigvals.sqrt()) @ eigvecs.transpose(-1, -2)
            self.A_nhalf.append(A_nhalf)
        self.transform = transform

    @staticmethod
    def _normalize_smooth_mode(mode: Optional[str]) -> str:
        key = (mode or "none").strip().lower()
        if key not in {"none", "ema", "kalman"}:
            raise ValueError(f"tsm_smooth_mode must be one of none/ema/kalman, got {mode}")
        return key

    @staticmethod
    def _smooth_single_sequence(
        seq: torch.Tensor,
        mode: str,
        ema_alpha: float,
        kalman_qr: float,
        valid_len: Optional[int] = None,
    ) -> torch.Tensor:
        if mode == "none":
            return seq
        t_len = int(seq.shape[0])
        if valid_len is None:
            valid_len = t_len
        valid_len = max(0, min(valid_len, t_len))
        if valid_len == 0:
            return torch.zeros_like(seq)

        out = torch.zeros_like(seq)
        x = seq[:valid_len]
        if mode == "ema":
            alpha = float(ema_alpha)
            out[:valid_len] = x
            for t in range(1, valid_len):
                out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
            return out

        # kalman (scalar per element, independent)
        q = float(kalman_qr)
        r = 1.0
        state = x[0].clone()
        p = torch.ones_like(state)
        out[0] = state
        for t in range(1, valid_len):
            p = p + q
            k = p / (p + r)
            state = state + k * (x[t] - state)
            p = (1.0 - k) * p
            out[t] = state
        return out

    @classmethod
    def _smooth_sequence(
        cls,
        seq: torch.Tensor,
        mode: str,
        ema_alpha: float,
        kalman_qr: float,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mode == "none":
            return seq
        if seq.ndim == 4:
            valid_len = None
            if padding_mask is not None:
                valid_len = int((~padding_mask).sum().item())
            return cls._smooth_single_sequence(seq, mode, ema_alpha, kalman_qr, valid_len)
        if seq.ndim != 5:
            raise ValueError(f"unsupported seq shape for smoothing: {seq.shape}")
        if padding_mask is None:
            return torch.stack(
                [
                    cls._smooth_single_sequence(s, mode, ema_alpha, kalman_qr, None)
                    for s in seq
                ],
                dim=0,
            )
        if padding_mask.shape[:2] != seq.shape[:2]:
            raise ValueError("padding_mask shape mismatch for smoothing")
        smoothed = []
        for s, mask in zip(seq, padding_mask):
            valid_len = int((~mask).sum().item())
            smoothed.append(cls._smooth_single_sequence(s, mode, ema_alpha, kalman_qr, valid_len))
        return torch.stack(smoothed, dim=0)

    def _parse_manifest(self, data) -> Tuple[List[str], List[int]]:
        base_dir = self.manifest_path.parent
        if not isinstance(data, dict):
            raise ValueError("manifest must be a dict with file_paths and trials/labels")
        paths = data.get("file_paths") or data.get("paths") or data.get("files")
        if paths is None:
            raise ValueError("manifest missing file_paths")
        if "labels" in data:
            labels = data["labels"]
        elif "trials" in data:
            labels = [t["label"] for t in data["trials"]]
        else:
            raise ValueError("manifest missing labels/trials")
        resolved = []
        for p in paths:
            p = str(p)
            cand = base_dir / p
            if cand.exists():
                resolved.append(str(cand))
            elif (base_dir.parent / p).exists():
                resolved.append(str(base_dir.parent / p))
            else:
                resolved.append(p)
        return resolved, [int(l) for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        arr = np.load(self.file_paths[idx])  # [T, 62, 62, 5] or [T, 5, 62, 62]
        arr_t = torch.as_tensor(arr, dtype=torch.float64)
        if arr_t.ndim != 4:
            raise ValueError(f"expected 4D tensor, got shape {arr_t.shape}")
        # Ensure [T, 5, 62, 62]
        if arr_t.shape[1] == 62 and arr_t.shape[2] == 62:
            arr_t = arr_t.permute(0, 3, 1, 2)
        assert arr_t.shape[1:] == (5, 62, 62), f"bad shape after transpose: {arr_t.shape}"

        ref = self.reference_mean.to(arr_t.device, dtype=torch.float64)  # [5, 62, 62]
        t_len, bands, _, _ = arr_t.shape
        proj_list = []
        for b in range(bands):
            covs = arr_t[:, b, :, :]  # [T, 62, 62]
            # enforce symmetry to reduce numerical drift
            covs = 0.5 * (covs + covs.transpose(-1, -2))
            ref_b = ref[b]
            A_nhalf = self.A_nhalf[b].to(arr_t.device)
            centered = A_nhalf @ covs @ A_nhalf
            eig_c, vec_c = torch.linalg.eigh(centered)
            eig_c = torch.clamp(eig_c, min=1e-6)
            proj = vec_c @ torch.diag_embed(torch.log(eig_c)) @ vec_c.transpose(-1, -2)
            proj_list.append(proj.unsqueeze(1))  # [T,1,62,62]
        proj_tensor = torch.cat(proj_list, dim=1)  # [T,5,62,62] float64

        if self.tsm_smooth_mode != "none":
            proj_tensor = self._smooth_sequence(
                proj_tensor,
                self.tsm_smooth_mode,
                self.tsm_ema_alpha,
                self.tsm_kalman_qr,
                padding_mask=None,
            )
        if self.transform:
            proj_tensor = self.transform(proj_tensor)
        proj_tensor = proj_tensor.to(dtype=torch.float32)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return proj_tensor, label


def collate_fn_pad(batch):
    sequences, labels = zip(*batch)
    lengths = [seq.shape[0] for seq in sequences]
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)  # [B, maxT, 5, 62, 62]
    max_len = padded.shape[1]
    padding_mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l < max_len:
            padding_mask[i, l:] = True
    labels = torch.stack(labels)
    return padded, labels, padding_mask


if __name__ == "__main__":
    # Minimal smoke test with synthetic SPD data
    import tempfile
    import os
    from torch.utils.data import DataLoader

    tmpdir = tempfile.mkdtemp()
    manifest = {"file_paths": [], "trials": []}
    for i, t in enumerate([10, 15, 12, 20]):
        path = os.path.join(tmpdir, f"trial_{i}.npy")
        seq = []
        for _ in range(t):
            cov_bands = []
            for _ in range(5):
                A = np.random.randn(62, 62)
                cov = A @ A.T + np.eye(62) * 1e-3
                cov_bands.append(cov)
            cov_bands = np.stack(cov_bands, axis=0)  # [5,62,62]
            cov_bands = np.transpose(cov_bands, (1, 2, 0))  # [62,62,5]
            seq.append(cov_bands)
        seq = np.stack(seq, axis=0)  # [T,62,62,5]
        np.save(path, seq)
        manifest["file_paths"].append(path)
        manifest["trials"].append({"label": i % 3})
    manifest_path = os.path.join(tmpdir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    ref_mean = torch.stack([torch.eye(62) for _ in range(5)], dim=0)
    ds = ManifoldStreamingDataset(manifest_path, reference_mean=ref_mean)
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn_pad, shuffle=False)
    x, y, mask = next(iter(dl))
    print("batch x shape:", x.shape)       # [2, maxT, 5, 62, 62]
    print("labels shape:", y.shape)
    print("mask shape:", mask.shape)
    print("mask example:", mask[0])
