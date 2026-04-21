from __future__ import annotations

import os
import tempfile
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class HybridDataset(Dataset):
    def __init__(self, manifold_path: str, dcnet_path: str, labels: Sequence[int]):
        self.manifold = np.load(manifold_path, allow_pickle=True)
        self.dcnet = np.load(dcnet_path, allow_pickle=True)
        self.labels = list(labels)
        if len(self.manifold) != len(self.dcnet) or len(self.manifold) != len(self.labels):
            raise ValueError(
                "HybridDataset length mismatch: "
                f"manifold={len(self.manifold)} dcnet={len(self.dcnet)} labels={len(self.labels)}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        manifold_seq = torch.as_tensor(self.manifold[idx], dtype=torch.float32)
        dcnet_seq = torch.as_tensor(self.dcnet[idx], dtype=torch.float32)
        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return manifold_seq, dcnet_seq, label


def hybrid_collate_fn(batch):
    manifold_seqs, dcnet_seqs, labels = zip(*batch)
    lengths = torch.as_tensor([seq.shape[0] for seq in manifold_seqs], dtype=torch.long)

    padded_manifold = pad_sequence(manifold_seqs, batch_first=True, padding_value=0.0)
    padded_dcnet = pad_sequence(dcnet_seqs, batch_first=True, padding_value=0.0)

    max_len = padded_manifold.shape[1]
    device = padded_manifold.device
    # True means padding (ignore), False means valid data.
    padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1).to(device)

    stacked_labels = torch.stack(labels)
    return padded_manifold, padded_dcnet, padding_mask, stacked_labels


if __name__ == "__main__":
    lengths = [10, 15, 12, 20]
    manifold_list = [np.random.randn(t, 9765).astype(np.float32) for t in lengths]
    dcnet_list = [np.random.randn(t, 62, 5).astype(np.float32) for t in lengths]
    labels = [0, 1, 2, 1]

    with tempfile.TemporaryDirectory() as tmpdir:
        manifold_path = os.path.join(tmpdir, "train_manifold.npy")
        dcnet_path = os.path.join(tmpdir, "train_dcnet.npy")
        np.save(manifold_path, np.array(manifold_list, dtype=object), allow_pickle=True)
        np.save(dcnet_path, np.array(dcnet_list, dtype=object), allow_pickle=True)

        dataset = HybridDataset(manifold_path, dcnet_path, labels)
        loader = DataLoader(dataset, batch_size=2, collate_fn=hybrid_collate_fn, shuffle=False)

        padded_manifold, padded_dcnet, padding_mask, stacked_labels = next(iter(loader))
        print("manifold batch shape:", padded_manifold.shape)
        print("dcnet batch shape:", padded_dcnet.shape)
        print("mask shape:", padding_mask.shape)
        print("labels shape:", stacked_labels.shape)

        batch_lengths = lengths[: padded_manifold.shape[0]]
        for i, seq_len in enumerate(batch_lengths):
            tail_ok = padding_mask[i, seq_len:].all().item() if seq_len < padded_manifold.shape[1] else True
            head_ok = (~padding_mask[i, :seq_len]).all().item() if seq_len > 0 else True
            print(f"sample {i} length {seq_len} head_ok={head_ok} tail_ok={tail_ok}")
