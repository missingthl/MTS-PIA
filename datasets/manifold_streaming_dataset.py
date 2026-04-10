from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ManifoldStreamingDataset(Dataset):
    def __init__(self, manifest_path: str):
        manifest_path = Path(manifest_path)
        data = json.loads(manifest_path.read_text())
        self.file_paths, self.labels = self._parse_manifest(data, manifest_path)
        if len(self.file_paths) != len(self.labels):
            raise ValueError(
                "manifest length mismatch: "
                f"file_paths={len(self.file_paths)} labels={len(self.labels)}"
            )

    @staticmethod
    def _resolve_path(base_dir: Path, path: str) -> Path:
        raw = Path(path)
        if raw.is_absolute():
            return raw
        candidate = base_dir / raw
        if candidate.exists():
            return candidate
        candidate = base_dir.parent / raw
        if candidate.exists():
            return candidate
        return raw

    @staticmethod
    def _parse_manifest(data, manifest_path: Path) -> Tuple[List[str], List[int]]:
        base_dir = manifest_path.parent
        if isinstance(data, dict):
            file_paths = data.get("file_paths") or data.get("paths") or data.get("files")
            if file_paths is None:
                raise ValueError("manifest missing file_paths")
            if "labels" in data:
                labels = data["labels"]
            elif "trials" in data:
                labels = [trial["label"] for trial in data["trials"]]
            else:
                raise ValueError("manifest missing labels/trials")
            resolved = []
            for path in file_paths:
                path = str(path)
                resolved.append(str(ManifoldStreamingDataset._resolve_path(base_dir, path)))
            return resolved, [int(v) for v in labels]

        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                if "file_path" in data[0]:
                    file_paths = [item["file_path"] for item in data]
                elif "path" in data[0]:
                    file_paths = [item["path"] for item in data]
                else:
                    raise ValueError("manifest list entries missing file_path/path")
                if "label" not in data[0]:
                    raise ValueError("manifest list entries missing label")
                labels = [item["label"] for item in data]
                resolved = []
                for path in file_paths:
                    path = str(path)
                    resolved.append(str(ManifoldStreamingDataset._resolve_path(base_dir, path)))
                return resolved, [int(v) for v in labels]
            raise ValueError("manifest list must contain dict entries with file paths and labels")

        raise ValueError("unsupported manifest format")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = np.load(self.file_paths[idx])
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(int(self.labels[idx]), dtype=torch.long)
        return x_tensor, y_tensor
