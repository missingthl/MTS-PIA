import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.io import loadmat


def _list_mat_keys(mat: Dict) -> List[str]:
    return [k for k in mat.keys() if not k.startswith("__")]


def _shape_of(mat: Dict, key: str):
    if key not in mat:
        return None
    arr = mat[key]
    if isinstance(arr, np.ndarray):
        return list(arr.shape)
    return None


def _key_presence(keys: List[str], prefix: str) -> Dict[str, List[str]]:
    present = []
    missing = []
    for i in range(1, 16):
        key = f"{prefix}{i}"
        if key in keys:
            present.append(key)
        else:
            missing.append(key)
    return {"present": present, "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SEED official .mat structure.")
    parser.add_argument("--mat-root", required=True, help="ExtractedFeatures_1s or _4s")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-n", type=int, default=3)
    parser.add_argument("--out-json", required=True, help="output JSON path")
    args = parser.parse_args()

    mat_root = Path(args.mat_root)
    if not mat_root.is_dir():
        raise FileNotFoundError(f"mat_root not found: {mat_root}")

    files = sorted(mat_root.glob("*.mat"))
    if not files:
        raise ValueError(f"no mat files in {mat_root}")
    rng = np.random.default_rng(int(args.seed))
    sample_n = min(int(args.sample_n), len(files))
    sample_files = rng.choice(files, size=sample_n, replace=False)

    samples = []
    shape_lds = []
    shape_moving = []

    for path in sample_files:
        mat = loadmat(path)
        keys = _list_mat_keys(mat)
        lds_info = _key_presence(keys, "de_LDS")
        moving_info = _key_presence(keys, "de_movingAve")
        lds_shape = _shape_of(mat, "de_LDS1")
        moving_shape = _shape_of(mat, "de_movingAve1")
        if lds_shape:
            shape_lds.append(tuple(lds_shape))
        if moving_shape:
            shape_moving.append(tuple(moving_shape))
        samples.append(
            {
                "file": str(path),
                "keys_count": len(keys),
                "de_LDS": lds_info,
                "de_movingAve": moving_info,
                "de_LDS1_shape": lds_shape,
                "de_movingAve1_shape": moving_shape,
            }
        )

    lds_shape_mode = None
    moving_shape_mode = None
    if shape_lds:
        lds_shape_mode = list(Counter(shape_lds).most_common(1)[0][0])
    if shape_moving:
        moving_shape_mode = list(Counter(shape_moving).most_common(1)[0][0])

    report = {
        "mat_root": str(mat_root),
        "sample_n": int(sample_n),
        "samples": samples,
        "de_LDS1_shape_mode": lds_shape_mode,
        "de_movingAve1_shape_mode": moving_shape_mode,
        "sample_unit": "trial-level keys (de_LDS1..15, de_movingAve1..15)",
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"[mat_schema] root={mat_root} samples={sample_n}", flush=True)
    print(f"[mat_schema] de_LDS1_shape_mode={lds_shape_mode}", flush=True)
    print(f"[mat_schema] report={out_path}", flush=True)


if __name__ == "__main__":
    main()
