import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pickle


LABEL_KEYS = {
    "label",
    "labels",
    "y",
    "y_train",
    "y_test",
    "label_train",
    "label_test",
    "train_label",
    "test_label",
}


def _describe_array(arr: np.ndarray) -> Dict[str, object]:
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def _extract_labels(key: str, obj: Any) -> List[int]:
    if key.lower() not in LABEL_KEYS:
        return []
    if isinstance(obj, np.ndarray):
        flat = obj.reshape(-1)
        if flat.size == 0:
            return []
        return [int(x) for x in flat.tolist()]
    if isinstance(obj, (list, tuple)):
        return [int(x) for x in obj]
    return []


def _summarize_dict(d: Dict[str, Any]) -> Dict[str, object]:
    keys = list(d.keys())
    arrays = []
    labels = []
    for k in keys:
        v = d[k]
        if isinstance(v, np.ndarray):
            arrays.append((k, v))
            labels.extend(_extract_labels(k, v))
    arrays = arrays[:5]
    return {
        "keys": keys,
        "arrays_sample": {k: _describe_array(v) for k, v in arrays},
        "label_values": labels,
    }


def _infer_sample_unit(shapes: List[List[int]]) -> Tuple[str, str]:
    first_dims = [s[0] for s in shapes if s]
    if any(d >= 200 for d in first_dims):
        return "window-wise", "found first_dim>=200"
    if any(10 <= d <= 60 for d in first_dims):
        return "trial-wise", "found 10<=first_dim<=60"
    return "clip-wise", "no clear window/trial signature"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit EEG npz structure.")
    parser.add_argument("--npz-dir", required=True, help="eeg_used_1s or eeg_used_4s directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-n", type=int, default=3)
    parser.add_argument(
        "--out-json",
        required=True,
        help="output JSON path",
    )
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    if not npz_dir.is_dir():
        raise FileNotFoundError(f"npz_dir not found: {npz_dir}")

    files = sorted(npz_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"no npz files in {npz_dir}")
    rng = np.random.default_rng(int(args.seed))
    sample_n = min(int(args.sample_n), len(files))
    sample_files = rng.choice(files, size=sample_n, replace=False)

    samples = []
    all_label_values: List[int] = []
    all_shapes: List[List[int]] = []

    for path in sample_files:
        entry = {"file": str(path), "keys": []}
        with np.load(path, allow_pickle=True) as data:
            keys = list(data.keys())
            entry["keys"] = keys
            entry["items"] = {}
            for key in keys:
                obj = data[key]
                if isinstance(obj, np.ndarray) and obj.size == 1 and obj.shape == ():
                    item = obj.item()
                    if isinstance(item, (bytes, bytearray)):
                        try:
                            item = pickle.loads(item)
                        except Exception:
                            pass
                    if isinstance(item, dict):
                        summary = _summarize_dict(item)
                        entry["items"][key] = {
                            "type": "dict",
                            "keys": summary["keys"],
                            "arrays_sample": summary["arrays_sample"],
                        }
                        all_label_values.extend(summary["label_values"])
                        for desc in summary["arrays_sample"].values():
                            all_shapes.append(desc["shape"])
                        continue
                if isinstance(obj, np.ndarray):
                    entry["items"][key] = {
                        "type": "array",
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    }
                    all_shapes.append(list(obj.shape))
                    if obj.shape == () and obj.dtype.kind in {"S", "O"}:
                        item = obj.item()
                        if isinstance(item, (bytes, bytearray)):
                            try:
                                item = pickle.loads(item)
                            except Exception:
                                item = None
                        if item is not None:
                            if isinstance(item, dict):
                                summary = _summarize_dict(item)
                                entry["items"][key]["decoded_dict_keys"] = summary["keys"]
                                entry["items"][key]["decoded_arrays_sample"] = summary["arrays_sample"]
                                all_label_values.extend(summary["label_values"])
                                for desc in summary["arrays_sample"].values():
                                    all_shapes.append(desc["shape"])
                            else:
                                all_label_values.extend(_extract_labels(key, item))
                    else:
                        all_label_values.extend(_extract_labels(key, obj))
                else:
                    entry["items"][key] = {"type": type(obj).__name__}
        samples.append(entry)

    label_counts = Counter(all_label_values)
    sample_unit, unit_reason = _infer_sample_unit(all_shapes)

    report = {
        "npz_dir": str(npz_dir),
        "sample_n": int(sample_n),
        "samples": samples,
        "label_values": sorted(set(all_label_values)),
        "label_counts": dict(label_counts),
        "npz_sample_unit": sample_unit,
        "npz_sample_unit_reason": unit_reason,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"[npz_schema] dir={npz_dir} samples={sample_n}", flush=True)
    print(f"[npz_schema] label_values={sorted(set(all_label_values))}", flush=True)
    print(f"[npz_schema] sample_unit={sample_unit} ({unit_reason})", flush=True)
    print(f"[npz_schema] report={out_path}", flush=True)


if __name__ == "__main__":
    main()
