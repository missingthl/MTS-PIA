from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


# Inter-patient split used by AAMI-style heartbeat classification benchmarks.
AAMI_DS1_RECORDS: Tuple[int, ...] = (
    101,
    106,
    108,
    109,
    112,
    114,
    115,
    116,
    118,
    119,
    122,
    124,
    201,
    203,
    205,
    207,
    208,
    209,
    215,
    220,
    223,
    230,
)

AAMI_DS2_RECORDS: Tuple[int, ...] = (
    100,
    103,
    105,
    111,
    113,
    117,
    121,
    123,
    200,
    202,
    210,
    212,
    213,
    214,
    219,
    221,
    222,
    228,
    231,
    232,
    233,
    234,
)


_AAMI_N = {"N", "L", "R", "e", "j"}
_AAMI_S = {"A", "a", "J", "S"}
_AAMI_V = {"V", "E"}
_AAMI_F = {"F"}
_AAMI_Q = {"/", "f", "Q"}


def _resolve_mitbih_root(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"MIT-BIH root not found: {p}")
    rec_file = p / "RECORDS"
    if not rec_file.is_file():
        raise FileNotFoundError(f"RECORDS file not found under {p}")
    return p


def _read_records_file(root: Path) -> List[int]:
    recs = []
    for line in (root / "RECORDS").read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        recs.append(int(s))
    return recs


def _build_label_map(class_scheme: str) -> Tuple[Dict[str, int], List[str]]:
    key = class_scheme.strip().lower()
    if key not in {"nsvf", "nsvfq"}:
        raise ValueError(f"Unsupported class_scheme: {class_scheme}")

    class_names = ["N", "S", "V", "F"] if key == "nsvf" else ["N", "S", "V", "F", "Q"]
    label_map: Dict[str, int] = {}

    for sym in _AAMI_N:
        label_map[sym] = 0
    for sym in _AAMI_S:
        label_map[sym] = 1
    for sym in _AAMI_V:
        label_map[sym] = 2
    for sym in _AAMI_F:
        label_map[sym] = 3
    if key == "nsvfq":
        for sym in _AAMI_Q:
            label_map[sym] = 4

    return label_map, class_names


def _load_record_wfdb(root: Path, record_id: int, n_leads: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    try:
        import wfdb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "wfdb is required for MIT-BIH preprocessing. Install in your env, e.g. `pip install wfdb`."
        ) from exc

    rec_name = str(record_id)
    rec = wfdb.rdrecord(str(root / rec_name))
    ann = wfdb.rdann(str(root / rec_name), "atr")

    if rec.p_signal is None:
        raise ValueError(f"wfdb returned empty p_signal for record {record_id}")

    sig = np.asarray(rec.p_signal, dtype=np.float32)  # [T, C]
    if sig.ndim != 2 or sig.shape[1] < 1:
        raise ValueError(f"Unexpected signal shape for record {record_id}: {sig.shape}")

    c = min(int(n_leads), int(sig.shape[1]))
    sig = sig[:, :c].T  # [C, T]
    ann_samples = np.asarray(ann.sample, dtype=np.int64).reshape(-1)
    ann_symbols = list(ann.symbol)
    if ann_samples.shape[0] != len(ann_symbols):
        raise ValueError(f"Annotation length mismatch for record {record_id}")
    return sig, ann_samples, ann_symbols


def _extract_beats_from_record(
    *,
    signal: np.ndarray,
    ann_samples: np.ndarray,
    ann_symbols: Sequence[str],
    record_id: int,
    split: str,
    label_map: Dict[str, int],
    pre_samples: int,
    post_samples: int,
    normalize_mode: str,
    max_beats_per_record: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, Counter]:
    n_ch, n_t = signal.shape
    win = int(pre_samples + post_samples)
    if win <= 0:
        raise ValueError("Invalid beat window length.")

    beats: List[np.ndarray] = []
    labels: List[int] = []
    tids: List[str] = []
    records: List[int] = []
    symbol_counter: Counter = Counter()

    for i, (sample, sym) in enumerate(zip(ann_samples.tolist(), ann_symbols)):
        label = label_map.get(sym)
        if label is None:
            continue

        start = int(sample) - int(pre_samples)
        end = int(sample) + int(post_samples)
        if start < 0 or end > n_t:
            continue

        x = signal[:, start:end].astype(np.float32, copy=True)  # [C, win]
        if x.shape[1] != win:
            continue

        if normalize_mode == "per_beat_zscore":
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True) + 1e-6
            x = (x - mu) / sd
        elif normalize_mode == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")

        beats.append(x)
        labels.append(int(label))
        tids.append(f"mitbih_{split}_r{int(record_id):03d}_a{i:06d}_s{int(sample):07d}")
        records.append(int(record_id))
        symbol_counter[sym] += 1

        if max_beats_per_record > 0 and len(labels) >= max_beats_per_record:
            break

    if not beats:
        return (
            np.empty((0, n_ch, win), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
            np.empty((0,), dtype=np.int32),
            symbol_counter,
        )

    return (
        np.stack(beats, axis=0).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int64),
        tids,
        np.asarray(records, dtype=np.int32),
        symbol_counter,
    )


def _concat_parts(parts: List[np.ndarray], axis: int = 0, dtype=np.float32) -> np.ndarray:
    if not parts:
        return np.empty((0,), dtype=dtype)
    return np.concatenate(parts, axis=axis)


def preprocess_mitbih(
    *,
    root: str | Path,
    out_npz: str | Path,
    class_scheme: str = "nsvf",
    pre_samples: int = 90,
    post_samples: int = 144,
    normalize_mode: str = "per_beat_zscore",
    n_leads: int = 2,
    max_beats_per_record: int = -1,
) -> Dict[str, object]:
    root_p = _resolve_mitbih_root(root)
    all_records = set(_read_records_file(root_p))
    train_records = [r for r in AAMI_DS1_RECORDS if r in all_records]
    test_records = [r for r in AAMI_DS2_RECORDS if r in all_records]

    label_map, class_names = _build_label_map(class_scheme)

    X_train_parts: List[np.ndarray] = []
    y_train_parts: List[np.ndarray] = []
    rec_train_parts: List[np.ndarray] = []
    tid_train: List[str] = []
    train_sym_counter: Counter = Counter()

    X_test_parts: List[np.ndarray] = []
    y_test_parts: List[np.ndarray] = []
    rec_test_parts: List[np.ndarray] = []
    tid_test: List[str] = []
    test_sym_counter: Counter = Counter()

    for rec_id in train_records:
        sig, ann_samp, ann_sym = _load_record_wfdb(root_p, rec_id, n_leads=n_leads)
        X, y, tid, recs, sym_count = _extract_beats_from_record(
            signal=sig,
            ann_samples=ann_samp,
            ann_symbols=ann_sym,
            record_id=rec_id,
            split="train",
            label_map=label_map,
            pre_samples=pre_samples,
            post_samples=post_samples,
            normalize_mode=normalize_mode,
            max_beats_per_record=max_beats_per_record,
        )
        X_train_parts.append(X)
        y_train_parts.append(y)
        rec_train_parts.append(recs)
        tid_train.extend(tid)
        train_sym_counter.update(sym_count)

    for rec_id in test_records:
        sig, ann_samp, ann_sym = _load_record_wfdb(root_p, rec_id, n_leads=n_leads)
        X, y, tid, recs, sym_count = _extract_beats_from_record(
            signal=sig,
            ann_samples=ann_samp,
            ann_symbols=ann_sym,
            record_id=rec_id,
            split="test",
            label_map=label_map,
            pre_samples=pre_samples,
            post_samples=post_samples,
            normalize_mode=normalize_mode,
            max_beats_per_record=max_beats_per_record,
        )
        X_test_parts.append(X)
        y_test_parts.append(y)
        rec_test_parts.append(recs)
        tid_test.extend(tid)
        test_sym_counter.update(sym_count)

    X_train = _concat_parts(X_train_parts, axis=0, dtype=np.float32)
    y_train = _concat_parts(y_train_parts, axis=0, dtype=np.int64)
    rec_train = _concat_parts(rec_train_parts, axis=0, dtype=np.int32)
    X_test = _concat_parts(X_test_parts, axis=0, dtype=np.float32)
    y_test = _concat_parts(y_test_parts, axis=0, dtype=np.int64)
    rec_test = _concat_parts(rec_test_parts, axis=0, dtype=np.int32)

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise RuntimeError("Processed arrays have unexpected rank; expected [N, C, T].")
    if X_train.shape[1] != X_test.shape[1] or X_train.shape[2] != X_test.shape[2]:
        raise RuntimeError("Train/test shape mismatch after preprocessing.")

    out_npz_p = Path(out_npz).expanduser().resolve()
    out_npz_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz_p,
        X_train=X_train,
        y_train=y_train,
        record_train=rec_train,
        tid_train=np.asarray(tid_train),
        X_test=X_test,
        y_test=y_test,
        record_test=rec_test,
        tid_test=np.asarray(tid_test),
        sfreq=np.asarray([360.0], dtype=np.float32),
    )

    class_counter_train = Counter(y_train.tolist())
    class_counter_test = Counter(y_test.tolist())
    meta = {
        "root": str(root_p),
        "out_npz": str(out_npz_p),
        "protocol": "aami_inter_patient",
        "class_scheme": class_scheme,
        "class_names": class_names,
        "train_records": train_records,
        "test_records": test_records,
        "pre_samples": int(pre_samples),
        "post_samples": int(post_samples),
        "window_len": int(pre_samples + post_samples),
        "normalize_mode": normalize_mode,
        "n_leads_used": int(X_train.shape[1] if X_train.shape[0] else X_test.shape[1]),
        "n_train_samples": int(y_train.shape[0]),
        "n_test_samples": int(y_test.shape[0]),
        "class_count_train": {str(k): int(v) for k, v in sorted(class_counter_train.items())},
        "class_count_test": {str(k): int(v) for k, v in sorted(class_counter_test.items())},
        "raw_symbol_count_train": dict(train_sym_counter),
        "raw_symbol_count_test": dict(test_sym_counter),
    }
    meta_path = out_npz_p.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIT-BIH beat-level preprocessing for MTS classification.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/MITBIH",
        help="MIT-BIH root containing RECORDS and .dat/.hea/.atr files.",
    )
    parser.add_argument(
        "--out-npz",
        type=str,
        default="data/MITBIH/processed/mitbih_aami44_nsvf_beats.npz",
    )
    parser.add_argument("--class-scheme", type=str, default="nsvf", choices=["nsvf", "nsvfq"])
    parser.add_argument("--pre-samples", type=int, default=90)
    parser.add_argument("--post-samples", type=int, default=144)
    parser.add_argument("--normalize-mode", type=str, default="per_beat_zscore", choices=["per_beat_zscore", "none"])
    parser.add_argument("--n-leads", type=int, default=2)
    parser.add_argument("--max-beats-per-record", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    meta = preprocess_mitbih(
        root=args.root,
        out_npz=args.out_npz,
        class_scheme=args.class_scheme,
        pre_samples=args.pre_samples,
        post_samples=args.post_samples,
        normalize_mode=args.normalize_mode,
        n_leads=args.n_leads,
        max_beats_per_record=args.max_beats_per_record,
    )
    print(f"[done] npz={meta['out_npz']}")
    print(f"[done] train={meta['n_train_samples']} test={meta['n_test_samples']}")
    print(f"[done] class_count_train={meta['class_count_train']}")
    print(f"[done] class_count_test={meta['class_count_test']}")


if __name__ == "__main__":
    main()

