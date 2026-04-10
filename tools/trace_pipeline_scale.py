from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.seed_raw_cnt import build_eeg62_view, load_one_raw
from manifold_raw.features import bandpass, parse_band_spec
from manifold_raw.scale_trace import print_stats


class Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _unit_summary(raw) -> str:
    unit = None
    unit_mul = None
    chs = raw.info.get("chs")
    if chs:
        unit = chs[0].get("unit")
        unit_mul = chs[0].get("unit_mul")
    orig_units = getattr(raw, "_orig_units", None)
    if isinstance(orig_units, dict):
        sample = list(orig_units.items())[:5]
        orig_units_summary = f"sample={sample} total={len(orig_units)}"
    else:
        orig_units_summary = str(orig_units)
    return f"unit={unit} unit_mul={unit_mul} orig_units={orig_units_summary}"


def _channel_flags(names) -> str:
    names_upper = [str(n).upper() for n in names]
    has_eog = any("EOG" in n for n in names_upper)
    has_ref = any(n in {"REF", "A1", "A2", "M1", "M2"} or "REF" in n for n in names_upper)
    return f"has_eog={has_eog} has_ref={has_ref}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="data/SEED/SEED_EEG/SEED_RAW_EEG/1_1.cnt",
    )
    parser.add_argument("--backend", type=str, default="cnt")
    parser.add_argument("--locs", type=str, default="data/SEED/channel_62_pos.locs")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument(
        "--bands",
        type=str,
        default="delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50",
    )
    parser.add_argument("--out", type=str, default="logs/trace_pipeline_scale_1_1.txt")
    parser.add_argument("--data-format", type=str, default=None)
    parser.add_argument("--filter-chunk", type=int, default=0)
    parser.add_argument("--resample-fs", type=float, default=0.0)
    args = parser.parse_args()

    path = _resolve_path(args.path)
    locs_path = _resolve_path(args.locs)
    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_one_raw(
        str(path),
        backend=args.backend,
        preload=False,
        data_format=args.data_format,
    )
    sfreq = float(raw.info.get("sfreq", 0.0))
    nchan = int(raw.info.get("nchan", len(raw.ch_names)))
    trace_stop = int(round(float(args.seconds) * sfreq)) if sfreq > 0 else 0

    with out_path.open("w", encoding="utf-8") as f:
        tee = Tee(sys.stdout, f)
        with contextlib.redirect_stdout(tee):
            print(f"[trace] path={path}")
            print(f"[trace] backend={args.backend} sfreq={sfreq} nchan={nchan}")
            print(f"[trace] seconds={args.seconds} trace_stop={trace_stop}")
            print(f"[trace] {_unit_summary(raw)}")
            if trace_stop <= 0:
                print("[trace] invalid trace window; check sfreq/seconds")
                return

            raw_slice = raw.get_data(start=0, stop=trace_stop)
            print(
                f"[scale][after_load_one_raw] dtype={raw_slice.dtype} "
                f"shape={raw_slice.shape}",
                flush=True,
            )
            print_stats("after_load_one_raw", raw_slice, force=True)

            raw62, meta = build_eeg62_view(raw, locs_path=str(locs_path))
            selected = meta.get("selected_names", [])
            raw62_slice = raw62.get_data(start=0, stop=trace_stop)
            print(
                f"[scale][after_build_eeg62_view] dtype={raw62_slice.dtype} "
                f"shape={raw62_slice.shape} selected={len(selected)} "
                f"{_channel_flags(selected)}",
                flush=True,
            )
            print_stats("after_build_eeg62_view", raw62_slice, force=True)

            trace_seg = raw62_slice
            fs_trace = sfreq
            if args.resample_fs and sfreq > 0:
                import mne

                trace_seg = mne.filter.resample(
                    np.asarray(trace_seg, dtype=np.float64),
                    up=float(args.resample_fs),
                    down=float(sfreq),
                    axis=-1,
                    npad="auto",
                    n_jobs=1,
                ).astype(np.float32, copy=False)
                fs_trace = float(args.resample_fs)
                print(f"[trace] resampled_fs={fs_trace}", flush=True)

            bands = parse_band_spec(args.bands)
            for band in bands:
                band_data = bandpass(
                    trace_seg,
                    fs_trace,
                    band,
                    chunk_size=int(args.filter_chunk or 0),
                )
                print_stats(f"after_bandpass_{band.name}", band_data, force=True)

            if hasattr(raw, "close"):
                raw.close()
            if hasattr(raw62, "close"):
                raw62.close()
            print(f"[trace] wrote {out_path}")


if __name__ == "__main__":
    main()
